"""
Pass 1 of the two-pass architecture (docs/SPEC.md §2): run the whole detection
pipeline over a film and produce a ScanResult — a record of what is IN it.

Order, and why it is this order:

  fingerprint   cheap, and a shared scan file is matched by identity, not name.
  transcribe    whisper + subtitle cross-check. Everything language-related
                needs it, and it is the one artifact worth reusing verbatim
                across rescans (it is taxonomy-independent).
  language      the wordlist + LLM word pass and the dialogue sweep, over the
                transcript. Runs before the visual work so the timeline fills
                with language markers while the slow part is still running.
  sweep         the cheap full-film trigger pass. Decides WHERE to look.
  identify      one classify call per candidate range. Decides WHAT is there.

The scan writes no opinions: no severity, no "should this be censored". It
emits tag ids from the fixed vocabulary (docs/TAXONOMY.md) plus a plain-language
description, so the same file serves a viewer who censors everything and one who
censors nothing.

Everything is resumable and crash-tolerant: the status file tracks progress and
the partial scan file next to the video is rewritten after every detection, so
an interrupted run never loses more than the segment it was working on.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
from pathlib import Path

from nuclearcutter.detection.language import detect_language, load_wordlist
from nuclearcutter.detection.transcribe import (
    TranscriptionStopped, find_subtitle_file, parse_subtitles, read_transcript_cache,
    transcribe, transcribe_killable, write_transcript_cache,
)
from nuclearcutter.detection.visual_scan import (
    DEFAULT_SWEEP_INTERVAL, CandidateRange, SweepWindow, VisualScanDetector,
)
from nuclearcutter.fingerprint.fingerprint import (
    cache_fingerprint, compute_fingerprint, load_cached_fingerprint,
)
from nuclearcutter.schema import (
    SCHEMA_VERSION, FilmIdentity, LanguageDetection, ScanResult, VisualDetection,
)
from nuclearcutter.taxonomy import Minors, Rendering
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig
from nuclearcutter.utils.model_server import WhisperConfig
from nuclearcutter.utils.scan_status import ScanStatus


def _load_resume_state(status_path: Path | str | None, video_name: str) -> ScanStatus | None:
    """Load an in-progress status file for `video_name`, or None if none exists.

    Used to resume an interrupted scan. A COMPLETED scan's status file is also
    resumable: each phase is skipped when its saved progress already shows it
    done (transcript cache present, sweep frames complete, every candidate
    classified), so clearing just one section in the GUI re-runs only that piece
    instead of the whole pipeline.
    """
    if status_path is None:
        return None
    sp = Path(status_path)
    if not sp.exists():
        return None
    try:
        st = ScanStatus.load(sp)
    except (OSError, ValueError):
        return None
    # Only resume a status file that belongs to THIS video. Status files are
    # named by video stem, and two films can share a stem — never resume across
    # films, or one film's detections land on another's timeline.
    if st.video and st.video != video_name:
        return None
    # RESUME ONLY A RUN FROM THIS EXACT SCHEMA — in both directions.
    #
    # "Newer than us" is obviously unsafe; "older" is the sneaky one, because it
    # looks fine: a v1 status file happily reports frames_done=50/50 from a sweep
    # that asked a different question with a different vocabulary. Resuming it
    # declares footage swept that this taxonomy never looked at, every detection
    # in that stretch is silently missing, and the finished scan reads as clean.
    # Note this reads the file, not `st.to_dict()` — to_dict() stamps the CURRENT
    # schema and would make this check always pass.
    try:
        on_disk = json.loads(sp.read_text()).get("schema", 0)
    except (OSError, ValueError):
        return None
    if int(on_disk or 0) != ScanStatus.SCHEMA:
        if int(on_disk or 0):
            print(f"[resume] {sp.name} is schema v{on_disk} and this build is v"
                  f"{ScanStatus.SCHEMA}, so its progress does not describe the same "
                  f"search — scanning from the beginning instead of resuming. "
                  f"(A scan made with the old categories did not look for the ones "
                  f"this build knows about.)", file=sys.stderr, flush=True)
        return None
    return st


def _window_from_candidate(c: dict) -> SweepWindow | None:
    """Rebuild a raw sweep window from a saved status record (resume path)."""
    try:
        return SweepWindow(
            start=float(c["start"]), end=float(c["end"]),
            triggers=[str(x) for x in (c.get("triggers") or ["doubt"])],
            rendering=str(c.get("rendering") or Rendering.LIVE_ACTION.value),
            subjects={"adult_female": 0, "adult_male": 0,
                      "minors": str(c.get("minors") or Minors.NO.value)},
            confidence=float(c.get("confidence", 0.5)))
    except (KeyError, TypeError, ValueError):
        return None


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (tmp + rename) so a reader never sees partial data.

    If the target dir is unwritable (e.g. the SMB share dropped), raise OSError —
    the caller decides whether to swallow it.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


class ScanStopped(RuntimeError):
    """Raised when the user hits Stop — progress has been saved to the status
    file and partial result, and can be resumed by starting the scan again."""


def scan(
    video_path: Path,
    llm_config: LLMConfig | None = None,
    title: str | None = None,
    year: int | None = None,
    progress_callback=None,
    whisper_cfg: WhisperConfig | None = None,
    sweep_interval: float | None = None,
    sweep_resolution: str | None = None,
    identify_scale: str | None = None,
    status_path: Path | str | None = None,
    partial_result_path: Path | str | None = None,
    scale: str | None = None,
    stop_event: threading.Event | None = None,
    run_dialogue_sweep: bool = True,
    client_factory=None,
) -> ScanResult:
    """Scan a film end to end. Returns the ScanResult (also saved by the caller).

    `progress_callback(name, detail)` drives the GUI/TUI bars:
      ("transcribing", frac) ("language", (done,total)) ("visual_sweep",
      (done,total)) ("visual_identify", (done,total)) plus ("candidate", {...})
      and ("visual_detection", {...}) / ("language_detection", [...]) markers.
    """
    llm_config = llm_config or LLMConfig()
    client = (client_factory(llm_config) if client_factory is not None
              else LLMClient(llm_config))
    client.test_connection()

    status = None
    resume = _load_resume_state(status_path, Path(video_path).name)
    resume_from = 0
    existing_windows: list[SweepWindow] = []
    identified_keys: set = set()
    resume_visual: list[VisualDetection] = []
    resume_language: list[LanguageDetection] = []
    if resume is not None:
        resume_from = resume.frames_done or 0
        # frames_done counts SAMPLES at the interval the interrupted run used.
        # Resuming at a DIFFERENT interval must preserve the swept POSITION in
        # seconds, not the sample count — otherwise doubling the sweep interval
        # between runs (a common speed-up move) would resume twice as deep and
        # silently skip un-swept footage.
        eff_interval = sweep_interval if sweep_interval is not None else DEFAULT_SWEEP_INTERVAL
        saved_interval = resume.sweep_interval or 0.0
        if resume_from and saved_interval and abs(saved_interval - eff_interval) > 1e-6:
            rescaled = int(round(resume_from * saved_interval / eff_interval))
            print(f"  sweep interval changed ({saved_interval:g}s -> {eff_interval:g}s): "
                  f"resuming at frame {rescaled} to keep position "
                  f"~{resume_from * saved_interval:.0f}s swept", file=sys.stderr)
            resume_from = rescaled
        for c in resume.visual_candidates:
            w = _window_from_candidate(c)
            if w is not None:
                existing_windows.append(w)
        # Detections already classified in a prior run — do not re-spend a
        # classify call on them. Keyed on the merged RANGE, not on tags: if a
        # re-scan gives different tags for the same range, that is a new answer
        # we want, so the key is span-only.
        for d in resume.visual_detections:
            key = (round(float(d["start"]), 1), round(float(d["end"]), 1))
            identified_keys.add(key)
            resume_visual.append(VisualDetection.from_dict(d))
        for d in resume.language_detections:
            resume_language.append(LanguageDetection.from_dict(d))
        print(f"[resume] loaded saved scan for {Path(video_path).name} from "
              f"{Path(status_path).name} — continuing from frame {resume_from}/"
              f"{resume.frames_total or '?'} ({len(resume_visual)} visual, "
              f"{len(resume_language)} language)", file=sys.stderr)

    if status_path:
        from nuclearcutter.utils.scan_status import _now_iso
        status = ScanStatus(video=Path(video_path).name, pid=os.getpid(),
                            started_at=_now_iso())
        status.sweep_interval = sweep_interval or DEFAULT_SWEEP_INTERVAL
        status_path = Path(status_path)
        if resume is not None:
            status.frames_done = resume.frames_done
            status.frames_total = resume.frames_total
            status.position_seconds = resume.position_seconds
            status.visual_candidates = list(resume.visual_candidates)
            status.visual_detections = list(resume.visual_detections)
            status.language_detections = list(resume.language_detections)

    def _write_status():
        if status is not None:
            try:
                status.write(status_path)
            except Exception as exc:  # noqa: BLE001 — a status write must never kill a scan
                print(f"warning: status write failed: {exc}", file=sys.stderr)

    def _phase(phase: str):
        if status is not None:
            status.set_phase(phase)
            _write_status()

    # Partial result: keeps a usable `<movie>.nuclearcutter.json` next to the
    # video, refreshed as detections are found, so an interrupted scan's file is
    # never empty. Written atomically.
    partial_path = Path(partial_result_path) if partial_result_path else None

    def _write_partial_result():
        if partial_path is None:
            return
        try:
            partial = ScanResult(
                schema_version=SCHEMA_VERSION,
                identity=identity,
                visual_detections=list(visual_detections),
                language_detections=list(language_detections),
                generator=_generator_info(llm_config, partial=True))
            _atomic_write_json(partial_path, partial.to_dict())
        except OSError as exc:
            # Never let a save failure kill the scan; the CLI's final save
            # retries with its own fallback logic.
            print(f"warning: could not write partial scan result: {exc}", file=sys.stderr)

    def _stopped(when: str):
        return stop_event is not None and stop_event.is_set()

    # ------------------------------------------------------------------ identity
    if progress_callback:
        progress_callback("fingerprinting", None)
    _phase("fingerprinting")
    if _stopped("before starting"):
        raise ScanStopped("scan stopped before starting")
    cached = load_cached_fingerprint(video_path)
    if cached:
        duration, phash_samples = cached
        print(f"[fingerprinting] loaded cached fingerprint ({len(phash_samples)} "
              f"samples, {duration:.0f}s)")
    else:
        duration, phash_samples = compute_fingerprint(video_path)
        cache_fingerprint(video_path, duration, phash_samples)
    identity = FilmIdentity(title=title, year=year, duration_seconds=duration,
                            phash_samples=[s.to_dict() for s in phash_samples])
    if status is not None:
        status.duration_seconds = duration
        _write_status()

    visual_detections: list[VisualDetection] = list(resume_visual)
    language_detections: list[LanguageDetection] = list(resume_language)
    _write_partial_result()

    # --------------------------------------------------------------- transcript
    if progress_callback:
        progress_callback("transcribing", None)
    _phase("transcribing")
    if _stopped("before transcription"):
        raise ScanStopped("scan stopped before transcription")

    transcript_path = Path(video_path).with_suffix(".nuclearcutter.transcript.json")
    utterances = read_transcript_cache(transcript_path, video_path)
    if utterances is not None:
        if progress_callback:
            progress_callback("transcribing", 1.0)
        print(f"[transcribing] loaded saved transcript for {Path(video_path).name} "
              f"from {transcript_path.name} ({len(utterances)} utterances) — skipping "
              f"whisper", file=sys.stderr)
    else:
        if transcript_path.exists():
            print("[transcribing] saved transcript found but the video changed "
                  "(size/mtime) — re-transcribing", file=sys.stderr)

        def _whisper_progress(frac):
            if progress_callback:
                progress_callback("transcribing", frac)

        try:
            if stop_event is not None:
                # GUI scans run whisper in a killable child process so Stop
                # actually stops it mid-transcription.
                utterances = transcribe_killable(video_path, whisper_cfg,
                                                 progress_callback=_whisper_progress,
                                                 stop_event=stop_event)
            else:
                utterances = transcribe(video_path, whisper_cfg,
                                        progress_callback=_whisper_progress)
        except TranscriptionStopped as exc:
            raise ScanStopped(str(exc)) from exc
        write_transcript_cache(transcript_path, video_path, utterances)
        # whisper skips silent windows, so its last reported fraction can be
        # below 1.0 even though it finished — force the bar closed.
        if progress_callback:
            progress_callback("transcribing", 1.0)

    subtitle_path = find_subtitle_file(video_path)
    subtitle_utterances = parse_subtitles(subtitle_path) if subtitle_path else []

    # Whisper is out of scope now; free its memory before the VLM starts holding
    # frames. (On a 20 GB box this is the difference between scanning and OOM.)
    gc.collect()

    # ---------------------------------------------------------------- language
    if progress_callback:
        progress_callback("language_detection", None)
    _phase("language")
    if _stopped("before language detection"):
        _write_status()
        _write_partial_result()
        raise ScanStopped("scan stopped before language detection")
    if resume_language:
        language_detections = list(resume_language)
    else:
        def _language_progress(done: int, total: int):
            if progress_callback:
                progress_callback("language", (done, total))

        language_detections = detect_language(
            utterances, client, load_wordlist(), subtitle_utterances,
            run_dialogue_sweep=run_dialogue_sweep, on_progress=_language_progress)
        for d in language_detections:
            if status is not None:
                status.add_language_detection(d.text, d.start, d.end, d.utterance_start,
                                              d.utterance_end, d.tags)
    if progress_callback:
        progress_callback("language_detections", [d.to_dict() for d in language_detections])
    _write_status()
    _write_partial_result()

    # ------------------------------------------------------------------- sweep
    if progress_callback:
        progress_callback("visual_sweep", None)
    _phase("sweep")
    detector = VisualScanDetector(client, scale=scale, sweep_resolution=sweep_resolution,
                                  identify_scale=identify_scale)

    def _on_flagged_window(start, end, triggers, confidence):
        if status is not None:
            status.add_candidate(start, end, triggers, confidence)
            _write_status()
        if progress_callback:
            progress_callback("candidate", {"start": start, "end": end,
                                            "triggers": list(triggers),
                                            "confidence": confidence})

    def _on_sweep_progress(done, total):
        if status is not None:
            status.set_sweep(done, total)
            _write_status()
        if progress_callback:
            progress_callback("visual_sweep", (done, total))

    candidates, stats = detector.sweep(
        video_path, sample_interval=sweep_interval or DEFAULT_SWEEP_INTERVAL,
        on_flagged_window=_on_flagged_window, on_progress=_on_sweep_progress,
        resume_from=resume_from, existing_windows=existing_windows,
        stop_event=stop_event)
    print(f"[sweep] {len(candidates)} candidate range(s) from "
          f"{stats.batches_queried} batch(es)", file=sys.stderr)

    if (stats.batches_queried and stats.batches_failed == stats.batches_queried):
        # Every VLM query failed. The candidates above are all 'doubt' fill, so
        # the identify pass is about to do real work on the whole film — and if
        # the server is truly down it will fail too. Stop here with a clear
        # message instead of producing a garbage scan file.
        raise RuntimeError(
            f"VLM sweep failed: all {stats.batches_failed} batch(es) returned "
            f"unparseable or empty responses. The model server is not answering "
            f"(is the model loaded, with vision support?) — check {llm_config.base_url}")

    # ---------------------------------------------------------------- identify
    if progress_callback:
        progress_callback("visual_identify", None)
    _phase("identify")
    if _stopped("before identify"):
        _write_status()
        _write_partial_result()
        raise ScanStopped("scan stopped before identify")

    invented_tags: list[str] = []
    query_failures = 0
    unclassified = 0
    for i, candidate in enumerate(candidates):
        if stop_event is not None and stop_event.is_set():
            _write_status()
            _write_partial_result()
            raise ScanStopped("scan stopped during identify")
        key = (round(candidate.start, 1), round(candidate.end, 1))
        if key in identified_keys:
            if progress_callback:
                progress_callback("visual_identify", (i + 1, len(candidates)))
            continue
        dialogue = _dialogue_in_range(utterances, candidate.start, candidate.end)
        detection, classified_ok = detector.identify(video_path, candidate, dialogue,
                                                     invented_tags=invented_tags)
        if not classified_ok:
            query_failures += 1
        visual_detections.append(detection)
        if "safety.unclassified" in detection.tags:
            unclassified += 1
        if status is not None:
            status.add_visual_detection(
                detection.start, detection.end, detection.tags,
                detection.description or "", detection.confidence,
                detection.rendering, detection.minors)
            _write_status()
        if progress_callback:
            progress_callback("visual_detection", detection.to_dict())
        _write_partial_result()
        if progress_callback:
            progress_callback("visual_identify", (i + 1, len(candidates)))

    if candidates and query_failures == len(candidates):
        raise RuntimeError(
            f"All {query_failures} classify queries failed. Cannot produce a "
            f"reliable scan. Check that your inference server is running and "
            f"accessible at {llm_config.base_url}")

    if invented_tags:
        uniq = sorted(set(invented_tags))
        print(f"[identify] NOTE: the model invented {len(invented_tags)} tag id(s) that are "
              f"not in the vocabulary ({len(uniq)} distinct). They were folded into their "
              f"family's 'unclear' category so those scenes stay censorable. Distinct ids: "
              f"{', '.join(uniq[:12])}{'...' if len(uniq) > 12 else ''}", file=sys.stderr)
    if candidates:
        print(f"[identify] {len(visual_detections)} visual detection(s), "
              f"{unclassified} of them unclassifiable", file=sys.stderr)

    result = ScanResult(
        schema_version=SCHEMA_VERSION, identity=identity,
        visual_detections=visual_detections,
        language_detections=language_detections,
        generator=_generator_info(llm_config))

    if status is not None:
        status.set_phase("done")
        if status.position_seconds is None and duration:
            status.position_seconds = duration
        _write_status()

    if progress_callback:
        progress_callback("done", None)

    return result


def _generator_info(llm_config: LLMConfig, partial: bool = False) -> dict:
    from nuclearcutter import __version__ as nc_version
    from nuclearcutter.taxonomy import TAXONOMY_VERSION
    info = {
        "nuclearcutter": nc_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "vlm_model": llm_config.vlm_model,
        "text_model": llm_config.text_model,
    }
    if partial:
        info["partial"] = True
    return info


def _dialogue_in_range(utterances, start: float, end: float) -> str:
    lines = [u.text for u in utterances if u.start < end and u.end > start]
    return " ".join(lines)
