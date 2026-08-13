"""
Pass 1 of the two-pass architecture (docs/SPEC.md section 2): orchestrates
the full detection pipeline (unified VLM visual sweep for nudity/gore/
violence, Whisper transcription, subtitle cross-check, profanity wordlist +
LLM check) and produces a ScanResult.

Visual detection is a single full-film VLM sweep (see detection/vlm_confirm.py)
— NudeNet was removed because it missed a real nude scene entirely; the VLM
sweep is the only visual detector and catches nudity, gore, and violence in
one pass.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
from pathlib import Path

from nuclearcutter.detection.profanity import detect_foul_language, load_wordlist
from nuclearcutter.detection.transcribe import find_subtitle_file, parse_subtitles, transcribe
from nuclearcutter.detection.vlm_confirm import DEFAULT_SWEEP_INTERVAL, VisualSweepDetector
from nuclearcutter.fingerprint.fingerprint import cache_fingerprint, compute_fingerprint, load_cached_fingerprint
from nuclearcutter.schema import (
    Category, FilmIdentity, LanguageDetection, ScanResult, SeverityLevel, VisualDetection,
)
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig
from nuclearcutter.utils.scan_status import ScanStatus


def _load_resume_state(status_path: Path | str | None, video_name: str) -> ScanStatus | None:
    """Load an in-progress status file for `video_name`, or None if none exists.

    Used to resume an interrupted scan. A COMPLETED scan's status file is also
    resumable: each phase is skipped when its saved progress already shows it
    done (transcribe cache present, sweep frames complete, every candidate
    confirmed), so clearing just one section (e.g. transcription) re-runs only
    that piece on the next start instead of the whole pipeline."""
    if not status_path:
        return None
    sp = Path(status_path)
    if not sp.exists():
        return None
    try:
        st = ScanStatus.load(sp)
    except Exception:
        return None
    # Only resume a status file that belongs to THIS video. (The CLI names
    # status files by video stem, but a different video could share the same
    # stem — never resume across films.)
    if st.video and st.video != video_name:
        return None
    return st


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (tmp + rename) so a reader never sees partial data.
    Mirrors ScanStatus.write(). If the target dir is unwritable (e.g. the SMB
    share dropped), raise OSError — the caller decides whether to swallow it."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


class ScanStopped(RuntimeError):
    """Raised when the user hits Stop — progress has been saved to the status
    file and partial result, and can be resumed by starting the scan again."""


def scan(
    video_path: Path,
    llm_config: LLMConfig = None,
    title: str = None,
    year: int = None,
    progress_callback=None,
    whisper_model: str = None,
    sweep_interval: float = None,
    status_path: Path | str = None,
    category_prompts: dict = None,
    partial_result_path: Path | str = None,
    scale: str = None,
    stop_event: threading.Event = None,
) -> ScanResult:
    llm_config = llm_config or LLMConfig()
    client = LLMClient(llm_config)
    client.test_connection()

    # Optional live status file for `nuclearcutter tui` (see utils/scan_status.py).
    status = None
    # Resume state from a prior interrupted run (if any).
    resume = _load_resume_state(status_path, Path(video_path).name)
    resume_from = 0
    existing_windows: list = []
    confirmed_keys: set = set()
    resume_visual: list[VisualDetection] = []
    resume_language: list[LanguageDetection] = []
    if resume is not None:
        resume_from = resume.frames_done or 0
        # Raw sweep windows already found (category/start/end/confidence/level),
        # fed back into the merge so candidate ranges stay complete.
        for c in resume.visual_candidates:
            existing_windows.append((
                float(c["start"]), float(c["end"]),
                Category.from_legacy(c["category"]), "",
                float(c.get("confidence", 0.5)),
                SeverityLevel.from_any(c.get("level")),
            ))
        # Confirmed detections from a prior run — skip re-confirming these.
        for d in resume.visual_detections:
            key = (d["category"], round(float(d["start"]), 1), round(float(d["end"]), 1))
            confirmed_keys.add(key)
            resume_visual.append(VisualDetection(
                category=Category.from_legacy(d["category"]),
                start=float(d["start"]), end=float(d["end"]),
                description=d.get("description", ""),
                confidence=float(d.get("confidence", 0.5)),
                level=SeverityLevel.from_any(d.get("level")),
                stage_a_score=None,
            ))
        # Language detections already found — skip re-running the LLM pass.
        for d in resume.language_detections:
            resume_language.append(LanguageDetection(
                start=float(d["start"]), end=float(d["end"]),
                utterance_start=float(d.get("utterance_start", d["start"])),
                utterance_end=float(d.get("utterance_end", d["end"])),
                word=d.get("word", ""),
                transcript_source="whisper",
                llm_confirmed=True,
                level=SeverityLevel.MED,
            ))
        print(
            f"[resume] loaded saved scan for {Path(video_path).name} from "
            f"{Path(status_path).name} — continuing from frame {resume_from}/"
            f"{resume.frames_total or '?'} "
            f"({len(resume_visual)} visual confirmed, {len(resume_language)} language)…",
            file=sys.stderr,
        )

    if status_path:
        from nuclearcutter.utils.scan_status import _now_iso

        status = ScanStatus(video=Path(video_path).name, pid=os.getpid(), started_at=_now_iso())
        status.sweep_interval = sweep_interval or DEFAULT_SWEEP_INTERVAL
        status_path = Path(status_path)
        # Seed the fresh status object with prior progress so the live file/TUI
        # keeps showing what's already been found across a resume.
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
            except Exception as exc:  # never let status-write failures kill the scan
                print(f"warning: status write failed: {exc}", file=sys.stderr)

    def _phase(phase: str):
        if status is not None:
            status.set_phase(phase)
            _write_status()

    # Partial result writer: keeps a usable `<movie>.nuclearcutter.json` next to
    # the video, refreshed as detections are found. So even if the scan is
    # interrupted (crash, Ctrl-C, share drop), the movie-folder scan file is
    # never empty — it holds everything confirmed so far. Written atomically.
    partial_path = Path(partial_result_path) if partial_result_path else None

    def _write_partial_result():
        if partial_path is None:
            return
        try:
            partial = ScanResult(
                schema_version=1,
                identity=identity,
                visual_detections=list(visual_detections),
                language_detections=list(language_detections),
                generator={
                    "vlm_model": llm_config.vlm_model,
                    "text_model": llm_config.text_model,
                    "partial": True,
                },
            )
            _atomic_write_json(partial_path, partial.to_dict())
        except OSError as exc:
            # Never let a save failure kill the scan; the final save in the CLI
            # will retry properly with its own fallback logic.
            print(f"warning: could not write partial scan result: {exc}", file=sys.stderr)

    if progress_callback:
        progress_callback("fingerprinting", None)
    _phase("fingerprinting")
    if stop_event is not None and stop_event.is_set():
        raise ScanStopped("scan stopped before starting")
    cached = load_cached_fingerprint(video_path)
    if cached:
        duration, phash_samples = cached
        print(f"[fingerprinting] loaded cached fingerprint ({len(phash_samples)} samples, {duration:.0f}s)")
    else:
        duration, phash_samples = compute_fingerprint(video_path)
        cache_fingerprint(video_path, duration, phash_samples)
    identity = FilmIdentity(
        title=title, year=year, duration_seconds=duration,
        phash_samples=[s.to_dict() for s in phash_samples],
    )
    if status is not None:
        status.duration_seconds = duration
        _write_status()

    # Emit an early partial result (identity only, plus anything already
    # confirmed from a prior resumed run) so the movie-folder scan file exists
    # from the start of the run.
    visual_detections = list(resume_visual)
    language_detections = list(resume_language)
    _write_partial_result()

    if progress_callback:
        progress_callback("transcribing", None)
    _phase("transcribing")
    if stop_event is not None and stop_event.is_set():
        raise ScanStopped("scan stopped before transcription")

    from nuclearcutter.detection.transcribe import (
        TranscriptionStopped, read_transcript_cache, transcribe_killable,
        write_transcript_cache,
    )

    # Resume-friendly transcription: if we already transcribed THIS exact file
    # (same size + mtime), reuse it instead of running whisper again.
    transcript_path = Path(video_path).with_suffix(".nuclearcutter.transcript.json")
    utterances = read_transcript_cache(transcript_path, video_path)
    if utterances is not None:
        if progress_callback:
            progress_callback("transcribing", 1.0)
        print(
            f"[transcribing] loaded saved transcript for {Path(video_path).name} "
            f"from {transcript_path.name} ({len(utterances)} utterances) — skipping whisper",
            file=sys.stderr,
        )
    else:
        if transcript_path.exists():
            print(
                "[transcribing] saved transcript found but the video changed "
                "(size/mtime) — re-transcribing",
                file=sys.stderr,
            )

        def _whisper_progress(frac):
            if progress_callback:
                progress_callback("transcribing", frac)

        try:
            if stop_event is not None:
                # Interactive (GUI) scans run whisper in a killable child
                # process so Stop actually stops it mid-transcription.
                utterances = transcribe_killable(
                    video_path, model=whisper_model,
                    progress_callback=_whisper_progress, stop_event=stop_event,
                )
            else:
                utterances = transcribe(
                    video_path, model=whisper_model, progress_callback=_whisper_progress
                )
        except TranscriptionStopped as exc:
            raise ScanStopped(str(exc)) from exc
        write_transcript_cache(transcript_path, video_path, utterances)
        # whisper skips progress updates for silent windows, so its final
        # reported fraction can be below 100% even though it finished — force
        # the bar to 100% now that transcription is actually done.
        if progress_callback:
            progress_callback("transcribing", 1.0)

    subtitle_path = find_subtitle_file(video_path)
    subtitle_utterances = parse_subtitles(subtitle_path) if subtitle_path else []

    # Whisper model is now out of scope; free its memory before loading VLM frames.
    gc.collect()

    # ------------------------------------------------------------------
    # Foul-language detection (right after transcription, so language marks
    # appear on the timeline early — it's independent of the visual sweep).
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("language_detection", None)
    _phase("language_detection")
    if stop_event is not None and stop_event.is_set():
        _write_status()
        _write_partial_result()
        raise ScanStopped("scan stopped before language detection")
    if resume_language:
        # Already completed this pass in a prior run.
        language_detections = list(resume_language)
    else:
        wordlist = load_wordlist()
        foul_prompt = (category_prompts or {}).get("foul_language")
        language_detections = detect_foul_language(utterances, client, wordlist, subtitle_utterances, foul_language_prompt=foul_prompt)
        for d in language_detections:
            if status is not None:
                status.add_language_detection(d.word, d.start, d.end, d.utterance_start, d.utterance_end)
    if progress_callback:
        progress_callback("language_detections", [d.to_dict() for d in language_detections])
    _write_status()
    _write_partial_result()

    # ------------------------------------------------------------------
    # Unified full-film VLM sweep (the only visual detector — no NudeNet).
    # One sweep pass catches nudity, gore, AND violence.
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("visual_sweep", None)
    _phase("visual_sweep")
    sweep_detector = VisualSweepDetector(client, prompts=category_prompts, scale=scale)

    def _on_flagged_window(start, end, category, confidence, level="low"):
        if status is not None:
            status.add_candidate(start, end, category, confidence, level)
            _write_status()
        if progress_callback:
            progress_callback("candidate", {
                "start": start, "end": end,
                "category": category, "confidence": confidence, "level": level,
            })

    def _on_sweep_progress(done, total):
        if status is not None:
            status.set_sweep(done, total)
            _write_status()
        if progress_callback:
            progress_callback("visual_sweep", (done, total))

    sweep_ranges = sweep_detector.sweep(
        video_path,
        sample_interval=sweep_interval or DEFAULT_SWEEP_INTERVAL,
        on_flagged_window=_on_flagged_window,
        on_progress=_on_sweep_progress,
        resume_from=resume_from,
        existing_windows=existing_windows,
        stop_event=stop_event,
    )
    print(
        f"[visual_sweep] {len(sweep_ranges)} candidate range(s) from VLM sweep",
        file=sys.stderr,
    )

    vlm_failures = 0
    _phase("visual_confirm")
    for i, candidate in enumerate(sweep_ranges):
        if stop_event is not None and stop_event.is_set():
            _write_status()
            _write_partial_result()
            raise ScanStopped("scan stopped during visual confirm")
        # Skip candidates already confirmed in a prior run.
        key = (candidate.category.value, round(candidate.start, 1), round(candidate.end, 1))
        if key in confirmed_keys:
            if progress_callback:
                progress_callback("visual_confirm", (i + 1, len(sweep_ranges)))
            continue
        dialogue = _dialogue_in_range(utterances, candidate.start, candidate.end)
        detection = sweep_detector.confirm_and_describe(video_path, candidate, dialogue)
        if detection:
            visual_detections.append(detection)
            if status is not None:
                status.add_visual_detection(
                    detection.category, detection.start, detection.end,
                    detection.description or "", detection.confidence,
                    detection.level.value,
                )
                _write_status()
            if progress_callback:
                progress_callback("visual_detection", detection.to_dict())
            _write_partial_result()  # refresh the movie-folder scan file
        else:
            vlm_failures += 1
        if progress_callback:
            progress_callback("visual_confirm", (i + 1, len(sweep_ranges)))

    if sweep_ranges and vlm_failures == len(sweep_ranges):
        raise RuntimeError(
            f"All {vlm_failures} VLM confirmation queries failed. Cannot produce a reliable scan.\n"
            f"Check that your inference server is running and accessible at {llm_config.base_url}"
        )

    result = ScanResult(
        schema_version=1,
        identity=identity,
        visual_detections=visual_detections,
        language_detections=language_detections,
        generator={
            "vlm_model": llm_config.vlm_model,
            "text_model": llm_config.text_model,
        },
    )

    if status is not None:
        status.set_phase("done")
        if status.position_seconds is None and duration:
            status.position_seconds = duration
        _write_status()

    if progress_callback:
        progress_callback("done", None)

    return result


def _dialogue_in_range(utterances, start: float, end: float) -> str:
    lines = [u.text for u in utterances if u.start < end and u.end > start]
    return " ".join(lines)
