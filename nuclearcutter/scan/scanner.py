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
import os
import sys
from pathlib import Path

from nuclearcutter.detection.profanity import detect_foul_language, load_wordlist
from nuclearcutter.detection.transcribe import find_subtitle_file, parse_subtitles, transcribe
from nuclearcutter.detection.vlm_confirm import DEFAULT_SWEEP_INTERVAL, VisualSweepDetector
from nuclearcutter.fingerprint.fingerprint import cache_fingerprint, compute_fingerprint, load_cached_fingerprint
from nuclearcutter.schema import FilmIdentity, ScanResult
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig
from nuclearcutter.utils.scan_status import ScanStatus


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
) -> ScanResult:
    llm_config = llm_config or LLMConfig()
    client = LLMClient(llm_config)
    client.test_connection()

    # Optional live status file for `nuclearcutter tui` (see utils/scan_status.py).
    status = None
    if status_path:
        from nuclearcutter.utils.scan_status import _now_iso

        status = ScanStatus(video=Path(video_path).name, pid=os.getpid(), started_at=_now_iso())
        status.sweep_interval = sweep_interval or DEFAULT_SWEEP_INTERVAL
        status_path = Path(status_path)

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

    if progress_callback:
        progress_callback("fingerprinting", None)
    _phase("fingerprinting")
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

    if progress_callback:
        progress_callback("transcribing", None)
    _phase("transcribing")
    utterances = transcribe(video_path, model=whisper_model)

    subtitle_path = find_subtitle_file(video_path)
    subtitle_utterances = parse_subtitles(subtitle_path) if subtitle_path else []

    # Whisper model is now out of scope; free its memory before loading VLM frames.
    gc.collect()

    # ------------------------------------------------------------------
    # Unified full-film VLM sweep (the only visual detector — no NudeNet).
    # One sweep pass catches nudity, gore, AND violence.
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("visual_sweep", None)
    _phase("visual_sweep")
    sweep_detector = VisualSweepDetector(client, prompts=category_prompts)

    def _on_flagged_window(start, end, category, confidence, level="med"):
        if status is not None:
            status.add_candidate(start, end, category, confidence, level)
            _write_status()

    def _on_sweep_progress(done, total):
        if status is not None:
            status.set_sweep(done, total)
            _write_status()

    sweep_ranges = sweep_detector.sweep(
        video_path,
        sample_interval=sweep_interval or DEFAULT_SWEEP_INTERVAL,
        on_flagged_window=_on_flagged_window,
        on_progress=_on_sweep_progress,
    )
    print(
        f"[visual_sweep] {len(sweep_ranges)} candidate range(s) from VLM sweep",
        file=sys.stderr,
    )

    visual_detections = []
    vlm_failures = 0
    _phase("visual_confirm")
    for i, candidate in enumerate(sweep_ranges):
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
        else:
            vlm_failures += 1
        if progress_callback:
            progress_callback("visual_confirm", (i + 1, len(sweep_ranges)))

    if sweep_ranges and vlm_failures == len(sweep_ranges):
        raise RuntimeError(
            f"All {vlm_failures} VLM confirmation queries failed. Cannot produce a reliable scan.\n"
            f"Check that your inference server is running and accessible at {llm_config.base_url}"
        )

    if progress_callback:
        progress_callback("language_detection", None)
    _phase("language_detection")
    wordlist = load_wordlist()
    foul_prompt = (category_prompts or {}).get("foul_language")
    language_detections = detect_foul_language(utterances, client, wordlist, subtitle_utterances, foul_language_prompt=foul_prompt)
    for d in language_detections:
        if status is not None:
            status.add_language_detection(d.word, d.start, d.end, d.utterance_start, d.utterance_end)
    _write_status()

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
