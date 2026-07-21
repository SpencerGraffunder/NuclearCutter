"""
Pass 1 of the two-pass architecture (docs/SPEC.md section 2): orchestrates
the full detection pipeline (Stage A/B nudity+intimate detection, Whisper
transcription, subtitle cross-check, profanity wordlist + LLM check) and
produces a ScanResult.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

from nuclearcutter.detection.nsfw_classifier import NsfwClassifier, CandidateRange
from nuclearcutter.detection.profanity import detect_foul_language, load_wordlist
from nuclearcutter.detection.transcribe import find_subtitle_file, parse_subtitles, transcribe
from nuclearcutter.detection.vlm_confirm import VlmConfirmer
from nuclearcutter.fingerprint.fingerprint import cache_fingerprint, compute_fingerprint, load_cached_fingerprint
from nuclearcutter.schema import FilmIdentity, ScanResult
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig


def scan(
    video_path: Path,
    llm_config: LLMConfig = None,
    title: str = None,
    year: int = None,
    progress_callback=None,
    whisper_model: str = None,
) -> ScanResult:
    llm_config = llm_config or LLMConfig()
    client = LLMClient(llm_config)
    client.test_connection()

    if progress_callback:
        progress_callback("fingerprinting", None)
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

    if progress_callback:
        progress_callback("nsfw_stage_a", None)

    stage_a_results_file = video_path.with_suffix(".stage_a_results.json")

    # Check if Stage A already completed in a prior run.
    if stage_a_results_file.exists():
        try:
            saved = json.loads(stage_a_results_file.read_text())
            candidates = [CandidateRange(**r) for r in saved["ranges"]]
            print(
                f"[nsfw_stage_a] loaded {len(candidates)} candidate ranges from "
                f"{stage_a_results_file.name}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"Warning: corrupt Stage A results ({exc}), re-scanning.",
                file=sys.stderr,
            )
            stage_a_results_file.unlink(missing_ok=True)
            candidates = None
    else:
        candidates = None

    if candidates is None:
        classifier = NsfwClassifier()
        candidates = classifier.scan(
            video_path,
            progress_callback=lambda t, total: progress_callback("nsfw_stage_a", (t, total)) if progress_callback else None,
        )
        # Save results immediately so later stages can crash without losing
        # Stage A progress.
        stage_a_results_file.write_text(json.dumps(
            {"ranges": [
                {"start": r.start, "end": r.end, "peak_score": r.peak_score}
                for r in candidates
            ]}
        ))
        del classifier
        gc.collect()

    if progress_callback:
        progress_callback("transcribing", None)
    utterances = transcribe(video_path, model=whisper_model) if whisper_model else transcribe(video_path)

    subtitle_path = find_subtitle_file(video_path)
    subtitle_utterances = parse_subtitles(subtitle_path) if subtitle_path else []

    # Whisper model is now out of scope; free its memory before loading VLM frames.
    gc.collect()

    if progress_callback:
        progress_callback("nsfw_stage_b", (0, len(candidates)))
    confirmer = VlmConfirmer(client)
    visual_detections = []
    for i, candidate in enumerate(candidates):
        dialogue = _dialogue_in_range(utterances, candidate.start, candidate.end)
        detection = confirmer.confirm_and_describe(video_path, candidate, dialogue)
        if detection:
            visual_detections.append(detection)
        if progress_callback:
            progress_callback("nsfw_stage_b", (i + 1, len(candidates)))

    if progress_callback:
        progress_callback("language_detection", None)
    wordlist = load_wordlist()
    language_detections = detect_foul_language(utterances, client, wordlist, subtitle_utterances)

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

    # Full scan complete — clean up Stage A results file so next run rescans.
    stage_a_results = video_path.with_suffix(".stage_a_results.json")
    stage_a_results.unlink(missing_ok=True)

    if progress_callback:
        progress_callback("done", None)

    return result


def _dialogue_in_range(utterances, start: float, end: float) -> str:
    lines = [u.text for u in utterances if u.start < end and u.end > start]
    return " ".join(lines)
