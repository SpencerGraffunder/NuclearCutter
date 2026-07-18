"""
Pass 1 of the two-pass architecture (docs/SPEC.md section 2): orchestrates
the full detection pipeline (Stage A/B nudity+intimate detection, Whisper
transcription, subtitle cross-check, profanity wordlist + LLM check) and
produces a ScanResult.
"""

from __future__ import annotations

from pathlib import Path

from nuclearcutter.detection.nsfw_classifier import NsfwClassifier
from nuclearcutter.detection.profanity import detect_foul_language, load_wordlist
from nuclearcutter.detection.transcribe import find_subtitle_file, parse_subtitles, transcribe
from nuclearcutter.detection.vlm_confirm import VlmConfirmer
from nuclearcutter.fingerprint.fingerprint import compute_fingerprint
from nuclearcutter.schema import FilmIdentity, ScanResult
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig


def scan(
    video_path: Path,
    llm_config: LLMConfig = None,
    title: str = None,
    year: int = None,
    progress_callback=None,
) -> ScanResult:
    llm_config = llm_config or LLMConfig()
    client = LLMClient(llm_config)

    if progress_callback:
        progress_callback("fingerprinting", None)
    duration, phash_samples = compute_fingerprint(video_path)
    identity = FilmIdentity(
        title=title, year=year, duration_seconds=duration,
        phash_samples=[s.to_dict() for s in phash_samples],
    )

    if progress_callback:
        progress_callback("nsfw_stage_a", None)
    classifier = NsfwClassifier()
    candidates = classifier.scan(
        video_path,
        progress_callback=lambda t, total: progress_callback("nsfw_stage_a", (t, total)) if progress_callback else None,
    )

    if progress_callback:
        progress_callback("transcribing", None)
    utterances = transcribe(video_path)

    subtitle_path = find_subtitle_file(video_path)
    subtitle_utterances = parse_subtitles(subtitle_path) if subtitle_path else []

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

    if progress_callback:
        progress_callback("done", None)

    return result


def _dialogue_in_range(utterances, start: float, end: float) -> str:
    lines = [u.text for u in utterances if u.start < end and u.end > start]
    return " ".join(lines)
