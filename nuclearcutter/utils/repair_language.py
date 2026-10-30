"""
Repair a scan JSON that was produced with the broken json_object handling
(empty `{}` from text_query_json → all language detections dropped).

The scan's visual detections are kept; only the language pass is redone:
re-transcribe the film (via the whisper.cpp server), then run the tag-based
language detector with the remote VLM client, and write the updated ScanResult
back to the same path.

A v1 scan file (category/level records) is REFUSED rather than half-converted:
its language rows name a severity level, not what was said, and there is no
honest mapping from "foul_language / med" to a tag id. Re-scan those films.

All model calls go to already-running remote servers (no local spawning):

    python -m nuclearcutter.utils.repair_language SCAN_JSON MOVIE_PATH \
        --base-url http://127.0.0.1:8080/v1 --vlm-model <id> \
        --whisper-base-url http://127.0.0.1:8081 \
        [--whisper-model <id-or-path>] [--whisper-models-dir <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nuclearcutter.detection.language import detect_language, load_wordlist
from nuclearcutter.detection.transcribe import find_subtitle_file, parse_subtitles, transcribe
from nuclearcutter.schema import ScanResult
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig
from nuclearcutter.utils.model_server import (
    DEFAULT_BASE_URL, WHISPER_DEFAULT_BASE_URL, WHISPER_DEFAULT_MODELS_DIR,
    WhisperConfig, require_server_up,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="repair_language",
        description="Re-run the language pass of a scan JSON against a "
                    "remote VLM + whisper server, replacing its tag rows.",
    )
    parser.add_argument("scan_json", help="Path to the scan JSON to repair")
    parser.add_argument("movie", help="Path to the source movie file")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"OpenAI-compatible /v1 model server (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--vlm-model", required=True, help="VLM model id served by --base-url")
    parser.add_argument("--text-model", default=None, help="Text model id (default: --vlm-model)")
    parser.add_argument("--whisper-base-url", default=WHISPER_DEFAULT_BASE_URL,
                        help=f"whisper.cpp server (default: {WHISPER_DEFAULT_BASE_URL})")
    parser.add_argument("--whisper-model", default="",
                        help="whisper model id/path; empty = use the loaded model")
    parser.add_argument("--whisper-models-dir", default=WHISPER_DEFAULT_MODELS_DIR,
                        help=f"dir of the whisper server's .bin models (default: {WHISPER_DEFAULT_MODELS_DIR})")
    args = parser.parse_args()

    scan_path = Path(args.scan_json)
    video_path = Path(args.movie).resolve()
    if not video_path.exists():
        print(f"error: movie not found: {video_path}", file=sys.stderr)
        return 1

    result = ScanResult.load(scan_path)  # raises ValueError on a v1 file
    print(f"Loaded scan: {len(result.visual_detections)} visual detections, "
          f"{len(result.language_detections)} language detections.")

    print(f"Connecting to model server: {args.base_url} ...")
    require_server_up(args.base_url)
    vlm_model = args.vlm_model
    text_model = args.text_model or vlm_model
    client = LLMClient(LLMConfig(base_url=args.base_url, vlm_model=vlm_model, text_model=text_model))
    client.test_connection()

    whisper_cfg = WhisperConfig(
        base_url=args.whisper_base_url,
        models_dir=args.whisper_models_dir,
        model=args.whisper_model or "",
    )
    print("Re-transcribing (whisper)...")
    utterances = transcribe(video_path, whisper_cfg)
    print(f"  {len(utterances)} utterances")

    subtitle_path = find_subtitle_file(video_path)
    subtitle_utterances = parse_subtitles(subtitle_path) if subtitle_path else []

    print("Running language detection (wordlist pass + dialogue sweep)...")
    language_detections = detect_language(
        utterances, client, load_wordlist(), subtitle_utterances,
        on_progress=lambda done, total: print(
            f"\r  dialogue sweep {done}/{total}", end="", flush=True) if total else None)
    print(f"\r  {len(language_detections)} detections          ")
    for d in language_detections:
        print(f"    [{d.start:.1f}-{d.end:.1f}] {d.text!r} -> {', '.join(d.tags)}"
              f"{'' if d.llm_confirmed else ' (unconfirmed)'}")

    result.language_detections = language_detections
    result.save(scan_path)
    print(f"Updated {scan_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
