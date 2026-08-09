"""
Repair a scan JSON that was produced with the broken mlx-vlm json_object
handling (empty `{}` from text_query_json → all language detections dropped).

The scan's visual detections are kept; only the language pass is redone:
re-transcribe the film (whisper), then run detect_foul_language with the fixed
client, and write the updated ScanResult back to the same path.

Usage:
    python -m nuclearcutter.utils.repair_language SCAN_JSON MOVIE_PATH
"""

from __future__ import annotations

import sys
from pathlib import Path

from nuclearcutter.detection.profanity import detect_foul_language, load_wordlist
from nuclearcutter.detection.transcribe import find_subtitle_file, parse_subtitles, transcribe
from nuclearcutter.schema import ScanResult
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig
from nuclearcutter.utils.model_server import ModelServerConfig, ensure_backend


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python -m nuclearcutter.utils.repair_language SCAN_JSON MOVIE_PATH", file=sys.stderr)
        return 1
    scan_path = Path(sys.argv[1])
    video_path = Path(sys.argv[2])

    result = ScanResult.load(scan_path)
    print(f"Loaded scan: {len(result.visual_detections)} visual detections, "
          f"{len(result.language_detections)} language detections.")

    # Backend: same defaults as the CLI (mlx-vlm auto-start).
    server_cfg = ModelServerConfig(backend="mlx-vlm")
    proc = ensure_backend(server_cfg, log_path=Path("/tmp/nuclearcutter_mlx_vlm.log"))
    llm_config = LLMConfig(
        base_url=server_cfg.base_url,
        vlm_model=server_cfg.model_path,
        text_model=server_cfg.model_path,
    )
    client = LLMClient(llm_config)
    client.test_connection()

    print("Re-transcribing (whisper)...")
    utterances = transcribe(video_path, model="mlx-community/whisper-small-mlx")
    print(f"  {len(utterances)} utterances")

    subtitle_path = find_subtitle_file(video_path)
    subtitle_utterances = parse_subtitles(subtitle_path) if subtitle_path else []

    print("Running language detection...")
    wordlist = load_wordlist()
    language_detections = detect_foul_language(utterances, client, wordlist, subtitle_utterances)
    print(f"  {len(language_detections)} detections")
    for d in language_detections:
        print(f"    [{d.start:.1f}-{d.end:.1f}] {d.word} (llm_confirmed={d.llm_confirmed})")

    result.language_detections = language_detections
    result.save(scan_path)
    print(f"Updated {scan_path}")

    if proc is not None:
        proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
