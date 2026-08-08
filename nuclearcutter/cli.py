"""
CLI entrypoint. See docs/SPEC.md section 7 — CLI-only for v1, two primary
commands mapping to the two-pass architecture.

Usage:
    nuclearcutter scan MOVIE.mkv --vlm-model qwen3.5 --text-model qwen3.5 --whisper-model mlx-community/whisper-small-mlx
    nuclearcutter render MOVIE.mkv [--scan MOVIE.nuclearcutter.json] [--prefs prefs.json]

Example:
    nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \\
        --base-url http://localhost:1234/v1 \\
        --vlm-model qwen3.5 --text-model qwen3.5 \\
        --whisper-model mlx-community/whisper-small-mlx
    nuclearcutter render "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \\
        --nudity blur --intimate-scenes blur --foul-language mute

Models are never defaulted — scan always requires --vlm-model, --text-model,
and --whisper-model to be passed explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nuclearcutter.render.renderer import build_output_path, render as render_pass
from nuclearcutter.scan.repo_match import find_matching_scan, spot_check_match
from nuclearcutter.scan.scanner import scan as scan_pass
from nuclearcutter.schema import Preferences, ScanResult
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig


def _default_scan_path(video_path: Path) -> Path:
    return video_path.with_suffix(".nuclearcutter.json")


def cmd_scan(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 1

    # No hardcoded model defaults: the scan needs an explicit VLM and text
    # model (and a Whisper model for transcription). Fail fast with a clear
    # message instead of guessing.
    missing = []
    if not args.vlm_model:
        missing.append("--vlm-model (vision model, e.g. qwen3.5)")
    if not args.text_model:
        missing.append("--text-model (text model, e.g. qwen3.5)")
    if not args.whisper_model:
        missing.append("--whisper-model (e.g. mlx-community/whisper-small-mlx)")
    if missing:
        print(
            "error: scan requires explicit models — no defaults are built in.\n"
            "  Provide: " + "\n           ".join(missing),
            file=sys.stderr,
        )
        return 1

    llm_config = LLMConfig(base_url=args.base_url, vlm_model=args.vlm_model, text_model=args.text_model)
    if args.vision_timeout is not None:
        llm_config.vision_timeout = args.vision_timeout

    whisper_model = args.whisper_model

    if not args.force_rescan and args.timestamps_dir:
        timestamps_dir = Path(args.timestamps_dir)
        print(f"Checking {timestamps_dir} for an existing scan...")
        candidate = find_matching_scan(video_path, timestamps_dir)
        if candidate:
            print("Fingerprint match found. Spot-checking with VLM before trusting it...")
            client = LLMClient(llm_config)
            if spot_check_match(video_path, candidate, client):
                out_path = Path(args.output) if args.output else _default_scan_path(video_path)
                try:
                    candidate.save(out_path)
                    print(f"Verified match. Wrote scan result to {out_path} (skipped full rescan).")
                except PermissionError:
                    fallback = Path.cwd() / out_path.name
                    print(f"warning: no write permission at {out_path.parent}, falling back to {fallback}")
                    candidate.save(fallback)
                    print(f"Verified match. Wrote scan result to {fallback} (skipped full rescan).")
                return 0
            else:
                print("Spot-check failed to confirm match closely enough — falling back to full scan.")

    def progress(stage: str, detail):
        if detail is None:
            print(f"[{stage}]")
        elif stage in ("visual_sweep",) and detail:
            t, total = detail
            print(f"\r[{stage}] {t:.0f}s / {total:.0f}s", end="", flush=True)
        elif stage == "visual_confirm" and detail:
            i, total = detail
            print(f"\r[{stage}] {i} / {total} candidates", end="", flush=True)

    print(f"Scanning {video_path.name}...")
    try:
        result: ScanResult = scan_pass(
            video_path,
            llm_config=llm_config,
            title=args.title,
            year=args.year,
            progress_callback=progress,
            whisper_model=whisper_model,
            sweep_interval=args.sweep_interval,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print()

    out_path = Path(args.output) if args.output else _default_scan_path(video_path)
    try:
        result.save(out_path)
    except PermissionError:
        fallback = Path.cwd() / out_path.name
        print(f"warning: no write permission at {out_path.parent}, falling back to {fallback}")
        result.save(fallback)
        out_path = fallback
    print(f"Scan complete: {len(result.visual_detections)} visual detections, "
          f"{len(result.language_detections)} language detections.")
    print(f"Wrote {out_path}")

    if args.timestamps_dir:
        print(f"\nTo share this scan, copy {out_path.name} into {args.timestamps_dir} and open a PR.")

    return 0


def cmd_render(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 1

    scan_path = Path(args.scan) if args.scan else _default_scan_path(video_path)
    if not scan_path.exists():
        print(f"error: scan file not found: {scan_path} (run `nuclearcutter scan` first)", file=sys.stderr)
        return 1

    scan_result = ScanResult.load(scan_path)

    if args.prefs:
        prefs = Preferences.load(Path(args.prefs))
    else:
        prefs = Preferences(
            nudity_action=args.nudity,
            nudity_blur_mute_audio=args.nudity_blur_mute_audio,
            intimate_scenes_action=args.intimate_scenes,
            intimate_scenes_blur_mute_audio=args.intimate_scenes_blur_mute_audio,
            gore_violence_action=args.gore_violence,
            gore_violence_blur_mute_audio=args.gore_violence_blur_mute_audio,
            foul_language_action=args.foul_language,
            foul_language_mute_scope=args.mute_scope,
        )

    output_path = Path(args.output) if args.output else build_output_path(video_path)
    print(f"Rendering {video_path.name} -> {output_path.name}...")
    render_pass(video_path, scan_result, prefs, output_path=output_path, font_path=args.font)
    print(f"Done: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nuclearcutter", description="Detect and censor content in local movie files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan a movie file and produce a timestamp JSON.")
    p_scan.add_argument("video")
    p_scan.add_argument("--output", "-o", help="Output path for scan JSON (default: MOVIE.nuclearcutter.json)")
    p_scan.add_argument("--timestamps-dir", help="Directory of shared scan JSONs to check for an existing match")
    p_scan.add_argument("--force-rescan", action="store_true", help="Skip the shared-repo match check")
    p_scan.add_argument("--title", help="Movie title, stored in the scan file for reference")
    p_scan.add_argument("--year", type=int, help="Release year, stored in the scan file for reference")
    p_scan.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-compatible API base URL (Ollama/LM Studio)")
    p_scan.add_argument("--vlm-model", help="Vision model name (required, e.g. qwen3.5)")
    p_scan.add_argument("--text-model", help="Text model name for profanity context checks (required, e.g. qwen3.5)")
    p_scan.add_argument("--whisper-model", help="Whisper model for transcription (required, e.g. mlx-community/whisper-small-mlx)")
    p_scan.add_argument("--vision-timeout", type=int, default=None,
                        help="Timeout in seconds for VLM vision requests (default: 300)")
    p_scan.add_argument("--sweep-interval", type=float, default=None,
                        help="Seconds between sampled frames in the full-film VLM sweep "
                             "(default: 5). Smaller catches shorter scenes but makes more "
                             "VLM calls; larger is faster but can miss brief flashes.")
    p_scan.set_defaults(func=cmd_scan)

    p_render = sub.add_parser("render", help="Render a cleaned copy of a movie file using a scan JSON + preferences.")
    p_render.add_argument("video")
    p_render.add_argument("--scan", "-s", help="Path to scan JSON (default: MOVIE.nuclearcutter.json)")
    p_render.add_argument("--output", "-o", help="Output path (default: MOVIE_cleaned.ext)")
    p_render.add_argument("--prefs", help="Path to a saved Preferences JSON, overrides individual flags below")
    p_render.add_argument("--nudity", choices=["blur", "none"], default="blur")
    p_render.add_argument("--nudity-blur-mute-audio", action="store_true")
    p_render.add_argument("--intimate-scenes", choices=["blur", "none"], default="blur")
    p_render.add_argument("--intimate-scenes-blur-mute-audio", action="store_true")
    p_render.add_argument("--gore-violence", choices=["blur", "none"], default="blur")
    p_render.add_argument("--gore-violence-blur-mute-audio", action="store_true")
    p_render.add_argument("--foul-language", choices=["mute", "none"], default="mute")
    p_render.add_argument("--mute-scope", choices=["word", "utterance"], default="word")
    p_render.add_argument("--font", help="Path to a .ttf font file for blur overlay text (optional)")
    p_render.set_defaults(func=cmd_render)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
