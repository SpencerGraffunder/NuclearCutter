#!/usr/bin/env python3
"""Benchmark a vision query on two model servers using 32 frames at 480p.

Usage:
    python3 scripts/bench_models.py <video_path> [--frames 32] [--scale 480]
                                    [--mlx-url http://127.0.0.1:1235/v1]
                                    [--lm-url http://127.0.0.1:1234/v1]
                                    [--mlx-model Qwen3.5-9B-MLX-4bit]
                                    [--lm-model qwen3.6-27b]

Runs the SAME sweep-style vision query (4 frames per call, 8 batches = 32
frames) against both servers and reports per-call and total latency.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuclearcutter.detection.vlm_confirm import _category_definitions, build_sweep_prompt
from nuclearcutter.utils.ffmpeg import extract_frame_at

SWEEP_FRAMES_PER_CALL = 4


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _query(url: str, model: str, prompt: str, image_paths: list[Path]) -> float:
    """Send one vision chat-completion; return elapsed seconds."""
    content = [{"type": "text", "text": prompt}]
    content += [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(p)}"}}
        for p in image_paths
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/chat/completions", json=payload, timeout=600)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--scale", type=int, default=480)
    ap.add_argument("--mlx-url", default="http://127.0.0.1:1235/v1")
    ap.add_argument("--lm-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--mlx-model", default="/Users/spencer/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit")
    ap.add_argument("--lm-model", default="qwen3.6-27b")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 1

    # Build the same sweep prompt the real pipeline uses.
    defs = _category_definitions(None)
    prompt = build_sweep_prompt(defs, SWEEP_FRAMES_PER_CALL)

    # Extract frames at the target scale, evenly spaced across the film.
    import subprocess
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(args.video)],
        capture_output=True, text=True, check=True,
    ).stdout
    dur = float(json.loads(raw)["format"]["duration"])
    n = args.frames
    timestamps = [dur * (i + 0.5) / n for i in range(n)]

    frames: list[Path] = []
    for ts in timestamps:
        try:
            frames.append(extract_frame_at(args.video, ts, scale_height=args.scale))
        except Exception as exc:
            print(f"  frame extraction failed at {ts:.1f}s: {exc}", file=sys.stderr)

    batches = [frames[i:i + SWEEP_FRAMES_PER_CALL] for i in range(0, len(frames), SWEEP_FRAMES_PER_CALL)]

    def bench(label: str, url: str, model: str):
        print(f"\n=== {label} ({model}) — {len(frames)} frames, {len(batches)} calls x {SWEEP_FRAMES_PER_CALL} ===")
        times = []
        for i, batch in enumerate(batches):
            try:
                dt = _query(url, model, prompt, batch)
                times.append(dt)
                print(f"  call {i + 1}/{len(batches)}: {dt:.2f}s")
            except Exception as exc:
                print(f"  call {i + 1}/{len(batches)}: FAILED — {exc}")
        if times:
            print(f"  total: {sum(times):.2f}s | mean/call: {sum(times)/len(times):.2f}s | min: {min(times):.2f}s | max: {max(times):.2f}s")

    try:
        bench("mlx-vlm (Qwen3.5-9B-MLX-4bit)", args.mlx_url, args.mlx_model)
    finally:
        for p in frames:
            p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
