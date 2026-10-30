#!/usr/bin/env python3
"""Measure where VLM scan time actually goes on a given model server.

The one-question-that-matters for scan architecture: is the cost of a sweep
call driven by *pixels*, by *prompt tokens*, or by *per-call overhead*? And
does the server cache a repeated prompt prefix (which decides whether a big
taxonomy prompt is affordable on every batch of the film, or only on the few
batches that got flagged)?

Run it on the box that hosts the model server:

    .venv/bin/python scripts/probe_vlm_cost.py \
        --video /path/to/movie.mkv \
        --server http://localhost:8080 --api-key KEY --model MODEL

Everything is measured against real frames of the film, with the exact request
shape production uses (text first, then images, thinking disabled), so the
numbers transfer to the sweep. It writes a JSON report next to the video.

Experiments:
  A  pixels       fixed 4 frames + short prompt, per-image pixel cap swept
  B  prompt_size  fixed 4 frames @102k px, text prompt size swept, 4 reps each
                  (rep 1 vs reps 2-4 reveals prefix caching)
  C  frames       fixed 102k px + short prompt, frames per call swept
  D  identify     a realistic stage-2 call: taxonomy-sized prompt + 8 hi-res frames
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nuclearcutter.utils.ffmpeg import extract_frame_at, probe_duration  # noqa: E402

SHORT_PROMPT = (
    "Look at the attached frames. Respond with ONLY a JSON object: "
    '{"flagged": true or false, "frames": [0-based indices of frames with content]}'
)


def taxonomy_like_text(target_tokens: int) -> str:
    """A stable, repeatable text block of roughly `target_tokens` tokens.

    Stable matters: the whole point of experiment B is to send the *same* long
    prompt repeatedly, the way the sweep sends one fixed prompt 900 times.
    """
    unit = (
        "- nudity.breasts_full_female: one or both bare breasts are visible on a "
        "female character, whether or not the nipples can be seen. | "
    )
    reps = max(1, int(target_tokens / 24))
    return (
        "You are reviewing sampled frames from a movie. Classify what is visible "
        "using ONLY these categories:\n" + (unit * reps) +
        '\nRespond with ONLY a JSON object: {"tags": ["id"], "confidence": 0.0}'
    )


def encode(path: Path, max_pixels: int) -> str:
    img = Image.open(path)
    if img.size[0] * img.size[1] > max_pixels:
        r = (max_pixels / (img.size[0] * img.size[1])) ** 0.5
        img = img.resize((int(img.size[0] * r), int(img.size[1] * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def call(server: str, key: str, model: str, prompt: str, frames: list[Path],
         max_pixels: int, timeout: int) -> dict:
    content = [{"type": "text", "text": prompt}]
    for f in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode(f, max_pixels)}"}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer directly and concisely."},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    r = requests.post(f"{server}/chat/completions", json=payload, timeout=timeout,
                      headers={"Authorization": f"Bearer {key}"} if key else {})
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    data = r.json()
    u = data.get("usage") or {}
    return {
        "elapsed": round(elapsed, 2),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "raw_keys": sorted(data.keys()),
    }


def run(name: str, label: str, results: list[dict]) -> dict:
    el = [r["elapsed"] for r in results]
    pt = [r["prompt_tokens"] for r in results if r["prompt_tokens"]]
    warm = el[1:] if len(el) > 1 else el
    out = {
        "case": label,
        "n": len(el),
        "median_s": round(statistics.median(el), 2),
        "first_s": round(el[0], 2),
        "warm_median_s": round(statistics.median(warm), 2),
        "prompt_tokens": pt[0] if pt else None,
        "completion_tokens": results[0]["completion_tokens"],
    }
    print(f"  {label:34s} med {out['median_s']:6.2f}s  first {out['first_s']:6.2f}s  "
          f"warm {out['warm_median_s']:6.2f}s  ptok {out['prompt_tokens']}  "
          f"ctok {out['completion_tokens']}")
    results.clear()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    video = Path(args.video)
    duration = probe_duration(video)
    # Frames from mid-film (away from credits/black), plus a couple elsewhere.
    stamps = [duration * f for f in (0.30, 0.34, 0.41, 0.47, 0.55, 0.62, 0.70, 0.77)]
    print(f"extracting {len(stamps)} probe frames from {video.name} ({duration:.0f}s)...")
    frames = []
    for ts in stamps:
        try:
            frames.append(extract_frame_at(video, ts, scale_height=1080))
        except Exception as exc:  # noqa: BLE001
            print(f"  frame at {ts:.0f}s failed: {exc}")
    if len(frames) < 4:
        print("need at least 4 frames", file=sys.stderr)
        return 1
    # Probe at the largest size we test; each call downscales in-memory.
    report: dict = {"video": video.name, "duration": duration, "model": args.model,
                    "server": args.server, "experiments": []}

    probe = call(args.server, args.api_key, args.model, SHORT_PROMPT, frames[:1],
                 921_600, args.timeout)
    print("response keys:", probe["raw_keys"], "(no 'timings' = speed not reported)")

    print("\nA) pixels per image (4 frames, short prompt)")
    rows = []
    for px in (57_600, 102_400, 230_400, 409_440, 921_600):
        got = []
        for _ in range(args.reps):
            got.append(call(args.server, args.api_key, args.model, SHORT_PROMPT,
                            frames[:4], px, args.timeout))
        rows.append(run("A", f"{px//1000}k px/img (total {px*4/1e6:.2f}MP)", got))
    report["experiments"].append({"name": "pixels", "rows": rows})

    print("\nB) text prompt size (4 frames @102k px, 4 reps each)")
    rows = []
    for tok in (200, 1_500, 4_000, 9_000, 18_000):
        got = []
        for _ in range(4):
            got.append(call(args.server, args.api_key, args.model, taxonomy_like_text(tok),
                            frames[:4], 102_400, args.timeout))
        rows.append(run("B", f"~{tok} tok prompt", got))
    report["experiments"].append({"name": "prompt_size", "rows": rows})

    print("\nC) frames per call (102k px, short prompt)")
    rows = []
    for n in (2, 4, 8):
        got = []
        for _ in range(args.reps):
            got.append(call(args.server, args.api_key, args.model, SHORT_PROMPT,
                            frames[:n], 102_400, args.timeout))
        rows.append(run("C", f"{n} frames ({n*102400/1e6:.2f}MP total)", got))
    report["experiments"].append({"name": "frames", "rows": rows})

    print("\nD) realistic stage-2 'identify' call (taxonomy prompt + hi-res frames)")
    rows = []
    for n, px, tok in ((6, 409_440, 4_000), (8, 409_440, 9_000), (8, 230_400, 9_000)):
        got = []
        for _ in range(args.reps):
            got.append(call(args.server, args.api_key, args.model, taxonomy_like_text(tok),
                            frames[:n], px, args.timeout))
        rows.append(run("D", f"{n} frames @ {px//1000}k px, {tok} tok", got))
    report["experiments"].append({"name": "identify", "rows": rows})

    out = video.with_suffix(".vlm_cost_probe.json")
    try:
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    except OSError:
        Path("vlm_cost_probe.json").write_text(json.dumps(report, indent=2))
        print("\nwrote ./vlm_cost_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
