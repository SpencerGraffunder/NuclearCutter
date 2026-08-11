#!/usr/bin/env python3
"""Benchmark multiple LM Studio models on the EXACT scan prompts.

Builds a 32-frame collection from a video (known-flagged frames + random
frames), then runs each model through the real sweep + confirm prompts used by
`nuclearcutter scan`, with thinking disabled. Reports per-model speed and
accuracy (did it flag the known-flagged frames? correct category?).

Usage:
    python3 scripts/bench_models_v2.py <video_path> <scan_json> [--model ID ...]

Example:
    python3 scripts/bench_models_v2.py "/Volumes/.../movie.mkv.iso" \\
        "/Volumes/.../movie.nuclearcutter.json" \\
        --model qwen/qwen3.6-35b-a3b qwen/qwen3-vl-8b qwen/qwen3-vl-30b
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuclearcutter.detection.vlm_confirm import (
    _category_definitions, build_confirm_prompt, build_sweep_prompt,
)
from nuclearcutter.schema import Category, SeverityLevel
from nuclearcutter.utils.ffmpeg import extract_frame_at

BASE_URL = "http://127.0.0.1:1234/v1"
SWEEP_FRAMES_PER_CALL = 4
SWEEP_SCALE = 480
CONFIRM_SCALE = 720
N_FLAGGED = 8
N_TOTAL = 32
RANDOM_SEED = 20260811


def _b64_uri(path: Path) -> str:
    import base64
    from PIL import Image
    import io
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _query(url: str, model: str, messages: list, max_tokens: int = 4096, timeout: int = 600):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/chat/completions", json=payload, timeout=timeout)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    usage = data.get("usage", {})
    return content, dt, usage


def _parse_json_loose(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except Exception:
        return {}


def _flag_windows(scan: dict) -> list[tuple[float, float, str]]:
    out = []
    for v in scan.get("visual_detections", []):
        out.append((float(v["start"]), float(v["end"]), v.get("category", "")))
    return out


def _build_collection(video: Path, duration: float, windows: list[tuple[float, float, str]]):
    """Return (frames, timestamps) — N_FLAGGED timestamps inside known windows
    plus random timestamps, paired 1:1 with their extracted frames."""
    rng = random.Random(RANDOM_SEED)
    flagged = []
    for start, end, _cat in windows[:N_FLAGGED]:
        flagged.append((start + end) / 2.0)  # mid-window

    rand_ts = []
    attempts = 0
    while len(rand_ts) < N_TOTAL - N_FLAGGED and attempts < N_TOTAL * 40:
        attempts += 1
        ts = rng.uniform(30.0, duration - 30.0)
        if any(s - 5 <= ts <= e + 5 for s, e, _ in windows):
            continue
        rand_ts.append(ts)

    all_ts = flagged + rand_ts
    rng.shuffle(all_ts)

    frames = []
    kept_ts = []
    for ts in all_ts:
        try:
            frames.append(extract_frame_at(video, ts, scale_height=SWEEP_SCALE))
            kept_ts.append(ts)
        except Exception as exc:
            print(f"  warn: frame at {ts:.1f}s failed: {exc}", file=sys.stderr)
    return frames, kept_ts


def _truth_for_batch(batch_ts, windows) -> str | None:
    """Return the expected category if this batch contains a flagged timestamp,
    else None (expected clean)."""
    for ts in batch_ts:
        for s, e, cat in windows:
            if s <= ts <= e:
                return cat
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("scan_json", type=Path)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--url", default=BASE_URL)
    args = ap.parse_args()

    if not args.video.exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 1
    if not args.scan_json.exists():
        print(f"error: scan json not found: {args.scan_json}", file=sys.stderr)
        return 1

    scan = json.loads(args.scan_json.read_text())
    windows = _flag_windows(scan)
    duration = float(scan["identity"]["duration_seconds"])
    print(f"known flagged windows: {len(windows)}")

    frames, all_ts = _build_collection(args.video, duration, windows)
    n_flagged = sum(1 for t in all_ts if any(s <= t <= e for s, e, _ in windows))
    print(f"collection: {len(frames)} frames ({n_flagged} in flagged windows, {len(frames)-n_flagged} random)")
    print(f"timestamps: {[round(t,1) for t in all_ts]}")

    # Group into sweep batches of 4 (frames and timestamps stay aligned).
    batches = [frames[i:i + SWEEP_FRAMES_PER_CALL] for i in range(0, len(frames), SWEEP_FRAMES_PER_CALL)]
    batches_ts = [all_ts[i:i + SWEEP_FRAMES_PER_CALL] for i in range(0, len(all_ts), SWEEP_FRAMES_PER_CALL)]

    defs = _category_definitions(None)
    sweep_prompt = build_sweep_prompt(defs, SWEEP_FRAMES_PER_CALL)

    if not args.model:
        print("no --model given; nothing to do", file=sys.stderr)
        return 1

    for model in args.model:
        print(f"\n{'='*70}\nMODEL: {model}\n{'='*70}")

        # --- Sweep pass: 8 batches x 4 frames, exact sweep prompt ---
        print("-- sweep (8 batches x 4 frames) --")
        sweep_times, sweep_correct = [], 0
        for bi, batch in enumerate(batches):
            content, dt, usage = _query(
                args.url, model,
                [{"role": "system", "content":
                  "Answer directly and concisely. Do not provide lengthy reasoning or "
                  "chain-of-thought; output only the requested result."},
                 {"role": "user", "content":
                  [{"type": "text", "text": sweep_prompt}] +
                  [{"type": "image_url", "image_url": {"url": _b64_uri(p)}} for p in batch]}],
                max_tokens=4096,
            )
            parsed = _parse_json_loose(content)
            expected = _truth_for_batch(batches_ts[bi], windows)
            flagged = bool(parsed.get("contains_flagged_content"))
            cat = parsed.get("category")
            ok = (flagged and expected is not None) or (not flagged and expected is None)
            sweep_correct += 1 if ok else 0
            sweep_times.append(dt)
            truth = expected or "clean"
            mark = "OK " if ok else "MISS"
            print(f"  batch {bi+1}: {dt:6.1f}s | truth={truth:8s} got={cat or ('clean' if not flagged else 'flagged')} "
                  f"[{usage.get('completion_tokens','?')} tok] {mark}")
            if not ok:
                print(f"      raw: {content[:200]!r}")

        print(f"  sweep accuracy: {sweep_correct}/{len(batches)} | "
              f"mean/call {sum(sweep_times)/len(sweep_times):.1f}s | total {sum(sweep_times):.1f}s")

        # --- Confirm pass: exact confirm prompt on flagged candidates ---
        print("-- confirm (flagged candidates, 6 frames @720p) --")
        conf_times = []
        conf_results = []
        for start, end, cat in windows[:4]:
            mid = (start + end) / 2.0
            # sample 6 frames across the window
            c_frames = []
            for k in range(6):
                ts = start + (end - start) * (k + 0.5) / 6
                try:
                    c_frames.append(extract_frame_at(args.video, ts, scale_height=CONFIRM_SCALE))
                except Exception:
                    pass
            if not c_frames:
                continue
            cat_enum = Category.from_legacy(cat)
            definition = defs.get(cat_enum, "")
            prompt = build_confirm_prompt(cat_enum, definition, "(no dialogue)")
            content, dt, usage = _query(
                args.url, model,
                [{"role": "system", "content":
                  "Answer directly and concisely. Do not provide lengthy reasoning or "
                  "chain-of-thought; output only the requested result."},
                 {"role": "user", "content":
                  [{"type": "text", "text": prompt}] +
                  [{"type": "image_url", "image_url": {"url": _b64_uri(p)}} for p in c_frames]}],
                max_tokens=4096,
            )
            parsed = _parse_json_loose(content)
            conf_times.append(dt)
            matches = {k: bool(parsed.get(k)) for k in
                       ("matches_low", "matches_med", "matches_high", "matches_exhigh")}
            any_match = any(matches.values())
            flagged = bool(parsed.get("contains_flagged_content", any_match))
            conf_results.append(flagged)
            print(f"  {cat:8s} [{start:.0f}-{end:.0f}]: {dt:6.1f}s | flagged={flagged} "
                  f"matches={[k.split('_')[1] for k,v in matches.items() if v]} "
                  f"[{usage.get('completion_tokens','?')} tok]")
            for p in c_frames:
                p.unlink(missing_ok=True)

        if conf_times:
            print(f"  confirm: {sum(conf_times)/len(conf_times):.1f}s mean/call | "
                  f"{sum(conf_results)}/{len(conf_results)} candidates flagged")

    for p in frames:
        p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
