"""
VLM benchmark — the "test/benchmark vlm" button in the web GUI.

Takes a small frame collection from a movie and runs the REAL scan prompts
(the sweep prompt + confirm prompt from prompts.json) against the selected
model, reporting speed and accuracy. If a scan file exists for the movie, a
few frames are sampled from its flagged windows so accuracy can be measured
against known truth; the rest are random frames from the rest of the film.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import time
from pathlib import Path

from nuclearcutter.detection.vlm_confirm import (
    _category_definitions, build_confirm_prompt, build_sweep_prompt,
)
from nuclearcutter.schema import ScanResult
from nuclearcutter.utils.ffmpeg import extract_frame_at

N_TOTAL_FRAMES = 12  # 3 sweep batches of 4 frames
BATCH_SIZE = 4
RANDOM_SEED = 20260811


def _random_timestamps(duration: float, n: int, avoid: list[tuple[float, float]]) -> list[float]:
    rng = random.Random(RANDOM_SEED)
    # Clamp the sample window to the file — for short videos the old
    # (10, duration-10) window could sit entirely past EOF, failing every
    # frame extraction.
    lo = min(5.0, max(duration * 0.1, 0.0))
    hi = max(lo + 1.0, min(duration - 5.0, duration - 0.1))
    if hi <= lo:
        hi = max(duration - 0.1, 0.0)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        ts = rng.uniform(lo, hi)
        if any(s - 5 <= ts <= e + 5 for s, e in avoid):
            continue
        out.append(ts)
    # Fall back to filling with mid-film timestamps if the movie is tiny.
    while len(out) < n:
        out.append(max(0.0, min(duration / 2, max(duration - 0.1, 0.0))))
    return out


def _timings_of(data: dict) -> dict:
    """Extract pp/gen speeds from a server response, when the server reports them.

    llama.cpp and LM Studio include a `timings` object with prompt_n/prompt_ms
    and predicted_n/predicted_ms; mlx-vlm does not. Returns {} when absent.
    """
    t = data.get("timings") or {}
    out = {}
    if t.get("prompt_ms") and t.get("prompt_n"):
        out["pp_tok_s"] = round(t["prompt_n"] / (t["prompt_ms"] / 1000.0), 1)
    if t.get("predicted_ms") and t.get("predicted_n"):
        out["gen_tok_s"] = round(t["predicted_n"] / (t["predicted_ms"] / 1000.0), 1)
    return out


def run_benchmark(
    video_path: Path,
    client,
    scan: ScanResult | None = None,
    scale: str = "480p",
    n_total: int = N_TOTAL_FRAMES,
    stop_event=None,
) -> dict:
    """Run the benchmark and return {summary, batches, confirms}."""
    from nuclearcutter.detection.vlm_confirm import scale_height_for

    scale_height = scale_height_for(scale)
    duration = scan.identity.duration_seconds if scan else _probe_duration(video_path)

    flagged_windows: list[tuple[float, float, str]] = []
    if scan:
        flagged_windows = [
            (d.start, d.end, d.category.value) for d in scan.visual_detections
        ]

    # Frame collection: midpoints of known flagged windows + random frames.
    rng = random.Random(RANDOM_SEED)
    flagged_ts = []
    for start, end, _cat in flagged_windows[:4]:
        flagged_ts.append((start + end) / 2.0)
    rand_ts = _random_timestamps(duration, n_total - len(flagged_ts), [(s, e) for s, e, _ in flagged_windows])
    all_ts = flagged_ts + rand_ts
    rng.shuffle(all_ts)
    all_ts = all_ts[:n_total]

    tmp_dir = Path(tempfile.mkdtemp(prefix="cleancut_bench_"))
    try:
        frames = []
        kept_ts = []
        for ts in all_ts:
            try:
                frames.append(extract_frame_at(video_path, ts, scale_height=scale_height))
                kept_ts.append(ts)
            except Exception as exc:
                print(f"  warn: frame at {ts:.1f}s failed: {exc}")

        defs = _category_definitions(None)

        batches: list[dict] = []
        for bi in range(0, len(frames), BATCH_SIZE):
            if stop_event is not None and stop_event.is_set():
                break
            batch = frames[bi:bi + BATCH_SIZE]
            batch_ts = kept_ts[bi:bi + BATCH_SIZE]
            if not batch:
                continue
            prompt = build_sweep_prompt(defs, len(batch))
            t0 = time.monotonic()
            data = _query_with_usage(client, prompt, batch)
            elapsed = time.monotonic() - t0
            parsed = data.get("parsed", {})
            usage = data.get("usage", {})
            expected = _expected_category(batch_ts, flagged_windows)
            flagged = bool(parsed.get("contains_flagged_content"))
            got_cat = parsed.get("category")
            correct = (flagged and expected is not None) or (not flagged and expected is None)
            batches.append({
                "batch": bi // BATCH_SIZE + 1,
                "elapsed": round(elapsed, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                **data.get("timings", {}),
                "flagged": flagged,
                "category": got_cat,
                "expected": expected,
                "correct": correct if expected is not None else None,
            })

        # Confirm pass on up to 2 known flagged windows (accuracy on the
        # per-category prompt).
        confirms = []
        for start, end, cat in flagged_windows[:2]:
            if stop_event is not None and stop_event.is_set():
                break
            mid = (start + end) / 2.0
            c_frames = []
            for k in range(6):
                ts = start + (end - start) * (k + 0.5) / 6
                try:
                    c_frames.append(extract_frame_at(video_path, ts, scale_height=scale_height))
                except Exception:
                    pass
            if not c_frames:
                continue
            from nuclearcutter.schema import Category

            cat_enum = Category.from_legacy(cat)
            prompt = build_confirm_prompt(cat_enum, defs.get(cat_enum, ""), "(no dialogue)")
            t0 = time.monotonic()
            data = _query_with_usage(client, prompt, c_frames)
            elapsed = time.monotonic() - t0
            parsed = data.get("parsed", {})
            matches = {k: bool(parsed.get(k)) for k in
                       ("matches_low", "matches_med", "matches_high", "matches_exhigh")}
            confirms.append({
                "category": cat,
                "window": [round(start, 1), round(end, 1)],
                "elapsed": round(elapsed, 1),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                **data.get("timings", {}),
                "flagged": any(matches.values()),
                "levels": [k.split("_")[1] for k, v in matches.items() if v],
            })
            for p in c_frames:
                p.unlink(missing_ok=True)

        judged = [b for b in batches if b.get("correct") is not None]
        summary = {
            "frames": len(frames),
            "batches": len(batches),
            "accuracy": f"{sum(b['correct'] for b in judged)}/{len(judged)}" if judged else "n/a",
            "mean_sweep_s": round(sum(b["elapsed"] for b in batches) / len(batches), 1) if batches else 0.0,
            "confirm_count": len(confirms),
        }
        return {"summary": summary, "batches": batches, "confirms": confirms}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _probe_duration(video_path: Path) -> float:
    from nuclearcutter.utils.ffmpeg import probe_duration

    return probe_duration(video_path)


def _query_with_usage(client, prompt: str, frames) -> dict:
    """Run a vision query through the SAME entry point a real scan uses
    (`LLMClient.vision_query_json`), so the benchmark's requests are identical
    to an actual sweep/confirm call — same thinking-off flags, same prompt
    plumbing, same image encoding.
    """
    parsed = client.vision_query_json(prompt, frames)
    return {
        "parsed": parsed,
        "usage": getattr(client, "_last_usage", {}) or {},
        "timings": getattr(client, "_last_timings", {}) or {},
    }


def _expected_category(batch_ts: list[float], windows: list[tuple[float, float, str]]) -> str | None:
    for ts in batch_ts:
        for s, e, cat in windows:
            if s <= ts <= e:
                return cat
    return None


def attach_usage_hook(client) -> None:
    """Register a usage hook on a client so _query_with_usage can read back the
    last request's usage/timings (stored on the client instance itself).

    Chains onto any existing usage_callback (e.g. the server's model-stats
    tracker), so benchmark traffic also feeds the model status panel.
    """
    existing = client.usage_callback

    def _hook(data, elapsed):
        client._last_usage = data.get("usage", {}) or {}
        client._last_timings = _timings_of(data)
        if existing:
            try:
                existing(data, elapsed)
            except Exception:
                pass

    client.usage_callback = _hook
