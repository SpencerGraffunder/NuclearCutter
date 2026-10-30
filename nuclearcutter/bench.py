"""
VLM benchmark — the "test/benchmark vlm" button in the web GUI.

Takes a small frame collection from a movie and runs the REAL scan prompts (the
trigger sweep + the tag classify call from prompts.json) against the selected
model, reporting speed and accuracy. If a scan file exists for the movie, frames
are sampled from its detected windows so accuracy can be measured against known
truth; the rest are random frames from the rest of the film.

Accuracy is measured per pass, because the two passes answer different
questions:

  sweep     did a batch containing a known detection get flagged at all?
            (a yes/no question, and the one that decides whether footage gets
            a closer look — a miss here is a scene that never gets classified)
  identify  of the tags the scan file holds for that window, how many did the
            model name? Reported as tag overlap, not as a single verdict,
            because "it said shirtless_male where the scan said
            underwear_male" is a different problem from "it said nothing".
"""

from __future__ import annotations

import random
import shutil
import tempfile
import time
from pathlib import Path

from nuclearcutter.detection.visual_scan import (
    SWEEP_QUALITY_PIXELS, build_identify_prompt, build_sweep_prompt, scale_height_for,
)
from nuclearcutter.schema import ScanResult
from nuclearcutter.taxonomy import TAGS
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
    and predicted_n/predicted_ms; some backends do not. Returns {} when absent.
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
    sweep_resolution: str | None = None,
    n_total: int = N_TOTAL_FRAMES,
    stop_event=None,
) -> dict:
    """Run the benchmark and return {summary, batches, identifies}."""
    scale_height = scale_height_for(scale)
    # Same sweep-only pixel cap the real sweep will use (classify stays full).
    sweep_max_pixels = SWEEP_QUALITY_PIXELS.get(
        (sweep_resolution or "full").strip().lower()
    )
    duration = scan.identity.duration_seconds if scan else _probe_duration(video_path)

    # Known windows and their tags, from the film's own scan file.
    windows: list[tuple[float, float, list[str]]] = [
        (d.start, d.end, list(d.tags)) for d in (scan.visual_detections if scan else [])
    ]

    rng = random.Random(RANDOM_SEED)
    flagged_ts = [(s + e) / 2.0 for s, e, _ in windows[:4]]
    rand_ts = _random_timestamps(duration, n_total - len(flagged_ts),
                                 [(s, e) for s, e, _ in windows])
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

        batches: list[dict] = []
        for bi in range(0, len(frames), BATCH_SIZE):
            if stop_event is not None and stop_event.is_set():
                break
            batch = frames[bi:bi + BATCH_SIZE]
            batch_ts = kept_ts[bi:bi + BATCH_SIZE]
            if not batch:
                continue
            prompt = build_sweep_prompt(len(batch))
            t0 = time.monotonic()
            data = _query_with_usage(client, prompt, batch, max_pixels=sweep_max_pixels)
            elapsed = time.monotonic() - t0
            parsed = data.get("parsed", {}) or {}
            usage = data.get("usage", {}) or {}
            triggers = parsed.get("triggers")
            if not isinstance(triggers, list):
                triggers = ["doubt"] if parsed.get("contains_flagged_content") else []
            expected = _expected_tags(batch_ts, windows)
            # Any trigger at all, 'doubt' included, means the model asked for a
            # closer look — 'doubt' is it admitting it could not exclude the
            # batch, which is a pass, not a miss.
            flagged = bool(triggers)
            correct = (flagged and expected is not None) or (not flagged and expected is None)
            batches.append({
                "batch": bi // BATCH_SIZE + 1,
                "elapsed": round(elapsed, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                **data.get("timings", {}),
                "flagged": flagged,
                "triggers": [str(t) for t in triggers],
                "expected": sorted({TAGS[t].label for t in expected if t in TAGS}) if expected else None,
                "correct": correct if expected is not None else None,
            })

        # Classify pass on up to 2 known windows: same call a real scan makes.
        identifies = []
        for start, end, tags in windows[:2]:
            if stop_event is not None and stop_event.is_set():
                break
            c_frames = []
            for k in range(6):
                ts = start + (end - start) * (k + 0.5) / 6
                try:
                    c_frames.append(extract_frame_at(video_path, ts, scale_height=scale_height))
                except Exception:
                    pass
            if not c_frames:
                continue
            prompt = build_identify_prompt(len(c_frames), "(no dialogue)")
            t0 = time.monotonic()
            data = _query_with_usage(client, prompt, c_frames)
            elapsed = time.monotonic() - t0
            parsed = data.get("parsed", {}) or {}
            got = [str(t) for t in (parsed.get("tags") or []) if str(t) in TAGS]
            known = [t for t in tags if t in TAGS]
            overlap = sorted(set(got) & set(known))
            identifies.append({
                "window": [round(start, 1), round(end, 1)],
                "elapsed": round(elapsed, 1),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                **data.get("timings", {}),
                "tags": got,
                "known": known,
                "overlap": [TAGS[t].label for t in overlap],
                # The number that matters: how much of the scan file's answer the
                # model reproduced. 0 with a non-empty `tags` list means the model
                # is describing the scene in vocabulary the scan did not use —
                # usually a resolution problem, not a comprehension problem.
                "recall": round(len(overlap) / len(known), 2) if known else None,
            })
            for p in c_frames:
                p.unlink(missing_ok=True)

        judged = [b for b in batches if b.get("correct") is not None]
        recalls = [i["recall"] for i in identifies if i.get("recall") is not None]
        summary = {
            "frames": len(frames),
            "batches": len(batches),
            "accuracy": f"{sum(b['correct'] for b in judged)}/{len(judged)}" if judged else "n/a",
            "mean_sweep_s": round(sum(b["elapsed"] for b in batches) / len(batches), 1) if batches else 0.0,
            "identify_count": len(identifies),
            "mean_tag_recall": round(sum(recalls) / len(recalls), 2) if recalls else None,
        }
        return {"summary": summary, "batches": batches, "identifies": identifies}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _probe_duration(video_path: Path) -> float:
    from nuclearcutter.utils.ffmpeg import probe_duration

    return probe_duration(video_path)


def _query_with_usage(client, prompt: str, frames, max_pixels: int | None = None) -> dict:
    """Run a vision query through the SAME entry point a real scan uses
    (`LLMClient.vision_query_json`), so the benchmark's requests are identical
    to an actual sweep/classify call — same thinking-off flags, same prompt
    plumbing, same image encoding. `max_pixels` mirrors the sweep-resolution cap
    so benchmark sweep timings reflect the configured sweep pass.
    """
    if max_pixels:
        parsed = client.vision_query_json(prompt, frames, max_pixels=max_pixels)
    else:
        parsed = client.vision_query_json(prompt, frames)
    return {
        "parsed": parsed,
        "usage": getattr(client, "_last_usage", {}) or {},
        "timings": getattr(client, "_last_timings", {}) or {},
    }


def _expected_tags(batch_ts: list[float],
                   windows: list[tuple[float, float, list[str]]]) -> list[str] | None:
    """Tags of every known detection overlapping this batch, or None if the batch
    is from unflagged footage (where the correct answer is to not ask)."""
    out: list[str] = []
    for ts in batch_ts:
        for s, e, tags in windows:
            if s <= ts <= e:
                out.extend(t for t in tags if t not in out)
    return out or None


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
