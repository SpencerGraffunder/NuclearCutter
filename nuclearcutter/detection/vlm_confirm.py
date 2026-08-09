"""
Unified VLM-driven visual detection — the ONLY visual detector.

History: the pipeline used NudeNet (a local CNN) as a cheap Stage A
gatekeeper, then VLM-confirmed its candidate ranges. But NudeNet can miss
a real scene entirely — it scored zero on every frame of a full nude scene
in The Martian (at 5863s) that a VLM flagged at 0.95 confidence. So NudeNet
is gone. Visual detection is now a single unified VLM sweep:

  1. `VisualSweepDetector.sweep()` samples frames across the WHOLE film at a
     configurable interval, sends small batches to the VLM, and asks one
     single-verdict question per batch: "does this batch contain ANY flagged
     content (nudity, intimate scenes, or gore/violence)?" Flagged batch
     windows are merged and padded generously before/after so a scene is
     never clipped and nothing is silently dropped.
  2. Each merged candidate range is then confirmed + described with the
     per-category confirm prompt. The nudity confirm prompt is unchanged
     (it deliberately flags underwear/swimwear/lingerie/suggestive clothing),
     and the sweep verdict is authoritative: if the confirm pass can't enrich
     a swept range, the sweep's own verdict is kept rather than dropping it.

The single-verdict batch format (rather than per-frame index JSON) is what
proved reliable with local reasoning VLMs — see docs/SPEC.md section 4.1.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nuclearcutter.schema import Category, VisualDetection
from nuclearcutter.utils.ffmpeg import extract_frame_at, extract_frames_in_range, probe_duration
from nuclearcutter.utils.llm_client import LLMClient

# How many frames to sample across a candidate range for the confirm/describe pass.
FRAMES_PER_RANGE = 6

# How many times to retry a VLM query that fails/returns unparseable output.
# Reasoning models intermittently emit empty responses; one retry usually
# succeeds, a small cap keeps a dead server from stalling the whole scan.
MAX_VLM_RETRIES = 3

# ---------------------------------------------------------------------------
# Unified full-film sweep
# ---------------------------------------------------------------------------

# Seconds between sampled frames in the full-film sweep. Smaller = catches
# shorter scenes but makes more VLM calls (each call reviews SWEEP_FRAMES_PER_CALL
# frames). 5.0s catches any scene that lasts >= ~5 seconds.
DEFAULT_SWEEP_INTERVAL_SECONDS = 2.0
DEFAULT_SWEEP_INTERVAL = DEFAULT_SWEEP_INTERVAL_SECONDS

# How many sampled frames go into a single batched VLM review call. Kept at 4:
# this is the batch size that proved reliable and accurate with local VLMs.
SWEEP_FRAMES_PER_CALL = 4

# Padding and merge distance are DERIVED from the sweep interval, not hardcoded:
# frames are sampled every `interval` seconds, and any sampled frame that was
# NOT in a flagged batch is a verified-clean frame we must never blur past. So
# we pad by exactly one interval (reaching up to, not past, the next clean
# sample), and merge two flagged batches if they're within one interval of each
# other (i.e. adjacent bad frames whose padding just touches). This keeps the
# blur window tight around actual content and never extends over clean footage.
def _padding_for_interval(interval: float) -> float:
    return interval


def _merge_gap_for_interval(interval: float) -> float:
    return interval
UNIFIED_SWEEP_PROMPT = """You are reviewing {n} sampled frames from a movie, shown in \
chronological order, to help a parental-content-filtering tool decide what is in this batch.

Look at ALL the attached frames together and respond with a JSON object with these exact fields:

- "contains_flagged_content": true or false — true if ANY of the frames shows visible nudity, \
a sexually intimate scene, or graphic gore/violence.
  - visible nudity: exposed genitals/breasts/buttocks, people clearly nude or in underwear/\
lingerie/swimwear in an intimate context. Scenes in underwear, swimwear, lingerie, or otherwise \
suggestive clothing should be treated as flagged content.
  - intimate scene: a sexually intimate moment (kissing/embracing with clothes on does NOT count).
  - graphic gore/violence: blood, wounds, dismemberment, mutilation, exposed organs, brutal \
fighting, murder, torture, assault. Everyday non-graphic action, cartoonish/implicit violence, \
or blood-free suspense should NOT be flagged.
- "category": "nudity", "intimate_scenes", or "gore_violence" if contains_flagged_content is \
true, otherwise null.
- "confidence": a number from 0 to 1.
- "description": a SHORT, clean, matter-of-fact summary of what the flagged frame(s) show, \
suitable for display as text on a black screen in place of the actual footage. If \
contains_flagged_content is false, set this to an empty string.

Respond with ONLY the JSON object, no other text."""

CONFIRM_PROMPT = """You are reviewing frames sampled from a short segment of a movie, \
to help a parental-content-filtering tool decide what's in this segment.

Dialogue spoken during this segment (may be empty if none):
---
{dialogue}
---

Look at the attached frames (sampled in chronological order across the segment) and \
respond with a JSON object with these exact fields:

- "contains_flagged_content": true or false — true only if there is visible nudity \
or a sexually intimate scene (not just kissing/embracing with clothes on — that alone \
does not count). Scenes in underwear, swimwear, lingerie, or otherwise suggestive \
clothing should be treated as flagged content.
- "category": either "nudity" (visible nudity, not necessarily sexual in nature) or \
"intimate_scenes" (a sexual/intimate scene, whether or not nudity is visible). Use \
"nudity" if visible nudity or underwear/swimwear/lingerie is the main flagged element, \
and "intimate_scenes" for other clothed sexual/intimate scenes. Omit or use null if \
contains_flagged_content is false.
- "confidence": a number from 0 to 1.
- "description": a SHORT, clean, matter-of-fact summary of what happens in this segment, \
suitable for display as text on a black screen in place of the actual footage. It should \
describe what happens visually AND include any plot-relevant content from the dialogue, \
so a viewer who reads this instead of watching does not miss story information. Do not \
be graphic or explicit in the description itself — describe the situation plainly, the \
way a content-rating summary would (e.g. "Two characters kiss and undress; they discuss \
their plan to leave town in the morning" rather than an explicit description). If \
contains_flagged_content is false, set this to an empty string.

Respond with ONLY the JSON object, no other text."""

GORE_CONFIRM_PROMPT = """You are reviewing frames sampled from a short segment of a movie, \
to help a parental-content-filtering tool decide what's in this segment.

Look at the attached frames (sampled in chronological order across the segment) and respond with \
a JSON object with these exact fields:
- "contains_flagged_content": true or false — true only if the segment shows graphic gore \
(blood, wounds, dismemberment, mutilation, exposed organs) or graphic violence (brutal fighting, \
murder, torture, assault).
- "category": "gore_violence" if contains_flagged_content is true, otherwise null.
- "confidence": a number from 0 to 1.
- "description": a SHORT, clean, matter-of-fact summary of what happens in this segment, suitable \
for display as text on a black screen in place of the actual footage. Do not be graphic or \
explicit. If contains_flagged_content is false, set this to an empty string.

Respond with ONLY the JSON object, no other text."""


@dataclass
class SweepRange:
    """A candidate visual-detection range found by the full-film sweep.

    Carries the sweep's own verdict so that — if the later confirm pass can't
    enrich it — the sweep's finding is still kept (never silently dropped).
    """
    start: float
    end: float
    category: Category
    description: str
    confidence: float


def _category_from_str(category_str: str | None) -> Category:
    if category_str == "intimate_scenes":
        return Category.INTIMATE_SCENES
    if category_str == "gore_violence":
        return Category.GORE_VIOLENCE
    return Category.NUDITY


class VisualSweepDetector:
    """Unified full-film VLM sweep + confirm. Replaces NudeNet and both old sweeps."""

    def __init__(self, client: LLMClient):
        self.client = client

    def sweep(
        self,
        video_path: Path,
        sample_interval: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
        on_flagged_window=None,
        on_progress=None,
    ) -> list[SweepRange]:
        """Sample the whole film; return merged, padded candidate ranges that contain
        nudity, intimate scenes, or gore/violence.

        `on_flagged_window(start, end, category, confidence)` is called with each
        raw flagged batch window as it's found (for live status/markers).
        `on_progress(done, total)` is called after each batch (for live progress).
        Both are optional.
        """
        duration = probe_duration(video_path)
        flagged_windows: list[tuple[float, float, Category, str, float]] = []
        tmp_dir = Path(tempfile.mkdtemp(prefix="cleancut_sweep_"))
        try:
            timestamps = [t for t in _arange(0.0, duration - 0.5, sample_interval)]
            for i in range(0, len(timestamps), SWEEP_FRAMES_PER_CALL):
                batch_ts = timestamps[i:i + SWEEP_FRAMES_PER_CALL]
                frame_paths = []
                for ts in batch_ts:
                    try:
                        frame_paths.append(extract_frame_at(video_path, ts))
                    except Exception as exc:
                        logging.warning("Sweep: frame extraction failed at %.1fs — %s", ts, exc)

                if not frame_paths:
                    continue

                try:
                    result = self._query_sweep_batch(frame_paths)
                finally:
                    for p in frame_paths:
                        p.unlink(missing_ok=True)

                if result and result.get("contains_flagged_content"):
                    category = _category_from_str(result.get("category"))
                    confidence = float(result.get("confidence", 0.5))
                    flagged_windows.append((
                        batch_ts[0],
                        batch_ts[-1],
                        category,
                        result.get("description", ""),
                        confidence,
                    ))
                    if on_flagged_window:
                        on_flagged_window(
                            start=batch_ts[0],
                            end=batch_ts[-1],
                            category=category.value,
                            confidence=confidence,
                        )

                if done := i + len(batch_ts):
                    if on_progress:
                        on_progress(done, len(timestamps))
                    if done % 100 == 0 or done >= len(timestamps):
                        print(
                            f"[sweep] {done} / {len(timestamps)} frames",
                            file=sys.stderr,
                        )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        padding = _padding_for_interval(sample_interval)
        merge_gap = _merge_gap_for_interval(sample_interval)
        return _merge_flagged_windows(flagged_windows, duration, padding=padding, merge_gap=merge_gap)

    def _query_sweep_batch(self, frame_paths: list[Path]) -> dict | None:
        """Ask the VLM whether a batch contains ANY flagged content (single verdict)."""
        prompt = UNIFIED_SWEEP_PROMPT.format(n=len(frame_paths))
        for attempt in range(1, MAX_VLM_RETRIES + 1):
            try:
                return self.client.vision_query_json(prompt, frame_paths)
            except Exception as exc:
                logging.warning(
                    "Sweep VLM query failed (attempt %d/%d): %s",
                    attempt, MAX_VLM_RETRIES, exc,
                )
                if attempt == MAX_VLM_RETRIES:
                    return None
        return None

    def confirm_and_describe(
        self,
        video_path: Path,
        candidate: SweepRange,
        dialogue_text: str = "",
    ) -> VisualDetection:
        """Enrich a swept range with the per-category confirm prompt.

        The sweep verdict is authoritative: if the confirm pass errors out or
        returns unparseable output, we keep the sweep's own finding rather than
        dropping the range (the whole point of the redesign is to never miss).
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="cleancut_confirm_"))
        try:
            fps = FRAMES_PER_RANGE / max(candidate.end - candidate.start, 0.1)
            frame_paths = extract_frames_in_range(video_path, candidate.start, candidate.end, fps, tmp_dir)
            if not frame_paths:
                return _sweep_as_detection(candidate)
            frame_paths = frame_paths[:FRAMES_PER_RANGE]

            if candidate.category == Category.GORE_VIOLENCE:
                prompt = GORE_CONFIRM_PROMPT
            else:
                prompt = CONFIRM_PROMPT.format(dialogue=dialogue_text or "(no dialogue)")

            result = None
            for attempt in range(1, MAX_VLM_RETRIES + 1):
                try:
                    result = self.client.vision_query_json(prompt, frame_paths)
                    break
                except Exception as exc:
                    logging.warning(
                        "Confirm query failed for [%.1f, %.1f] (attempt %d/%d): %s",
                        candidate.start, candidate.end, attempt, MAX_VLM_RETRIES, exc,
                    )
                    if attempt == MAX_VLM_RETRIES:
                        return _sweep_as_detection(candidate)

            if not result.get("contains_flagged_content"):
                # Confirm pass thinks the range is clean. Still keep the sweep's
                # verdict — never miss a scene the sweep already caught.
                return _sweep_as_detection(candidate)

            return VisualDetection(
                category=_category_from_str(result.get("category")),
                start=candidate.start,
                end=candidate.end,
                description=result.get("description", "") or candidate.description,
                confidence=float(result.get("confidence", candidate.confidence)),
                stage_a_score=None,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _sweep_as_detection(candidate: SweepRange) -> VisualDetection:
    """Build a VisualDetection straight from the sweep's own verdict."""
    return VisualDetection(
        category=candidate.category,
        start=candidate.start,
        end=candidate.end,
        description=candidate.description,
        confidence=candidate.confidence,
        stage_a_score=None,
    )


def _arange(start: float, stop: float, step: float) -> list[float]:
    out = []
    t = start
    while t < stop:
        out.append(t)
        t += step
    return out


def _merge_flagged_windows(
    flagged_windows: list[tuple[float, float, Category, str, float]],
    duration: float,
    padding: float,
    merge_gap: float,
) -> list[SweepRange]:
    """Merge adjacent flagged batch windows into padded candidate ranges.

    `padding` is added to both ends of a merged range (derived from the sweep
    interval so it reaches up to, not past, the next verified-clean sampled
    frame). `merge_gap` is how close two flagged windows must be to merge.
    """
    if not flagged_windows:
        return []
    flagged_windows = sorted(flagged_windows, key=lambda w: w[0])

    ranges: list[SweepRange] = []
    start, end, category, description, confidence = flagged_windows[0]
    peak_confidence = confidence

    def _flush():
        nonlocal start, end, category, description, peak_confidence
        ranges.append(SweepRange(
            start=max(0.0, start - padding),
            end=min(duration, end + padding),
            category=category,
            description=description,
            confidence=peak_confidence,
        ))

    for w_start, w_end, w_cat, w_desc, w_conf in flagged_windows[1:]:
        if w_start - end <= merge_gap:
            end = max(end, w_end)
            peak_confidence = max(peak_confidence, w_conf)
            if not description and w_desc:
                description = w_desc
        else:
            _flush()
            start, end, category, description, peak_confidence = (
                w_start, w_end, w_cat, w_desc, w_conf
            )
    _flush()

    return ranges
