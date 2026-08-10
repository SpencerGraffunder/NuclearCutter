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
     content (nudity, gore, or violence)?" Flagged batch
     windows are merged and padded generously before/after so a scene is
     never clipped and nothing is silently dropped.
  2. Each merged candidate range is then confirmed + described with the
     per-category confirm prompt. The nudity definition deliberately flags
     underwear/swimwear/lingerie/suggestive clothing (default medium
     sensitivity), and the sweep verdict is authoritative: if the confirm
     pass can't enrich a swept range, the sweep's own verdict is kept rather
     than dropping it.

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

from nuclearcutter.schema import Category, SeverityLevel, VisualDetection
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
# ---------------------------------------------------------------------------
# Per-category LEVEL SCALES (the fixed, standardized definitions)
# ---------------------------------------------------------------------------
# Each category is classified into a FIXED severity scale: low / med / high /
# exhigh. This is NOT user-editable per category — it's what makes scan files
# shareable: the scan records how BAD each scene is (which level it fits), and
# each user picks their own THRESHOLD (config `nudity_level` etc.).
#
# Semantics of a level (from the user's config, this is how much censorship a
# threshold means):
#   low    = LOW censorship   -> only the WORST content is caught
#   med    = medium censorship
#   high   = high censorship  -> a lot gets caught
#   exhigh = MAX censorship   -> basically everything gets caught
#
# A detection is assigned the LOWEST (most restrictive) level whose definition
# its content matches: the worst content gets "low", and content that only
# barely qualifies gets "exhigh".

def _level_scale_text(scale: dict) -> str:
    lines = []
    for lv in (SeverityLevel.LOW, SeverityLevel.MED, SeverityLevel.HIGH, SeverityLevel.EXHIGH):
        lines.append(f"- {lv.value}: {scale[lv]}")
    return "\n".join(lines)


# The definitions ARE the old sensitivity presets (low = strictest) plus a new
# exhigh that is even more permissive than the old high.
DEFAULT_LEVEL_SCALES: dict = {
    Category.NUDITY: {
        SeverityLevel.LOW: (
            "Flag as nudity any of these: bare genitals, breasts, or buttocks "
            "(female nipples count; a bare male chest does not); a person "
            "visibly nude or mostly nude; a couple in bed or having sex where "
            "any inappropriate part of them can be seen, even briefly or partly "
            "covered. Do NOT flag: clothed people, kissing/embracing, bare male "
            "chests alone."
        ),
        SeverityLevel.MED: (
            "Flag as nudity any of these: bare genitals, breasts, or buttocks; "
            "people in underwear, swimwear, or lingerie, or otherwise partly "
            "covering private parts in an intimate context; a couple in bed or "
            "having sex where any part of them is shown or can be seen, even if "
            "briefly. Do NOT flag: fully clothed people, bare male chests alone, "
            "kissing/embracing."
        ),
        SeverityLevel.HIGH: (
            "Flag as nudity ANY of these: bare or partly covered genitals, "
            "breasts, or buttocks; revealing clothing or suggestive state: "
            "shirtless men, women in bikinis or lingerie, people in underwear, "
            "anything that shows skin in an intimate or sexualized way; a "
            "couple in bed or having sex, whether or not anything is fully "
            "visible. Do NOT flag: fully clothed people in non-intimate settings."
        ),
        SeverityLevel.EXHIGH: (
            "Flag as nudity or immodesty ANY of these: any skin that is not "
            "fully covered by normal everyday clothing — bare shoulders, bare "
            "arms, bare legs, bare midriffs, low-cut or tight clothing; people "
            "in swimwear, underwear, lingerie, pajamas, towels, or any state of "
            "partial undress; shirtless men; anyone in bed, embracing, kissing, "
            "or any romantic or intimate situation; anything that could be "
            "considered revealing or suggestive. The goal is to censor "
            "essentially everything that isn't fully modest, so err on the side "
            "of flagging."
        ),
    },
    Category.GORE: {
        SeverityLevel.LOW: (
            "Flag as gore only EXTREME graphic content: exposed organs, "
            "dismemberment, mutilation, mass casualties with visible flesh. Do "
            "NOT flag ordinary blood, small wounds, or medical procedures."
        ),
        SeverityLevel.MED: (
            "Flag as gore: visible blood, open wounds, injuries to flesh, "
            "surgery or medical procedures showing blood/incisions, dead bodies "
            "with wounds, or mutilation. Do NOT flag blood-free violence "
            "(that's the \"violence\" category)."
        ),
        SeverityLevel.HIGH: (
            "Flag as gore: ANY visible blood or wound, however small — a cut, "
            "a bruise, a bloody nose, a surgery scene, a corpse, or any blood "
            "on screen."
        ),
        SeverityLevel.EXHIGH: (
            "Flag as gore or anything related to injury/illness: any blood, any "
            "wound, any bandage, any bruise, any scar, any hospital or medical "
            "scene, any illness, any hint of blood, or anything that could be "
            "considered graphic or disturbing. The goal is to censor "
            "essentially anything remotely medical or bloody, so err on the "
            "side of flagging."
        ),
    },
    Category.VIOLENCE: {
        SeverityLevel.LOW: (
            "Flag as violence only GRAPHIC or brutal acts: brutal fighting, "
            "murder, torture, assault with a weapon, or anything clearly "
            "life-threatening. Do NOT flag shouting, mild shoving, or "
            "cartoonish/implicit violence."
        ),
        SeverityLevel.MED: (
            "Flag as violence: characters deliberately hurting each other — "
            "fighting, punching, kicking, hitting, attacks with weapons, "
            "murder, torture, or attempted murder. Do NOT flag gore itself "
            "(that's the \"gore\" category) or everyday non-aggressive action."
        ),
        SeverityLevel.HIGH: (
            "Flag as violence: ANY intentional physical aggression between "
            "characters — a slap, a shove, a punch, a kick, grabbing someone "
            "roughly, or a threat with a weapon — whether or not anyone is "
            "visibly injured."
        ),
        SeverityLevel.EXHIGH: (
            "Flag as violence or aggression ANY of these: any forceful physical "
            "contact — shoving, pushing, grabbing, slapping, punching, kicking; "
            "any yelling, shouting, arguing, threatening, or intimidation; any "
            "weapon shown or implied; any tense or hostile confrontation — even "
            "if no one is visibly injured. The goal is to censor essentially "
            "any aggression or hostility, so err on the side of flagging."
        ),
    },
}

# Ordering of the scale as shown in prompts (lowest to highest).
_LEVELS_IN_ORDER = [
    SeverityLevel.LOW, SeverityLevel.MED, SeverityLevel.HIGH, SeverityLevel.EXHIGH,
]

_VISUAL_CATEGORIES = [Category.NUDITY, Category.GORE, Category.VIOLENCE]


def _category_definitions(prompts: dict | None) -> dict:
    """Return {Category: definition-text}, merging user custom prompts.

    Default = the built-in fixed level scale for that category (standardized
    and shareable). If the user supplies a custom prompt for a category, it
    replaces the level scale entirely (best-effort; level still returned but
    defaults to MED if unparseable).
    """
    out = {
        cat: _level_scale_text(DEFAULT_LEVEL_SCALES[cat])
        for cat in _VISUAL_CATEGORIES
    }
    if prompts:
        for key, val in prompts.items():
            cat = key if isinstance(key, Category) else Category.from_legacy(str(key))
            if val and cat in out:
                out[cat] = str(val)
    return out


def build_sweep_prompt(category_defs: dict, n: int) -> str:
    """Compose the unified sweep prompt from the per-category definitions.

    One question per sampled batch: "does this batch contain ANY flagged
    content?" — where "flagged" is defined by each category's level scale.
    The VLM also returns WHICH level the content fits into.
    """
    bullets = "\n".join(f"- {cat.value}:\n{category_defs[cat]}" for cat in _VISUAL_CATEGORIES)
    return f"""You are reviewing {{n}} sampled frames from a movie, shown in \
chronological order, to help a parental-content-filtering tool decide what is in this batch.

Flag the batch if ANY of the following categories applies. For each category, \
here is the severity scale (low = the WORST content, exhigh = the mildest \
content that still counts) — this is the ONLY measure of what counts:
{bullets}

Respond with a JSON object with these exact fields:

- "contains_flagged_content": true or false — true if ANY of the frames shows \
content matching any category above.
- "category": "nudity", "gore", or "violence" if contains_flagged_content is \
true, otherwise null. If multiple apply, pick the most severe one.
- "level": which severity level the content fits — "low", "med", "high", or \
"exhigh". Assign the LOWEST (most restrictive) level whose definition the \
content matches: the worst content gets "low", and content that only barely \
qualifies gets "exhigh". If contains_flagged_content is false, set this to null.
- "confidence": a number from 0 to 1.
- "description": a SHORT, clean, matter-of-fact summary of what the flagged \
frame(s) show, suitable for display as text on a black screen in place of the \
actual footage. If contains_flagged_content is false, set this to an empty string.

Respond with ONLY the JSON object, no other text.""".replace("{{n}}", str(n))


def build_confirm_prompt(category: Category, definition: str, dialogue_text: str) -> str:
    """Per-category confirm prompt: verify a swept range matches that category's
    level scale and return the exact level + the on-screen summary."""
    return f"""You are reviewing frames sampled from a short segment of a movie that \
was flagged as possibly containing: {category.value}.

Here is the severity scale for {category.value} (low = the WORST content, \
exhigh = the mildest content that still counts) — the ONLY measure of what counts:
{definition}

Dialogue spoken during this segment (may be empty if none):
---
{dialogue_text}
---

Look at the attached frames (sampled in chronological order across the segment) and \
respond with a JSON object with these exact fields:

- "contains_flagged_content": true or false — true only if the segment matches \
the scale above.
- "category": "{category.value}" if contains_flagged_content is true, otherwise null.
- "level": which severity level the segment fits — "low", "med", "high", or \
"exhigh". Assign the LOWEST (most restrictive) level whose definition the \
content matches: the worst content gets "low", and content that only barely \
qualifies gets "exhigh". If contains_flagged_content is false, set this to null.
- "confidence": a number from 0 to 1.
- "description": a SHORT, clean, matter-of-fact summary of what happens in this segment, \
suitable for display as text on a black screen in place of the actual footage. It should \
describe what happens visually AND include any plot-relevant content from the dialogue, \
so a viewer who reads this instead of watching does not miss story information. Do not \
be graphic or explicit in the description itself — describe the situation plainly, the \
way a content-rating summary would. If contains_flagged_content is false, set this to \
an empty string.

Respond with ONLY the JSON object, no other text."""


@dataclass
class SweepRange:
    """A candidate visual-detection range found by the full-film sweep.

    Carries the sweep's own verdict (category + level) so that — if the later
    confirm pass can't enrich it — the sweep's finding is still kept (never
    silently dropped).
    """
    start: float
    end: float
    category: Category
    description: str
    confidence: float
    level: SeverityLevel = SeverityLevel.MED


def _category_from_str(category_str: str | None) -> Category:
    if not category_str:
        return Category.NUDITY
    try:
        return Category.from_legacy(category_str)
    except ValueError:
        return Category.NUDITY


def _level_from_str(level_str: str | None) -> SeverityLevel:
    return SeverityLevel.from_any(level_str)


class VisualSweepDetector:
    """Unified full-film VLM sweep + confirm. Replaces NudeNet and both old sweeps."""

    def __init__(self, client: LLMClient, prompts: dict | None = None):
        self.client = client
        # Per-category definitions (user overrides merged over defaults).
        self.category_defs = _category_definitions(prompts)

    def sweep(
        self,
        video_path: Path,
        sample_interval: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
        on_flagged_window=None,
        on_progress=None,
    ) -> list[SweepRange]:
        """Sample the whole film; return merged, padded candidate ranges that contain
        nudity, gore, or violence.

        `on_flagged_window(start, end, category, confidence)` is called with each
        raw flagged batch window as it's found (for live status/markers).
        `on_progress(done, total)` is called after each batch (for live progress).
        Both are optional.
        """
        duration = probe_duration(video_path)
        flagged_windows: list[tuple[float, float, Category, str, float, SeverityLevel]] = []
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
                    level = _level_from_str(result.get("level"))
                    flagged_windows.append((
                        batch_ts[0],
                        batch_ts[-1],
                        category,
                        result.get("description", ""),
                        confidence,
                        level,
                    ))
                    if on_flagged_window:
                        on_flagged_window(
                            start=batch_ts[0],
                            end=batch_ts[-1],
                            category=category.value,
                            confidence=confidence,
                            level=level.value,
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
        prompt = build_sweep_prompt(self.category_defs, len(frame_paths))
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

            definition = self.category_defs.get(candidate.category, "")
            prompt = build_confirm_prompt(candidate.category, definition, dialogue_text or "(no dialogue)")

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
                level=_level_from_str(result.get("level")) or candidate.level,
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
        level=candidate.level,
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
    flagged_windows: list[tuple[float, float, Category, str, float, SeverityLevel]],
    duration: float,
    padding: float,
    merge_gap: float,
) -> list[SweepRange]:
    """Merge adjacent flagged batch windows into padded candidate ranges.

    `padding` is added to both ends of a merged range (derived from the sweep
    interval so it reaches up to, not past, the next verified-clean sampled
    frame). `merge_gap` is how close two flagged windows must be to merge.
    When merging, the WORST level seen is kept (lowest rank = low, since low
    is the most restrictive / most severe) so severity is never downgraded.
    """
    if not flagged_windows:
        return []
    flagged_windows = sorted(flagged_windows, key=lambda w: w[0])

    ranges: list[SweepRange] = []
    start, end, category, description, confidence, level = flagged_windows[0]
    peak_confidence = confidence
    peak_level = level

    def _flush():
        nonlocal start, end, category, description, peak_confidence, peak_level
        ranges.append(SweepRange(
            start=max(0.0, start - padding),
            end=min(duration, end + padding),
            category=category,
            description=description,
            confidence=peak_confidence,
            level=peak_level,
        ))

    for w_start, w_end, w_cat, w_desc, w_conf, w_level in flagged_windows[1:]:
        if w_start - end <= merge_gap:
            end = max(end, w_end)
            peak_confidence = max(peak_confidence, w_conf)
            if w_level.rank < peak_level.rank:
                peak_level = w_level  # keep the WORST (lowest rank) level
            if not description and w_desc:
                description = w_desc
        else:
            _flush()
            start, end, category, description, peak_confidence, peak_level = (
                w_start, w_end, w_cat, w_desc, w_conf, w_level
            )
    _flush()

    return ranges
