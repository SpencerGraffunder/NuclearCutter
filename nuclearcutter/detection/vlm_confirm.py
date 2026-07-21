"""
Stage B of the nudity/intimate-scene detection pipeline: for each candidate
range flagged cheaply by Stage A (nsfw_classifier.py), use a VLM to:

  1. Confirm or reject the flag (Stage A is deliberately trigger-happy).
  2. Classify it as `nudity` vs `intimate_scenes`.
  3. Generate the human-readable scene description used later for blurred
     segments — weaving together what's visually happening AND relevant
     dialogue/plot content from that time range (see docs/SPEC.md 4.1).

The dialogue content for a candidate range is pulled from the Whisper
transcript (see detection/transcribe.py) covering the same time span, and
passed to the VLM alongside the sampled frames so it can write a summary
that captures plot-relevant conversation, not just visuals.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nuclearcutter.detection.nsfw_classifier import CandidateRange
from nuclearcutter.schema import Category, VisualDetection
from nuclearcutter.utils.ffmpeg import extract_frames_in_range
from nuclearcutter.utils.llm_client import LLMClient

FRAMES_PER_RANGE = 6  # sampled evenly across the candidate range for VLM review

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


@dataclass
class DialogueContext:
    """Minimal interface expected from the transcript module for a time range."""
    text: str


class VlmConfirmer:
    def __init__(self, client: LLMClient):
        self.client = client

    def confirm_and_describe(
        self,
        video_path: Path,
        candidate: CandidateRange,
        dialogue_text: str = "",
    ) -> VisualDetection | None:
        """Run Stage B on one candidate range. Returns a VisualDetection, or None if rejected."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="cleancut_vlm_"))
        try:
            fps = FRAMES_PER_RANGE / max(candidate.end - candidate.start, 0.1)
            frame_paths = extract_frames_in_range(video_path, candidate.start, candidate.end, fps, tmp_dir)
            if not frame_paths:
                return None

            # Cap to avoid pathological over-extraction on longer ranges.
            frame_paths = frame_paths[:FRAMES_PER_RANGE]

            prompt = CONFIRM_PROMPT.format(dialogue=dialogue_text or "(no dialogue)")
            try:
                result = self.client.vision_query_json(prompt, frame_paths)
            except Exception as exc:
                logging.warning(
                    "VLM query failed for candidate [%.1f, %.1f] — skipping: %s",
                    candidate.start, candidate.end, exc,
                )
                return None

            if not result.get("contains_flagged_content"):
                return None

            category_str = result.get("category", "nudity")
            category = Category.INTIMATE_SCENES if category_str == "intimate_scenes" else Category.NUDITY

            return VisualDetection(
                category=category,
                start=candidate.start,
                end=candidate.end,
                description=result.get("description", ""),
                confidence=float(result.get("confidence", 0.5)),
                stage_a_score=candidate.peak_score,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
