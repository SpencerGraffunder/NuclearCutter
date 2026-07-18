"""
Stage A of the nudity/intimate-scene detection pipeline: a cheap, fast,
local image classifier that scans sampled frames across the whole movie
and flags *candidate* time ranges. This exists purely so we don't burn
expensive VLM calls on the ~99% of a film with nothing to flag — see
docs/SPEC.md section 4.1.

Uses NudeNet (ONNX-based, runs fine on CPU or via CoreML/ANE on Apple
Silicon through onnxruntime's CoreML execution provider). Any classifier
that returns a per-frame nudity score would slot in here; NudeNet is the
default because it's small, fast, and has no GPU dependency requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nuclearcutter.utils.ffmpeg import probe_duration

# NudeDetector is imported lazily inside NsfwClassifier.__init__ rather than
# at module load time, so importing nuclearcutter.detection.nsfw_classifier (and
# anything that transitively imports it, like the CLI) doesn't require
# nudenet/onnxruntime to be installed just to e.g. run `nuclearcutter --help` or
# use the render pipeline on its own.

# Labels from NudeNet's default model that count as "candidate" for our
# purposes. NudeNet returns many fine-grained classes (exposed vs covered,
# by body region); we treat any exposed-* label as worth flagging to the
# VLM stage, which will make the real judgment call including context
# (e.g. medical/nature-documentary nudity vs sexual content — matters for
# fewer titles but the VLM stage is where that nuance belongs, not here).
CANDIDATE_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_BREAST_EXPOSED",
}

DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
DEFAULT_SCORE_THRESHOLD = 0.5

# When merging adjacent flagged frames into a candidate range, allow this
# many seconds of gap (frames below threshold) before treating it as a
# separate range. Prevents fragmenting one continuous scene into dozens of
# tiny candidate ranges due to a single frame dipping below threshold.
MERGE_GAP_SECONDS = 3.0

# Padding added to both ends of a merged candidate range before sending to
# the VLM stage, so we don't clip the start/end of the actual scene.
RANGE_PADDING_SECONDS = 2.0


@dataclass
class CandidateRange:
    start: float
    end: float
    peak_score: float


class NsfwClassifier:
    def __init__(self):
        from nudenet import NudeDetector
        self._detector = NudeDetector()

    def score_frame(self, frame_path: Path) -> float:
        """Return the max confidence across candidate labels for a single frame."""
        detections = self._detector.detect(str(frame_path))
        best = 0.0
        for det in detections:
            if det["class"] in CANDIDATE_LABELS:
                best = max(best, det["score"])
        return best

    def scan(
        self,
        video_path: Path,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
        progress_callback=None,
    ) -> list[CandidateRange]:
        """Scan the full video at sample_interval, return merged candidate ranges.

        progress_callback, if given, is called with (current_second, total_seconds)
        after each sampled frame — useful for CLI progress bars on long scans.
        """
        from nuclearcutter.utils.ffmpeg import extract_frame_at

        duration = probe_duration(video_path)
        flagged_points: list[tuple[float, float]] = []  # (timestamp, score)

        t = 0.0
        while t < duration:
            frame_path = extract_frame_at(video_path, t)
            try:
                score = self.score_frame(frame_path)
            finally:
                frame_path.unlink(missing_ok=True)

            if score >= threshold:
                flagged_points.append((t, score))

            if progress_callback:
                progress_callback(t, duration)

            t += sample_interval

        return _merge_points_to_ranges(flagged_points, duration)


def _merge_points_to_ranges(flagged_points: list[tuple[float, float]], duration: float) -> list[CandidateRange]:
    if not flagged_points:
        return []

    ranges: list[CandidateRange] = []
    range_start, range_end, peak = flagged_points[0][0], flagged_points[0][0], flagged_points[0][1]

    for t, score in flagged_points[1:]:
        if t - range_end <= MERGE_GAP_SECONDS:
            range_end = t
            peak = max(peak, score)
        else:
            ranges.append(_pad_range(range_start, range_end, peak, duration))
            range_start, range_end, peak = t, t, score

    ranges.append(_pad_range(range_start, range_end, peak, duration))
    return ranges


def _pad_range(start: float, end: float, peak: float, duration: float) -> CandidateRange:
    return CandidateRange(
        start=max(0.0, start - RANGE_PADDING_SECONDS),
        end=min(duration, end + RANGE_PADDING_SECONDS),
        peak_score=peak,
    )
