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

import gc
import json
import sys
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
# by body region); Stage A is deliberately set up to be overinclusive, flagging
# anything that might indicate nudity, swimwear, underwear, or intimate
# visual content. The VLM stage makes the real judgment call including
# context, such as whether the scene is sexual/intimate or just a non-flagged
# wardrobe choice.
CANDIDATE_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
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

    # Number of frames between checkpoint saves. Frequent enough that a crash
    # doesn't lose much progress, not so frequent that it thrashes the disk.
    CHECKPOINT_INTERVAL = 100

    def _checkpoint_path(self, video_path: Path) -> Path:
        return video_path.with_suffix(".stage_a_checkpoint.json")

    def _results_path(self, video_path: Path) -> Path:
        return video_path.with_suffix(".stage_a_results.json")

    def _save_checkpoint(
        self,
        video_path: Path,
        duration: float,
        sample_interval: float,
        last_timestamp: float,
        frame_count: int,
        flagged_points: list,
    ) -> None:
        import os as _os
        st = _os.stat(str(video_path))
        data = {
            "version": 1,
            "video_path": str(video_path),
            "file_size": st.st_size,
            "file_mtime_ns": st.st_mtime_ns,
            "duration": duration,
            "sample_interval": sample_interval,
            "last_timestamp": last_timestamp,
            "frame_count": frame_count,
            "flagged_points": flagged_points,
        }
        self._checkpoint_path(video_path).write_text(json.dumps(data))

    def _load_checkpoint(self, video_path: Path, sample_interval: float, duration: float) -> tuple[float, int, list] | None:
        """Try to resume from a checkpoint. Returns (last_timestamp, frame_count, flagged_points) or None."""
        cp = self._checkpoint_path(video_path)
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text())
        except Exception:
            return None

        # Validate the checkpoint is for this file (same path, size, mtime).
        import os as _os
        st = _os.stat(str(video_path))
        if (data.get("file_size") != st.st_size or
            data.get("file_mtime_ns") != st.st_mtime_ns or
            data.get("sample_interval") != sample_interval or
            abs(data.get("duration", 0) - duration) > 0.5):
            print(f"Warning: checkpoint mismatch for {video_path}, starting fresh.", file=sys.stderr)
            cp.unlink(missing_ok=True)
            return None

        last_t = data["last_timestamp"]
        fc = data.get("frame_count", 0)
        pts = data.get("flagged_points", [])
        print(f"Resuming Stage A from {last_t:.0f}s (frame {fc}) — "
              f"{len(pts)} flagged points so far.", file=sys.stderr)
        return last_t, fc, pts

    def scan(
        self,
        video_path: Path,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
        progress_callback=None,
    ) -> list[CandidateRange]:
        """Scan the full video at sample_interval, return merged candidate ranges.

        Tolerant of individual frame-extraction failures: logs a warning and
        skips the frame rather than aborting. Also saves periodic checkpoints
        so a crash mid-scan can resume from the last saved position — re-run
        the exact same command and it auto-detects the checkpoint.

        After Stage A completes, the merged candidate ranges are saved to a
        separate results file (Movie.stage_a_results.json). If the scan later
        crashes during transcription or Stage B, re-running will skip Stage A
        entirely and load the saved results. Delete that file to force a
        full rescan of Stage A.
        """
        from nuclearcutter.utils.ffmpeg import extract_frame_at

        duration = probe_duration(video_path)

        # If Stage A already completed in a previous run, load saved results.
        results_file = self._results_path(video_path)
        if results_file.exists():
            try:
                saved = json.loads(results_file.read_text())
                ranges = [CandidateRange(**r) for r in saved["ranges"]]
                print(
                    f"Loaded saved Stage A results ({len(ranges)} candidate ranges).",
                    file=sys.stderr,
                )
                return ranges
            except Exception as exc:
                print(
                    f"Warning: could not load saved Stage A results ({exc}), "
                    f"re-scanning.",
                    file=sys.stderr,
                )
                results_file.unlink(missing_ok=True)

        flagged_points: list[tuple[float, float]] = []  # (timestamp, score)

        # Resume from mid-scan checkpoint if one exists.
        resume = self._load_checkpoint(video_path, sample_interval, duration)
        if resume:
            resume_t, frame_count, flagged_points = resume
            t = resume_t + sample_interval
        else:
            frame_count = 0
            t = 0.0

        # Stop sampling half a second before the end — ffmpeg can't reliably
        # extract a frame at exactly the final frame, and we'd rather skip one
        # end-of-video sample than crash on every long scan.
        END_MARGIN = 0.5
        # Periodically recreate the ONNX session to flush its internal memory
        # arena. ONNX Runtime accumulates cached allocations across inference
        # calls that never shrink — over 7200 frames (2h movie @ 1 fps) this
        # can silently consume gigabytes.
        RECREATE_INTERVAL = 500
        skipped = 0
        while t < duration - END_MARGIN:
            # --- tolerant frame extraction ---
            try:
                frame_path = extract_frame_at(video_path, t)
            except Exception as exc:
                print(
                    f"  Warning: skipped frame at {t:.0f}s — {exc}",
                    file=sys.stderr,
                )
                skipped += 1
                t += sample_interval
                continue

            try:
                score = self.score_frame(frame_path)
            except Exception as exc:
                print(
                    f"  Warning: classifier failed on frame at {t:.0f}s — {exc}",
                    file=sys.stderr,
                )
                skipped += 1
                t += sample_interval
                continue
            finally:
                frame_path.unlink(missing_ok=True)

            if score >= threshold:
                flagged_points.append((t, score))

            if progress_callback:
                progress_callback(t, duration)

            # Flush ONNX Runtime memory arena periodically.
            frame_count += 1
            if frame_count % RECREATE_INTERVAL == 0:
                del self._detector
                gc.collect()
                from nudenet import NudeDetector
                self._detector = NudeDetector()

            # Periodic checkpoint.
            if frame_count % self.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(
                    video_path, duration, sample_interval,
                    t, frame_count, flagged_points,
                )

            t += sample_interval

        # Stage A complete — save final results so later stages can crash
        # without losing progress, then clean up the mid-scan checkpoint.
        ranges = _merge_points_to_ranges(flagged_points, duration)
        results_file.write_text(json.dumps(
            {"ranges": [{"start": r.start, "end": r.end, "peak_score": r.peak_score} for r in ranges]}
        ))
        self._checkpoint_path(video_path).unlink(missing_ok=True)
        if skipped:
            print(f"  {skipped} frame(s) skipped due to errors.", file=sys.stderr)

        return ranges


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
