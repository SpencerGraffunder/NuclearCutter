"""Tests for the VLM sweep's progress reporting (nuclearcutter/detection/vlm_confirm.py).

The GUI scan bar must reach 100% whenever the sweep completes — even if the
final batch's frames fail to extract — and must NOT claim 100% when the sweep
was stopped. These guard the "stuck at 98%" and "scan bar 0% on resume" bugs.
"""

import subprocess
import threading

from nuclearcutter.detection.vlm_confirm import VisualSweepDetector


def _tiny_video(tmp_path) -> str:
    video = tmp_path / "tiny.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(video)],
        check=True,
    )
    return str(video)


class _NoopClient:
    """Sweep only talks to the client when frames extract successfully."""

    def vision_query_json(self, prompt, frames):
        raise AssertionError("should not be called when frames fail to extract")


def _fail_extraction(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("frame extraction failed")

    monkeypatch.setattr("nuclearcutter.detection.vlm_confirm.extract_frame_at", _boom)


def test_sweep_reports_100_even_when_frames_fail(tmp_path, monkeypatch):
    """Extraction failing on the last batch must not stall the scan bar below 100%."""
    video = _tiny_video(tmp_path)
    _fail_extraction(monkeypatch)
    det = VisualSweepDetector(_NoopClient())
    progress = []
    det.sweep(video, sample_interval=2.0, on_progress=lambda d, t: progress.append((d, t)))
    assert progress, "progress must be reported even when no frames extract"
    done, total = progress[-1]
    assert done == total > 0  # ended at 100%


def test_sweep_resume_with_nothing_left_reports_100(tmp_path, monkeypatch):
    """Resuming when the sweep already finished must still show 100%, not 0%."""
    video = _tiny_video(tmp_path)
    _fail_extraction(monkeypatch)
    det = VisualSweepDetector(_NoopClient())
    progress = []
    det.sweep(video, sample_interval=2.0, resume_from=9999,
              on_progress=lambda d, t: progress.append((d, t)))
    assert progress
    done, total = progress[-1]
    assert done == total > 0


def test_sweep_stopped_does_not_claim_100(tmp_path, monkeypatch):
    """A stopped sweep keeps partial progress — it must not report 100%."""
    video = _tiny_video(tmp_path)
    _fail_extraction(monkeypatch)
    stop = threading.Event()
    stop.set()
    det = VisualSweepDetector(_NoopClient())
    progress = []
    det.sweep(video, sample_interval=2.0, stop_event=stop,
              on_progress=lambda d, t: progress.append((d, t)))
    assert progress == []  # stopped before any batch: no progress, no 100% claim


# ---------------------------------------------------------------------------
# Frame-level localization: blur only around the flagged frame(s), not the
# whole 4-frame batch.
# ---------------------------------------------------------------------------

from nuclearcutter.detection.vlm_confirm import (
    _flagged_timestamps, _frame_padding_for_interval, _merge_flagged_windows,
    _padding_for_interval,
)
from nuclearcutter.schema import Category, SeverityLevel


class _FlaggingClient:
    """Fake VLM: returns a flagged verdict, optionally localizing flagged_frames."""

    def __init__(self, flagged_frames=None):
        self._ff = flagged_frames

    def vision_query_json(self, prompt, frames):
        d = {"contains_flagged_content": True, "category": "nudity",
             "confidence": 0.9, "description": "x"}
        if self._ff is not None:
            d["flagged_frames"] = self._ff
        return d


def test_flagged_timestamps_maps_indices():
    ts = [0.0, 2.0, 4.0, 6.0]
    assert _flagged_timestamps(ts, [1]) == [2.0]
    assert _flagged_timestamps(ts, [0, 3]) == [0.0, 6.0]
    assert _flagged_timestamps(ts, [1, 1, 2.0]) == [2.0, 4.0]  # dedupe; float ok
    assert _flagged_timestamps(ts, [9, -1, "1", True, 2.5]) == []  # bad indices
    assert _flagged_timestamps(ts, None) == []  # no localization -> fallback


def test_merge_frame_level_is_tight_batch_is_wide():
    """A single flagged frame blurs ~1 interval; a whole-batch flag blurs more."""
    win = (4.0, 4.0, Category.NUDITY, "x", 0.9, SeverityLevel.LOW)
    fr = _merge_flagged_windows([win], 100.0,
                                padding=_frame_padding_for_interval(2.0), merge_gap=2.0)[0]
    assert abs(fr.start - 3.0) < 1e-6 and abs(fr.end - 5.0) < 1e-6  # ~2s

    batch = (0.0, 6.0, Category.NUDITY, "x", 0.9, SeverityLevel.LOW)
    br = _merge_flagged_windows([batch], 100.0,
                                padding=_padding_for_interval(2.0), merge_gap=2.0)[0]
    assert br.start == 0.0 and abs(br.end - 8.0) < 1e-6  # ~8s


def test_sweep_localized_frame_blurs_less_than_batch(tmp_path):
    video = _tiny_video(tmp_path)  # 3s -> sample timestamps [0, 2]
    # Flagged frame 1 (at ~2s) -> tight ~2s candidate around it.
    det = VisualSweepDetector(_FlaggingClient(flagged_frames=[1]))
    tight = det.sweep(video, sample_interval=2.0)
    assert tight
    assert (tight[0].end - tight[0].start) <= 3.0

    # No localization -> whole batch (0..2) padded -> wider candidate.
    det2 = VisualSweepDetector(_FlaggingClient(flagged_frames=None))
    wide = det2.sweep(video, sample_interval=2.0)
    assert wide
    assert (wide[0].end - wide[0].start) >= (tight[0].end - tight[0].start)
