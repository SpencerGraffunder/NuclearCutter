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
