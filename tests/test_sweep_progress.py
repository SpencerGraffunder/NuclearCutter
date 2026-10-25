"""Tests for the VLM sweep's progress reporting (nuclearcutter/detection/vlm_confirm.py).

The GUI scan bar must reach 100% whenever the sweep completes — even if the
final batch's frames fail to extract — and must NOT claim 100% when the sweep
was stopped. These guard the "stuck at 98%" and "scan bar 0% on resume" bugs.
"""

import subprocess
import threading
from pathlib import Path

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
    # Chunked extraction is the primary path now; when it fails the sweep
    # falls back to per-frame extraction — fail BOTH so "no frames at all".
    monkeypatch.setattr("nuclearcutter.detection.vlm_confirm.extract_frames_chunk", _boom)


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


# ---------------------------------------------------------------------------
# Chunked + prefetched extraction: the sweep grabs one ffmpeg decode pass per
# chunk (frames for the next chunk extracted while the VLM chews the current
# batch). Per-frame seeking dominated full-film scan time (~30s/batch vs
# ~9s/batch VLM), so the chunk path must be the primary source of frames,
# with per-frame extraction kept only as the fallback for missing samples.
# ---------------------------------------------------------------------------


class _CleanClient:
    def vision_query_json(self, prompt, frames):
        return {"contains_flagged_content": False}


def test_sweep_uses_chunked_extraction(tmp_path, monkeypatch):
    """The sweep asks for one chunk covering the whole film, and every
    sample timestamp is delivered (fallback covers what ffmpeg fell short of)."""
    from nuclearcutter.detection import vlm_confirm

    video = _tiny_video(tmp_path)  # 3s -> sample timestamps [0, 1, 2] at 1s interval
    chunk_calls = []
    delivered = []

    def fake_chunk(vp, tss, out_dir, scale_height=None, prefix="chunk_%05d.jpg"):
        chunk_calls.append(list(tss))
        return {}  # pretend ffmpeg produced nothing -> per-frame fallback fills in

    def fake_frame(vp, ts, scale_height=None):
        p = tmp_path / f"f_{ts:.3f}.jpg"
        p.write_bytes(b"jpegbytes")
        delivered.append(round(ts, 3))
        return p

    monkeypatch.setattr(vlm_confirm, "extract_frames_chunk", fake_chunk)
    monkeypatch.setattr(vlm_confirm, "extract_frame_at", fake_frame)

    det = vlm_confirm.VisualSweepDetector(_CleanClient())
    det.sweep(video, sample_interval=1.0)

    assert chunk_calls == [[0.0, 1.0, 2.0]]  # ONE chunk (8 batches > all frames)
    assert delivered == [0.0, 1.0, 2.0]      # fallback filled every missing sample


def test_chunk_frames_used_and_only_missing_fall_back(tmp_path, monkeypatch):
    """Samples the chunk pass produced are used as-is; only samples it did
    NOT produce get a per-frame seek."""
    from nuclearcutter.detection import vlm_confirm

    video = _tiny_video(tmp_path)  # ts [0, 1, 2]
    fallback_ts = []

    def fake_chunk(vp, tss, out_dir, scale_height=None, prefix="chunk_%05d.jpg"):
        out = {}
        for t in tss:
            if t % 2 == 0:  # produce 0.0 and 2.0, fall short on 1.0
                p = tmp_path / f"c_{t:.3f}.jpg"
                p.write_bytes(b"x")
                out[t] = p
        return out

    def fake_frame(vp, ts, scale_height=None):
        fallback_ts.append(ts)
        p = tmp_path / f"f_{ts:.3f}.jpg"
        p.write_bytes(b"y")
        return p

    monkeypatch.setattr(vlm_confirm, "extract_frames_chunk", fake_chunk)
    monkeypatch.setattr(vlm_confirm, "extract_frame_at", fake_frame)

    det = vlm_confirm.VisualSweepDetector(_CleanClient())
    det.sweep(video, sample_interval=1.0)
    assert fallback_ts == [1.0]


def test_extract_frames_chunk_real_ffmpeg(tmp_path):
    """extract_frames_chunk runs ffmpeg once and maps produced frames to the
    requested timestamps in order — including when ffmpeg falls short of the
    requested count at the end of the file (trailing timestamps absent)."""
    from nuclearcutter.utils.ffmpeg import extract_frames_chunk

    video = _tiny_video(tmp_path)  # 3s long
    out_dir = tmp_path / "chunked"

    got = extract_frames_chunk(Path(video), [0.0, 1.0, 2.0], out_dir, scale_height=120)
    assert sorted(got.keys()) == [0.0, 1.0, 2.0]
    for ts, p in got.items():
        assert p.exists() and p.stat().st_size > 0

    # Past the end of the 3s video ffmpeg can only produce the first couple
    # of samples — the rest must be ABSENT from the mapping (not an error),
    # so the sweep's per-frame fallback picks them up.
    got2 = extract_frames_chunk(Path(video), [2.0, 3.0, 4.0, 5.0], out_dir,
                                scale_height=120, prefix="tail_%05d.jpg")
    assert 2.0 in got2
    assert 5.0 not in got2


def test_sweep_resolution_applies_pixel_cap_to_sweep_queries(monkeypatch, tmp_path):
    """A reduced sweep_resolution sends a per-image pixel cap with sweep
    queries; the default "full" omits the kwarg so the client's configured
    cap applies (and fake/stub clients predating the kwarg keep working)."""
    from nuclearcutter.detection import vlm_confirm

    captured = []

    class _CapClient:
        def vision_query_json(self, prompt, frames, max_pixels=None):
            captured.append(max_pixels)
            return {"contains_flagged_content": False}

    def fake_chunk(vp, timestamps, out_dir, **kw):
        return {ts: (tmp_path / f"c_{ts:.3f}.jpg") for ts in timestamps}

    monkeypatch.setattr(vlm_confirm, "extract_frames_chunk", fake_chunk)

    video = _tiny_video(tmp_path)  # 3s -> one 3-frame batch at a 1s interval
    vlm_confirm.VisualSweepDetector(_CapClient(), sweep_resolution="240p").sweep(
        video, sample_interval=1.0)
    vlm_confirm.VisualSweepDetector(_CapClient()).sweep(video, sample_interval=1.0)
    # One sweep batch each: the 240p preset caps images at 102,400 px;
    # "full" sends no cap.
    assert captured == [102_400, None]
