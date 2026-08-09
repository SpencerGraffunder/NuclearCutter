"""Tests for the live scan status writer (nuclearcutter/utils/scan_status.py)
and the TUI log/status parsing (nuclearcutter/tui.py)."""

import json

import pytest

from nuclearcutter.utils.scan_status import ScanStatus
from nuclearcutter.tui import View, build_panel, view_from_log, view_from_status


# ---------------------------------------------------------------------------
# scan_status
# ---------------------------------------------------------------------------


def test_status_roundtrip(tmp_path):
    p = tmp_path / "status.json"
    st = ScanStatus(video="M.mkv", pid=42, duration_seconds=100.0, sweep_interval=2.0)
    st.set_phase("visual_sweep")
    st.set_sweep(50, 100)
    st.add_candidate(10.0, 12.0, "NUDITY", 0.9)
    st.add_visual_detection("GORE_VIOLENCE", 30.0, 40.0, "desc", 0.7)
    st.add_language_detection("fuck", 5.0, 5.2, 4.0, 6.0)
    st.write(p)

    loaded = ScanStatus.load(p)
    assert loaded.phase == "visual_sweep"
    assert loaded.frames_done == 50
    assert loaded.frames_total == 100
    assert loaded.position_seconds == 100.0  # 50 * 2.0
    assert loaded.visual_candidates == [{"category": "NUDITY", "start": 10.0,
                                         "end": 12.0, "confidence": 0.9}]
    assert loaded.visual_detections[0]["category"] == "GORE_VIOLENCE"
    assert loaded.language_detections[0]["word"] == "fuck"


def test_status_write_is_valid_json(tmp_path):
    p = tmp_path / "status.json"
    ScanStatus(video="M.mkv").write(p)
    assert json.loads(p.read_text())["schema"] == 1


def test_status_add_visual_detection_accepts_enum(tmp_path):
    from nuclearcutter.schema import Category

    p = tmp_path / "status.json"
    st = ScanStatus()
    st.add_visual_detection(Category.NUDITY, 1.0, 2.0, "d", 0.5)
    st.write(p)
    loaded = ScanStatus.load(p)
    assert loaded.visual_detections[0]["category"] == "nudity"  # Category.NUDITY.value


# ---------------------------------------------------------------------------
# tui — log attach + status parsing
# ---------------------------------------------------------------------------


def _write_log(tmp_path, lines):
    p = tmp_path / "scan.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_view_from_log_sweep(tmp_path):
    p = _write_log(tmp_path, [
        "[fingerprinting] loaded cached fingerprint (16 samples, 8496s)",
        "[sweep] 100 / 4248 frames",
        "[sweep] 200 / 4248 frames",
    ])
    v = view_from_log(p, interval=2.0)
    assert v.phase == "visual_sweep"
    assert v.duration == 8496.0
    assert v.frames_done == 200
    assert v.frames_total == 4248
    assert v.position == 400.0


def test_view_from_log_phase_priority(tmp_path):
    # "done" must win over a stale [sweep] line.
    p = _write_log(tmp_path, [
        "[sweep] 100 / 4248 frames",
        "Scan complete: 3 visual detections, 5 language detections.",
    ])
    v = view_from_log(p, interval=2.0)
    assert v.phase == "done"


def test_view_from_log_missing(tmp_path):
    v = view_from_log(tmp_path / "nope.log", interval=2.0)
    assert v.phase == "starting"
    assert v.frames_done == 0


def test_view_from_status(tmp_path):
    p = tmp_path / "status.json"
    st = ScanStatus(video="M.mkv", duration_seconds=100.0, sweep_interval=2.0)
    st.set_phase("visual_sweep")
    st.set_sweep(25, 50)
    st.add_candidate(10.0, 12.0, "NUDITY", 0.9)
    st.write(p)

    v = view_from_status(p)
    assert v.phase == "visual_sweep"
    assert v.position == 50.0
    assert len(v.candidates) == 1


def test_build_panel_renders(tmp_path):
    from rich.console import Console

    v = View(phase="visual_sweep", duration=100.0, position=50.0,
             frames_done=25, frames_total=50)
    panel = build_panel(v, {"cpu": 10.0, "mem_pct": 20.0, "proc_cpu": 5.0,
                            "proc_mem": None, "gpu": None, "temp": None})
    console = Console(record=True, width=100)
    console.print(panel)
    out = console.export_text()
    assert "visual_sweep" in out
    assert "Timeline" in out
