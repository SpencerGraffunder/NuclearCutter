"""Tests for the live scan status writer (nuclearcutter/utils/scan_status.py) —
the JSON file the web GUI polls and the resume logic reads."""

import json

from nuclearcutter.utils.scan_status import ScanStatus


def test_status_roundtrip(tmp_path):
    p = tmp_path / "status.json"
    st = ScanStatus(video="M.mkv", pid=42, duration_seconds=100.0, sweep_interval=2.0)
    st.set_phase("visual_sweep")
    st.set_sweep(50, 100)
    st.add_candidate(10.0, 12.0, "NUDITY", 0.9)
    st.add_visual_detection("GORE", 30.0, 40.0, "desc", 0.7)
    st.add_language_detection("fuck", 5.0, 5.2, 4.0, 6.0)
    st.write(p)

    loaded = ScanStatus.load(p)
    assert loaded.phase == "visual_sweep"
    assert loaded.frames_done == 50
    assert loaded.frames_total == 100
    assert loaded.position_seconds == 100.0  # 50 * 2.0
    assert loaded.visual_candidates == [{"category": "NUDITY", "start": 10.0,
                                         "end": 12.0, "confidence": 0.9,
                                         "level": "med"}]
    assert loaded.visual_detections[0]["category"] == "GORE"
    assert loaded.language_detections[0]["word"] == "fuck"


def test_status_write_is_valid_json(tmp_path):
    p = tmp_path / "status.json"
    st = ScanStatus(video="M.mkv", pid=1)
    st.add_candidate(1.0, 2.0, "VIOLENCE", 0.8, level="high")
    st.write(p)
    data = json.loads(p.read_text())
    assert data["schema"] == 1
    assert data["video"] == "M.mkv"
    # No tmp file left behind after the atomic rename.
    assert not (tmp_path / "status.json.tmp").exists()


def test_load_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        ScanStatus.load(tmp_path / "nope.json")


def test_set_sweep_updates_position():
    st = ScanStatus(sweep_interval=2.0)
    st.set_sweep(25, 100)
    assert st.frames_done == 25
    assert st.position_seconds == 50.0


def test_to_dict_includes_schema():
    st = ScanStatus(video="x.mkv")
    d = st.to_dict()
    assert d["schema"] == 1
    assert d["video"] == "x.mkv"
