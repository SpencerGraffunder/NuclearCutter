"""Tests for the live scan status writer (nuclearcutter/utils/scan_status.py) —
the JSON file the web GUI polls and the resume logic reads.

The status file is the only thing a user sees while a multi-hour scan runs, and
it is what a resume reads. Both paths only work if the shape is stable and the
records carry tag ids, so the GUI can name what it found in the same vocabulary
it will use for the chooser.
"""

import json

import pytest

from nuclearcutter.utils.scan_status import ScanStatus


def test_status_roundtrip(tmp_path):
    p = tmp_path / "status.json"
    st = ScanStatus(video="M.mkv", pid=42, duration_seconds=100.0, sweep_interval=2.0)
    st.set_phase("sweep")
    st.set_sweep(50, 100)
    st.add_candidate(10.0, 12.0, ["skin"], 0.9)
    st.add_visual_detection(30.0, 40.0, ["gore.blood_moderate"], "desc", 0.7)
    st.add_language_detection("damn", 5.0, 5.2, 4.0, 6.0,
                              tags=["language.profanity_common"])
    st.write(p)

    loaded = ScanStatus.load(p)
    assert loaded.phase == "sweep"
    assert loaded.frames_done == 50
    assert loaded.frames_total == 100
    assert loaded.position_seconds == 100.0  # 50 * 2.0
    # A candidate carries TRIGGER names, never tag ids: the sweep only decides
    # where to look, and a trigger where a tag belongs would imply a
    # classification that has not happened yet.
    assert loaded.visual_candidates == [{"triggers": ["skin"], "start": 10.0,
                                         "end": 12.0, "confidence": 0.9}]
    assert loaded.visual_detections[0]["tags"] == ["gore.blood_moderate"]
    assert loaded.language_detections[0]["tags"] == ["language.profanity_common"]


def test_status_write_is_valid_json(tmp_path):
    p = tmp_path / "status.json"
    st = ScanStatus(video="M.mkv", pid=1)
    st.add_candidate(1.0, 2.0, ["blood", "weapon"], 0.8)
    st.write(p)
    data = json.loads(p.read_text())
    assert data["schema"] == ScanStatus.SCHEMA
    assert data["video"] == "M.mkv"
    # No tmp file left behind after the atomic rename.
    assert not (tmp_path / "status.json.tmp").exists()


def test_candidates_merge_touching_ranges():
    """The sweep reports per-frame hits; two adjacent frames of one gesture must
    not become three dim markers that the classify pass then visits three times."""
    st = ScanStatus(sweep_interval=2.0)
    st.add_candidate(10.0, 12.0, ["skin"], 0.9)
    st.add_candidate(12.0, 14.0, ["skin"], 0.6)
    assert len(st.visual_candidates) == 1
    assert st.visual_candidates[0]["start"] == 10.0
    assert st.visual_candidates[0]["end"] == 14.0
    # The stronger sighting wins the confidence: a range is as sure as its best frame.
    assert st.visual_candidates[0]["confidence"] == 0.9
    assert st.visual_candidates[0]["triggers"] == ["skin"]


def test_candidates_with_different_triggers_stay_separate():
    """Merging is only for the same trigger set: 'skin' and 'blood' are different
    questions and the classify pass needs to see both."""
    st = ScanStatus()
    st.add_candidate(10.0, 12.0, ["skin"], 0.9)
    st.add_candidate(12.0, 14.0, ["blood"], 0.9)
    assert len(st.visual_candidates) == 2


def test_load_missing_file_raises(tmp_path):
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
    assert d["schema"] == ScanStatus.SCHEMA
    assert d["video"] == "x.mkv"


def test_schema_constant_matches_the_scan_file():
    """One schema number for both files. A status file that agrees with the scan
    file about nothing else would still agree about this, which is what makes
    'resume only an exactly-compatible run' a single check."""
    from nuclearcutter.schema import SCHEMA_VERSION

    assert ScanStatus.SCHEMA == SCHEMA_VERSION
