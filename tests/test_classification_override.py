"""Tests for the per-detection classification override (issue #10).

The timeline popup can re-gate a detection (low/med/high/exhigh) or dismiss it
("removed"). A removed detection stays on the timeline (dark grey) but the
render must leave it alone no matter the settings; a level override re-gates
that single detection against the user's threshold. The override is stored on
the detection as a plain string sentinel and written to the scan file.
"""

import json

from fastapi.testclient import TestClient

from nuclearcutter.render.renderer import (
    _language_mute_ranges_within,
    plan_timeline,
)
from nuclearcutter.schema import (
    AudioAction,
    Category,
    FilmIdentity,
    LanguageDetection,
    LEVEL_OVERRIDE_CHOICES,
    LEVEL_OVERRIDE_REMOVED,
    Preferences,
    ScanResult,
    SeverityLevel,
    VisualAction,
    VisualDetection,
)
from nuclearcutter.server import AppState, create_app


# ---------------------------------------------------------------------------
# schema: the override field + helpers
# ---------------------------------------------------------------------------

def _identity(duration=100.0):
    return FilmIdentity(title="T", year=2024, duration_seconds=duration, phash_samples=[])


def test_level_override_choices_constants():
    # The dropdown options are the four levels plus the "removed" sentinel.
    assert LEVEL_OVERRIDE_REMOVED == "removed"
    assert set(LEVEL_OVERRIDE_CHOICES) == {"low", "med", "high", "exhigh", "removed"}
    # The override is a plain string sentinel, NOT a SeverityLevel member.
    assert LEVEL_OVERRIDE_REMOVED not in {lv.value for lv in SeverityLevel}


def test_visual_override_round_trip():
    vd = VisualDetection(category=Category.NUDITY, start=1.0, end=2.0, description="d",
                         confidence=0.9, level=SeverityLevel.EXHIGH, level_override="low")
    d = vd.to_dict()
    assert d["level"] == "exhigh"          # scanned level unchanged
    assert d["level_override"] == "low"    # override persisted alongside
    loaded = VisualDetection.from_dict(d)
    assert loaded.level_override == "low"
    assert loaded.level == SeverityLevel.EXHIGH


def test_language_override_round_trip():
    ld = LanguageDetection(start=1.0, end=1.2, utterance_start=0.5, utterance_end=2.0,
                           word="fuck", transcript_source="whisper", llm_confirmed=True,
                           level=SeverityLevel.LOW, level_override="high")
    loaded = LanguageDetection.from_dict(ld.to_dict())
    assert loaded.level_override == "high"
    assert loaded.level == SeverityLevel.LOW


def test_override_defaults_to_none():
    vd = VisualDetection(category=Category.NUDITY, start=1.0, end=2.0, description="d", confidence=0.9)
    assert vd.level_override is None
    assert vd.is_removed is False
    # effective_level falls back to the scanned level when there's no override.
    vd.level = SeverityLevel.HIGH
    assert vd.effective_level() == SeverityLevel.HIGH


def test_override_parse_is_tolerant():
    # Missing / blank / unknown overrides all normalize to None (no override).
    for bad in (None, "", "   ", "bogus", "REMOVED!"):
        vd = VisualDetection.from_dict({"category": "nudity", "start": 1.0, "end": 2.0,
                                        "description": "d", "confidence": 0.9,
                                        "level_override": bad})
        assert vd.level_override is None, bad
        ld = LanguageDetection.from_dict({"start": 1.0, "end": 2.0, "utterance_start": 0.5,
                                          "utterance_end": 2.0, "word": "x",
                                          "transcript_source": "whisper", "llm_confirmed": True,
                                          "level_override": bad})
        assert ld.level_override is None, bad


def test_override_parse_lowercases_levels():
    vd = VisualDetection.from_dict({"category": "gore", "start": 1.0, "end": 2.0,
                                    "description": "d", "confidence": 0.9,
                                    "level_override": "  High  "})
    assert vd.level_override == "high"
    assert vd.effective_level() == SeverityLevel.HIGH


def test_override_removed_property():
    vd = VisualDetection(category=Category.NUDITY, start=1.0, end=2.0, description="d",
                         confidence=0.9, level_override="removed")
    assert vd.is_removed is True
    # A "removed" override does not act as a level override.
    assert vd.effective_level() == SeverityLevel.MED  # falls back to scanned level

    ld = LanguageDetection(start=1.0, end=1.2, utterance_start=0.5, utterance_end=2.0,
                           word="x", transcript_source="whisper", llm_confirmed=True,
                           level_override="removed")
    assert ld.is_removed is True


def test_scan_still_stores_no_actions_with_override():
    """SPEC.md section 6 invariant must hold even with overrides set: the scan
    records the override (a level sentinel), never a user-chosen action."""
    vd = VisualDetection(category=Category.NUDITY, start=1.0, end=2.0, description="d",
                         confidence=0.9, level_override="removed")
    ld = LanguageDetection(start=1.0, end=1.2, utterance_start=0.5, utterance_end=2.0,
                           word="damn", transcript_source="whisper", llm_confirmed=True,
                           level_override="med")
    result = ScanResult(schema_version=1, identity=_identity(),
                        visual_detections=[vd], language_detections=[ld])
    raw = json.dumps(result.to_dict())
    for action in ("blur", "black", "skip", "mute"):
        assert action not in raw, action


# ---------------------------------------------------------------------------
# renderer: removed detections are skipped; overrides re-gate the render
# ---------------------------------------------------------------------------

def test_plan_timeline_skips_removed_visual_detection():
    identity = _identity(100.0)
    # Nudity is at the default med threshold and would normally be blurred.
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a",
                         confidence=0.9, level=SeverityLevel.LOW, level_override="removed")
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[])
    prefs = Preferences(nudity_visual=VisualAction.BLUR)

    timeline = plan_timeline(scan, prefs, 100.0)
    cats = [s.category for s in timeline if s.category is not None]
    # Removed -> no segment at all, even though the settings would catch it.
    assert Category.NUDITY not in cats
    assert timeline[0].visual == VisualAction.NONE


def test_plan_timeline_re_gates_visual_by_override_level():
    identity = _identity(100.0)
    # Scanned level LOW (worst) would be caught at the default med threshold,
    # but the user overrode it to exhigh (mild) -> no longer caught.
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a",
                         confidence=0.9, level=SeverityLevel.LOW, level_override="exhigh")
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[])
    prefs = Preferences(nudity_visual=VisualAction.BLUR)  # default med threshold

    timeline = plan_timeline(scan, prefs, 100.0)
    cats = [s.category for s in timeline if s.category is not None]
    assert Category.NUDITY not in cats  # exhigh is below the med threshold


def test_plan_timeline_override_promotes_visual_to_caught():
    identity = _identity(100.0)
    # Scanned level exhigh (mild) would NOT be caught at the default med
    # threshold, but the user overrode it to low (worst) -> now caught.
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a",
                         confidence=0.9, level=SeverityLevel.EXHIGH, level_override="low")
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[])
    prefs = Preferences(nudity_visual=VisualAction.BLUR)  # default med threshold

    timeline = plan_timeline(scan, prefs, 100.0)
    nud = next(s for s in timeline if s.category == Category.NUDITY)
    assert nud.visual == VisualAction.BLUR


def test_plan_timeline_skips_removed_language_detection():
    identity = _identity(100.0)
    # LOW "fuck" (worst) is confirmed and would be muted at the default med
    # threshold, but the user removed it.
    ld = LanguageDetection(start=40.0, end=40.3, utterance_start=39.0, utterance_end=42.0,
                           word="fuck", transcript_source="whisper", llm_confirmed=True,
                           level=SeverityLevel.LOW, level_override="removed")
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])

    # Default: foul_language_visual NONE -> only the audio mute path matters.
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_WORD)
    assert _language_mute_ranges_within(scan, prefs, _seg(0.0, 100.0)) == []


def test_plan_timeline_re_gates_language_by_override():
    identity = _identity(100.0)
    # "crap" scanned HIGH (mild) would not be muted at the med threshold; the
    # user overrode it to LOW (worst) -> now muted.
    ld = LanguageDetection(start=40.0, end=40.3, utterance_start=39.0, utterance_end=42.0,
                           word="crap", transcript_source="whisper", llm_confirmed=True,
                           level=SeverityLevel.HIGH, level_override="low")
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_WORD)  # default med threshold

    ranges = _language_mute_ranges_within(scan, prefs, _seg(0.0, 100.0))
    assert len(ranges) == 1  # the overridden-to-low word is now muted


def _seg(start, end, category=None):
    from nuclearcutter.render.renderer import TimelineSegment
    return TimelineSegment(start=start, end=end, visual=VisualAction.NONE,
                           audio=AudioAction.NONE, category=category)


# ---------------------------------------------------------------------------
# server: POST /api/detection/override
# ---------------------------------------------------------------------------

def _state_with_scan(tmp):
    st = AppState()
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a",
                         confidence=0.9, level=SeverityLevel.LOW)
    ld = LanguageDetection(start=40.0, end=40.3, utterance_start=39.0, utterance_end=42.0,
                           word="damn", transcript_source="whisper", llm_confirmed=True,
                           level=SeverityLevel.LOW)
    result = ScanResult(schema_version=1, identity=_identity(),
                        visual_detections=[vd], language_detections=[ld])
    path = tmp / "scan.json"
    result.save(path)
    st.scan_result = result
    st.scan_result_path = str(path)
    return st, path


def test_override_endpoint_sets_and_persists(tmp_path):
    st, path = _state_with_scan(tmp_path)
    c = TestClient(create_app(st))

    r = c.post("/api/detection/override", json={"type": "visual", "index": 0, "level": "high"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["level"] == "high"
    assert body["saved"] is True

    # It landed on the in-memory detection...
    assert st.scan_result.visual_detections[0].level_override == "high"
    # ...and was persisted to the scan file the render reads.
    reloaded = ScanResult.load(path)
    assert reloaded.visual_detections[0].level_override == "high"


def test_override_endpoint_sets_removed(tmp_path):
    st, path = _state_with_scan(tmp_path)
    c = TestClient(create_app(st))

    r = c.post("/api/detection/override", json={"type": "language", "index": 0, "level": "removed"})
    assert r.status_code == 200
    assert r.json()["level"] == "removed"
    assert st.scan_result.language_detections[0].is_removed is True
    reloaded = ScanResult.load(path)
    assert reloaded.language_detections[0].level_override == "removed"


def test_override_endpoint_clears_override(tmp_path):
    st, path = _state_with_scan(tmp_path)
    c = TestClient(create_app(st))
    st.scan_result.visual_detections[0].level_override = "high"

    r = c.post("/api/detection/override", json={"type": "visual", "index": 0, "level": None})
    assert r.status_code == 200
    assert r.json()["level"] is None
    assert st.scan_result.visual_detections[0].level_override is None
    assert ScanResult.load(path).visual_detections[0].level_override is None


def test_override_endpoint_rejects_bad_level(tmp_path):
    st, _ = _state_with_scan(tmp_path)
    c = TestClient(create_app(st))
    r = c.post("/api/detection/override", json={"type": "visual", "index": 0, "level": "bogus"})
    assert r.status_code == 400


def test_override_endpoint_rejects_bad_type(tmp_path):
    st, _ = _state_with_scan(tmp_path)
    c = TestClient(create_app(st))
    r = c.post("/api/detection/override", json={"type": "audio", "index": 0, "level": "high"})
    assert r.status_code == 400


def test_override_endpoint_bad_index_404(tmp_path):
    st, _ = _state_with_scan(tmp_path)
    c = TestClient(create_app(st))
    r = c.post("/api/detection/override", json={"type": "visual", "index": 5, "level": "high"})
    assert r.status_code == 404


def test_override_endpoint_no_scan_400():
    st = AppState()  # no scan loaded
    c = TestClient(create_app(st))
    r = c.post("/api/detection/override", json={"type": "visual", "index": 0, "level": "high"})
    assert r.status_code == 400


def test_timeline_payload_includes_override():
    """The timeline the GUI renders must expose index + level_override so the
    popup can offer the override controls and pre-fill the stored value."""
    st = _build_state_inline()
    c = TestClient(create_app(st))
    tl = c.get("/api/state").json()["timeline"]
    vd = tl["visual_detections"][0]
    assert "index" in vd and vd["index"] == 0
    assert "level_override" in vd
    ld = tl["language_detections"][0]
    assert "index" in ld and ld["index"] == 0
    assert "level_override" in ld


def _build_state_inline():
    """Build an in-memory AppState with a scan (no file) for the payload test."""
    st = AppState()
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a",
                         confidence=0.9, level=SeverityLevel.LOW, level_override="removed")
    ld = LanguageDetection(start=40.0, end=40.3, utterance_start=39.0, utterance_end=42.0,
                           word="damn", transcript_source="whisper", llm_confirmed=True,
                           level=SeverityLevel.LOW, level_override="high")
    st.scan_result = ScanResult(schema_version=1, identity=_identity(),
                                visual_detections=[vd], language_detections=[ld])
    return st
