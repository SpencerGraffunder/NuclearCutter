import json
import tempfile
from pathlib import Path

import pytest

from nuclearcutter.schema import (
    Action, Category, FilmIdentity, LanguageDetection, Preferences, ScanResult, VisualDetection,
)


def make_scan_result():
    identity = FilmIdentity(title="Test", year=2024, duration_seconds=100.0, phash_samples=[])
    vd = VisualDetection(category=Category.NUDITY, start=1.0, end=2.0, description="d", confidence=0.9)
    ld = LanguageDetection(
        start=3.0, end=3.5, utterance_start=2.5, utterance_end=4.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    return ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[ld])


def test_scan_result_round_trip():
    result = make_scan_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scan.json"
        result.save(path)
        loaded = ScanResult.load(path)

    assert loaded.identity.title == "Test"
    assert loaded.visual_detections[0].category == Category.NUDITY
    assert loaded.language_detections[0].word == "damn"
    assert loaded.language_detections[0].llm_confirmed is True


def test_scan_result_does_not_store_actions():
    """Critical invariant from SPEC.md section 6: scan results must never
    encode which action a user chose, only what's in the film."""
    result = make_scan_result()
    raw = json.dumps(result.to_dict())
    assert "blur" not in raw
    assert "skip" not in raw
    assert "mute" not in raw


def test_preferences_round_trip():
    prefs = Preferences(
        nudity_action=Action.BLUR,
        nudity_blur_mute_audio=True,
        intimate_scenes_action=Action.BLUR,
        foul_language_action=Action.MUTE,
        foul_language_mute_scope="utterance",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prefs.json"
        prefs.save(path)
        loaded = Preferences.load(path)

    assert loaded.nudity_action == Action.BLUR
    assert loaded.nudity_blur_mute_audio is True
    assert loaded.foul_language_mute_scope == "utterance"


def test_preferences_defaults():
    prefs = Preferences()
    assert prefs.nudity_action == Action.BLUR
    assert prefs.foul_language_action == Action.MUTE
    assert prefs.action_for(Category.NUDITY) == Action.BLUR
    assert prefs.action_for(Category.FOUL_LANGUAGE) == Action.MUTE


def test_action_for_all_categories():
    prefs = Preferences(
        nudity_action=Action.BLUR,
        intimate_scenes_action=Action.NONE,
        foul_language_action=Action.MUTE,
    )
    assert prefs.action_for(Category.NUDITY) == Action.BLUR
    assert prefs.action_for(Category.INTIMATE_SCENES) == Action.NONE
    assert prefs.action_for(Category.FOUL_LANGUAGE) == Action.MUTE
