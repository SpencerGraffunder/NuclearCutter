import json
import tempfile
from pathlib import Path

import pytest

from nuclearcutter.schema import (
    AudioAction, Category, FilmIdentity, LanguageDetection, Preferences, ScanResult, SeverityLevel,
    VisualAction, VisualDetection,
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
        nudity_visual=VisualAction.BLUR,
        nudity_audio=AudioAction.MUTE_SCENE,
        nudity_level=SeverityLevel.HIGH,
        gore_visual=VisualAction.BLACK,
        gore_audio=AudioAction.NONE,
        gore_level=SeverityLevel.LOW,
        violence_visual=VisualAction.NONE,
        violence_audio=AudioAction.MUTE_SCENE,
        violence_level=SeverityLevel.EXHIGH,
        foul_language_visual=VisualAction.NONE,
        foul_language_audio=AudioAction.MUTE_PHRASE,
        foul_language_level=SeverityLevel.HIGH,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prefs.json"
        prefs.save(path)
        loaded = Preferences.load(path)

    assert loaded.nudity_visual == VisualAction.BLUR
    assert loaded.nudity_audio == AudioAction.MUTE_SCENE
    assert loaded.gore_visual == VisualAction.BLACK
    assert loaded.violence_audio == AudioAction.MUTE_SCENE
    assert loaded.foul_language_audio == AudioAction.MUTE_PHRASE
    assert loaded.nudity_level == SeverityLevel.HIGH
    assert loaded.gore_level == SeverityLevel.LOW
    assert loaded.violence_level == SeverityLevel.EXHIGH
    assert loaded.foul_language_level == SeverityLevel.HIGH


def test_preferences_defaults():
    prefs = Preferences()
    assert prefs.nudity_visual == VisualAction.BLUR
    assert prefs.nudity_audio == AudioAction.NONE
    assert prefs.gore_visual == VisualAction.BLUR
    assert prefs.violence_visual == VisualAction.BLUR
    assert prefs.foul_language_visual == VisualAction.NONE
    assert prefs.foul_language_audio == AudioAction.MUTE_PHRASE
    assert prefs.visual_for(Category.NUDITY) == VisualAction.BLUR
    assert prefs.audio_for(Category.FOUL_LANGUAGE) == AudioAction.MUTE_PHRASE
    assert prefs.level_for(Category.NUDITY) == SeverityLevel.MED
    assert prefs.level_for(Category.FOUL_LANGUAGE) == SeverityLevel.MED


def test_severity_level_ordering():
    assert SeverityLevel.LOW.rank < SeverityLevel.MED.rank
    assert SeverityLevel.MED.rank < SeverityLevel.HIGH.rank
    assert SeverityLevel.HIGH.rank < SeverityLevel.EXHIGH.rank
    # low = least censorship (only the worst content); exhigh = most.
    # A detection is corrected if its level is at/above (<= rank) the threshold.
    assert SeverityLevel.LOW.is_corrected_by(SeverityLevel.LOW) is True
    assert SeverityLevel.LOW.is_corrected_by(SeverityLevel.MED) is True
    assert SeverityLevel.MED.is_corrected_by(SeverityLevel.LOW) is False
    assert SeverityLevel.HIGH.is_corrected_by(SeverityLevel.MED) is False
    assert SeverityLevel.EXHIGH.is_corrected_by(SeverityLevel.EXHIGH) is True
    assert SeverityLevel.from_any("high") == SeverityLevel.HIGH
    assert SeverityLevel.from_any("bogus") == SeverityLevel.MED
    assert SeverityLevel.from_any(None) == SeverityLevel.MED


def test_visual_detection_round_trip_preserves_level():
    vd = VisualDetection(category=Category.NUDITY, start=1.0, end=2.0, description="d",
                         confidence=0.9, level=SeverityLevel.EXHIGH)
    d = vd.to_dict()
    assert d["level"] == "exhigh"
    loaded = VisualDetection.from_dict(d)
    assert loaded.level == SeverityLevel.EXHIGH


def test_visual_detection_legacy_defaults_level_to_med():
    vd = VisualDetection.from_dict({"category": "nudity", "start": 1.0, "end": 2.0,
                                    "description": "d", "confidence": 0.9})
    assert vd.level == SeverityLevel.MED


def test_language_detection_round_trip_preserves_level():
    ld = LanguageDetection(start=1.0, end=1.2, utterance_start=0.5, utterance_end=2.0,
                           word="fuck", transcript_source="whisper", llm_confirmed=True,
                           level=SeverityLevel.HIGH)
    loaded = LanguageDetection.from_dict(ld.to_dict())
    assert loaded.level == SeverityLevel.HIGH


def test_visual_and_audio_for_all_categories():
    prefs = Preferences(
        nudity_visual=VisualAction.BLUR,
        nudity_audio=AudioAction.NONE,
        gore_visual=VisualAction.BLACK,
        gore_audio=AudioAction.MUTE_SCENE,
        violence_visual=VisualAction.NONE,
        violence_audio=AudioAction.NONE,
        foul_language_visual=VisualAction.NONE,
        foul_language_audio=AudioAction.MUTE_WORD,
    )
    assert prefs.visual_for(Category.NUDITY) == VisualAction.BLUR
    assert prefs.audio_for(Category.NUDITY) == AudioAction.NONE
    assert prefs.visual_for(Category.GORE) == VisualAction.BLACK
    assert prefs.audio_for(Category.GORE) == AudioAction.MUTE_SCENE
    assert prefs.visual_for(Category.VIOLENCE) == VisualAction.NONE
    assert prefs.audio_for(Category.FOUL_LANGUAGE) == AudioAction.MUTE_WORD


def test_audio_action_mutes_audio_includes_mute_scene():
    assert AudioAction.MUTE_SCENE.mutes_audio is True
    assert AudioAction.MUTE_WORD.mutes_audio is True
    assert AudioAction.NONE.mutes_audio is False


def test_category_from_legacy_maps_old_names():
    assert Category.from_legacy("intimate_scenes") == Category.NUDITY
    assert Category.from_legacy("gore_violence") == Category.GORE
    assert Category.from_legacy("nudity") == Category.NUDITY
    assert Category.from_legacy("gore") == Category.GORE
    assert Category.from_legacy("violence") == Category.VIOLENCE
    assert Category.from_legacy("foul_language") == Category.FOUL_LANGUAGE
