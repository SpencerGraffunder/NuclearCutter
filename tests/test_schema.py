"""The scan file: what is stored, what round-trips, and what is refused.

A scan file is the whole product — it is written once, read on every render, and
meant to be shared between people with different tastes. So the rules under test
are: it records facts (tags), never opinions (levels); loading it cannot invent a
protection the file does not have; and a file from the old taxonomy is REFUSED
rather than quietly reinterpreted.
"""

from __future__ import annotations

import json

import pytest

from nuclearcutter.schema import (
    SCHEMA_VERSION, AudioAction, FilmIdentity, LanguageDetection, Preferences,
    ScanResult, UnsupportedScanVersion, VisualAction, VisualDetection,
)
from nuclearcutter.taxonomy import Minors, Rendering, Selection


def _vis(**kw) -> VisualDetection:
    return VisualDetection(
        start=kw.get("start", 10.0), end=kw.get("end", 13.0),
        description=kw.get("description", "a shirtless man walks in"),
        confidence=kw.get("confidence", 0.8),
        tags=kw.get("tags", ["nudity.shirtless_male"]),
        rendering=kw.get("rendering", Rendering.LIVE_ACTION.value),
        minors=kw.get("minors", Minors.NO.value),
        uncertain_tags=kw.get("uncertain_tags", []),
        triggers=kw.get("triggers", ["skin"]),
        removed=kw.get("removed", False))


def _lang(**kw) -> LanguageDetection:
    return LanguageDetection(
        start=kw.get("start", 5.0), end=kw.get("end", 5.4),
        utterance_start=kw.get("utterance_start", 4.0),
        utterance_end=kw.get("utterance_end", 7.0),
        text=kw.get("text", "damn"), tags=kw.get("tags", ["language.profanity_mild"]))


def _scan(**kw) -> ScanResult:
    return ScanResult(
        schema_version=SCHEMA_VERSION,
        identity=FilmIdentity(title=kw.get("title", "Test"), year=2026,
                              duration_seconds=kw.get("duration", 100.0),
                              phash_samples=[]),
        visual_detections=kw.get("visual", []),
        language_detections=kw.get("language", []),
        generator={"nuclearcutter": "0.2.0", "taxonomy_version": 2})


# --------------------------------------------------------------------------
# Version discipline
# --------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "movie.nuclearcutter.json"
    original = _scan(visual=[_vis()], language=[_lang()])
    original.save(path)
    back = ScanResult.load(path)
    assert back.visual_detections[0].tags == ["nudity.shirtless_male"]
    assert back.language_detections[0].tags == ["language.profanity_mild"]
    assert back.identity.title == "Test"


def test_old_severity_file_is_refused_not_guessed(tmp_path):
    # v1 stored category + level. There is no honest way to turn "nudity/high"
    # back into "what is in the frame", and guessing would silently change what
    # gets censored. So: refuse, and say what to do.
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "identity": {"title": "Old", "duration_seconds": 10.0, "phash_samples": []},
        "visual_detections": [{"start": 1.0, "end": 2.0, "category": "NUDITY",
                               "level": "high", "description": "x",
                               "confidence": 0.9}],
        "language_detections": [],
    }))
    with pytest.raises(UnsupportedScanVersion) as exc:
        ScanResult.load(path)
    assert "scan" in str(exc.value).lower()  # tells you what to do about it


def test_missing_version_is_refused(tmp_path):
    path = tmp_path / "nover.json"
    path.write_text(json.dumps({"identity": {"title": "x", "duration_seconds": 1.0,
                                             "phash_samples": []},
                                "visual_detections": [], "language_detections": []}))
    with pytest.raises(UnsupportedScanVersion):
        ScanResult.load(path)


def test_future_version_is_refused(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 99,
                                "identity": {"title": "x", "duration_seconds": 1.0,
                                             "phash_samples": []},
                                "visual_detections": [], "language_detections": []}))
    with pytest.raises(UnsupportedScanVersion):
        ScanResult.load(path)


def test_written_file_declares_its_version_and_taxonomy(tmp_path):
    path = tmp_path / "m.json"
    _scan(visual=[_vis()]).save(path)
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == 2
    assert raw["generator"]["taxonomy_version"] == 2
    # No opinion-shaped key may survive anywhere in the file.
    text = json.dumps(raw).lower()
    for banned in ("severity", '"level"', "exhigh", '"category"'):
        assert banned not in text, banned


# --------------------------------------------------------------------------
# Unknown tag ids survive a load (they are someone else's vocabulary, not junk)
# --------------------------------------------------------------------------

def test_unknown_tag_ids_are_preserved_on_load(tmp_path):
    # A scan file written by a newer build stays readable, and its unfamiliar
    # tags stay censorable rather than being deleted as invalid.
    path = tmp_path / "newer.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "identity": {"title": "x", "duration_seconds": 10.0, "phash_samples": []},
        "visual_detections": [{"start": 1.0, "end": 2.0, "tags": ["nudity.future_tag"],
                               "description": "", "confidence": 0.5,
                               "rendering": "live_action", "minors": "no",
                               "uncertain_tags": [], "triggers": [],
                               "dismissed_tags": []}],
        "language_detections": [],
    }))
    det = ScanResult.load(path).visual_detections[0]
    assert det.tags == ["nudity.future_tag"]
    assert det.censorable(Selection()) == ["nudity.future_tag"]


def test_minor_protection_is_reapplied_on_load(tmp_path):
    # A hand-edited (or buggy) file that dropped the protection tag must not be
    # able to un-censor a scene involving a child.
    path = tmp_path / "edited.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "identity": {"title": "x", "duration_seconds": 10.0, "phash_samples": []},
        "visual_detections": [{"start": 1.0, "end": 2.0,
                               "tags": ["nudity.full_nude_female"],
                               "description": "", "confidence": 0.5,
                               "rendering": "live_action", "minors": "yes",
                               "uncertain_tags": [], "triggers": [],
                               "dismissed_tags": []}],
        "language_detections": [],
    }))
    det = ScanResult.load(path).visual_detections[0]
    assert "safety.exposure_with_minor" in det.tags
    assert det.censorable(Selection()) == ["safety.exposure_with_minor"]


# --------------------------------------------------------------------------
# censorable(): the one matching rule
# --------------------------------------------------------------------------

def test_censorable_is_empty_when_nothing_matches():
    assert _vis().censorable(Selection()) == []


def test_censorable_matches_a_selected_tag():
    sel = Selection(tags={"nudity.shirtless_male"})
    assert _vis().censorable(sel) == ["nudity.shirtless_male"]


def test_censorable_is_empty_when_rendering_is_excluded():
    sel = Selection(tags={"nudity.shirtless_male"}, renderings={"anime"})
    assert _vis(rendering="live_action").censorable(sel) == []


def test_censorable_anime_detection_matches_when_anime_included():
    det = _vis(rendering="anime")
    sel = Selection(tags={"nudity.shirtless_male"}, renderings={"anime"})
    assert det.censorable(sel) == ["nudity.shirtless_male"]


def test_a_removed_detection_is_not_censorable():
    det = _vis(removed=True)
    sel = Selection(tags={"nudity.shirtless_male"})
    assert det.censorable(sel) == []


def test_removing_one_tag_does_not_remove_the_others():
    # The popup unticks what is not there; the rest of the detection still acts.
    det = _vis(tags=["nudity.shirtless_male", "gore.blood_moderate"])
    sel = Selection(tags={"nudity.shirtless_male", "gore.blood_moderate"})
    assert det.censorable(sel) == ["nudity.shirtless_male", "gore.blood_moderate"]
    det.tags = ["gore.blood_moderate"]
    assert det.censorable(sel) == ["gore.blood_moderate"]


def test_removal_cannot_walk_around_the_child_floor():
    # The one thing "I looked at it and I allow it" is not allowed to do.
    det = _vis(tags=["nudity.full_nude_female", "safety.exposure_with_minor"],
               minors=Minors.YES.value, removed=True)
    assert det.censorable(Selection()) == ["safety.exposure_with_minor"]
    assert det.is_removed is True  # the timeline still shows it as allowed


def test_removed_language_detection_is_not_censorable():
    d = _lang()
    d.removed = True
    assert d.censorable(Selection(tags={"language.profanity_mild"})) == []


def test_language_detection_censorable():
    d = _lang()
    assert d.censorable(Selection(tags={"language.profanity_mild"})) == \
        ["language.profanity_mild"]
    assert d.censorable(Selection()) == []


def test_uncertain_is_a_fact_not_a_verdict():
    det = _vis(tags=["nudity.unclear"], uncertain_tags=["nudity.unclear"])
    assert det.is_uncertain is True
    # Being uncertain does not censor anything by itself; `nudity.unclear` is an
    # ordinary checkbox the viewer can turn on.
    assert det.censorable(Selection()) == []


# --------------------------------------------------------------------------
# ScanResult summaries
# --------------------------------------------------------------------------

def test_tag_counts_for_the_scan_summary():
    scan = _scan(visual=[_vis(tags=["nudity.shirtless_male", "gore.blood_moderate"]),
                         _vis(tags=["nudity.shirtless_male"], start=20.0, end=22.0)],
                 language=[_lang()])
    counts = scan.tag_counts()
    assert counts["nudity.shirtless_male"] == 2
    assert counts["gore.blood_moderate"] == 1
    assert counts["language.profanity_mild"] == 1


def test_empty_scan_is_valid():
    scan = _scan()
    assert scan.visual_detections == []
    assert scan.tag_counts() == {}


# --------------------------------------------------------------------------
# Preferences: what to DO about a match
# --------------------------------------------------------------------------

def test_defaults_are_blur_for_visual_and_mute_phrase_for_language():
    prefs = Preferences()
    assert prefs.visual_for("nudity") is VisualAction.BLUR
    assert prefs.visual_for("language") is VisualAction.NONE
    assert prefs.audio_for("language") is AudioAction.MUTE_PHRASE
    assert prefs.audio_for("nudity") is AudioAction.NONE


def test_visual_for_accepts_enum_or_string():
    from nuclearcutter.taxonomy import Family

    prefs = Preferences(visual_actions={"gore": "black"})
    assert prefs.visual_for(Family.GORE) is VisualAction.BLACK
    assert prefs.visual_for("gore") is VisualAction.BLACK


def test_unknown_family_falls_back_to_a_safe_action():
    prefs = Preferences()
    assert prefs.visual_for("alien_family") in (VisualAction.BLUR, VisualAction.BLACK)
    assert prefs.audio_for("alien_family") is AudioAction.NONE


def test_preferences_round_trip_keeps_selection_and_actions(tmp_path):
    prefs = Preferences(selection=Selection(tags={"gore.blood_moderate"},
                                            renderings={"live_action"}),
                        visual_actions={"violence": "black"},
                        blur_strength=2.0, blur_padding=0.5)
    path = tmp_path / "prefs.json"
    prefs.save(path)
    back = Preferences.load(path)
    assert back.selection.tags == {"gore.blood_moderate"}
    assert back.selection.renderings == {"live_action"}
    assert back.visual_for("violence") is VisualAction.BLACK
    assert back.blur_strength == 2.0
    assert back.blur_padding == 0.5


def test_preferences_load_of_a_file_with_levels_is_ignored_not_fatal(tmp_path):
    # Someone's old settings.json has nudity_level: "high". There is no
    # equivalent, so the default is used; crashing on a stale preference file
    # would be a worse outcome than silently resetting one screen.
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"nudity_level": "high", "selection": ["gore.blood_moderate"]}))
    back = Preferences.load(path)
    assert back.visual_for("nudity") is VisualAction.BLUR
    assert back.selection.tags == {"gore.blood_moderate"}


def test_selection_accepts_a_bare_list_of_ids():
    prefs = Preferences.from_dict({"selection": ["nudity.shirtless_male"]})
    assert prefs.selection.tags == {"nudity.shirtless_male"}


def test_audio_actions_replace_is_reported_as_unimplemented_muting():
    # The GUI can offer it; the renderer mutes. The enum must keep the value so
    # a saved preference file still loads.
    assert AudioAction.REPLACE_WORD.value == "replace_word"
    assert AudioAction.MUTE_WORD.mutes_audio is True
    assert AudioAction.NONE.mutes_audio is False
