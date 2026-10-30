"""Timeline planning: the moment a scan file and a viewer's choices meet.

`plan_timeline` is the only place a detection becomes an instruction to hide
something, so this is where the redesign is either real or cosmetic. Under v1 it
read a severity level and compared it against a threshold ("is this scene worse
than the user's nudity tolerance?"). Now it asks one question — does the viewer's
selection contain any of this detection's tags? — and then applies the strongest
action among the families the detection matched.

These tests are pure-function tests: no ffmpeg, no model, no files.
"""

from __future__ import annotations

import pytest

from nuclearcutter.render.renderer import TimelineSegment, plan_timeline
from nuclearcutter.schema import (
    SCHEMA_VERSION, AudioAction, FilmIdentity, LanguageDetection, Preferences,
    ScanResult, VisualAction, VisualDetection,
)
from nuclearcutter.taxonomy import Family, Minors, Rendering, Selection


def _vis(tags, start=10.0, end=13.0, description="something happens",
         rendering=Rendering.LIVE_ACTION.value, minors=Minors.NO.value,
         removed=False) -> VisualDetection:
    return VisualDetection(start=start, end=end, description=description,
                           confidence=0.8, tags=list(tags), rendering=rendering,
                           minors=minors, triggers=["skin"], removed=removed)


def _lang(text="damn", tags=("language.profanity_mild",), start=5.0, end=5.4,
          utt=(4.0, 7.0), removed=False) -> LanguageDetection:
    return LanguageDetection(start=start, end=end, utterance_start=utt[0],
                             utterance_end=utt[1], text=text, tags=list(tags),
                             llm_confirmed=True, removed=removed)


def _scan(visual=(), language=()) -> ScanResult:
    return ScanResult(
        schema_version=SCHEMA_VERSION,
        identity=FilmIdentity(title="T", year=None, duration_seconds=100.0,
                              phash_samples=[]),
        visual_detections=list(visual), language_detections=list(language))


def _prefs(selected=(), visual_actions=None, audio_actions=None, **kw) -> Preferences:
    return Preferences(selection=Selection(tags=set(selected)),
                       visual_actions=visual_actions or {},
                       audio_actions=audio_actions or {}, **kw)


def _censored(timeline: list[TimelineSegment]) -> list[TimelineSegment]:
    return [s for s in timeline if s.visual != VisualAction.NONE]


def _clean(timeline: list[TimelineSegment]) -> list[TimelineSegment]:
    return [s for s in timeline if s.visual == VisualAction.NONE]


# --------------------------------------------------------------------------
# Nothing selected = nothing touched (the user's explicit default)
# --------------------------------------------------------------------------

def test_nothing_selected_leaves_the_whole_film_alone():
    scan = _scan(visual=[_vis(["nudity.full_nude_female"]),
                         _vis(["gore.blood_heavy"], start=40.0, end=44.0)],
                 language=[_lang()])
    timeline = plan_timeline(scan, _prefs(), duration=100.0)
    assert _censored(timeline) == []
    assert len(timeline) == 1  # one untouched segment covering the film
    assert timeline[0].start == 0.0 and timeline[0].end == 100.0


def test_selecting_one_tag_censors_only_matching_detections():
    scan = _scan(visual=[_vis(["nudity.shirtless_male"], start=10.0, end=12.0),
                         _vis(["gore.blood_heavy"], start=50.0, end=52.0)])
    timeline = plan_timeline(scan, _prefs({"nudity.shirtless_male"}), duration=100.0)
    hits = _censored(timeline)
    assert len(hits) == 1
    assert hits[0].start == pytest.approx(10.0)
    assert hits[0].families == ["nudity"]


def test_the_severity_machine_is_unreachable_from_the_package():
    # Checked by IMPORTING every module, not by grepping: prose is allowed to
    # explain what was replaced ("...what a SeverityLevel used to mean..."), but
    # no module may still be able to reach the old machinery.
    import importlib
    import pkgutil

    import nuclearcutter

    offenders: list[str] = []
    for info in pkgutil.walk_packages(nuclearcutter.__path__, "nuclearcutter."):
        if ".tests" in info.name:
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception:  # a module that will not import is a failure anyway
            offenders.append(f"{info.name}: import error")
            continue
        for sym in ("SeverityLevel", "effective_level", "level_for",
                    "is_corrected_by", "DEFAULT_LEVEL_SCALES",
                    "DEFAULT_FOUL_LANGUAGE_SCALE", "Category"):
            if hasattr(mod, sym):
                offenders.append(f"{info.name}: {sym}")
    assert offenders == [], offenders


# --------------------------------------------------------------------------
# One detection, several families: strongest action wins
# --------------------------------------------------------------------------

def test_a_detection_matching_two_families_takes_the_strongest_visual_action():
    det = _vis(["gore.wound_detail", "violence.beating_prolonged"], start=20.0, end=24.0)
    prefs = _prefs({"gore.wound_detail", "violence.beating_prolonged"},
                   visual_actions={"gore": "blur", "violence": "black"})
    hits = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0))
    assert len(hits) == 1
    assert hits[0].visual is VisualAction.BLACK
    assert set(hits[0].families) == {"gore", "violence"}


def test_a_detection_matching_a_family_set_to_none_still_uses_the_other():
    # Blood that is also violence, where the viewer filters gore but not
    # violence: the scene is still hidden, because gore asked for it.
    det = _vis(["gore.blood_moderate", "violence.fight_armed"])
    prefs = _prefs({"gore.blood_moderate"},
                   visual_actions={"gore": "blur", "violence": "none"})
    hits = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0))
    assert len(hits) == 1
    assert hits[0].visual is VisualAction.BLUR


def test_scene_mute_comes_from_a_visual_family_asking_for_it():
    det = _vis(["violence.beating_prolonged"])
    prefs = _prefs({"violence.beating_prolonged"},
                   visual_actions={"violence": "black"},
                   audio_actions={"violence": "mute_scene"})
    seg = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0))[0]
    assert seg.scene_mute is True
    assert seg.audio is AudioAction.MUTE_SCENE


def test_audio_is_left_alone_unless_a_family_asks():
    det = _vis(["gore.blood_moderate"])
    prefs = _prefs({"gore.blood_moderate"}, audio_actions={"gore": "none"})
    seg = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0))[0]
    assert seg.audio is AudioAction.NONE


# --------------------------------------------------------------------------
# Rendering narrows
# --------------------------------------------------------------------------

def test_anime_detection_is_untouched_when_only_live_action_is_included():
    det = _vis(["nudity.full_nude_female"], rendering=Rendering.ANIME.value)
    prefs = _prefs({"nudity.full_nude_female"})
    prefs.selection.renderings = {Rendering.LIVE_ACTION.value}
    assert _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0)) == []


def test_stage_rendering_can_be_excluded_from_a_filter():
    # The Hadestown case: a filmed stage show where nothing is actually bare.
    det = _vis(["nudity.shoulders_back_bare_female"], rendering=Rendering.STAGE.value)
    prefs = _prefs({"nudity.shoulders_back_bare_female"})
    prefs.selection.renderings = {Rendering.LIVE_ACTION.value, Rendering.ANIMATED.value}
    assert _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0)) == []

    prefs.selection.renderings = {Rendering.STAGE.value}
    assert len(_censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0))) == 1


# --------------------------------------------------------------------------
# Removal and the child floor
# --------------------------------------------------------------------------

def test_a_removed_detection_is_not_planned():
    det = _vis(["gore.blood_heavy"], removed=True)
    prefs = _prefs({"gore.blood_heavy"})
    assert _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0)) == []


def test_removing_a_child_protection_scene_still_censors_the_protection():
    det = _vis(["nudity.full_nude_female", "safety.exposure_with_minor"],
               minors=Minors.YES.value, removed=True)
    prefs = _prefs()  # nothing selected at all
    hits = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0))
    assert len(hits) == 1, "the enforced tag must survive removal"
    assert "safety.exposure_with_minor" in hits[0].tags


def test_uncertain_child_age_is_an_ordinary_choice():
    det = _vis(["safety.exposure_age_uncertain"], minors=Minors.UNCERTAIN.value)
    assert _censored(plan_timeline(_scan(visual=[det]), _prefs(), 60.0)) == []
    prefs = _prefs({"safety.exposure_age_uncertain"})
    assert len(_censored(plan_timeline(_scan(visual=[det]), prefs, 60.0))) == 1


# --------------------------------------------------------------------------
# Geometry: coverage, overlap, padding
# --------------------------------------------------------------------------

def test_timeline_covers_the_film_with_no_gaps_or_overlaps():
    scan = _scan(visual=[_vis(["gore.blood_heavy"], start=10.0, end=20.0),
                         _vis(["violence.gunfire"], start=50.0, end=55.0)])
    prefs = _prefs({"gore.blood_heavy", "violence.gunfire"})
    timeline = plan_timeline(scan, prefs, duration=100.0)
    assert timeline[0].start == 0.0
    assert timeline[-1].end == 100.0
    for a, b in zip(timeline, timeline[1:]):
        assert a.end == pytest.approx(b.start)


def test_overlapping_detections_do_not_double_cover_or_crash():
    scan = _scan(visual=[_vis(["gore.blood_heavy"], start=10.0, end=30.0),
                         _vis(["violence.gunfire"], start=20.0, end=40.0)])
    prefs = _prefs({"gore.blood_heavy", "violence.gunfire"})
    timeline = plan_timeline(scan, prefs, duration=100.0)
    for a, b in zip(timeline, timeline[1:]):
        assert a.end <= b.start + 1e-6
    total = sum(s.source_duration for s in timeline)
    assert total == pytest.approx(100.0)


def test_a_detection_fully_inside_another_does_not_produce_a_zero_length_segment():
    scan = _scan(visual=[_vis(["gore.blood_heavy"], start=10.0, end=40.0),
                         _vis(["violence.gunfire"], start=20.0, end=22.0)])
    prefs = _prefs({"gore.blood_heavy", "violence.gunfire"})
    timeline = plan_timeline(scan, prefs, duration=100.0)
    assert all(s.end > s.start for s in timeline)


def test_out_of_order_detections_are_planned_in_time_order():
    scan = _scan(visual=[_vis(["violence.gunfire"], start=60.0, end=62.0),
                         _vis(["gore.blood_heavy"], start=10.0, end=12.0)])
    prefs = _prefs({"gore.blood_heavy", "violence.gunfire"})
    hits = _censored(plan_timeline(scan, prefs, duration=100.0))
    assert [h.start for h in hits] == sorted(h.start for h in hits)


def test_blur_padding_extends_both_sides_and_clamps_to_the_film():
    det = _vis(["gore.blood_heavy"], start=1.0, end=3.0)
    prefs = _prefs({"gore.blood_heavy"}, blur_padding=1.5)
    seg = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=100.0))[0]
    assert seg.start == pytest.approx(0.0)  # clamped, not negative
    assert seg.end == pytest.approx(4.5)


def test_negative_padding_is_treated_as_zero():
    det = _vis(["gore.blood_heavy"], start=10.0, end=12.0)
    prefs = _prefs({"gore.blood_heavy"}, blur_padding=-5.0)
    seg = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=100.0))[0]
    assert seg.start == pytest.approx(10.0) and seg.end == pytest.approx(12.0)


def test_a_detection_at_the_very_end_does_not_run_past_the_film():
    det = _vis(["gore.blood_heavy"], start=98.0, end=100.0)
    prefs = _prefs({"gore.blood_heavy"}, blur_padding=2.0)
    timeline = plan_timeline(_scan(visual=[det]), prefs, duration=100.0)
    assert timeline[-1].end == 100.0
    assert all(s.end <= 100.0 + 1e-6 for s in timeline)


# --------------------------------------------------------------------------
# Language: audio-only by default, and narrow
# --------------------------------------------------------------------------

def test_language_detection_does_not_create_a_video_segment_by_default():
    scan = _scan(language=[_lang()])
    timeline = plan_timeline(scan, _prefs({"language.profanity_mild"}), duration=60.0)
    assert _censored(timeline) == [], "swearing must not blank the screen"


def test_language_can_hide_the_picture_when_asked():
    scan = _scan(language=[_lang()])
    prefs = _prefs({"language.profanity_mild"}, visual_actions={"language": "blur"})
    hits = _censored(plan_timeline(scan, prefs, duration=60.0))
    assert len(hits) == 1
    # The whole sentence, not just the word: hiding the picture while a line is
    # being said has to cover the line.
    assert hits[0].start == pytest.approx(4.0)
    assert hits[0].end == pytest.approx(7.0)


def test_mute_windows_are_word_sized_for_mute_word():
    from nuclearcutter.render.renderer import _language_mute_ranges_within

    scan = _scan(language=[_lang(start=5.0, end=5.4, utt=(4.0, 8.0))])
    prefs = _prefs({"language.profanity_mild"}, audio_actions={"language": "mute_word"},
                   mute_padding=0.0)
    seg = TimelineSegment(start=0.0, end=20.0, visual=VisualAction.NONE,
                          audio=AudioAction.NONE)
    ranges = _language_mute_ranges_within(scan, prefs, seg)
    assert ranges == [(5.0, 5.4)]


def test_mute_windows_are_sentence_sized_for_mute_phrase():
    from nuclearcutter.render.renderer import _language_mute_ranges_within

    scan = _scan(language=[_lang(start=5.0, end=5.4, utt=(4.0, 8.0))])
    prefs = _prefs({"language.profanity_mild"},
                   audio_actions={"language": "mute_phrase"}, mute_padding=0.25)
    seg = TimelineSegment(start=0.0, end=20.0, visual=VisualAction.NONE,
                          audio=AudioAction.NONE)
    ranges = _language_mute_ranges_within(scan, prefs, seg)
    assert ranges == [(pytest.approx(3.75), pytest.approx(8.25))]


def test_mute_windows_are_clamped_to_the_segment():
    from nuclearcutter.render.renderer import _language_mute_ranges_within

    scan = _scan(language=[_lang(start=9.0, end=12.0, utt=(8.0, 30.0))])
    prefs = _prefs({"language.profanity_mild"}, audio_actions={"language": "mute_word"},
                   mute_padding=0.0)
    seg = TimelineSegment(start=0.0, end=10.0, visual=VisualAction.NONE,
                          audio=AudioAction.NONE)
    assert _language_mute_ranges_within(scan, prefs, seg) == [(9.0, 10.0)]


def test_mute_nothing_for_a_family_with_audio_none():
    from nuclearcutter.render.renderer import _language_mute_ranges_within

    scan = _scan(language=[_lang()])
    prefs = _prefs({"language.profanity_mild"}, audio_actions={"language": "none"})
    seg = TimelineSegment(start=0.0, end=30.0, visual=VisualAction.NONE,
                          audio=AudioAction.NONE)
    assert _language_mute_ranges_within(scan, prefs, seg) == []


def test_a_removed_language_detection_is_not_muted():
    from nuclearcutter.render.renderer import _language_mute_ranges_within

    scan = _scan(language=[_lang(removed=True)])
    prefs = _prefs({"language.profanity_mild"}, audio_actions={"language": "mute_word"})
    seg = TimelineSegment(start=0.0, end=30.0, visual=VisualAction.NONE,
                          audio=AudioAction.NONE)
    assert _language_mute_ranges_within(scan, prefs, seg) == []


def test_two_different_language_tags_mute_independently():
    from nuclearcutter.render.renderer import _language_mute_ranges_within

    scan = _scan(language=[_lang("damn", ["language.profanity_mild"], 5.0, 5.4),
                           _lang("nword", ["language.slur_racial"], 20.0, 20.6)])
    # Filter slurs only: mild swearing stays audible.
    prefs = _prefs({"language.slur_racial"}, audio_actions={"language": "mute_word"},
                   mute_padding=0.0)
    seg = TimelineSegment(start=0.0, end=40.0, visual=VisualAction.NONE,
                          audio=AudioAction.NONE)
    assert _language_mute_ranges_within(scan, prefs, seg) == [(20.0, 20.6)]


# --------------------------------------------------------------------------
# Card text
# --------------------------------------------------------------------------

def test_card_text_prefers_the_description():
    seg = TimelineSegment(start=0, end=1, visual=VisualAction.BLUR,
                          audio=AudioAction.NONE, description="a man draws a knife",
                          kinds="Weapon")
    assert seg.card_text == "a man draws a knife"


def test_card_text_falls_back_to_naming_what_was_filtered():
    # An unclassified detection has no description, and a blank card tells the
    # viewer nothing about what they missed.
    seg = TimelineSegment(start=0, end=1, visual=VisualAction.BLACK,
                          audio=AudioAction.NONE, description="",
                          kinds="Blood, Wound")
    assert seg.card_text == "Blood, Wound"


def test_card_text_never_empty():
    seg = TimelineSegment(start=0, end=1, visual=VisualAction.BLUR,
                          audio=AudioAction.NONE)
    assert seg.card_text.strip()


def test_plan_carries_tag_labels_for_the_caption():
    det = _vis(["gore.blood_heavy", "violence.gunfire"], description="")
    prefs = _prefs({"gore.blood_heavy", "violence.gunfire"})
    seg = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0))[0]
    assert seg.kinds and "," in seg.kinds
    assert seg.tags == ["gore.blood_heavy", "violence.gunfire"]


def test_unclear_detection_still_gets_a_card():
    det = _vis(["safety.unclassified"], description="")
    prefs = _prefs({"safety.unclassified"})
    seg = _censored(plan_timeline(_scan(visual=[det]), prefs, duration=60.0))[0]
    assert "could not name it" in seg.card_text.lower() or seg.kinds
