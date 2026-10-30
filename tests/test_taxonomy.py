"""The tag vocabulary: invariants that must hold or the whole design leaks.

These tests are mostly *table* tests, and that is deliberate. The vocabulary in
`nuclearcutter/taxonomy.py` is simultaneously the model's output alphabet, the
GUI's checkbox list, and the render's matching rule. A typo there is not a
cosmetic bug: an id in the prompt but not in TAGS gets folded into `unclear` on
every scan, and a tag whose family disagrees with its id prefix matches nothing.
"""

from __future__ import annotations

import pytest

from nuclearcutter.taxonomy import (
    FAMILY_ORDER, NON_CENSORABLE, TAGS, Family, Minors, Rendering, Selection,
    all_tag_ids, assert_vocabulary_sane, audio_options, default_selection,
    enforce_minor_protection, family_of, minor_locked_tags, normalize_tags,
    rendering_allowed, selection_named, tags_hit, tags_in_family,
)


class _Det:
    """Duck-typed stand-in so the taxonomy is testable without the schema."""

    def __init__(self, tags, minors=Minors.NO.value):
        self.tags = list(tags)
        self.minors = minors
        self.uncertain_tags = []


# --------------------------------------------------------------------------
# Vocabulary integrity
# --------------------------------------------------------------------------

def test_assert_vocabulary_sane_passes():
    assert_vocabulary_sane()  # raises AssertionError if the table is broken


def test_every_id_is_prefixed_with_its_own_family():
    for tag in TAGS.values():
        assert tag.id.startswith(tag.family.value + "."), tag.id


def test_ids_are_unique_and_match_the_dict():
    ids = all_tag_ids()
    assert len(ids) == len(set(ids))
    assert set(ids) == set(TAGS.keys())


def test_every_family_has_tags():
    for fam in Family:
        assert tags_in_family(fam), f"{fam.value} has no tags"


def test_families_cover_every_family():
    assert {f for f in FAMILY_ORDER} == set(Family)


def test_gendered_tags_are_only_where_gender_is_meaningful():
    # The user asked for men/women to be separable for modesty. Gore and
    # violence must NOT be gender-split — a wound is a wound, and splitting them
    # would double the chooser for no benefit.
    for tag in tags_in_family(Family.GORE) + tags_in_family(Family.VIOLENCE):
        assert tag.gender == "any", tag.id
    for tag in tags_in_family(Family.NUDITY):
        assert tag.gender in ("any", "male", "female"), tag.id


def test_nudity_has_parallel_male_and_female_tags():
    genders = {tag.gender for tag in tags_in_family(Family.NUDITY)}
    assert {"male", "female"} <= genders


def test_every_tag_acts_on_picture_or_sound():
    for tag in TAGS.values():
        assert tag.visual or tag.audio, f"{tag.id} can neither hide picture nor sound"


def test_language_family_is_audio_only():
    for tag in tags_in_family(Family.LANGUAGE):
        assert tag.audio and not tag.visual, tag.id


def test_slurs_are_their_own_tags_not_buried_in_profanity():
    slurs = [t.id for t in tags_in_family(Family.LANGUAGE) if t.id.startswith("language.slur_")]
    assert len(slurs) >= 5
    # The old scale filed slurs and "damn" in the same bucket; the new vocabulary
    # must keep them separable or the redesign has lost its main win.
    assert "language.profanity_mild" not in slurs


# --------------------------------------------------------------------------
# normalize_tags: model output can degrade, never disappear
# --------------------------------------------------------------------------

def test_normalize_keeps_known_ids_and_dedupes():
    out = normalize_tags(["nudity.shirtless_male", "nudity.shirtless_male",
                          "gore.blood_moderate"])
    assert out == ["nudity.shirtless_male", "gore.blood_moderate"]


def test_normalize_folds_unknown_id_into_family_residual():
    unknown: list[str] = []
    out = normalize_tags(["nudity.nude_woman_topless_half"], unknown)
    assert out == ["nudity.unclear"]
    assert unknown == ["nudity.nude_woman_topless_half"]


def test_normalize_folds_unrecognised_family_into_unclassified():
    assert normalize_tags(["wobble.wobble"]) == ["safety.unclassified"]


def test_normalize_survives_garbage_types():
    assert normalize_tags([None, 3, {"a": 1}, "gore.blood_moderate"]) == ["gore.blood_moderate"]


def test_normalize_accepts_case_and_separator_variants():
    assert normalize_tags([" Nudity-Shirtless Male "]) == ["nudity.shirtless_male"]


def test_empty_output_is_the_callers_problem_not_normally_nothing():
    # normalize_tags CAN return [] (the model said "nothing here"); the identify
    # pass turns that into safety.unclassified. Tested in test_visual_scan.py.
    assert normalize_tags([]) == []


def test_harm_family_has_no_residual_so_unknown_becomes_unclassified():
    assert normalize_tags(["harm.thing_not_in_table"]) == ["safety.unclassified"]


# --------------------------------------------------------------------------
# Selection: on/off, no ordering, no threshold
# --------------------------------------------------------------------------

def test_default_selection_censors_nothing():
    sel = default_selection()
    assert not sel.tags
    det = _Det(["gore.blood_moderate"])
    assert tags_hit(det.tags, sel) == []


def test_tags_hit_matches_selected_tag():
    sel = Selection(tags={"gore.blood_moderate"})
    assert tags_hit(["gore.blood_moderate", "violence.verbal_threat"], sel) == ["gore.blood_moderate"]


def test_tags_hit_always_includes_enforced_tag():
    sel = Selection(tags=set())  # nothing selected at all
    hit = tags_hit(["nudity.breasts_full_female", "safety.exposure_with_minor"], sel)
    assert "safety.exposure_with_minor" in hit


def test_tags_hit_keeps_unknown_stored_ids_censorable():
    # A scan file from a newer vocabulary must not become un-censorable here.
    assert tags_hit(["future.new_thing"], Selection(tags=set())) == ["future.new_thing"]


def test_selection_has_no_ordering_anywhere():
    # Guards against re-introducing the old rank comparison.
    sel = default_selection()
    assert not any(hasattr(sel, a) for a in ("level", "threshold", "rank"))
    src = __import__("inspect").getsource(Selection)
    assert "low" not in src and "exhigh" not in src


def test_selection_round_trips_through_dict():
    sel = Selection(tags={"nudity.shirtless_male"}, renderings={"anime"})
    back = Selection.from_dict(sel.to_dict())
    assert back.tags == {"nudity.shirtless_male"}
    assert back.renderings == {"anime"}


def test_from_dict_drops_unknown_ids_and_restores_default_renderings():
    sel = Selection.from_dict({"tags": ["gore.blood_moderate", "bogus.thing"]})
    assert sel.tags == {"gore.blood_moderate"}
    assert sel.renderings == {r.value for r in Rendering}


def test_presets_select_whole_families():
    graphic = selection_named("graphic")
    assert "gore.blood_moderate" in graphic.tags
    assert "violence.beating_prolonged" in graphic.tags or any(
        t.startswith("violence.") for t in graphic.tags)
    assert not any(t.startswith("nudity.") for t in graphic.tags)


def test_preset_with_something_on_turns_unclassified_on():
    # A scan that could not name a moment must not slip through a filter that is
    # switched on for everything else.
    for name in ("nudity", "graphic", "language"):
        assert "safety.unclassified" in selection_named(name).tags, name
    assert "safety.unclassified" not in selection_named("none").tags


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        selection_named("make_it_tasteful")


def test_non_censorable_tags_are_the_enforced_ones():
    assert NON_CENSORABLE == {t.id for t in TAGS.values() if t.always_censor}


# --------------------------------------------------------------------------
# Rendering narrows, never widens
# --------------------------------------------------------------------------

def test_rendering_filter_excludes_a_style():
    sel = Selection(tags={"gore.blood_moderate"}, renderings={"live_action"})
    assert rendering_allowed("anime", sel) is False
    assert rendering_allowed("live_action", sel) is True


def test_unknown_rendering_value_matches_everything():
    # A wrong guess by the model must never silently un-censor.
    assert rendering_allowed("mime_performance", default_selection()) is True
    assert rendering_allowed(None, default_selection()) is True


# --------------------------------------------------------------------------
# Per-family actions
# --------------------------------------------------------------------------

def test_language_offers_word_granularity_and_visual_none():
    assert "mute_word" in audio_options(Family.LANGUAGE)
    assert Family.LANGUAGE.default_visual == "none"
    assert Family.LANGUAGE.takes_audio is True


def test_visual_families_only_offer_scene_mute():
    for fam in (Family.NUDITY, Family.GORE, Family.VIOLENCE, Family.HARM):
        assert audio_options(fam) == ["none", "mute_scene"], fam
        assert fam.takes_visual is True


def test_default_visual_is_blur_for_visual_families():
    assert Family.GORE.default_visual == "blur"
    assert Family.LANGUAGE.default_visual == "none"


def test_family_media_defaults_are_valid_actions():
    from nuclearcutter.schema import AudioAction, VisualAction

    for fam in Family:
        assert VisualAction(fam.default_visual)
        assert AudioAction(fam.default_audio)


# --------------------------------------------------------------------------
# The child-protection floor
# --------------------------------------------------------------------------

def test_minor_exposure_is_enforced_and_undismissable():
    det = _Det(["nudity.breasts_full_female"], minors=Minors.YES.value)
    enforce_minor_protection(det)
    assert "safety.exposure_with_minor" in det.tags
    assert "safety.exposure_with_minor" in minor_locked_tags(det)


def test_uncertain_age_is_a_checkbox_not_a_force():
    det = _Det(["nudity.legs_thigh_bare_female"], minors=Minors.UNCERTAIN.value)
    enforce_minor_protection(det)
    assert "safety.exposure_age_uncertain" in det.tags
    assert "safety.exposure_with_minor" not in det.tags
    # ...and unlike the enforced tag, it CAN be dismissed.
    assert "safety.exposure_age_uncertain" not in minor_locked_tags(det)


def test_uncertain_minor_without_exposure_gets_no_protection_tag():
    """`minors == uncertain` with nothing exposed must stay untagged.

    A local VLM guessing ages from thumbnails says "possible child" for crowds,
    doorways and toys; forcing the floor there would censor a large share of
    ordinary scenes and the tool would get switched off. `yes` is NOT held to this
    standard (see the test below) — it is the model's confident claim that a child
    is in frame, and that is precisely the case the floor exists for.
    """
    violent = "violence.beating_prolonged" if "violence.beating_prolonged" in TAGS \
        else tags_in_family(Family.VIOLENCE)[0].id
    det = _Det([violent], minors=Minors.UNCERTAIN.value)
    enforce_minor_protection(det)
    assert det.tags == [violent], "nothing exposed, so nothing to protect"


def test_confident_minor_gets_the_floor_even_without_an_exposure_tag():
    """The floor keys on `minors`, not on a tag surviving later.

    Two reasons, both about attacks on the rule rather than aesthetics:
      * a scan that says "a child is in this scene" next to a beating is exactly
        what a family buys the filter for; the model naming undress or not is a
        detail we should not bet a child's protection on.
      * tag lists are REPLACED when a viewer corrects the model, so a floor that
        needed a nudity tag to still be there could be switched off by deleting
        that tag. `minors` is not part of a tag edit.
    """
    det = _Det(["violence.beating_prolonged"], minors=Minors.YES.value)
    enforce_minor_protection(det)
    assert "safety.exposure_with_minor" in det.tags


def test_tag_edit_cannot_walk_under_the_floor():
    """The concrete version of the same rule, through the API a viewer uses."""
    from nuclearcutter.schema import VisualDetection, replace_tags

    det = VisualDetection(start=1.0, end=2.0, tags=["nudity.breasts_full_female"],
                          description="x", confidence=0.9, minors=Minors.YES.value)
    replace_tags(det, ["violence.argument"])
    assert "safety.exposure_with_minor" in det.tags, \
        "unticking the exposure tags must not remove the floor with them"

    # Clearing the minors flag is the honest way out, and it does work.
    det.minors = Minors.NO.value
    replace_tags(det, ["violence.argument"])
    assert "safety.exposure_with_minor" not in det.tags


def test_enforce_is_idempotent():
    det = _Det(["nudity.cleavage_female"], minors=Minors.YES.value)
    enforce_minor_protection(det)
    enforce_minor_protection(det)
    assert det.tags.count("safety.exposure_with_minor") == 1


def test_enforced_tag_is_censorable_with_nothing_selected():
    det = _Det(["nudity.cleavage_female", "safety.exposure_with_minor"])
    assert "safety.exposure_with_minor" in tags_hit(det.tags, default_selection())


# --------------------------------------------------------------------------
# Prompt blocks are generated, never hand-copied
# --------------------------------------------------------------------------

def test_taxonomy_prompt_lists_ids_that_exist():
    from nuclearcutter.taxonomy import taxonomy_prompt

    block = taxonomy_prompt([Family.NUDITY])
    for tid in all_tag_ids():
        if tid.startswith("nudity."):
            assert tid in block, tid
    assert "gore." not in block


def test_taxonomy_prompt_narrows_by_gender():
    from nuclearcutter.taxonomy import taxonomy_prompt

    male_only = taxonomy_prompt([Family.NUDITY], genders=("male",))
    female_ids = [t.id for t in tags_in_family(Family.NUDITY) if t.gender == "female"]
    male_ids = [t.id for t in tags_in_family(Family.NUDITY) if t.gender == "male"]
    for tid in male_ids:
        assert tid in male_only
    for tid in female_ids:
        assert tid not in male_only, tid


def test_sweep_triggers_are_lowercase_ids():
    from nuclearcutter.taxonomy import SWEEP_TRIGGERS

    names = [t[0] for t in SWEEP_TRIGGERS]
    assert len(names) == len(set(names))
    assert all(n == n.lower() and " " not in n for n in names)
    assert "doubt" in names  # the model's escape hatch must exist


def test_family_of_helper():
    assert family_of("gore.blood_moderate") is Family.GORE
    assert family_of("nope.nope") is None


# ---------------------------------------------------------------------------
# Invariants docs/TAXONOMY.md states in tables. A doc nobody checks is a bug
# with citations, so the numbers and rules it promises are asserted here.
# ---------------------------------------------------------------------------

def test_documented_tag_and_preset_counts():
    from nuclearcutter.taxonomy import PRESET_NAMES, NON_CENSORABLE

    per_family = {f.value: len(tags_in_family(f)) for f in Family}
    assert per_family == {"nudity": 35, "intimacy": 11, "violence": 26, "gore": 21,
                          "harm": 5, "language": 16, "substance": 4, "safety": 3}
    assert len(TAGS) == 121
    counts = {n: len(selection_named(n).tags) for n in PRESET_NAMES}
    assert counts == {"none": 0, "nudity": 37, "sexual": 48, "language": 22,
                      "graphic": 49, "everything": 120}


def test_presets_never_remove_what_an_earlier_one_picked():
    """The buttons nest, so tapping a second preset only ever adds.

    A preset that silently *unticked* an earlier choice would be the worst kind of
    surprise: "nudity" then "graphic" losing the nudity box is the sort of thing a
    viewer notices only after the file is rendered.
    """
    from nuclearcutter.taxonomy import PRESET_NAMES

    picked = {n: set(selection_named(n).tags) for n in PRESET_NAMES}
    assert picked["nudity"] <= picked["sexual"] <= picked["everything"]
    assert picked["graphic"] <= picked["everything"]
    assert picked["language"] <= picked["everything"]


def test_graphic_preset_excludes_self_harm():
    """"Graphic" means fighting and blood, never suicide detail.

    This is the preset a viewer reaches for when a film has a battle in it.
    Inheriting `harm.suicide_method` from a button labelled "graphic" is the most
    harmful surprise this product could produce — the family is 5 tags and has to
    be ticked knowingly. (docs/TAXONOMY.md §4)
    """
    graphic = selection_named("graphic").tags
    assert not [t for t in graphic if t.startswith("harm.")]
    assert any(t.startswith("violence.") for t in graphic)
    assert any(t.startswith("gore.") for t in graphic)


def test_every_preset_omits_the_child_floor():
    """Presets set *choices*. The always-censor tag is not a choice, so a preset
    that listed it would render a checkbox the viewer cannot untick.

    Checked at `selection_named()` rather than at the API layer on purpose: the
    CLI's --preset writes the same list into a scan's preferences, so a filter
    applied only in the HTTP response would leave the two paths disagreeing.
    """
    from nuclearcutter.taxonomy import PRESET_NAMES

    for name in PRESET_NAMES:
        assert "safety.exposure_with_minor" not in selection_named(name).tags


def test_residuals_exist_where_declared():
    """A declared residual that is not a real tag silently sends unknown output to
    `safety.unclassified`, losing which family was unsure — and so losing the
    viewer's ability to censor *only* that family."""
    from nuclearcutter.taxonomy import _RESIDUAL

    for fam, residual in _RESIDUAL.items():
        if residual is None:
            continue
        assert residual in TAGS, f"{fam.value}: {residual} is declared but not a tag"
        assert TAGS[residual].family is fam
    # harm/substance opt out on purpose; if that changes, docs §3.3 must change.
    assert _RESIDUAL[Family.HARM] is None
    assert _RESIDUAL[Family.SUBSTANCE] is None


def test_invented_language_tag_finds_the_language_residual():
    """The KeyError-shaped hole this file's own doc found: `language.unclear`
    existed as a tag but was missing from _RESIDUAL, so an invented language tag
    degraded to `safety.unclassified` and the family was lost."""
    out = normalize_tags(["language.rude_tone_generic"])
    assert out == ["language.unclear"]
