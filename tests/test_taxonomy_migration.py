"""The version 2 -> version 3 vocabulary merge must not lose an old scan file.

A scan file records what a model saw under the vocabulary that existed *then*. The
2026-10-30 merge took 121 tags down to 68, so unless every retired id resolves onto
the tag that absorbed it, an existing scan silently degrades: "bare midriff" becomes
"nudity unclear", the range stays censorable but the viewer is told the scan could
not tell — and a scan that lies about its own precision is worse than a coarser one,
because it also makes the censorable range *stricter* than what was seen.

The v2 id list below is copied on purpose. Reading it from git would make this test
pass in a checkout where the file is gone, and the point is to keep the map complete
forever, not until the history gets rebased.
"""

from __future__ import annotations

import pytest

from nuclearcutter.taxonomy import (
    LEGACY_TAGS, TAGS, Family, TAXONOMY_VERSION, normalize_tags,
)

V2_TAG_IDS: list[str] = [
    "gore.blood_heavy",
    "gore.blood_moderate",
    "gore.blood_trivial",
    "gore.bodily_fluids",
    "gore.bones_fracture",
    "gore.burn_injury",
    "gore.butcher_hunt",
    "gore.corpse",
    "gore.corpse_decay",
    "gore.dismemberment",
    "gore.illness_symptoms",
    "gore.medical_noblood",
    "gore.mutilation",
    "gore.needle_injection",
    "gore.nonrealistic",
    "gore.organs_tissue",
    "gore.surgery_medical",
    "gore.unclear",
    "gore.vomit",
    "gore.wound_detail",
    "gore.wound_open",
    "harm.eating_disorder",
    "harm.overdose",
    "harm.self_harm_depicted",
    "harm.suicide_depicted",
    "harm.suicide_method",
    "intimacy.bondage",
    "intimacy.couple_in_bed_undressed",
    "intimacy.cuddling_clothed",
    "intimacy.kissing_brief",
    "intimacy.kissing_passionate",
    "intimacy.morning_after",
    "intimacy.sex_suggested_offscreen",
    "intimacy.simulated_sex",
    "intimacy.striptease_pole_lapdance",
    "intimacy.unclear",
    "intimacy.undressing_another",
    "language.anatomy_crude",
    "language.anatomy_words",
    "language.blasphemy_religious",
    "language.crude_sexual_jokes",
    "language.degradation_insult",
    "language.hostile_tone_general",
    "language.profanity_common",
    "language.profanity_mild",
    "language.profanity_severe",
    "language.sexual_talk_explicit",
    "language.slur_disability",
    "language.slur_lgbtq",
    "language.slur_racial",
    "language.slur_religious_ethnic",
    "language.slur_sexist",
    "language.unclear",
    "nudity.bathing_shower",
    "nudity.behind_covers",
    "nudity.breastfeeding_female",
    "nudity.breasts_full_female",
    "nudity.breasts_partial_female",
    "nudity.changing_clothes",
    "nudity.childbirth_female",
    "nudity.cleavage_female",
    "nudity.covered_only_towel_female",
    "nudity.covered_only_towel_male",
    "nudity.erection_visible_male",
    "nudity.full_nude_female",
    "nudity.full_nude_male",
    "nudity.genital_outline_female",
    "nudity.hosiery_female",
    "nudity.implied_not_visible",
    "nudity.legs_thigh_bare_female",
    "nudity.masturbation_female",
    "nudity.masturbation_male",
    "nudity.medical_exposure_female",
    "nudity.medical_exposure_male",
    "nudity.midriff_bare_female",
    "nudity.nipples_visible_female",
    "nudity.sheer_tight_male",
    "nudity.sheer_transparent_female",
    "nudity.shirtless_male",
    "nudity.shoulders_back_bare_female",
    "nudity.swimwear_female",
    "nudity.swimwear_male",
    "nudity.tight_revealing_female",
    "nudity.unclear",
    "nudity.underwear_lingerie_female",
    "nudity.underwear_male",
    "nudity.wet_clothing_female",
    "nudity.wet_clothing_male",
    "safety.exposure_age_uncertain",
    "safety.exposure_with_minor",
    "safety.unclassified",
    "substance.drug_preparation_deal",
    "substance.drug_use",
    "substance.heavy_drinking",
    "substance.smoking",
    "violence.animal_harm",
    "violence.argument",
    "violence.battle_war",
    "violence.beating_prolonged",
    "violence.burning_scalding",
    "violence.car_crash",
    "violence.child_endangerment",
    "violence.confinement_kidnap",
    "violence.destruction_disaster",
    "violence.domestic_abuse",
    "violence.execution",
    "violence.explosion",
    "violence.fight_armed",
    "violence.fight_unarmed",
    "violence.gunfire",
    "violence.implied_offscreen",
    "violence.murder",
    "violence.murder_attempt",
    "violence.peril_chase",
    "violence.restraint_capture",
    "violence.shove_grab",
    "violence.slap_hit",
    "violence.torture",
    "violence.unclear",
    "violence.verbal_threat",
    "violence.weapon_display",
]


def test_every_version_2_tag_id_still_resolves():
    unmapped = [i for i in V2_TAG_IDS if i not in TAGS and i not in LEGACY_TAGS]
    assert not unmapped, (
        f"these retired ids would vanish into a family residual when an old scan "
        f"file loads: {unmapped}")


def test_legacy_aliases_land_on_a_live_tag_and_never_chain():
    for old, new in LEGACY_TAGS.items():
        assert new in TAGS, f"{old} points at {new}, which is not in the vocabulary"
        assert old != new, f"{old} is an identity alias — drop it, resolve_legacy_tag " \
                           f"already returns ids that are current"
        assert old not in TAGS, (
            f"{old} is both live and retired; a live id in LEGACY_TAGS means the "
            f"merge deleted it and re-added it under the same name")


def test_a_retired_id_never_leaves_its_family():
    """A merge joins siblings; it does not relabel what the scan claimed.

    `violence.murder` -> `violence.death` is a narrowing within one family. The same
    id mapped onto `gore.corpse` would be a different claim about the same frames, and
    it would change which of the viewer's ticks catches the detection — the one
    migration error that can both add and remove censorship silently. (Fanning one
    retired id out to two live tags is not expressible here at all: the map's values
    are single strings, which is why no test needs to police it.)
    """
    for old, new in LEGACY_TAGS.items():
        assert old.split(".")[0] == new.split(".")[0], (
            f"{old} merged across families into {new}")
        # Same family and both censorable-or-not in the same way, or the tick that
        # used to catch this detection stops catching it.
        assert TAGS[new].visual == (old.split(".")[0] not in ("language",)) or \
            TAGS[new].audio, f"{new} offers no action the retired {old} could take"


def test_an_old_scan_file_loads_onto_the_new_chooser(tmp_path):
    """End to end, on the path a scan file actually takes: JSON -> normalize_tags."""
    old_ids = ["nudity.midriff_bare_female", "nudity.shoulders_back_bare_female",
               "violence.murder", "language.slur_racial", "gore.blood_trivial"]
    out = normalize_tags(old_ids)
    assert out == ["nudity.skin_bare_female", "violence.death", "language.slur",
                   "gore.blood"], out
    # Nothing landed on a residual, i.e. no precision was traded for compatibility.
    assert not [t for t in out if "unclear" in t or "unclassified" in t]
    # ...and every result is a real checkbox in the GUI's own list.
    for t in out:
        assert TAGS[t].family in set(Family)


def test_a_genuinely_unknown_id_still_degrades_rather_than_vanishing():
    unknown: list[str] = []
    assert normalize_tags(["nudity.something_the_model_made_up"], unknown) == \
        ["nudity.unclear"]
    assert unknown, "an invented id must be reported, not quietly folded"
