"""
The tag vocabulary: NuclearCutter's single source of truth for *what is in a
scene*.

This module replaces the 2026-08…2026-10 severity taxonomy. The old design
asked the model "how bad is this, on a low/med/high/exhigh scale of how much
censorship it needs" and threw away what it actually saw. That made the scan
file a judgement, so it could only be shared by people who happened to share
that judgement, and it made every threshold a coin flip against a paragraph of
pre-written prose (`DEFAULT_LEVEL_SCALES`).

The new design asks one factual question — *what is in this frame?* — and
records the answer as a set of tag ids from the vocabulary below. Deciding
what deserves censorship is then a separate, local, revisitable choice: a
`Selection` is a set of tag ids, nothing more. The scan file stops being an
opinion, which is what makes it worth sharing (docs/SPEC.md §6).

Consequences this module encodes:

  * Tags are descriptive facts, never verdicts. A tag's `hint` (what the model
    reads) says when the thing IS visible, never that it is offensive. Words
    like "flag", "inappropriate", "sensitive", "err on the side of" are banned
    from hints on purpose — they are what turned a woman dancing with a fan
    into `violence/low` (= "brutal acts, murder, torture") in a real scan.

  * Gender exists only where a body state is gendered, and only in the tag id,
    so the chooser is one checkbox per fact. Interactions are deliberately NOT
    gendered: getting that right needs both subjects identified, and one flip
    silently uncensors a scene (docs/TAXONOMY.md §gender).

  * Every family has an `unclear` residual and unknown model output degrades
    *to* it, never to nothing. A dropped tag means an uncensored scene, which
    is the one failure mode this tool cannot have.

  * `always_censor` tags are enforced in the renderer no matter what the
    selection says. There is exactly one, and it is not configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Version of the TAG VOCABULARY, stored in every scan file next to
# SCHEMA_VERSION. They change for different reasons: the file format changes
# shape, while the vocabulary can also change shape (a tag added, renamed or
# removed) with the format identical. Bumping this is a signal that an older
# scan's tag ids may no longer all mean what they used to — a scan file from a
# previous vocabulary still LOADS (unknown ids fold into their family residual,
# so nothing becomes uncensorable), but a re-scan is what gets the new
# categories.
TAXONOMY_VERSION = 2


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

class Family(str, Enum):
    NUDITY = "nudity"
    INTIMACY = "intimacy"
    VIOLENCE = "violence"
    GORE = "gore"
    HARM = "harm"
    LANGUAGE = "language"
    SUBSTANCE = "substance"
    SAFETY = "safety"

    @property
    def label(self) -> str:
        return _FAMILY_LABELS[self]

    @property
    def is_visual(self) -> bool:
        """True if a detection with this family's tags hides picture."""
        return self not in (Family.LANGUAGE,)

    @property
    def is_audio(self) -> bool:
        """True if this family's tags can act on audio."""
        return self in (Family.LANGUAGE, Family.SUBSTANCE, Family.HARM)

    @property
    def blurb(self) -> str:
        """One-line description for the chooser, shown under the family heading."""
        return _FAMILY_BLURBS[self]

    @property
    def takes_visual(self) -> bool:
        """Whether the GUI should offer a blur/black control for this family."""
        return _FAMILY_MEDIA[self][0] != "none"

    @property
    def takes_audio(self) -> bool:
        """Whether the GUI should offer a mute control for this family."""
        return _FAMILY_MEDIA[self][1] != "none"

    @property
    def default_visual(self) -> str:
        # _FAMILY_MEDIA's first slot is a MEDIA KIND ("visual"/"none"), not an
        # action: it says whether this family hides picture at all. The default
        # way to hide it is a blur.
        return "blur" if self.takes_visual else "none"

    @property
    def default_audio(self) -> str:
        return _FAMILY_MEDIA[self][1]


_FAMILY_LABELS = {
    Family.NUDITY: "Nudity & modesty",
    Family.INTIMACY: "Sexual situations",
    Family.VIOLENCE: "Violence",
    Family.GORE: "Blood & injury",
    Family.HARM: "Self-harm & suicide",
    Family.LANGUAGE: "Language",
    Family.SUBSTANCE: "Drugs, drinking, smoking",
    Family.SAFETY: "Scan was unsure",
}

_FAMILY_BLURBS = {
    Family.NUDITY: "How much of a body, or what kind of clothing, is visible.",
    Family.INTIMACY: "Kissing, undressing and sex — with or without nudity.",
    Family.VIOLENCE: "One person hurting, or threatening to hurt, another.",
    Family.GORE: "Wounds, blood and bodies that are damaged.",
    Family.HARM: "Someone hurting themselves, and how it is done.",
    Family.LANGUAGE: "Swearing, slurs, insults and explicit talk.",
    Family.SUBSTANCE: "Smoking, heavy drinking and drug use.",
    Family.SAFETY: "The scan noticed something but could not name it.",
}

# What the renderer does for each family when a tag in it is selected.
# Language/substance are audio-only; everything else hides picture. Harm is
# visual because the depiction itself is the thing being censored.
_FAMILY_MEDIA = {
    Family.NUDITY: ("visual", "none"),
    Family.INTIMACY: ("visual", "none"),
    Family.VIOLENCE: ("visual", "none"),
    Family.GORE: ("visual", "none"),
    Family.HARM: ("visual", "none"),
    Family.LANGUAGE: ("none", "mute_phrase"),
    Family.SUBSTANCE: ("visual", "none"),
    Family.SAFETY: ("visual", "none"),
}


# ---------------------------------------------------------------------------
# Orthogonal qualifiers
# ---------------------------------------------------------------------------

class Rendering(str, Enum):
    """How the content is presented. A filter *modifier*, not a category.

    Cross-producting this into tag ids would quadruple the vocabulary, so it is
    stored per detection and applied as a small set of checkboxes instead.
    Defaults are all-on: narrowing what counts is the user's choice, and a
    wrong `rendering` guess must never silently un-censor anything.
    """
    LIVE_ACTION = "live_action"
    ANIMATED = "animated"           # animation / CGI / VFX creature
    ANIME = "anime"                 # anime-style specifically
    ARTWORK = "artwork"             # painting, drawing, statue, stained glass
    ARCHIVAL = "archival"           # newsreel, documentary file footage
    FANTASY = "fantasy"             # dream/flashback/imagined, or non-real fluids
    STAGE = "stage"                 # filmed stage/theatrical performance

    @property
    def label(self) -> str:
        return _RENDERING_LABELS[self]


_RENDERING_LABELS = {
    Rendering.LIVE_ACTION: "Live action (real people, filmed)",
    Rendering.ANIMATED: "Animated or CGI",
    Rendering.ANIME: "Anime style",
    Rendering.ARTWORK: "Painting, drawing or statue",
    Rendering.ARCHIVAL: "Archival / news footage",
    Rendering.FANTASY: "Dream, imagined, or non-real fluids",
    Rendering.STAGE: "Filmed stage performance",
}


class Minors(str, Enum):
    """Whether a person who appears to be a child is involved.

    A local VLM estimating age is genuinely unreliable, so this is a three-way
    answer and each branch behaves differently:
      NO        -> nothing extra happens
      YES      -> `exposure_with_minor` is enforced, un-dismissably
      UNCERTAIN -> `exposure_age_uncertain` applies (a normal, off-able tag)
    Making UNCERTAIN force-censor too would censor half of all distant adult
    figures, so it does not; it is surfaced as a checkbox instead.
    """
    NO = "no"
    YES = "yes"
    UNCERTAIN = "uncertain"

    @property
    def label(self) -> str:
        return _MINORS_LABELS[self]

    @property
    def description(self) -> str:
        return _MINORS_DESCRIPTIONS[self]


_MINORS_LABELS = {
    Minors.NO: "No child in the scene",
    Minors.YES: "A child appears to be present",
    Minors.UNCERTAIN: "A child might be present",
}

_MINORS_DESCRIPTIONS = {
    Minors.NO: "Everyone visible reads as an adult.",
    Minors.YES: ("A person who appears to be under 18 is involved. Any exposure "
                 "here is censored no matter what is selected."),
    Minors.UNCERTAIN: ("Small, distant or obscured figures the model could not age. "
                       "Censored only if you turn on 'Exposure where a child might be present'."),
}


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tag:
    id: str
    label: str          # UI label, plain English
    family: Family
    hint: str           # the ONLY definition the model is given
    gender: str = "any"        # "female" | "male" | "any"
    low_precision: bool = False  # model is known to be shaky here -> tell the user
    always_censor: bool = False  # enforced even when not selected
    visual: bool = True
    audio: bool = False

    @property
    def prefix(self) -> str:
        return self.id.split(".", 1)[0]


def _t(tid: str, label: str, family: Family, hint: str, **kw) -> Tag:
    fam_prefix = tid.split(".", 1)[0]
    if fam_prefix != family.value:
        raise ValueError(f"tag {tid!r} has family {family.value!r} but a foreign id prefix")
    return Tag(id=tid, label=label, family=family, hint=hint, **kw)


F = Family


_TAGS: list[Tag] = [
    # ------------------------------------------------------------------ nudity
    # Body-state facts, women. Split from men because a user may only want to
    # filter female modesty (or male), and because the lines people draw are
    # genuinely different for each.
    _t("nudity.cleavage_female", "Cleavage (women)", F.NUDITY,
       "A female character's cleavage is visible: skin between the breasts above a "
       "low neckline. Breasts are otherwise covered.", gender="female"),
    _t("nudity.shoulders_back_bare_female", "Bare shoulders or back (women)", F.NUDITY,
       "A female character's shoulders, upper back or neckline are bare (off-shoulder, "
       "backless, halter). No chest exposure beyond this.", gender="female"),
    _t("nudity.midriff_bare_female", "Bare midriff (women)", F.NUDITY,
       "A female character's stomach or waist is bare (crop top, midriff-revealing outfit).",
       gender="female"),
    _t("nudity.legs_thigh_bare_female", "Bare legs or thighs (women)", F.NUDITY,
       "A female character's legs or upper thighs are bare: short skirt or shorts, a high "
       "slit, bare legs with no tights.", gender="female"),
    _t("nudity.breasts_partial_female", "Partially exposed breasts (women)", F.NUDITY,
       "Part of a female character's bare breast is visible - side breast, under breast, "
       "breast fold - but the nipple area is covered by a hand, arm, hair or clothing.",
       gender="female"),
    _t("nudity.breasts_full_female", "Bare breasts (women)", F.NUDITY,
       "A female character's bare breast is fully or almost fully visible, with or without "
       "the nipple visible.", gender="female"),
    _t("nudity.nipples_visible_female", "Visible nipples (women)", F.NUDITY,
       "A female character's nipple or areola is visible, including through sheer or wet "
       "material, or unpainted through thin fabric.", gender="female"),
    _t("nudity.full_nude_female", "Fully nude / genitals or buttocks (women)", F.NUDITY,
       "A female character shows genitals, pubic hair, buttocks or anus, or is shown nude "
       "with no garment covering those parts. Bare buttocks count as this category.",
       gender="female"),
    _t("nudity.swimwear_female", "Swimwear / bikini (women)", F.NUDITY,
       "A female character is in a bikini, tankini, one-piece or other swimwear, worn as "
       "swimwear or as clothing.", gender="female"),
    _t("nudity.underwear_lingerie_female", "Underwear or lingerie (women)", F.NUDITY,
       "A female character is in a bra, panties, slips, lingerie or babydoll as her only "
       "or main clothing. Not swimwear.", gender="female"),
    _t("nudity.hosiery_female", "Stockings, garters or fishnets (women)", F.NUDITY,
       "A female character wears stockings, a garter belt, hold-ups or fishnet tights as "
       "lingerie or as the main visible garment, not everyday tights with clothing.",
       gender="female"),
    _t("nudity.sheer_transparent_female", "See-through clothing (women)", F.NUDITY,
       "A female character's clothing is transparent or mesh enough that her skin or under-"
       "garments show through it.", gender="female"),
    _t("nudity.wet_clothing_female", "Soaked or clinging wet clothing (women)", F.NUDITY,
       "A female character's clothing is soaked and clinging, showing the shape of her body "
       "or under-garments.", gender="female"),
    _t("nudity.tight_revealing_female", "Skin-tight or fetish clothing (women)", F.NUDITY,
       "A female character wears skin-tight bodycon, wet-look, latex, vinyl, a catsuit or a "
       "body stocking that traces the whole body.", gender="female"),
    _t("nudity.covered_only_towel_female", "Undressed but hidden by hand, towel or sheet (women)",
       F.NUDITY,
       "A female character is plainly not wearing clothes but a hand, arm, towel, sheet or "
       "prop hides the private parts.", gender="female"),
    _t("nudity.genital_outline_female", "Genital outline through clothing (women)", F.NUDITY,
       "The outline of a female character's genitals or pubic area shows through tight or "
       "thin clothing.", gender="female"),
    _t("nudity.breastfeeding_female", "Breastfeeding", F.NUDITY,
       "A female character is nursing a baby at the breast.", gender="female"),
    _t("nudity.childbirth_female", "Childbirth", F.NUDITY,
       "A birth or labour scene in which a woman's body is exposed for delivery.",
       gender="female"),
    _t("nudity.medical_exposure_female", "Medical exposure (women)", F.NUDITY,
       "A female character's body is exposed in a medical context: being examined, treated, "
       "operated on, or an incision site being worked on.", gender="female"),
    _t("nudity.masturbation_female", "Self-touch / masturbation (women)", F.NUDITY,
       "A female character touches her own private parts, or her hand moves under covers or "
       "clothing in a sexual way.", gender="female"),

    # Body-state facts, men. Deliberately shorter than the women's list: there is
    # no male cleavage, midriff or nipple line that people filter, and inventing
    # male twins for female tags would add checkboxes no one turns on.
    _t("nudity.shirtless_male", "Shirtless (men)", F.NUDITY,
       "A male character's bare chest or torso is visible with no shirt, whether or not he "
       "is muscular.", gender="male"),
    _t("nudity.full_nude_male", "Fully nude / genitals or buttocks (men)", F.NUDITY,
       "A male character shows genitals, pubic area or buttocks, or is shown nude with "
       "nothing covering those parts.", gender="male"),
    _t("nudity.swimwear_male", "Swimwear (men)", F.NUDITY,
       "A male character is in swim briefs, trunks or board shorts as his only clothing.",
       gender="male"),
    _t("nudity.underwear_male", "Underwear (men)", F.NUDITY,
       "A male character is in boxers, briefs or boxer shorts as his only clothing.",
       gender="male"),
    _t("nudity.covered_only_towel_male", "Undressed but hidden by hand, towel or sheet (men)",
       F.NUDITY,
       "A male character is plainly not wearing clothes but a hand, towel or sheet hides "
       "the private parts.", gender="male"),
    _t("nudity.erection_visible_male", "Visible erection or bulge (men)", F.NUDITY,
       "A male character has a visible erection, or the shape of his genitals is visible "
       "through thin or tight clothing.", gender="male"),
    _t("nudity.masturbation_male", "Self-touch / masturbation (men)", F.NUDITY,
       "A male character touches his own private parts, or his hand moves under covers in a "
       "sexual way.", gender="male"),
    _t("nudity.medical_exposure_male", "Medical exposure (men)", F.NUDITY,
       "A male character's body is exposed in a medical context: being examined, treated or "
       "operated on.", gender="male"),
    _t("nudity.sheer_tight_male", "See-through or skin-tight clothing (men)", F.NUDITY,
       "A male character's clothing is sheer, mesh or so tight that it exposes the shape of "
       "his body or genitals.", gender="male"),
    _t("nudity.wet_clothing_male", "Soaked or clinging wet clothing (men)", F.NUDITY,
       "A male character's clothing is soaked and clinging enough to show his body or "
       "under-garments.", gender="male"),

    # Nudity facts that are not about a gendered body.
    _t("nudity.implied_not_visible", "Nudity implied, nothing shown", F.NUDITY,
       "Characters are clearly nude but nothing is on display: turned away, behind another "
       "person, in shadow, cut away from, or framed above the neck."),
    _t("nudity.bathing_shower", "Bathing or showering", F.NUDITY,
       "A character is bathing, in a bath, a shower or a hot tub, whether or not any private "
       "part is visible."),
    _t("nudity.changing_clothes", "Changing clothes", F.NUDITY,
       "A character is actively getting undressed or dressed, partially clothed."),
    _t("nudity.behind_covers", "Under bed covers", F.NUDITY,
       "A character is under bed covers with bare skin or an undressed shoulder visible, "
       "bodies entangled under the covers."),
    _t("nudity.unclear", "Nudity unclear", F.NUDITY,
       "A person appears to be undressed, underdressed or in an intimate state but the "
       "picture is too dark, brief, distant or blocked to say what is visible. Use this "
       "instead of inventing a specific body part.", low_precision=True),

    # --------------------------------------------------------------- intimacy
    # Its own family (not part of nudity) so that kissing and sex scenes with no
    # nudity at all are separable: "censor the sex, keep the kissing" and "censor
    # making out, nothing is nude" are both ordinary choices.
    _t("intimacy.kissing_brief", "Brief kissing", F.INTIMACY,
       "A quick or closed-mouth kiss, or a kiss that lasts under about two seconds."),
    _t("intimacy.kissing_passionate", "Passionate kissing", F.INTIMACY,
       "A prolonged kiss, open-mouth or with tongues, or characters kissing while falling, "
       "lying or moving together."),
    _t("intimacy.cuddling_clothed", "Cuddling or embracing intimately", F.INTIMACY,
       "Characters hold or cuddle each other in bed or in an intimate pose while fully "
       "clothed, with no undressing."),
    _t("intimacy.couple_in_bed_undressed", "Couple in bed, undressed", F.INTIMACY,
       "Two characters are in bed together undressed or partly undressed, whether or not "
       "anything is visible."),
    _t("intimacy.undressing_another", "Undressing or handling another person", F.INTIMACY,
       "One character pulls clothing off another, or hands move under another character's "
       "clothing or over their body."),
    _t("intimacy.simulated_sex", "Simulated sex", F.INTIMACY,
       "Characters appear to be having sex: bodies joined or intertwined, on top of each "
       "other, rhythmic movement, whether clothed or not, with no genitals visible."),
    _t("intimacy.sex_suggested_offscreen", "Sex suggested, shown off screen", F.INTIMACY,
       "Sex is clearly happening or about to happen without being shown: a door closing, "
       "candlelit bed, moaning or rustling over a shot of the room, two characters getting "
       "into bed and cutting away."),
    _t("intimacy.morning_after", "Afterwards scene", F.INTIMACY,
       "Characters in bed together after sex, in towels or under sheets, talking or "
       "waking up."),
    _t("intimacy.striptease_pole_lapdance", "Striptease, pole or lap dance", F.INTIMACY,
       "A character strips, dances on a pole, dances on someone's lap, or is on a stage in "
       "the act of removing clothing."),
    _t("intimacy.bondage", "Bondage or restraint in a sexual context", F.INTIMACY,
       "A character is tied, bound or handcuffed in a sexual or suggestive context. If the "
       "restraint is a arrest or a threat, that is the violence category instead."),
    _t("intimacy.unclear", "Sexual content unclear", F.INTIMACY,
       "The situation reads as romantic or sexual but nothing specific can be named from "
       "the picture.", low_precision=True),

    # --------------------------------------------------------------- violence
    _t("violence.peril_chase", "Being chased or in danger", F.VIOLENCE,
       "A character is pursued, hunted or in physical danger, with nobody touching them."),
    _t("violence.argument", "Heated argument", F.VIOLENCE,
       "Characters shout at or argue angrily with each other. No physical contact.",
       low_precision=True),
    _t("violence.verbal_threat", "Threats or intimidation", F.VIOLENCE,
       "A character threatens, menaces or intimidates another, in words or by advancing on "
       "them, without hitting them.", low_precision=True),
    _t("violence.weapon_display", "Weapon visible or aimed", F.VIOLENCE,
       "A gun, knife, blade, bat or other weapon is held, drawn, pointed or aimed, even if "
       "it is never used."),
    _t("violence.restraint_capture", "Grabbed, pinned or captured", F.VIOLENCE,
       "A character is seized, pinned down, held, restrained, handcuffed or captured."),
    _t("violence.confinement_kidnap", "Kidnapped or locked up", F.VIOLENCE,
       "A character is kidnapped, held captive, locked in a cell, room or cage, or bound."),
    _t("violence.shove_grab", "Shoving or rough handling", F.VIOLENCE,
       "Characters shove, push, grab, jostle or roughly pull each other."),
    _t("violence.slap_hit", "Slap or single blow", F.VIOLENCE,
       "Someone is slapped, backhanded, punched once, or hit with an open hand or object, "
       "without a fight developing."),
    _t("violence.fight_unarmed", "Fight without weapons", F.VIOLENCE,
       "Two or more characters exchange punches, kicks or grappling."),
    _t("violence.fight_armed", "Fight with weapons", F.VIOLENCE,
       "Characters fight with swords, axes, staffs, guns used as clubs or other weapons."),
    _t("violence.beating_prolonged", "Beating or sustained assault", F.VIOLENCE,
       "One character repeatedly hits, kicks, stomps or assaults another who is not fighting "
       "back, or a helpless person is attacked."),
    _t("violence.gunfire", "Shooting", F.VIOLENCE,
       "A gun is fired at a person, people are shot at, or a character is hit by gunfire."),
    _t("violence.explosion", "Explosion", F.VIOLENCE,
       "A bomb, blast, grenade, explosion or wreck burning, whether or not anyone is hurt."),
    _t("violence.battle_war", "Battle or war scene", F.VIOLENCE,
       "Many combatants fighting: a battle, a charge, a firefight, soldiers dying."),
    _t("violence.car_crash", "Crash or collision", F.VIOLENCE,
       "A vehicle crashes, rolls, is hit, or falls: car, train, plane or boat."),
    _t("violence.destruction_disaster", "Disaster or mass destruction", F.VIOLENCE,
       "A building, city or structure collapses, a natural disaster, a flood, an avalanche, "
       "or crowds being crushed."),
    _t("violence.burning_scalding", "Burning or scalding a person", F.VIOLENCE,
       "A person is burned, set on fire, branded, or scalded."),
    _t("violence.torture", "Torture", F.VIOLENCE,
       "A character is deliberately made to suffer: interrogated with pain, tied to a device, "
       "waterboarded, electrodes, teeth or nails pulled, whipped."),
    _t("violence.execution", "Execution", F.VIOLENCE,
       "A helpless or restrained character is deliberately killed, or is about to be, by "
       "shooting, hanging, guillotine or similar."),
    _t("violence.murder", "A character is killed", F.VIOLENCE,
       "A character is killed on screen, or has clearly just been killed: stabbed, shot, "
       "strangled, pushed to their death."),
    _t("violence.murder_attempt", "Killing attempted", F.VIOLENCE,
       "An attempt on a character's life that they survive or that is interrupted: hanging, "
       "drowning, strangling, a push that is survived."),
    _t("violence.animal_harm", "An animal is hurt or killed", F.VIOLENCE,
       "An animal is struck, injured, killed, suffering, or shown dead."),
    _t("violence.child_endangerment", "A child is threatened or in danger", F.VIOLENCE,
       "A child is threatened, frightened by an aggressor, neglected deliberately, or placed "
       "in physical danger."),
    _t("violence.domestic_abuse", "Abuse inside a household", F.VIOLENCE,
       "One character frightens, controls, threatens or hurts a partner or family member in a "
       "home setting.", low_precision=True),
    _t("violence.implied_offscreen", "Violence heard or implied off screen", F.VIOLENCE,
       "Violence happens out of shot: screaming, a gunshot with no shot shown, someone "
       "dragged away, blood on a wall afterwards."),
    _t("violence.unclear", "Violence unclear", F.VIOLENCE,
       "Something aggressive or dangerous seems to be happening but it cannot be named from "
       "the picture.", low_precision=True),

    # ------------------------------------------------------------------- gore
    _t("gore.blood_trivial", "A little blood", F.GORE,
       "A small amount of blood: a trickle, a nosebleed, a scrape, blood on a hand or a "
       "lip, a bandage with a spot of blood."),
    _t("gore.blood_moderate", "Blood from a wound", F.GORE,
       "A clear, visible amount of blood: a wound bleeding, blood soaking cloth, drips or "
       "splatter on the floor or clothing."),
    _t("gore.blood_heavy", "A lot of blood", F.GORE,
       "Heavy blood: pools, spurting, spraying, a body covered in blood, a blood-soaked room."),
    _t("gore.wound_open", "Open wound", F.GORE,
       "An open cut, gash, laceration, hole, tear or bite in skin, without showing inside "
       "the body."),
    _t("gore.wound_detail", "Wound shown in detail", F.GORE,
       "A wound seen close up or in detail: fat, muscle, tendon, a bullet hole, stitching or "
       "a wound being closed."),
    _t("gore.bones_fracture", "Bone or fracture", F.GORE,
       "Bone is visible through skin, a limb is bent wrongly, a bone is broken or set, or an "
       "X-ray shows a break."),
    _t("gore.organs_tissue", "Organs or insides", F.GORE,
       "Internal organs, intestines, brain, muscle tissue or the inside of a body is shown."),
    _t("gore.dismemberment", "Dismemberment", F.GORE,
       "A body part is cut or torn off, a body is in pieces, a severed head, limbs or a "
       "torso alone."),
    _t("gore.mutilation", "Mutilation or deliberate damage", F.GORE,
       "A body is deliberately damaged beyond a wound: eyes, tongue, ears, teeth or nails "
       "removed, skin cut away, a body defaced."),
    _t("gore.burn_injury", "Burned flesh", F.GORE,
       "Burned or blistered skin, charred tissue, the aftermath of a fire on a body."),
    _t("gore.needle_injection", "Needle or injection", F.GORE,
       "A needle, syringe, IV line or drug being injected into skin, whether medical or "
       "recreational."),
    _t("gore.corpse", "A dead body", F.GORE,
       "A dead body or corpse is clearly shown, without heavy damage: a body laid out, a "
       "corpse in a bed, on a slab, at a scene."),
    _t("gore.corpse_decay", "Decaying or skeletal body", F.GORE,
       "A body showing decay, decomposition, maggots, skeletal remains or a skeleton."),
    _t("gore.surgery_medical", "Surgery or medical procedure", F.GORE,
       "An operation, incision, suturing, a wound being treated, an operating theatre or a "
       "medical procedure with visible flesh."),
    _t("gore.vomit", "Vomit", F.GORE,
       "Someone vomits or retches, or vomit is shown."),
    _t("gore.bodily_fluids", "Other bodily fluids", F.GORE,
       "Pus, infected wounds, faeces, urine, spit or semen shown on screen."),
    _t("gore.medical_noblood", "Hospital or medical care, no blood", F.GORE,
       "A hospital, bandages, a cast, a stretcher, a scan or a monitor, with no blood or "
       "wound shown."),
    _t("gore.illness_symptoms", "Illness or symptoms", F.GORE,
       "A character is visibly sick: coughing blood, a seizure, fainting, vomiting from "
       "illness, a wound going septic."),
    _t("gore.butcher_hunt", "Butchering, hunting or dead animals", F.GORE,
       "An animal being slaughtered, butchered, skinned, dressed or field-dressed, or a dead "
       "animal shown with blood."),
    _t("gore.nonrealistic", "Non-real blood or fluid", F.GORE,
       "Blood or fluid that is plainly not real: green or blue fluid, oil, sludge, cartoon "
       "or stylised blood, a robot or creature leaking."),
    _t("gore.unclear", "Blood or injury unclear", F.GORE,
       "Something looks like blood, injury or bodily harm but it cannot be told from the "
       "picture - dark liquid, distance, brief shot.", low_precision=True),

    # ------------------------------------------------------------------- harm
    _t("harm.suicide_depicted", "Suicide depicted", F.HARM,
       "A character takes their own life, or is shown having just died by suicide."),
    _t("harm.suicide_method", "Suicide method shown in detail", F.HARM,
       "The means of suicide is shown or explained in a way that could be copied: preparing "
       "pills, a noose being made, a weapon being raised to oneself, a list of methods."),
    _t("harm.self_harm_depicted", "Deliberate self-injury", F.HARM,
       "A character deliberately injures themselves: cutting, burning, hitting themselves; "
       "or fresh marks are shown."),
    _t("harm.overdose", "Overdose or poisoning oneself", F.HARM,
       "A character takes an overdose or is shown collapsing from one, whether by accident "
       "or on purpose."),
    _t("harm.eating_disorder", "Eating disorder", F.HARM,
       "Restricting food, bingeing, purging, or a character's body weight and food being "
       "treated as a problem."),

    # --------------------------------------------------------------- language
    # Audio-family. Words are grouped by WHAT they are (a slur is a different
    # thing from an expletive), because a family may care enormously about slurs
    # and not at all about "damn" - which the old single scale could not express.
    _t("language.profanity_mild", "Mild swear words", F.LANGUAGE,
       "Mild expletives and their euphemisms: damn, hell, crap, ass, bastard, "
       "bloody, geez, damn it, heck.", visual=False, audio=True),
    _t("language.profanity_common", "Common swear words", F.LANGUAGE,
       "Common profanity: fuck and its forms, shit, bitch, asshole, dick, prick, bullshit, "
       "piss, goddamn.", visual=False, audio=True),
    _t("language.profanity_severe", "The most offensive swear words", F.LANGUAGE,
       "The strongest general expletive words, above ordinary profanity in how offensive "
       "they are.", visual=False, audio=True),
    _t("language.slur_racial", "Racial slur", F.LANGUAGE,
       "A racial or ethnic slur is spoken.", visual=False, audio=True),
    _t("language.slur_religious_ethnic", "Religious or ethnic slur", F.LANGUAGE,
       "A slur aimed at someone's religion or ethnicity is spoken.", visual=False, audio=True),
    _t("language.slur_lgbtq", "Slur about sexuality or gender", F.LANGUAGE,
       "A slur about someone's sexuality or gender identity is spoken.", visual=False, audio=True),
    _t("language.slur_disability", "Slur about disability or ability", F.LANGUAGE,
       "A slur about disability, or a word used to insult someone by comparing them to a "
       "disabled person, is spoken.", visual=False, audio=True),
    _t("language.slur_sexist", "Sexist or misogynistic slur", F.LANGUAGE,
       "A word used to put someone down for being a woman, or a sexual slur aimed at a "
       "person, is spoken.", visual=False, audio=True),
    _t("language.anatomy_words", "Words for body parts", F.LANGUAGE,
       "Clinical or everyday words for genitals, breasts or sex acts are spoken, including "
       "medical or educational discussion.", visual=False, audio=True),
    _t("language.anatomy_crude", "Crude words for body parts", F.LANGUAGE,
       "Slang or vulgar words for genitals, breasts, sex acts or bodily functions are "
       "spoken, without being an expletive.", visual=False, audio=True),
    _t("language.degradation_insult", "Cruel insult aimed at someone", F.LANGUAGE,
       "A character says something meant to degrade or humiliate another named person: "
       "calling them worthless, stupid, ugly, a burden, telling them to die.",
       visual=False, audio=True),
    _t("language.hostile_tone_general", "Rude, mean or dismissive tone", F.LANGUAGE,
       "Speech that is rude, mean, sarcastic, mocking or dismissive without being a swear "
       "word or a clear insult: shouting at someone, 'shut up', 'I hate you', sneering.",
       visual=False, audio=True, low_precision=True),
    _t("language.blasphemy_religious", "Religious language used as an expletive", F.LANGUAGE,
       "God, Christ, Jesus, the Virgin Mary, hell or damnation used as an exclamation or "
       "curse rather than as discussion of religion.", visual=False, audio=True),
    _t("language.sexual_talk_explicit", "Explicit talk about sex", F.LANGUAGE,
       "Characters describe sex acts, discuss their sex lives in detail, or ask explicit "
       "sexual questions, with nothing shown.", visual=False, audio=True),
    _t("language.crude_sexual_jokes", "Sexual jokes or innuendo", F.LANGUAGE,
       "A sexual joke, double entendre, or crude suggestive remark is made.",
       visual=False, audio=True),
    _t("language.unclear", "Language unclear", F.LANGUAGE,
       "Speech that appears offensive or adult but the words cannot be made out.",
       visual=False, audio=True, low_precision=True),

    # -------------------------------------------------------------- substance
    _t("substance.smoking", "Smoking or vaping", F.SUBSTANCE,
       "A character smokes a cigarette, cigar, pipe or vapes, or a lit cigarette is held or "
       "shown."),
    _t("substance.heavy_drinking", "Drinking to excess", F.SUBSTANCE,
       "Characters drink alcohol heavily or drunkenly: getting drunk, many glasses, "
       "compulsive drinking, alcoholism shown as a problem. A glass of wine at dinner does "
       "not count."),
    _t("substance.drug_use", "Drug use", F.SUBSTANCE,
       "A character uses illegal drugs or misuses medicine: smoking cannabis, snorting, "
       "injecting, popping pills to get high, being intoxicated."),
    _t("substance.drug_preparation_deal", "Dealing or preparing drugs", F.SUBSTANCE,
       "Drugs being grown, cooked, weighed, packaged, exchanged for money, or a transaction "
       "being discussed."),

    # ----------------------------------------------------------------- safety
    # Residuals and enforced protections. The reason these exist: a sweep batch
    # that the model cannot explain still must not be shown, and once categories
    # get specific, "did not fit a box" silently becomes "shown unblurred".
    _t("safety.unclassified", "Scan could not name it", F.SAFETY,
       "The scan picked this moment out and could not describe it as any specific thing.",
       visual=True),
    _t("safety.exposure_age_uncertain", "Exposure where a child might be present", F.SAFETY,
       "A person appears to be a child, or their age cannot be told, and there is undress or "
       "exposure.", visual=True),
    _t("safety.exposure_with_minor", "Exposure involving a child", F.SAFETY,
       "A person who appears to be a child is shown undressed or exposed.",
       always_censor=True, visual=True),
]


TAGS: dict[str, Tag] = {t.id: t for t in _TAGS}

# Order for the chooser: families in a sensible reading order, tags in the order
# written above (which runs mild -> explicit inside each family on purpose).
FAMILY_ORDER = [F.NUDITY, F.INTIMACY, F.VIOLENCE, F.GORE, F.HARM, F.LANGUAGE,
                F.SUBSTANCE, F.SAFETY]


def tags_by_family() -> dict[Family, list[Tag]]:
    out: dict[Family, list[Tag]] = {f: [] for f in FAMILY_ORDER}
    for t in _TAGS:
        out[t.family].append(t)
    return out


def tags_in_family(family: Family) -> list[Tag]:
    """Every tag of one family, in vocabulary order."""
    return [t for t in _TAGS if t.family is family]


def audio_options(family: Family) -> list[str]:
    """The audio actions that make sense for a family.

    Lives here so the GUI's dropdown, the server's validation and the schema all
    read the same table instead of each keeping a copy — v1 had the same four
    options written out in two files and they had already started to differ.

    Only LANGUAGE takes audio (`_FAMILY_MEDIA`), so it is the only family that gets
    the word/phrase set. Substance is on-screen smoking and drug use with no spoken
    counterpart — its detections are visual, so muting "the flagged word" would
    address a word that does not exist. (An earlier version of this function listed
    SUBSTANCE alongside LANGUAGE; that branch was unreachable, because the
    `takes_audio` test above already returns for it, which is exactly the drift this
    function exists to prevent.)
    """
    if not family.takes_audio:
        return ["none", "mute_scene"]
    return ["none", "mute_word", "mute_phrase", "replace_word", "replace_phrase"]


def default_tag_ids() -> list[str]:
    """Tag ids the chooser shows as ticked on a first run: nothing.

    `always_censor` tags are not included because they are not a choice — the
    chooser must not offer to untick them, and `tags_hit()` applies them anyway.
    """
    return []


# Excluded from the "censor everything" preset: enforced automatically, so a
# preset that lists them would imply the user chose them (and a GUI that renders
# a checkbox for them would offer an untick that does nothing).
NON_CENSORABLE = frozenset(t.id for t in _TAGS if t.always_censor)


def minor_locked_tags(det) -> set[str]:
    """Tags on a detection that the child-protection rule refuses to let go.

    Takes anything with `.tags` and `.minors` attributes (a VisualDetection) so
    this module does not have to import the schema.
    """
    if getattr(det, "minors", Minors.NO.value) != Minors.YES.value:
        return set()
    tags = set(getattr(det, "tags", []) or [])
    locked = {"safety.exposure_with_minor"}
    # An exposure-family tag next to a child is the case the rule exists for.
    for tid in tags:
        tag = TAGS.get(tid)
        if tag is not None and tag.family in (Family.NUDITY, Family.INTIMACY):
            locked.add(tid)
    return locked


def enforce_minor_protection(det) -> None:
    """Apply the one rule that is not the user's to make, in place.

    A filter a family turns on to protect children must not depend on that family
    also having thought to enable `nudity.*`: exposure of a child is not a taste
    question. When a person who appears to be a child is present,
    `safety.exposure_with_minor` is added here, the taxonomy marks it
    `always_censor`, and the chooser renders it locked on.

    WHICH FACT IS THE EVIDENCE
    --------------------------
    The trigger is the `minors` flag, not the presence of a nudity tag:

      * `minors == yes` means someone in this frame is a child. Nudity around a
        child is the exact case a family buys this tool for; whether the model
        also named the exposure tag is a detail.
      * tag lists get REPLACED — a viewer correcting what the model saw unticks
        tags whole. If the floor depended on a tag surviving that rewrite, deleting
        the exposure tags would delete the floor with it, and the "floor" would be
        one untick away from off. `minors` is not part of a tag edit; it is a
        separate, deliberate statement, and clearing it is the honest way to say
        the age estimate was wrong.

    So with `minors == yes` the floor tag is applied unconditionally.

    `minors == uncertain` is NOT forced — a local VLM estimating age from a distant
    blurry figure is unreliable, and forcing it would censor a large fraction of
    ordinary scenes (a crowd, a child in a doorway, a toy in the background). It
    adds `safety.exposure_age_uncertain`, an ordinary censorable tag, so the viewer
    decides — and only when something is actually exposed, because "a possible
    child in an ordinary scene" is not a hit.

    Called wherever a detection is created or edited (classify, scan-file load,
    per-detection edit), so the rule cannot be bypassed by hand-editing a scan file
    or by unticking a box in the popup. Duck-typed on `.tags` / `.minors` /
    `.uncertain_tags` so this module never imports the schema.
    """
    minors = getattr(det, "minors", Minors.NO.value)
    if minors not in (Minors.YES.value, Minors.UNCERTAIN.value):
        return
    if minors == Minors.YES.value:
        if "safety.exposure_with_minor" not in det.tags:
            det.tags.append("safety.exposure_with_minor")
        return
    exposed = any(TAGS.get(t) is not None and TAGS[t].family in (Family.NUDITY, Family.INTIMACY)
                  for t in (getattr(det, "tags", []) or []))
    if not exposed:
        return  # an uncertain child with nothing exposed is not a hit
    if "safety.exposure_age_uncertain" not in det.tags:
        det.tags.append("safety.exposure_age_uncertain")
    uncertain = getattr(det, "uncertain_tags", None)
    if uncertain is not None and "safety.exposure_age_uncertain" not in uncertain:
        uncertain.append("safety.exposure_age_uncertain")


def is_known_tag(tag_id: str) -> bool:
    return tag_id in TAGS


def family_of(tag_id: str) -> Family | None:
    t = TAGS.get(tag_id)
    return t.family if t else None


def all_tag_ids() -> list[str]:
    return list(TAGS.keys())


# ---------------------------------------------------------------------------
# Validation: unknown model output degrades to the family residual, never to nothing
# ---------------------------------------------------------------------------

_RESIDUAL = {
    F.NUDITY: "nudity.unclear",
    F.INTIMACY: "intimacy.unclear",
    F.VIOLENCE: "violence.unclear",
    F.GORE: "gore.unclear",
    F.LANGUAGE: "language.unclear",
    # harm and substance deliberately have NO residual, and this is the one place
    # the "uncertainty always degrades to a tag" rule bends. Both bends for the
    # same reason: their content is identified by CONTEXT, not by appearance, so a
    # blurry frame cannot support the guess. Is that an eating-disorder plot or an
    # athletic character? Is that medicine or a narcotic? A `harm.unclear` would
    # fire on dark clothing and a raised arm, and since it is censorable by default
    # it would silently delete plot lines the viewer sat down to watch - a filter
    # that censors the story gets switched off, which is worse for the people who
    # need it than the gap it leaves. `gore.unclear` is different: blood is a
    # visual property, so "looks like blood" is an honest observation.
    # Unnamed-but-flagged moments still land on safety.unclassified, so nothing is
    # dropped - only the specific *claim* about harm is withheld.
    F.HARM: None,
    F.SUBSTANCE: None,
    F.SAFETY: "safety.unclassified",
}

# A model that invents "nudity.nude_woman_topless_half" must not silently
# delete the finding. Unknown ids are folded into the family residual where one
# exists, and into safety.unclassified when even the family is unrecognised.
def normalize_tags(raw_ids: list, seen_unknown: list[str] | None = None) -> list[str]:
    """Validate model output against the vocabulary.

    Returns a de-duplicated list of real tag ids. An unknown id that at least
    names a known family becomes that family's `unclear` residual (so the scene
    stays censorable); an unknown id that names nothing becomes
    safety.unclassified. Nothing is ever dropped for being misspelt.
    """
    out: list[str] = []

    def _add(tid: str) -> None:
        if tid in TAGS and tid not in out:
            out.append(tid)

    for raw in raw_ids or []:
        if not isinstance(raw, str):
            continue
        tid = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if tid not in TAGS and "." not in tid and "_" in tid:
            # A model that writes `nudity_shirtless_male` (separator dropped) is
            # still naming a real category. Family names never contain an
            # underscore, so the first one is the separator.
            head, rest = tid.split("_", 1)
            if head in {f.value for f in Family}:
                tid = f"{head}.{rest}"
        if tid in TAGS:
            _add(tid)
            continue
        if seen_unknown is not None and raw not in seen_unknown:
            seen_unknown.append(raw)
        head = tid.split(".", 1)[0]
        try:
            fam = Family(head)
        except ValueError:
            _add("safety.unclassified")
            continue
        _add(_RESIDUAL.get(fam) or "safety.unclassified")
    return out


# ---------------------------------------------------------------------------
# Selection: the user's choice, expressed as a set of tag ids
# ---------------------------------------------------------------------------

@dataclass
class Selection:
    """Which tags the user wants censored, plus rendering modifiers.

    Replaces `Preferences.*_level`. There is deliberately no ordering and no
    threshold: a tag is either on or off. What to DO about a match (blur vs
    black, mute vs leave) is a separate per-family choice and lives in
    `Preferences`, so this stays a pure statement of what the user cares about.
    """
    tags: set[str] = field(default_factory=set)
    renderings: set[str] = field(default_factory=lambda: {r.value for r in Rendering})

    def is_on(self, tag_id: str) -> bool:
        return tag_id in self.tags or (TAGS.get(tag_id) is not None
                                       and TAGS[tag_id].always_censor)

    def toggle(self, tag_id: str, on: bool) -> None:
        if tag_id not in TAGS:
            raise ValueError(f"unknown tag id: {tag_id}")
        if TAGS[tag_id].always_censor:
            return  # not configurable
        if on:
            self.tags.add(tag_id)
        else:
            self.tags.discard(tag_id)

    def to_dict(self) -> dict:
        return {"tags": sorted(self.tags), "renderings": sorted(self.renderings)}

    @staticmethod
    def from_dict(d: dict) -> "Selection":
        sel = Selection()
        sel.tags = {t for t in (d.get("tags") or []) if t in TAGS}
        rs = d.get("renderings")
        sel.renderings = ({r for r in rs if r in {x.value for x in Rendering}}
                          if rs else {r.value for r in Rendering})
        return sel


# The presets, in the order a chooser should show them (least -> most covered).
# One list because the CLI's --preset choices, the server's /api/taxonomy payload
# and the GUI's buttons all have to agree; three hand-written lists drift, and the
# failure is a preset button that 400s or a flag nobody can guess the name of.
#
# Deliberately no "recommended" preset: choosing what is offensive is the whole
# decision this tool hands back to the viewer.
# There is deliberately no "adult" preset alongside "everything": SAFETY holds
# only the child floor (not a choice) plus two leftovers, so "all the adult
# families" and "everything" resolve to the same tag set. Two buttons that censor
# identically teach the viewer to distrust both.
PRESET_NAMES = ("none", "nudity", "sexual", "language", "graphic", "everything")


def selection_named(name: str) -> Selection:
    """Preset selections. Default is empty — nothing censored.

    Presets only ever flip whole groups and can be edited afterwards; they are a
    starting point, never a policy. There is no 'recommended' preset on purpose:
    the whole point of the redesign is that the tool does not decide what is bad.
    """
    sel = Selection()
    fams: set[Family] = set()
    if name in ("", "none", "nothing"):
        return sel
    if name in ("everything", "all"):
        fams = {f for f in Family}
    elif name == "nudity":
        fams = {F.NUDITY}
    elif name == "sexual":
        fams = {F.NUDITY, F.INTIMACY}
    elif name == "graphic":
        # Violence and gore, NOT harm. "Graphic" is what a viewer reaches for when
        # there is fighting in a movie; inheriting suicide-method detail and
        # self-injury from a button labelled "graphic" is the worst surprise this
        # product could produce, and those tags are small enough to choose
        # knowingly. See docs/TAXONOMY.md §4.
        fams = {F.VIOLENCE, F.GORE}
    elif name == "language":
        fams = {F.LANGUAGE, F.SUBSTANCE}
    else:
        raise ValueError(f"unknown preset: {name}")
    for t in _TAGS:
        if t.always_censor:
            # A Selection is a set of *choices*; `always_censor` is enforced by the
            # renderer regardless of what is in here (see Selection.enforced_tags /
            # schema.replace_tags). Listing it in a preset would make every preset
            # look like it had ticked the child floor, and a GUI that renders a
            # preset as a checkbox list would offer to untick it. Excluded at the
            # source so no caller has to remember to filter.
            continue
        if t.family in fams and t.visual:
            sel.tags.add(t.id)
        if t.family in fams and t.audio:
            sel.tags.add(t.id)
    # Safety residuals belong to no preset except 'everything'-style ones.
    if F.SAFETY not in fams:
        for t in _TAGS:
            if t.family is F.SAFETY and not t.always_censor:
                sel.tags.discard(t.id)
    # Unclassified is on for any preset that censors *something*: a scan that
    # could not name a moment should not slip through a filter that is on.
    if fams - {F.SAFETY}:
        sel.tags.add("safety.unclassified")
        sel.tags.add("safety.exposure_age_uncertain")
    return sel


def default_selection() -> Selection:
    """Nothing censored, except the one enforced protection.

    The user asked for this: 'default to nothing censored, it's easy enough to
    customize in the GUI.' A brand-new scan therefore renders the film exactly
    as it is until the user turns something on.
    """
    return Selection()


# ---------------------------------------------------------------------------
# What gets censored: the only matching rule in the program
# ---------------------------------------------------------------------------

def tags_hit(tags: list[str], sel: Selection) -> list[str]:
    """Which of a detection's tags the selection wants censored.

    Rendering narrows a match (a cartoon counts only if cartoons are included)
    and never widens one. An unrecognised rendering value matches everything,
    again so a bad guess cannot silently un-censor.
    """
    hit: list[str] = []
    for tid in tags or []:
        t = TAGS.get(tid)
        if t is None:
            # Unknown-but-stored ids stay censorable.
            hit.append(tid)
            continue
        if t.always_censor or tid in sel.tags:
            hit.append(tid)
    return hit


def rendering_allowed(rendering: str | None, sel: Selection) -> bool:
    if not rendering:
        return True
    vals = {r.value for r in Rendering}
    if rendering not in vals:
        return True
    return rendering in sel.renderings


# ---------------------------------------------------------------------------
# Model-facing prompt blocks (generated from the vocabulary, never hand-copied)
# ---------------------------------------------------------------------------

# SUBSTANCE belongs here even though it also takes an audio action: smoking,
# drinking and drug use are SEEN, the sweep triggers on them, and a classify pass
# that cannot name them turns every cigarette into "doubt" -> unclear. A family
# may appear in both prompt sets; what matters is that every tag with `visual`
# is offerable to the picture pass (asserted in tests/test_prompts.py).
_VISUAL_PROMPT_FAMILIES = [F.NUDITY, F.INTIMACY, F.VIOLENCE, F.GORE, F.HARM,
                           F.SUBSTANCE, F.SAFETY]
_AUDIO_PROMPT_FAMILIES = [F.LANGUAGE, F.SUBSTANCE]


def taxonomy_prompt(families: list[Family] | None = None,
                    genders: tuple[str, ...] | None = None,
                    include_ids: set[str] | None = None,
                    exclude_ids: set[str] | None = None) -> str:
    """Render the vocabulary into the block the model is given.

    Generated from TAGS so the prompt can never drift from the vocabulary — the
    old prompts and the old scale text were kept in sync by hand and did not.

    `genders` narrows the body-state tags: when the coarse pass already told us
    every subject is male, showing 19 female body tags invites the model to
    invent a woman. `any`-gendered tags are always included.

    `include_ids` / `exclude_ids` narrow by id. The dialogue pass uses them to
    ask about the things a wordlist cannot find, instead of paying ~500 prompt
    tokens for swear-word tiers it already knows how to detect for free.
    """
    fams = families or _VISUAL_PROMPT_FAMILIES
    lines: list[str] = []
    for fam in fams:
        tags = [t for t in _TAGS if t.family is fam]
        if genders:
            tags = [t for t in tags if t.gender == "any" or t.gender in genders]
        if include_ids is not None:
            tags = [t for t in tags if t.id in include_ids]
        if exclude_ids is not None:
            tags = [t for t in tags if t.id not in exclude_ids]
        if not tags:
            continue
        lines.append(f"[{fam.label}]")
        for t in tags:
            lines.append(f"- {t.id}: {t.hint}")
        lines.append("")
    return "\n".join(lines).strip()


def audio_taxonomy_prompt() -> str:
    return taxonomy_prompt(_AUDIO_PROMPT_FAMILIES)


# What the dialogue-group pass asks about: things a wordlist structurally
# cannot find, because they are not words. A sexual remark, a crude joke, a
# cruelty aimed at a person, a reference to drugs, and slurs the wordlist does
# not carry. Profanity tiers are deliberately NOT here — the wordlist finds
# those for free, and re-asking about them on every line of the film is how a
# cheap pass stops being cheap.
DIALOGUE_GROUP_IDS: set[str] = {
    "language.slur_racial", "language.slur_religious_ethnic", "language.slur_lgbtq",
    "language.slur_disability", "language.slur_sexist",
    "language.degradation_insult", "language.hostile_tone_general",
    "language.sexual_talk_explicit", "language.crude_sexual_jokes",
    "language.anatomy_crude",
    "substance.smoking", "substance.heavy_drinking", "substance.drug_use",
    "substance.drug_preparation_deal",
}


def dialogue_group_prompt() -> str:
    return taxonomy_prompt(_AUDIO_PROMPT_FAMILIES, include_ids=DIALOGUE_GROUP_IDS)


# The coarse trigger list for the full-film sweep. It is a *detector*
# vocabulary, not a filter vocabulary: each entry names something whose pixels
# are cheap to notice, and nothing in it asks the model to care. Derived from
# the real families so it can never drift out of step with them, and far
# shorter than the three level scales it replaces (those cost ~1.3k prompt
# tokens on every single batch of the film).
SWEEP_TRIGGERS: list[tuple[str, str]] = [
    ("skin", "a person with skin showing that everyday street clothing would "
             "cover: bare chest, shoulders, back, stomach, thighs or legs, "
             "underwear, swimwear, lingerie, towels, or someone undressed or partly undressed"),
    ("body_parts", "a visible nipple, breast, buttock or genital, or the outline of one"),
    ("intimate", "two people kissing, embracing in bed, undressing each other, or appearing "
                 "to have sex"),
    ("blood", "any blood or red fluid, on skin, clothing, a wall or the floor"),
    ("injury", "a wound, bandage, bruise, burn, needle, corpse, or a medical or surgical scene"),
    ("fighting", "any hitting, grappling, struggling, a weapon, shooting, or a battle"),
    ("hostile", "someone threatening, menacing or shouting at another person"),
    ("restraint", "someone grabbed, pinned, tied up, handcuffed or locked up"),
    ("danger", "a chase, a fall, a crash, an explosion, a fire, or a child in danger"),
    ("substance", "cigarettes, drinking, or drugs being used, prepared or exchanged"),
    ("offensive_speech", "any hint of adult conversation, crude remarks or arguments"),
    ("doubt", "anything else that looks like it might matter and cannot be placed in a box above"),
]


def sweep_trigger_block() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in SWEEP_TRIGGERS)


# ---------------------------------------------------------------------------
# Low-precision warnings shown in the chooser
# ---------------------------------------------------------------------------

def low_precision_ids() -> list[str]:
    return [t.id for t in _TAGS if t.low_precision]


def assert_vocabulary_sane() -> None:
    """Called from a test, and on import of the CLI, so a bad vocabulary can
    never reach a scan: ids must match families, residuals must exist, and no
    hint may contain the judgement words that broke the old scale."""
    banned = ("flag", "inappropriate", "offensive unless", "sensitive",
              "err on the side", "censor", "should be hidden", "inappropriate")
    for t in _TAGS:
        low = t.hint.lower()
        for word in banned:
            if word in low:
                raise AssertionError(
                    f"tag {t.id} hint contains judgement word {word!r}: {t.hint}")
        if t.family is Family.LANGUAGE and not t.audio:
            raise AssertionError(f"language tag {t.id} is not audio")
        if t.family is not Family.LANGUAGE and t.audio and not t.visual:
            raise AssertionError(f"visual tag {t.id} is audio-only but not language")
    for fam, res in _RESIDUAL.items():
        if res is not None and res not in TAGS:
            raise AssertionError(f"residual {res} for {fam} is not a real tag")
    if sum(1 for t in _TAGS if t.always_censor) != 1:
        raise AssertionError("exactly one always_censor tag is allowed")


assert_vocabulary_sane()
