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
TAXONOMY_VERSION = 3


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
    # Body-state facts. Women and men stay split where the modesty line a viewer
    # draws is genuinely different for each (that is a standing requirement); tags
    # that describe the same STATE for either sex are one tag with gender="any".
    #
    # 2026-10-30 merge: this family was 35 tags, and the chooser was a wall of
    # checkboxes where most adjacent pairs were the same decision wearing different
    # clinical names. Bare shoulders, bare midriff and bare thighs are all "skin is
    # showing"; see-through and soaked clothing are both "the fabric is not doing its
    # job". Each merged tag's hint still lists every case the old ones covered, so
    # the MODEL's precision did not have to be paid for with the viewer's simplicity.
    _t("nudity.skin_bare_female", "Bare skin: shoulders, back, midriff or legs (women)", F.NUDITY,
       "A female character's shoulders, upper back, midriff, stomach, waist, legs or "
       "upper thighs are bare: off-shoulder or backless dress, halter top, crop top, "
       "short skirt or shorts, a high slit, bare legs with no tights. Clothing still "
       "covers the chest and groin.", gender="female"),
    _t("nudity.breasts_partial_female", "Partially exposed breasts or cleavage (women)", F.NUDITY,
       "A female character's breasts are partly visible but the areolae are covered: "
       "cleavage above a low neckline, sideboob, a breast slipping from a top, "
       "nude-coloured or pastie coverage. Nipples are not clearly visible.",
       gender="female"),
    _t("nudity.breasts_bare_female", "Bare breasts (women)", F.NUDITY,
       "A female character's bare breasts, including visible nipples or areolae, "
       "whether fully bare or seen through fabric that leaves the areola visible. "
       "Side view, back view or brief frame counts.", gender="female"),
    _t("nudity.full_nude_female", "Fully nude (women)", F.NUDITY,
       "A female character is nude: genitals, pubic area or buttocks visible, or the "
       "whole body is bare. Buttocks count as full nudity even if the front is hidden.",
       gender="female"),
    _t("nudity.underwear_swimwear_female", "Underwear, lingerie or swimwear (women)", F.NUDITY,
       "A female character in underwear, lingerie, a bra, a bikini or a swimsuit, "
       "including topless-appearing swimwear that is still swimwear.", gender="female"),
    _t("nudity.revealing_clothing_female", "Revealing, sheer-cut or fetish clothing (women)", F.NUDITY,
       "A female character wears clothing that reads as sexualised coverage rather "
       "than exposing a body part: stockings, garters, fishnets, corset, mesh or cut-out "
       "garments, latex, PVC, or other skin-tight fetish wear.", gender="female"),
    _t("nudity.sheer_or_wet_female", "See-through or soaked clothing (women)", F.NUDITY,
       "A female character's clothing fails to conceal because of what it is made of or "
       "what happened to it: sheer, mesh, translucent or lace fabric, or wet, soaked, "
       "rain-soaked or pool-soaked clothing clinging to the body so its shape or "
       "underlying skin shows.", gender="female"),
    _t("nudity.shirtless_male", "Shirtless (men)", F.NUDITY,
       "A male character's bare chest is visible, whether or not the rest is clothed.",
       gender="male"),
    _t("nudity.full_nude_male", "Fully nude (men)", F.NUDITY,
       "A male character is nude: genitals, pubic area or buttocks visible, or the whole "
       "body is bare.", gender="male"),
    _t("nudity.underwear_swimwear_male", "Underwear or swimwear (men)", F.NUDITY,
       "A male character in underwear, boxers, briefs, swim trunks or speedos, with no "
       "shirt or with the lower body in undergarments only.", gender="male"),
    _t("nudity.sheer_or_wet_male", "See-through or soaked clothing (men)", F.NUDITY,
       "A male character's clothing shows what is underneath because it is sheer, "
       "skin-tight mesh or latex, or because it is wet and clinging.", gender="male"),
    _t("nudity.undressed_covered", "Undressed but covered", F.NUDITY,
       "A character is evidently undressed or getting undressed yet nothing is exposed: "
       "hiding behind a hand, a towel, a bed sheet, a doorway or a foreground object. "
       "Any sex is implied by the situation, not by what is visible."),
    _t("nudity.genital_shape", "Genital outline or erection through clothing", F.NUDITY,
       "The genital area is legible through clothing: a visible erection or bulge, a "
       "camel toe, or an outline pressed through tight or thin fabric. No bare genital "
       "skin."),
    _t("nudity.self_touch", "Self-touch or masturbation", F.NUDITY,
       "A character touches their own genitals or simulates masturbation, alone, with no "
       "other person involved and usually no full nudity shown."),
    _t("nudity.medical_or_maternal", "Medical or maternal exposure", F.NUDITY,
       "A body is exposed in a clinical or maternal context rather than a sexual one: a "
       "medical exam, surgery site, hospital gown, a pregnancy examination, childbirth, "
       "or breastfeeding."),
    _t("nudity.implied_not_visible", "Nudity implied, nothing shown", F.NUDITY,
       "The scene means nudity without showing it: characters in bed under covers, a "
       "shadow or silhouette, a shower running with nobody visible, someone stepping out "
       "of frame nude, reactions that assume nudity."),
    _t("nudity.vulnerable_context", "Bathing, changing or in bed", F.NUDITY,
       "A vulnerable-but-clothed context: showering or bathing (with modesty intact), "
       "changing clothes, getting into or out of bed, waking up in bed. Any nudity is "
       "obscured, off frame, or cut away."),
    _t("nudity.unclear", "Nudity unclear", F.NUDITY,
       "Something skin-like, flesh-toned or body-shaped is visible but its extent or "
       "location is ambiguous: a dark or fast frame, heavy shadow, a blur, an out-of-focus "
       "foreground, or a partial frame. Use this rather than nothing when the frame seems "
       "to show exposure that cannot be identified."),

    # ---------------------------------------------------------------- intimacy
    _t("intimacy.kissing", "Kissing", F.INTIMACY,
       "Characters kissing, at any length or intensity: a brief peck, a prolonged or "
       "passionate kiss, open-mouthed kissing. Kissing on the cheek or hand that reads "
       "as romantic counts if it is held."),
    _t("intimacy.undressing_another", "Undressing or handling another person", F.INTIMACY,
       "One character undresses another, or handles a body sexually: pulling down "
       "clothing, removing underwear, hands under clothing, a couple in bed with "
       "clothing off or coming off."),
    _t("intimacy.simulated_sex", "Sex, or its aftermath", F.INTIMACY,
       "Simulated sex, whether shown, implied through movement and sound, or conveyed by "
       "cutting away and showing the aftermath (two people in bed afterwards, tangled "
       "sheets, the morning after). Visible thrusting or an obvious rhythm is enough; a "
       "fade to black with a clear meaning counts."),
    _t("intimacy.cuddling_clothed", "Cuddling or embracing intimately", F.INTIMACY,
       "Two characters cuddle, embrace or lie together fully clothed, in a way that "
       "reads as intimate rather than comforting or congratulatory."),
    _t("intimacy.striptease_pole_lapdance", "Striptease, pole or lap dance", F.INTIMACY,
       "A strip tease, a pole or chair dance, a lap dance, or a striptease club and its "
       "stage/pole equipment. The dance itself is enough: a partially clothed dancer "
       "counts."),
    _t("intimacy.bondage", "Bondage or restraint in a sexual context", F.INTIMACY,
       "Restraint that is sexual rather than hostile: rope, handcuffs or ties worn as part "
       "of intimacy, blindfolds, being bound to a bed or chair in a sexual scene. Restraint "
       "meant to harm or imprison is violence.restraint_confinement, and both may apply."),
    _t("intimacy.unclear", "Sexual content unclear", F.INTIMACY,
       "Framing, sound, movement or editing that suggests intimacy without identifying it: "
       "heavy breathing over a dark frame, a bed scene with bodies not visible, a "
       "meaningful fade, two people heard but not seen."),

    # ---------------------------------------------------------------- violence
    _t("violence.threat_or_weapon", "Threats, intimidation or a weapon", F.VIOLENCE,
       "Someone threatens or intimidates another: a verbal threat, a hostile advance, a "
       "shove of power over someone, or a weapon visible, drawn, pointed or cocked at "
       "someone, whether or not it is fired. A heated argument with raised voices counts."),
    _t("violence.peril_chase", "Being chased or in danger", F.VIOLENCE,
       "A character is pursued, cornered, hunted, hanging on, about to fall or otherwise "
       "helpless against an approaching threat."),
    _t("violence.restraint_confinement", "Restrained, captured or locked up", F.VIOLENCE,
       "A character is grabbed and pinned, handcuffed, tied up, gagged, carried off, "
       "captured, imprisoned, locked in a cell, box or trunk, or otherwise held against "
       "their will."),
    _t("violence.fight", "A physical fight or assault", F.VIOLENCE,
       "One person hits another: a shove, a grab, a slap, a punch, a single blow, or a "
       "sustained unarmed or armed fight. Both parties fighting back and one-sided "
       "striking both count."),
    _t("violence.beating_abuse", "Beating or abuse", F.VIOLENCE,
       "Sustained one-sided violence: a beating, an assault while the victim is helpless, "
       "or abuse inside a household or relationship, including the intimidating pattern "
       "of it, not only the blow."),
    _t("violence.gunfire", "Shooting", F.VIOLENCE,
       "A gun fired at a person or a person shot, from a single shot to sustained "
       "firefight fire."),
    _t("violence.mass_scene", "Battle, disaster or destruction", F.VIOLENCE,
       "Large-scale violence or catastrophe: a battle or war scene, an explosion, a "
       "vehicle crash or collision, a building, bridge or city being destroyed, a natural "
       "disaster, or a crowd being swept up in it."),
    _t("violence.torture_burning", "Torture, or a person burned", F.VIOLENCE,
       "A person is made to suffer deliberately: torture, interrogation under duress, "
       "being tied to a device, waterboarding, branding, or being burned or scalded. Both "
       "the act and the immediate suffering count."),
    _t("violence.death", "Someone is killed", F.VIOLENCE,
       "A character is killed or a killing is depicted or attempted: murder, an execution, "
       "an assassination, a killing in self-defence, a fatal fall, or an attempt that fails."),
    _t("violence.animal_harm", "An animal is hurt or killed", F.VIOLENCE,
       "An animal is struck, injured, killed, shown dying, or is visibly dead."),
    _t("violence.child_endangerment", "A child is threatened or in danger", F.VIOLENCE,
       "A child is threatened, menaced, neglected in a dangerous way, separated from an "
       "adult by violence, or placed in peril."),
    _t("violence.implied_offscreen", "Violence heard or implied off screen", F.VIOLENCE,
       "Violence conveyed indirectly: screams, gunshots or thuds off screen, aftermath "
       "shots such as a body being carried away or a blood-spattered wall, or someone "
       "describing an attack."),
    _t("violence.unclear", "Violence unclear", F.VIOLENCE,
       "Contact or aggression is visible but ambiguous: a scruffle in a crowd, a fast cut, "
       "a blow that lands off frame, or someone reacting in pain with the cause unseen."),

    # -------------------------------------------------------------------- gore
    _t("gore.blood", "Blood", F.GORE,
       "Blood is visible in any amount: a trickle, a cut, a nosebleed, blood on hands, "
       "clothing, a floor or a wall, blood spurting or pooling. Do not classify the "
       "quantity."),
    _t("gore.blood_heavy", "A lot of blood", F.GORE,
       "Heavy blood: arterial spurting, a pool or spray, a body bleeding out, a large "
       "quantity of blood covering a person, room or weapon."),
    _t("gore.wound_bone", "A wound, or bone and fracture", F.GORE,
       "An injury is visible: a cut, laceration, puncture, abrasion, gash, ulcer, "
       "gunshot hole, exposed flesh or muscle, a broken bone, a limb bent the wrong way, "
       "or a wound examined, stitched or treated in close view."),
    _t("gore.dismemberment", "Dismemberment, organs or deliberate damage", F.GORE,
       "A body taken apart or opened: a severed limb or head, organs, intestines or "
       "insides visible, a body cavity, eyeball or tongue removal, mutilation, or a body "
       "eviscerated or ground up."),
    _t("gore.burn_injury", "Burned flesh", F.GORE,
       "Burn injury on a body: charred, blistered, blackened or melted skin, a burn "
       "victim, or a body that is visibly scorched."),
    _t("gore.corpse", "A dead body", F.GORE,
       "A dead human body, whether fresh, decayed, skeletal, mummified, a corpse in a "
       "morgue, a body in a bag, or bones in an attic."),
    _t("gore.medical", "Medical or drug-injection detail", F.GORE,
       "Medical detail a filter might want to cover: surgery, an operation in progress, "
       "a needle, syringe, IV or injection, blood being drawn, an endoscope, a wound being "
       "sutured, or an intensive-care machine."),
    _t("gore.illness", "Illness, hospital care without blood", F.GORE,
       "Sickness or medical context that is not gory: a character vomiting-free and ill, "
       "symptoms such as coughing, fainting, pallor or sweating, a hospital ward, an "
       "examination, a wheelchair or a diagnosis."),
    _t("gore.bodily_fluids", "Vomit or other bodily fluids", F.GORE,
       "Bodily fluid other than blood: vomit, saliva, mucus, pus, sweat shown heavily, "
       "semen or other sexual fluids, urine or faeces."),
    _t("gore.butcher_hunt", "Dead animals, butchery or hunting", F.GORE,
       "An animal carcass, meat being butchered or skinned, a hunted animal killed or "
       "dying, a kill shown, or animal parts on sale in a market."),
    _t("gore.nonrealistic", "Non-realistic blood or injury", F.GORE,
       "Injury or blood that does not look real: bright or stylised colours, cartoonish "
       "impact, an obviously fake squib, animated or video-game violence."),
    _t("gore.unclear", "Blood or injury unclear", F.GORE,
       "Something red, wet, damaged or body-like that could be blood or injury: a red "
       "liquid in dim light, a stain, a dark frame with a wound-shaped shape, a costume "
       "or prop that reads briefly as damage."),

    # -------------------------------------------------------------------- harm
    # Deliberately NOT merged further. `harm.suicide_method` is the one tag whose
    # detail is the harm: a viewer who accepts a character's death on screen may still
    # need the method not shown, and collapsing it into the depicted suicide would make
    # that uncensorable.
    _t("harm.suicide", "Suicide depicted", F.HARM,
       "A character dies by suicide, or a suicide is shown, in progress or completed, "
       "including an attempt that is interrupted."),
    _t("harm.suicide_method", "Suicide method shown in detail", F.HARM,
       "The method itself is legible and could serve as instruction: the method being "
       "prepared, written instructions or notes describing a method, a noose, pills being "
       "counted, a ledge or vehicle chosen, or the act depicted step by step."),
    _t("harm.self_injury", "Self-injury or overdose", F.HARM,
       "A character deliberately hurts themselves: cutting, burning or hitting themselves, "
       "scars shown, or taking an overdose or poisoning themselves deliberately."),
    _t("harm.eating_disorder", "An eating disorder", F.HARM,
       "Disordered eating shown or made explicit: restricting, purging, bingeing, "
       "body-checking in a mirror, calorie counting, or plot dialogue naming an eating "
       "disorder."),

    # ---------------------------------------------------------------- language
    _t("language.profanity", "Swearing", F.LANGUAGE,
       "Any swear word or mild expletive, from damn, hell, bastard and crap to Fuck and "
       "Shit. The intensity of the word is not part of this tag; only the strongest tier "
       "is separate.", visual=False, audio=True),
    _t("language.profanity_severe", "The most offensive swear words", F.LANGUAGE,
       "The strongest tier of profanity: the C-word, the N-word used non-hostilely, "
       "cunt, and other words that are widely treated as the worst in the language.", visual=False, audio=True),
    _t("language.slur", "A slur against a group", F.LANGUAGE,
       "A slur, epithet or derogatory term aimed at someone's race, ethnicity, national "
       "origin, religion, sexuality, gender identity, sex, disability or immigration "
       "status. Includes a slur used by a character who is meant to be wrong.", visual=False, audio=True),
    _t("language.anatomy", "Words for body parts or sex acts", F.LANGUAGE,
       "Vocabulary for genitals, bodily functions or sex acts, in any register from "
       "clinical to crude: cunt, cock, pussy, tits, balls, piss, shit, and explicit nouns "
       "for sex acts.", visual=False, audio=True),
    _t("language.cruel_tone", "Cruel insults or a hostile tone", F.LANGUAGE,
       "Dialogue meant to cut someone down: a cruel or demeaning insult about a body, an "
       "ability, looks, intelligence, or worth, or a persistent tone of contempt, mockery "
       "or dismissal directed at a character.", visual=False, audio=True),
    _t("language.blasphemy", "Religious language as an expletive", F.LANGUAGE,
       "Religious names or phrases used as swearing or exclamations: God damn it, Jesus "
       "Christ as an expletive, hell and damn as emphasis, or sacrilegious phrasing.", visual=False, audio=True),
    _t("language.sexual_talk", "Explicit sexual talk or jokes", F.LANGUAGE,
       "Speech about sex: explicit description of sex or of what someone wants to do, "
       "graphic sexual jokes or innuendo, discussion of a sexual encounter, or talk of "
       "body parts in a sexual way.", visual=False, audio=True),
    _t("language.unclear", "Language unclear", F.LANGUAGE,
       "Words that could be offensive but cannot be made out: audio clipped, music over "
       "the dialogue, a mumbled or bleeped word, a caption that is missing or wrong, or "
       "dialogue in a language the transcript did not resolve.", visual=False, audio=True),

    # --------------------------------------------------------------- substance
    _t("substance.smoking", "Smoking or vaping", F.SUBSTANCE,
       "A character smoking or vaping a cigarette, cigar, pipe, joint, vape or e-cigarette, "
       "including holding a lit one."),
    _t("substance.heavy_drinking", "Drinking to excess", F.SUBSTANCE,
       "Drinking shown as excess or as a problem: binge drinking, getting visibly drunk, "
       "drinking alone to cope, flask or bottle at lips repeatedly, drinking before driving, "
       "or alcoholism as a plot point. A glass of wine at dinner does not count."),
    _t("substance.drugs", "Drug use, dealing or preparation", F.SUBSTANCE,
       "Illegal or misused drugs: snorting, injecting, smoking or swallowing drugs, a "
       "line, needle, pipe or pill being used, drugs being weighed, cut, bagged, sold or "
       "bought, or a crop/lab operation."),

    # ------------------------------------------------------------------ safety
    # Not content categories: these are how the scan reports its own uncertainty.
    # They are never merged away, and exposure_with_minor is the single
    # always_censor floor that no selection can switch off.
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
# Old tag ids, mapped onto the tag that replaced them in the 2026-10-30 merge
# (v2's 121 tags -> v3's 68). Without this, a scan file made before the merge loses
# every merged-away id into its family residual: the film stays censorable, which is
# the safety requirement, but "bare midriff" becomes "nudity unclear" and the viewer
# is told the scan could not tell what it plainly told the previous scan. Folding onto
# the surviving tag keeps an old scan useful AND lets it drive the new chooser.
#
# Many-to-one is the only shape here; no id maps to two, because a scan file must not
# gain censorship it did not record.
LEGACY_TAGS: dict[str, str] = {
    # gore
    "gore.blood_moderate": "gore.blood",
    "gore.blood_trivial": "gore.blood",
    "gore.bones_fracture": "gore.wound_bone",
    "gore.corpse_decay": "gore.corpse",
    "gore.illness_symptoms": "gore.illness",
    "gore.medical_noblood": "gore.illness",
    "gore.mutilation": "gore.dismemberment",
    "gore.needle_injection": "gore.medical",
    "gore.organs_tissue": "gore.dismemberment",
    "gore.surgery_medical": "gore.medical",
    "gore.vomit": "gore.bodily_fluids",
    "gore.wound_detail": "gore.wound_bone",
    "gore.wound_open": "gore.wound_bone",
    # harm
    "harm.overdose": "harm.self_injury",
    "harm.self_harm_depicted": "harm.self_injury",
    "harm.suicide_depicted": "harm.suicide",
    # intimacy
    "intimacy.couple_in_bed_undressed": "intimacy.undressing_another",
    "intimacy.kissing_brief": "intimacy.kissing",
    "intimacy.kissing_passionate": "intimacy.kissing",
    "intimacy.morning_after": "intimacy.simulated_sex",
    "intimacy.sex_suggested_offscreen": "intimacy.simulated_sex",
    # language
    "language.anatomy_crude": "language.anatomy",
    "language.anatomy_words": "language.anatomy",
    "language.blasphemy_religious": "language.blasphemy",
    "language.crude_sexual_jokes": "language.sexual_talk",
    "language.degradation_insult": "language.cruel_tone",
    "language.hostile_tone_general": "language.cruel_tone",
    "language.profanity_common": "language.profanity",
    "language.profanity_mild": "language.profanity",
    "language.sexual_talk_explicit": "language.sexual_talk",
    "language.slur_disability": "language.slur",
    "language.slur_lgbtq": "language.slur",
    "language.slur_racial": "language.slur",
    "language.slur_religious_ethnic": "language.slur",
    "language.slur_sexist": "language.slur",
    # nudity
    "nudity.bathing_shower": "nudity.vulnerable_context",
    "nudity.behind_covers": "nudity.vulnerable_context",
    "nudity.breastfeeding_female": "nudity.medical_or_maternal",
    "nudity.breasts_full_female": "nudity.breasts_bare_female",
    "nudity.changing_clothes": "nudity.vulnerable_context",
    "nudity.childbirth_female": "nudity.medical_or_maternal",
    "nudity.cleavage_female": "nudity.breasts_partial_female",
    "nudity.covered_only_towel_female": "nudity.undressed_covered",
    "nudity.covered_only_towel_male": "nudity.undressed_covered",
    "nudity.erection_visible_male": "nudity.genital_shape",
    "nudity.genital_outline_female": "nudity.genital_shape",
    "nudity.hosiery_female": "nudity.revealing_clothing_female",
    "nudity.legs_thigh_bare_female": "nudity.skin_bare_female",
    "nudity.masturbation_female": "nudity.self_touch",
    "nudity.masturbation_male": "nudity.self_touch",
    "nudity.medical_exposure_female": "nudity.medical_or_maternal",
    "nudity.medical_exposure_male": "nudity.medical_or_maternal",
    "nudity.midriff_bare_female": "nudity.skin_bare_female",
    "nudity.nipples_visible_female": "nudity.breasts_bare_female",
    "nudity.sheer_tight_male": "nudity.sheer_or_wet_male",
    "nudity.sheer_transparent_female": "nudity.sheer_or_wet_female",
    "nudity.shoulders_back_bare_female": "nudity.skin_bare_female",
    "nudity.swimwear_female": "nudity.underwear_swimwear_female",
    "nudity.swimwear_male": "nudity.underwear_swimwear_male",
    "nudity.tight_revealing_female": "nudity.revealing_clothing_female",
    "nudity.underwear_lingerie_female": "nudity.underwear_swimwear_female",
    "nudity.underwear_male": "nudity.underwear_swimwear_male",
    "nudity.wet_clothing_female": "nudity.sheer_or_wet_female",
    "nudity.wet_clothing_male": "nudity.sheer_or_wet_male",
    # substance
    "substance.drug_preparation_deal": "substance.drugs",
    "substance.drug_use": "substance.drugs",
    # violence
    "violence.argument": "violence.threat_or_weapon",
    "violence.battle_war": "violence.mass_scene",
    "violence.beating_prolonged": "violence.beating_abuse",
    "violence.burning_scalding": "violence.torture_burning",
    "violence.car_crash": "violence.mass_scene",
    "violence.confinement_kidnap": "violence.restraint_confinement",
    "violence.destruction_disaster": "violence.mass_scene",
    "violence.domestic_abuse": "violence.beating_abuse",
    "violence.execution": "violence.death",
    "violence.explosion": "violence.mass_scene",
    "violence.fight_armed": "violence.fight",
    "violence.fight_unarmed": "violence.fight",
    "violence.murder": "violence.death",
    "violence.murder_attempt": "violence.death",
    "violence.restraint_capture": "violence.restraint_confinement",
    "violence.shove_grab": "violence.fight",
    "violence.slap_hit": "violence.fight",
    "violence.torture": "violence.torture_burning",
    "violence.verbal_threat": "violence.threat_or_weapon",
    "violence.weapon_display": "violence.threat_or_weapon",
}


def resolve_legacy_tag(tag_id: str) -> str | None:
    """The current id for a tag id from an older vocabulary, or None.

    An id that is still current resolves to itself, which is what makes this safe to
    call on every tag rather than only on files that claim to be old.
    """
    if tag_id in TAGS:
        return tag_id
    return LEGACY_TAGS.get(tag_id)


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
        legacy = LEGACY_TAGS.get(tid)
        if legacy:
            # A retired id from a scan file written before the merge: record the tag
            # that absorbed it, and do not report it as invented output -- the scan
            # said something precise, and folding it into `unclear` would claim the
            # opposite.
            _add(legacy)
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
    "language.slur", "language.cruel_tone", "language.sexual_talk", "language.anatomy",
    "substance.smoking", "substance.heavy_drinking", "substance.drugs",
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
