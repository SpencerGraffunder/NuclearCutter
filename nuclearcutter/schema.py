"""
Data models for NuclearCutter's two artifact types:

1. ScanResult   — the neutral, shareable record of what's in a movie.
                  Written by `nuclearcutter scan`, read by `nuclearcutter render`.
                  This is what gets committed to timestamps/ in the repo.

2. Preferences  — a user's personal choice of what action to take per
                  category. Never shared, never committed alongside a
                  ScanResult for someone else's use.

See docs/SPEC.md sections 2, 3, and 6 for the reasoning behind keeping
these two things strictly separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 1


class Category(str, Enum):
    NUDITY = "nudity"
    GORE = "gore"
    VIOLENCE = "violence"
    FOUL_LANGUAGE = "foul_language"

    @classmethod
    def from_legacy(cls, value: str) -> "Category":
        """Map old category names (from pre-2026-08 scan files) to the new set.

        NOTE: don't use `legacy.get(value, cls(value))` here — the default
        argument is evaluated eagerly, so an unknown-but-legacy value would
        raise before the mapping is consulted.
        """
        legacy = {
            "intimate_scenes": cls.NUDITY,
            "gore_violence": cls.GORE,
        }
        if value in legacy:
            return legacy[value]
        return cls(value)


class VisualAction(str, Enum):
    """What to do to the VIDEO for a flagged category."""
    NONE = "none"
    BLUR = "blur"  # intense box blur + clean text description overlay
    BLACK = "black"  # black screen + clean text description overlay


class AudioAction(str, Enum):
    """What to do to the AUDIO for a flagged category.

    Foul language has word-level timestamps, so it can mute/replace individual
    words or whole phrases. Visual categories (nudity/gore/violence) have NO
    per-sound recognition, so their only audio options are none / mute_scene
    (silence the whole flagged scene).
    """
    NONE = "none"
    MUTE_WORD = "mute_word"  # silence the flagged word (foul language only)
    MUTE_PHRASE = "mute_phrase"  # silence the whole utterance/phrase (foul language only)
    MUTE_SCENE = "mute_scene"  # silence the whole flagged scene (visual categories)
    REPLACE_WORD = "replace_word"  # (upcoming) AI voice replacement
    REPLACE_PHRASE = "replace_phrase"  # (upcoming) AI voice replacement

    @property
    def mutes_audio(self) -> bool:
        return self in (AudioAction.MUTE_WORD, AudioAction.MUTE_PHRASE,
                        AudioAction.MUTE_SCENE,
                        AudioAction.REPLACE_WORD, AudioAction.REPLACE_PHRASE)


# Module-level (NOT class attributes — a `str`-Enum would turn them into
# members). Valid audio actions by category type.
# Visual categories (nudity/gore/violence): no per-sound recognition, so only
# none / mute_scene. Foul language has word-level timestamps: full set.
VISUAL_AUDIO_ACTIONS = ("none", "mute_scene")
LANGUAGE_AUDIO_ACTIONS = ("none", "mute_word", "mute_phrase",
                          "replace_word", "replace_phrase")


class SeverityLevel(str, Enum):
    """The FIXED severity scale every detection is classified into.

    Standardized (NOT user-editable) so scan files can be shared. The level a
    detection is assigned = how MUCH censorship is needed to catch it:

      low    — the WORST content; caught even by the most restrictive filter
               (bare genitals/full nudity, extreme gore, murder, slurs)
      med    — caught by a moderate filter
      high   — caught by a broad filter (revealing clothing, any blood, slaps)
      exhigh — the MILDEST flagged content; only caught by a maximal filter
               (anything even slightly bad — a towel, a shove, "omg")

    A user's per-category threshold (`nudity_level` etc.) is how strict they
    want to be: `low` = low censorship (only the worst), `exhigh` = censor
    basically everything. Ordering by rank: low < med < high < exhigh.
    """
    LOW = "low"
    MED = "med"
    HIGH = "high"
    EXHIGH = "exhigh"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self.value]

    def is_corrected_by(self, threshold: "SeverityLevel") -> bool:
        """True if a detection at this level gets corrected by a user whose
        threshold is `threshold`. Content at/above the threshold's strictness
        is corrected — i.e. this level's rank <= the threshold's rank.

        `low` threshold -> only rank-0 (worst) content corrected (low censorship).
        `exhigh` threshold -> everything (rank <= 3) corrected (max censorship).
        """
        return self.rank <= threshold.rank

    @classmethod
    def from_any(cls, value) -> "SeverityLevel":
        """Tolerant parse (str, enum member, None). Unknown -> MED (safe default)."""
        if isinstance(value, SeverityLevel):
            return value
        if value is None:
            return SeverityLevel.MED
        try:
            return SeverityLevel(str(value).strip().lower())
        except ValueError:
            return SeverityLevel.MED


# Module-level (NOT a class attribute — a `str`-Enum would turn a class attr
# into a member). Used for ordering comparisons.
_SEVERITY_RANK = {"low": 0, "med": 1, "high": 2, "exhigh": 3}


@dataclass
class VisualDetection:
    """A single visual detection (nudity/gore/violence) from the VLM sweep."""

    category: Category
    start: float  # seconds
    end: float  # seconds
    description: str  # VLM-generated summary, visual + dialogue content woven together
    confidence: float  # 0-1, from the VLM confirmation stage
    level: SeverityLevel = SeverityLevel.MED  # how severe (low/med/high/exhigh)
    stage_a_score: Optional[float] = None  # retained for backward-compat; always None with the VLM sweep

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["level"] = self.level.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "VisualDetection":
        d = dict(d)
        d["category"] = Category.from_legacy(d["category"])
        d["level"] = SeverityLevel.from_any(d.get("level"))
        return VisualDetection(**d)


@dataclass
class LanguageDetection:
    """A single foul-language detection."""

    start: float  # seconds, tightest window (the word itself)
    end: float
    utterance_start: float  # seconds, start of the containing sentence/utterance
    utterance_end: float
    word: str
    transcript_source: str  # "whisper", "subtitle", or "whisper+subtitle"
    llm_confirmed: bool  # result of the always-on LLM context check
    llm_reasoning: Optional[str] = None
    level: SeverityLevel = SeverityLevel.MED  # severity of this word (low/med/high/exhigh)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "LanguageDetection":
        d = dict(d)
        d["level"] = SeverityLevel.from_any(d.get("level"))
        return LanguageDetection(**d)


@dataclass
class FilmIdentity:
    """Fingerprint data used to match a ScanResult to a local file.

    See docs/SPEC.md section 5. Duration + perceptual hashes at fixed
    percentage-of-runtime points, so this is resilient to different
    containers/frame-rates/encodes of the same underlying film.
    """

    title: Optional[str]
    year: Optional[int]
    duration_seconds: float
    phash_samples: list[dict]  # [{"pct": 0.1, "phash": "..."}, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FilmIdentity":
        return FilmIdentity(**d)


@dataclass
class ScanResult:
    schema_version: int
    identity: FilmIdentity
    visual_detections: list[VisualDetection] = field(default_factory=list)
    language_detections: list[LanguageDetection] = field(default_factory=list)
    generator: dict = field(default_factory=dict)  # model names/versions used, for provenance

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "visual_detections": [d.to_dict() for d in self.visual_detections],
            "language_detections": [d.to_dict() for d in self.language_detections],
            "generator": self.generator,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> "ScanResult":
        d = json.loads(path.read_text())
        return ScanResult(
            schema_version=d["schema_version"],
            identity=FilmIdentity.from_dict(d["identity"]),
            visual_detections=[VisualDetection.from_dict(x) for x in d["visual_detections"]],
            language_detections=[LanguageDetection.from_dict(x) for x in d["language_detections"]],
            generator=d.get("generator", {}),
        )


@dataclass
class Preferences:
    """User's personal choice of action per category. Never shared.

    Each category has TWO independent corrections: a visual one (what to do
    to the video) and an audio one (what to do to the audio). E.g. nudity
    might be visual=BLUR with audio=NONE, while foul language is visual=NONE
    with audio=MUTE_PHRASE. See VisualAction / AudioAction for options.
    """

    nudity_visual: VisualAction = VisualAction.BLUR
    nudity_audio: AudioAction = AudioAction.NONE
    nudity_level: SeverityLevel = SeverityLevel.MED  # correct nudity at/above this

    gore_visual: VisualAction = VisualAction.BLUR
    gore_audio: AudioAction = AudioAction.NONE
    gore_level: SeverityLevel = SeverityLevel.MED

    violence_visual: VisualAction = VisualAction.BLUR
    violence_audio: AudioAction = AudioAction.NONE
    violence_level: SeverityLevel = SeverityLevel.MED

    foul_language_visual: VisualAction = VisualAction.NONE
    foul_language_audio: AudioAction = AudioAction.MUTE_PHRASE
    foul_language_level: SeverityLevel = SeverityLevel.MED

    # Multiplier on blur intensity. 1.0 = the standard intense boxblur
    # (radius 30, 3 passes); 2.0 = twice as extreme (radius 60, 6 passes).
    blur_strength: float = 1.0
    # Extra seconds muted before/after each flagged word/utterance window.
    mute_padding: float = 0.5
    # Extra seconds blurred/blacked before/after each flagged visual segment.
    blur_padding: float = 0.0

    def visual_for(self, category: Category) -> VisualAction:
        if category == Category.NUDITY:
            return self.nudity_visual
        if category == Category.GORE:
            return self.gore_visual
        if category == Category.VIOLENCE:
            return self.violence_visual
        if category == Category.FOUL_LANGUAGE:
            return self.foul_language_visual
        raise ValueError(f"Unknown category: {category}")

    def audio_for(self, category: Category) -> AudioAction:
        if category == Category.NUDITY:
            return self.nudity_audio
        if category == Category.GORE:
            return self.gore_audio
        if category == Category.VIOLENCE:
            return self.violence_audio
        if category == Category.FOUL_LANGUAGE:
            return self.foul_language_audio
        raise ValueError(f"Unknown category: {category}")

    def level_for(self, category: Category) -> SeverityLevel:
        """The user's threshold level for a category — correct at/above this."""
        if category == Category.NUDITY:
            return self.nudity_level
        if category == Category.GORE:
            return self.gore_level
        if category == Category.VIOLENCE:
            return self.violence_level
        if category == Category.FOUL_LANGUAGE:
            return self.foul_language_level
        raise ValueError(f"Unknown category: {category}")

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("nudity_visual", "gore_visual", "violence_visual", "foul_language_visual"):
            d[key] = getattr(self, key).value
        for key in ("nudity_audio", "gore_audio", "violence_audio", "foul_language_audio"):
            d[key] = getattr(self, key).value
        for key in ("nudity_level", "gore_level", "violence_level", "foul_language_level"):
            d[key] = getattr(self, key).value
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> "Preferences":
        d = json.loads(path.read_text())
        return Preferences(
            nudity_visual=VisualAction(d.get("nudity_visual", "blur")),
            nudity_audio=AudioAction(d.get("nudity_audio", "none")),
            nudity_level=SeverityLevel.from_any(d.get("nudity_level")),
            gore_visual=VisualAction(d.get("gore_visual", "blur")),
            gore_audio=AudioAction(d.get("gore_audio", "none")),
            gore_level=SeverityLevel.from_any(d.get("gore_level")),
            violence_visual=VisualAction(d.get("violence_visual", "blur")),
            violence_audio=AudioAction(d.get("violence_audio", "none")),
            violence_level=SeverityLevel.from_any(d.get("violence_level")),
            foul_language_visual=VisualAction(d.get("foul_language_visual", "none")),
            foul_language_audio=AudioAction(d.get("foul_language_audio", "mute_phrase")),
            foul_language_level=SeverityLevel.from_any(d.get("foul_language_level")),
            blur_strength=float(d.get("blur_strength", 1.0)),
            mute_padding=float(d.get("mute_padding", 0.5)),
            blur_padding=float(d.get("blur_padding", 0.0)),
        )
