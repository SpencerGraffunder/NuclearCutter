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
    INTIMATE_SCENES = "intimate_scenes"
    FOUL_LANGUAGE = "foul_language"


class Action(str, Enum):
    BLUR = "blur"
    MUTE = "mute"
    NONE = "none"  # detected but user chose to leave it untouched


# Which actions are valid for which category — enforced at render time.
VALID_ACTIONS = {
    Category.NUDITY: {Action.BLUR, Action.NONE},
    Category.INTIMATE_SCENES: {Action.BLUR, Action.NONE},
    Category.FOUL_LANGUAGE: {Action.MUTE, Action.NONE},
}


@dataclass
class VisualDetection:
    """A single nudity/intimate_scenes detection from the Stage A+B pipeline."""

    category: Category
    start: float  # seconds
    end: float  # seconds
    description: str  # VLM-generated summary, visual + dialogue content woven together
    confidence: float  # 0-1, from the VLM confirmation stage
    stage_a_score: Optional[float] = None  # raw classifier score that triggered the candidate range

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "VisualDetection":
        d = dict(d)
        d["category"] = Category(d["category"])
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

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "LanguageDetection":
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
    """User's personal choice of action per category. Never shared."""

    nudity_action: Action = Action.BLUR
    nudity_blur_mute_audio: bool = False

    intimate_scenes_action: Action = Action.BLUR
    intimate_scenes_blur_mute_audio: bool = False

    foul_language_action: Action = Action.MUTE
    foul_language_mute_scope: str = "word"  # "word" or "utterance"

    def action_for(self, category: Category) -> Action:
        if category == Category.NUDITY:
            return self.nudity_action
        if category == Category.INTIMATE_SCENES:
            return self.intimate_scenes_action
        if category == Category.FOUL_LANGUAGE:
            return self.foul_language_action
        raise ValueError(f"Unknown category: {category}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nudity_action"] = self.nudity_action.value
        d["intimate_scenes_action"] = self.intimate_scenes_action.value
        d["foul_language_action"] = self.foul_language_action.value
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> "Preferences":
        d = json.loads(path.read_text())
        return Preferences(
            nudity_action=Action(d.get("nudity_action", "blur")),
            nudity_blur_mute_audio=d.get("nudity_blur_mute_audio", False),
            intimate_scenes_action=Action(d.get("intimate_scenes_action", "blur")),
            intimate_scenes_blur_mute_audio=d.get("intimate_scenes_blur_mute_audio", False),
            foul_language_action=Action(d.get("foul_language_action", "mute")),
            foul_language_mute_scope=d.get("foul_language_mute_scope", "word"),
        )
