"""
Data models for NuclearCutter's two artifact types:

1. ScanResult   — the neutral, shareable record of what's in a movie.
                  Written by `nuclearcutter scan`, read by `nuclearcutter render`.
                  This is what gets committed to timestamps/ in the repo.

2. Preferences  — a user's personal choice of what to censor and what to do
                  about it. Never shared, never committed alongside a
                  ScanResult for someone else's use.

Since schema v2 a detection carries a list of `tags` drawn from the fixed
vocabulary in `nuclearcutter/taxonomy.py` — a description of *what is there* —
instead of a severity level, which was a description of *how bad it was*. That
inversion is the whole redesign: the scan file stops being an opinion, so one
scan serves every viewer, and the viewer's choice is a set of tag ids
(`taxonomy.Selection`) rather than a threshold to argue with. See docs/SPEC.md
§2, §3 and §6, and docs/TAXONOMY.md for the vocabulary itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Optional

from nuclearcutter.taxonomy import (
    Family, Minors, Rendering, Selection, TAGS, enforce_minor_protection,
    family_of, normalize_tags, rendering_allowed,
)


SCHEMA_VERSION = 2

# A scan file written by an older schema is rejected outright, never coerced.
# v1 stored `category` + a low/med/high/exhigh `level`, which was a judgement
# about how much censorship a scene needed; there is no honest way to turn that
# back into "what is in the frame", and pretending otherwise would silently
# censor or silently spare scenes on a guess. Re-scan instead.
SUPPORTED_SCAN_VERSIONS = (SCHEMA_VERSION,)


class UnsupportedScanVersion(ValueError):
    """Raised when a scan file's schema_version is not supported."""


class VisualAction(str, Enum):
    """What to do to the VIDEO for a matched category."""
    NONE = "none"
    BLUR = "blur"  # intense box blur + clean text description overlay
    BLACK = "black"  # black screen + clean text description overlay


class AudioAction(str, Enum):
    """What to do to the AUDIO for a matched category.

    Foul language has word-level timestamps, so it can mute/replace individual
    words or whole phrases. Visual categories have no per-sound recognition, so
    their only audio options are none / mute_scene (silence the whole segment).
    """
    NONE = "none"
    MUTE_WORD = "mute_word"  # silence the flagged word (language detections)
    MUTE_PHRASE = "mute_phrase"  # silence the whole utterance/phrase (language)
    MUTE_SCENE = "mute_scene"  # silence the whole flagged segment (visual)
    REPLACE_WORD = "replace_word"  # (upcoming) AI voice replacement
    REPLACE_PHRASE = "replace_phrase"  # (upcoming) AI voice replacement

    @property
    def mutes_audio(self) -> bool:
        return self in (AudioAction.MUTE_WORD, AudioAction.MUTE_PHRASE,
                        AudioAction.MUTE_SCENE,
                        AudioAction.REPLACE_WORD, AudioAction.REPLACE_PHRASE)


# Module-level (NOT class attributes — a `str`-Enum would turn them into
# members). Valid audio actions by detection type.
VISUAL_AUDIO_ACTIONS = ("none", "mute_scene")
LANGUAGE_AUDIO_ACTIONS = ("none", "mute_word", "mute_phrase",
                          "replace_word", "replace_phrase")

# What a user may pick per family in the chooser. Family -> (visual options,
# audio options). Defaults come from taxonomy._FAMILY_MEDIA.
_FAMILY_MEDIA_DEFAULT = {
    "nudity": (VisualAction.BLUR, AudioAction.NONE),
    "intimacy": (VisualAction.BLUR, AudioAction.NONE),
    "violence": (VisualAction.BLUR, AudioAction.NONE),
    "gore": (VisualAction.BLUR, AudioAction.NONE),
    "harm": (VisualAction.BLUR, AudioAction.NONE),
    "language": (VisualAction.NONE, AudioAction.MUTE_PHRASE),
    "substance": (VisualAction.BLUR, AudioAction.NONE),
    "safety": (VisualAction.BLUR, AudioAction.NONE),
}


def replace_tags(det, new_tags: list[str]) -> None:
    """Rewrite a detection's tags, keeping the child-protection floor.

    A viewer correcting what the model saw replaces the tag list whole: unticking
    "breasts (women)" on a detection means "that is not on screen", and there is no
    partial-edit version of that. But `enforce_minor_protection()` decides whether
    to censor a child's exposure by looking for a nudity/intimacy tag on the
    detection — so a naive `det.tags = new` would let someone walk under the floor
    by deleting the exposure tags, which are the RECORD of why the floor applies.

    So the floor tag is carried across the replacement here, at the one place a
    tag list is rewritten wholesale, rather than in each caller that might forget.
    `minors` is the honest switch: set it to "no" if the age estimate was wrong.
    """
    floor = "safety.exposure_with_minor"
    had_floor = floor in (getattr(det, "tags", []) or [])
    det.tags = [t for t in new_tags]
    if had_floor and getattr(det, "minors", None) == Minors.YES.value \
            and floor not in det.tags:
        # Keep the floor while the reason for it still stands. Re-adding it AFTER
        # enforce_minor_protection() would resurrect it even when the viewer
        # cleared the minors flag, and there would be no way back out: clearing
        # `minors` is the honest way to say the age estimate was wrong.
        det.tags.append(floor)
    enforce_minor_protection(det)


def _parse_tags(raw) -> list[str]:
    """Read a stored tag list: keep every id, in order, without re-deciding it.

    This deliberately does NOT run `normalize_tags`. Folding an unfamiliar id
    into its family residual is the right call for a MODEL's output (a made-up
    category should become "something in this family"), but a scan file is not
    model output: it was written by some build's vocabulary, possibly a newer
    one. Rewriting `nudity.some_future_tag` to `nudity.unclear` here would
    destroy a fact the file actually knows, and the rewrite would be saved back
    on the next edit, so the information would be gone forever.

    An unknown id is still censorable — `tags_hit()` treats it as a hit whenever
    it appears — so keeping it verbatim is the fail-safe choice too. Only junk
    (non-strings, repeats) is dropped.
    """
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        tid = item.strip()
        if tid and tid not in out:
            out.append(tid)
    return out


@dataclass
class VisualDetection:
    """A single visual detection: one moment in the film plus what is in it."""

    start: float  # seconds
    end: float  # seconds
    description: str  # clean, factual summary (also the on-screen text)
    confidence: float  # 0-1, from the identify pass
    tags: list[str] = field(default_factory=list)  # vocabulary ids
    rendering: str = Rendering.LIVE_ACTION.value
    minors: str = Minors.NO.value
    uncertain_tags: list[str] = field(default_factory=list)  # subset of tags
    triggers: list[str] = field(default_factory=list)  # coarse sweep triggers,
    #   kept purely as provenance: which cheap cue let the model look here.
    removed: bool = False  # user dismissed it; timeline shows it, render skips it

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "VisualDetection":
        tags = _parse_tags(d.get("tags"))
        uncertain = [t for t in (d.get("uncertain_tags") or []) if t in TAGS]
        rendering = d.get("rendering") or Rendering.LIVE_ACTION.value
        if rendering not in {r.value for r in Rendering}:
            rendering = Rendering.LIVE_ACTION.value
        minors = d.get("minors") or Minors.NO.value
        if minors not in {m.value for m in Minors}:
            minors = Minors.NO.value
        return VisualDetection(
            start=float(d["start"]), end=float(d["end"]),
            description=d.get("description", ""),
            confidence=float(d.get("confidence", 0.5)),
            tags=tags,
            rendering=rendering,
            minors=minors,
            uncertain_tags=uncertain,
            triggers=[str(x) for x in (d.get("triggers") or [])],
            removed=bool(d.get("removed", False)),
        )

    # -- families ------------------------------------------------------------
    @property
    def families(self) -> list[Family]:
        out: list[Family] = []
        for t in self.tags:
            f = family_of(t)
            if f is not None and f not in out:
                out.append(f)
        return out

    @property
    def is_removed(self) -> bool:
        return self.removed

    @property
    def is_uncertain(self) -> bool:
        """True when the model itself said it was unsure about every tag it gave.

        Surfaced in the chooser: an uncertain tag set is what a human should
        look at first, and it is the honest successor to the old design hiding
        its uncertainty behind a permissive 'exhigh' threshold.
        """
        return bool(self.tags) and all(t in self.uncertain_tags for t in self.tags)

    def censorable(self, sel: Selection) -> list[str]:
        """Tags on this detection that the user's selection wants censored.

        Three rules, all fail-safe:
          * `rendering` may only NARROW a match. An unrecognised rendering value
            matches everything, so a bad guess cannot un-censor.
          * a detection the user dismissed ("removed") is not censored —
            that is their call about a scene they have looked at.
          * ...unless it carries an enforced tag. Removing a scene that shows
            exposure with a child present is not a call this tool makes on
            someone's behalf, so `removed` is ignored for the enforced tags and
            they still censor. The timeline keeps showing the marker as removed;
            the popup explains why it is still being hidden.
        """
        from nuclearcutter.taxonomy import tags_hit

        if not rendering_allowed(self.rendering, sel):
            return []
        hit = tags_hit(self.tags, sel)
        if self.removed:
            return [t for t in hit
                    if t in TAGS and TAGS[t].always_censor]
        return hit


@dataclass
class LanguageDetection:
    """A single language/substance detection from the transcript.

    v2 widened this from "one foul word" to "one line that contains something
    from the language/substance vocabulary", because things like a slur, a
    sexual remark or a crude joke are not single words. `text` is the word or
    phrase that carried it; `tags` say what kind.
    """

    start: float  # seconds, tightest window (the word/phrase itself)
    end: float
    utterance_start: float  # seconds, start of the containing sentence
    utterance_end: float
    text: str  # the offending word or phrase as spoken
    tags: list[str] = field(default_factory=list)
    # Where the WORDS came from, not which pass flagged them (that is
    # `llm_reasoning`): a whisper transcript cross-checked against a subtitle
    # file is more trustworthy, and the GUI shows it in the tooltip.
    transcript_source: str = "whisper"  # "whisper" | "subtitle" | "whisper+subtitle"
    llm_confirmed: bool = False
    llm_reasoning: Optional[str] = None
    removed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "LanguageDetection":
        return LanguageDetection(
            start=float(d["start"]), end=float(d["end"]),
            utterance_start=float(d.get("utterance_start", d["start"])),
            utterance_end=float(d.get("utterance_end", d["end"])),
            text=d.get("text", d.get("word", "")),
            tags=_parse_tags(d.get("tags")),
            transcript_source=d.get("transcript_source", "whisper"),
            llm_confirmed=bool(d.get("llm_confirmed", False)),
            llm_reasoning=d.get("llm_reasoning"),
            removed=bool(d.get("removed", False)),
        )

    @property
    def is_removed(self) -> bool:
        return self.removed

    def mute_window(self, padding: float, mute_whole_utterance: bool) -> tuple[float, float]:
        if mute_whole_utterance:
            return (self.utterance_start - padding, self.utterance_end + padding)
        return (self.start - padding, self.end + padding)

    def censorable(self, sel: Selection) -> list[str]:
        if self.removed:
            return []
        from nuclearcutter.taxonomy import tags_hit
        return tags_hit(self.tags, sel)


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
        # Tolerant on purpose: `year` and `source_path` are optional in practice
        # (a hand-written or older file may omit them), and a scan file that is
        # otherwise readable should not fail to open over a missing key.
        known = {f for f in ("title", "year", "duration_seconds", "phash_samples",
                             "source_path")}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs.setdefault("title", None)
        kwargs.setdefault("year", None)
        kwargs.setdefault("duration_seconds", 0.0)
        kwargs.setdefault("phash_samples", [])
        return FilmIdentity(**kwargs)


@dataclass
class ScanResult:
    schema_version: int
    identity: FilmIdentity
    visual_detections: list[VisualDetection] = field(default_factory=list)
    language_detections: list[LanguageDetection] = field(default_factory=list)
    generator: dict = field(default_factory=dict)  # models/versions, for provenance

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "visual_detections": [d.to_dict() for d in self.visual_detections],
            "language_detections": [d.to_dict() for d in self.language_detections],
            "generator": self.generator,
        }

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "ScanResult":
        d = json.loads(Path(path).read_text())
        version = int(d.get("schema_version") or 0)
        if version not in SUPPORTED_SCAN_VERSIONS:
            raise UnsupportedScanVersion(
                f"{Path(path).name}: scan file is schema v{version or '?'}, this build "
                f"reads v{'/'.join(str(v) for v in SUPPORTED_SCAN_VERSIONS)}. Scans made "
                f"before the tag-taxonomy redesign stored a severity level instead of "
                f"what is in the scene, which cannot be converted — re-scan the film.")
        visual = [VisualDetection.from_dict(x) for x in d.get("visual_detections", [])]
        # The child floor is re-applied on every load, not trusted from the file.
        # A scan file is plain JSON that people hand-edit, merge and share; the
        # protection has to survive any of that. `censorable()` independently
        # refuses to let `removed` hide an enforced tag, so this is the second
        # half of the same rule rather than the only defence.
        from nuclearcutter.taxonomy import enforce_minor_protection
        for det in visual:
            enforce_minor_protection(det)
        return ScanResult(
            schema_version=version,
            identity=FilmIdentity.from_dict(d["identity"]),
            visual_detections=visual,
            language_detections=[LanguageDetection.from_dict(x)
                                 for x in d.get("language_detections", [])],
            generator=d.get("generator", {}),
        )

    # -- chooser support -----------------------------------------------------
    def tag_counts(self) -> dict[str, int]:
        """How many detections carry each tag. Drives the chooser: a category
        that never occurred is shown greyed and collapsed, not as a wall of
        identical empty checkboxes."""
        counts: dict[str, int] = {}
        for d in self.visual_detections:
            for t in d.tags:
                counts[t] = counts.get(t, 0) + 1
        for d in self.language_detections:
            for t in d.tags:
                counts[t] = counts.get(t, 0) + 1
        return counts


@dataclass
class Preferences:
    """User's personal choice of what to censor, and what to do about it.

    Two orthogonal halves, deliberately kept apart:
      * `selection` — WHICH tags matter to this viewer (a set of ids).
      * per-family actions — what to DO when one matches (blur / black / none,
        mute / leave).
    Splitting them is what lets one scan file serve a strict viewer and a lax
    one, and lets the same person flip "blur" to "black" without re-deciding
    anything.
    """

    selection: Selection = field(default_factory=Selection)
    # NAMING TRAP: these two are `visual_actions` / `audio_actions` HERE, but
    # `family_actions: {family: {visual, audio}}` at the HTTP boundary the GUI
    # speaks (server.update_settings maps one to the other). The shapes differ
    # because the wire format groups by family — which is what a row in the GUI's
    # matrix is — while the schema splits by medium, which is what the renderer
    # needs. Both spellings are correct in their own layer; a rename that touches
    # only one silently drops every action override to defaults, so if either is
    # renamed, grep both `visual_actions` and `family_actions` in the same pass.
    visual_actions: dict[str, str] = field(
        default_factory=lambda: {k: v[0].value for k, v in _FAMILY_MEDIA_DEFAULT.items()})
    audio_actions: dict[str, str] = field(
        default_factory=lambda: {k: v[1].value for k, v in _FAMILY_MEDIA_DEFAULT.items()})

    # Multiplier on blur intensity. 1.0 = the standard intense boxblur
    # (radius 30, 3 passes); 2.0 = twice as extreme (radius 60, 6 passes).
    blur_strength: float = 1.0
    # Extra seconds muted before/after each flagged word/utterance window.
    mute_padding: float = 0.5
    # Extra seconds blurred/blacked before/after each flagged visual segment.
    blur_padding: float = 0.0

    def __post_init__(self) -> None:
        # A partially-specified Preferences is the normal case (the GUI sends the
        # families it has rows for), and a missing family must mean "the default",
        # never "the fallback in visual_for". Without this, `visual_actions={}`
        # silently BLURS every family, including language, which blanks the
        # picture while somebody swears.
        defaults_vis = {k: v[0].value for k, v in _FAMILY_MEDIA_DEFAULT.items()}
        defaults_aud = {k: v[1].value for k, v in _FAMILY_MEDIA_DEFAULT.items()}
        for key, val in defaults_vis.items():
            self.visual_actions.setdefault(key, val)
        for key, val in defaults_aud.items():
            self.audio_actions.setdefault(key, val)

    def visual_for(self, family: Family | str) -> VisualAction:
        key = family.value if isinstance(family, Family) else str(family)
        return VisualAction(self.visual_actions.get(key, VisualAction.BLUR.value))

    def audio_for(self, family: Family | str) -> AudioAction:
        key = family.value if isinstance(family, Family) else str(family)
        return AudioAction(self.audio_actions.get(key, AudioAction.NONE.value))

    def to_dict(self) -> dict:
        return {
            "selection": self.selection.to_dict(),
            "visual_actions": dict(self.visual_actions),
            "audio_actions": dict(self.audio_actions),
            "blur_strength": self.blur_strength,
            "mute_padding": self.mute_padding,
            "blur_padding": self.blur_padding,
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> "Preferences":
        d = json.loads(Path(path).read_text())
        return Preferences.from_dict(d)

    @staticmethod
    def from_dict(d: dict) -> "Preferences":
        sel = d.get("selection")
        vis = {k: v[0].value for k, v in _FAMILY_MEDIA_DEFAULT.items()}
        aud = {k: v[1].value for k, v in _FAMILY_MEDIA_DEFAULT.items()}
        for k, v in (d.get("visual_actions") or {}).items():
            if k in vis and v in {a.value for a in VisualAction}:
                vis[k] = v
        for k, v in (d.get("audio_actions") or {}).items():
            if k in aud and v in {a.value for a in AudioAction}:
                aud[k] = v
        return Preferences(
            # The GUI sends {"tags": [...], "renderings": [...]}; a hand-written
            # file or a script may send just the ids. Both mean the same thing.
            selection=(Selection.from_dict(sel) if isinstance(sel, dict)
                       else Selection.from_dict({"tags": sel}) if isinstance(sel, list)
                       else Selection()),
            visual_actions=vis,
            audio_actions=aud,
            blur_strength=float(d.get("blur_strength", 1.0)),
            mute_padding=float(d.get("mute_padding", 0.5)),
            blur_padding=float(d.get("blur_padding", 0.0)),
        )
