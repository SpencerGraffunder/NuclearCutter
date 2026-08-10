"""
Live scan status — a small JSON file the scan writes as it runs, which
`nuclearcutter tui` (nuclearcutter/tui.py) reads to render a live dashboard.

The status file is written atomically (tmp + rename) so a reader never sees a
half-written file. It's plain JSON and written only on phase changes, every
~100 sweep frames, and after each confirmed detection — cheap enough that a
multi-hour scan keeps it fresh with no meaningful overhead.

JSON shape (schema v1):

    {
      "schema": 1,
      "video": "Movie.mkv",
      "duration_seconds": 8496.0,
      "sweep_interval": 2.0,
      "started_at": "2026-08-08T19:15:00",
      "pid": 55436,
      "phase": "visual_sweep",   # fingerprinting|transcribing|visual_sweep|
                                 # visual_confirm|language_detection|done
      "frames_done": 700,
      "frames_total": 4248,
      "position_seconds": 1400.0,
      "visual_candidates":  [ {"category":"NUDITY","start":..,"end":..,"confidence":..}, ... ],
      "visual_detections":  [ {"category":"NUDITY","start":..,"end":..,"description":..,"confidence":..}, ... ],
      "language_detections":[ {"category":"FOUL_LANGUAGE","start":..,"end":..,"word":..,
                               "utterance_start":..,"utterance_end":..}, ... ]
    }

`visual_candidates` are raw, unconfirmed VLM sweep hits (shown dim in the TUI);
`visual_detections` are the confirmed/enriched detections added as the confirm
pass runs. Both are useful while the scan is in flight.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@dataclass
class ScanStatus:
    video: str = ""
    duration_seconds: float | None = None
    sweep_interval: float | None = None
    started_at: str = ""
    pid: int | None = None
    phase: str = "starting"
    frames_done: int = 0
    frames_total: int = 0
    position_seconds: float | None = None
    visual_candidates: list = field(default_factory=list)
    visual_detections: list = field(default_factory=list)
    language_detections: list = field(default_factory=list)

    # -- mutators ----------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def set_sweep(self, done: int, total: int) -> None:
        self.frames_done = done
        self.frames_total = total
        if self.sweep_interval:
            self.position_seconds = round(done * self.sweep_interval, 1)

    def add_candidate(self, start: float, end: float, category: str, confidence: float,
                      level: str = "med") -> None:
        """A raw, unconfirmed VLM sweep hit (shown as a dim marker in the TUI)."""
        self.visual_candidates.append(
            {"category": category, "start": round(start, 1), "end": round(end, 1),
             "confidence": round(confidence, 3), "level": level}
        )

    def add_visual_detection(self, category, start: float, end: float,
                             description: str = "", confidence: float = 0.5,
                             level: str = "med") -> None:
        """A confirmed detection from the confirm pass."""
        cat = category.value if hasattr(category, "value") else str(category)
        self.visual_detections.append(
            {"category": cat, "start": round(start, 1), "end": round(end, 1),
             "description": description, "confidence": round(confidence, 3),
             "level": level}
        )

    def add_language_detection(self, word: str, start: float, end: float,
                               utterance_start: float, utterance_end: float) -> None:
        self.language_detections.append(
            {"category": "FOUL_LANGUAGE", "word": word,
             "start": round(start, 1), "end": round(end, 1),
             "utterance_start": round(utterance_start, 1),
             "utterance_end": round(utterance_end, 1)}
        )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            **{k: v for k, v in asdict(self).items()},
        }

    def write(self, path: Path | str) -> None:
        """Atomic write: tmp file + rename, so readers never see partial JSON."""
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path | str) -> "ScanStatus":
        data = json.loads(Path(path).read_text())
        st = cls()
        for k, v in data.items():
            if hasattr(st, k) and k != "schema":
                setattr(st, k, v)
        return st
