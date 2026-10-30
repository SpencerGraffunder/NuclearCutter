"""
Live scan status — a small JSON file the scan writes as it runs, which the
web GUI and resume logic read.

The status file is written atomically (tmp + rename) so a reader never sees a
half-written file. It's plain JSON and written only on phase changes, every
~100 sweep frames, and after each confirmed detection — cheap enough that a
multi-hour scan keeps it fresh with no meaningful overhead.

JSON shape (schema v2 — tag taxonomy):

    {
      "schema": 2,
      "video": "Movie.mkv",
      "duration_seconds": 8496.0,
      "sweep_interval": 2.0,
      "started_at": "2026-10-29T19:15:00",
      "pid": 55436,
      "phase": "sweep",           # fingerprinting|transcribing|language|sweep|
                                  # identify|done
      "frames_done": 700,
      "frames_total": 4248,
      "position_seconds": 1400.0,
      "visual_candidates":  [ {"triggers":["skin"],"start":..,"end":..,"confidence":..}, ... ],
      "visual_detections":  [ {"tags":["nudity.shirtless_male"],"start":..,"end":..,
                                "description":..,"confidence":..,"rendering":"live_action",
                                "minors":"no"}, ... ],
      "language_detections":[ {"tags":["language.profanity_common"],"text":"..","start":..,
                                "end":..,"utterance_start":..,"utterance_end":..}, ... ]
    }

`visual_candidates` are raw sweep hits — which coarse TRIGGER fired, not what is
in the frame, because the sweep does not decide that (shown dim in the TUI).
`visual_detections` are the classified detections added as the identify pass
runs. Both are useful while the scan is in flight, and both carry tag ids so the
GUI can show what has been found so far in the same vocabulary it will use for
the chooser.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@dataclass
class ScanStatus:
    # The schema this build writes, and the only one it will RESUME from. A
    # status file from another schema describes a different search (see
    # scanner._load_resume_state for why resuming across it is dangerous).
    SCHEMA: int = SCHEMA_VERSION

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

    def add_candidate(self, start: float, end: float, triggers: list[str],
                      confidence: float) -> None:
        """A raw sweep hit (shown as a dim marker in the TUI).

        Carries TRIGGER names, not tag ids: the sweep only decides where to
        look, and writing a trigger name where a tag id belongs would imply a
        classification that has not happened yet.
        """
        # Merge with a touching candidate of the SAME triggers. The sweep reports
        # per-frame hits, so one gesture seen across three frames would otherwise
        # become three dim markers on the timeline AND three classify calls on the
        # same stretch of film. Different trigger sets stay separate: "skin" and
        # "blood" are different questions and both need answering.
        for c in self.visual_candidates:
            if (c.get("triggers") == list(triggers)
                    and start <= c["end"] + 0.01 and end >= c["start"] - 0.01):
                c["start"] = round(min(c["start"], start), 1)
                c["end"] = round(max(c["end"], end), 1)
                # A range is as sure as its best sighting, not its last one: the
                # final frame of a slow fade-out is often the blurriest.
                c["confidence"] = round(max(c.get("confidence", 0.0), confidence), 3)
                return
        self.visual_candidates.append(
            {"triggers": list(triggers), "start": round(start, 1),
             "end": round(end, 1), "confidence": round(confidence, 3)}
        )

    def add_visual_detection(self, start: float, end: float, tags: list[str],
                             description: str = "", confidence: float = 0.5,
                             rendering: str = "live_action", minors: str = "no") -> None:
        """A classified detection from the identify pass."""
        self.visual_detections.append(
            {"tags": list(tags), "start": round(start, 1), "end": round(end, 1),
             "description": description, "confidence": round(confidence, 3),
             "rendering": rendering, "minors": minors}
        )

    def add_language_detection(self, text: str, start: float, end: float,
                               utterance_start: float, utterance_end: float,
                               tags: list[str] | None = None) -> None:
        self.language_detections.append(
            {"tags": list(tags or []), "text": text,
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
