#!/usr/bin/env python3
"""Recover a completed scan result from a live status JSON.

If a scan finishes the pipeline but crashes before saving (e.g. the SMB share
disconnected while writing the output JSON), the full detection data is still
captured in the live status file. This reconstructs a proper ScanResult from
that status file — including the cached fingerprint for the identity — and
writes it to the same path the CLI would have used.

Usage:
    python3 scripts/reconstruct_scan.py <status.json> <video_path> [output.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuclearcutter.fingerprint.fingerprint import cache_path_for, load_cached_fingerprint
from nuclearcutter.schema import (
    Category, FilmIdentity, LanguageDetection, ScanResult, SeverityLevel, VisualDetection,
)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python3 scripts/reconstruct_scan.py STATUS_JSON VIDEO_PATH [OUTPUT]", file=sys.stderr)
        return 1
    status_path = Path(sys.argv[1])
    video_path = Path(sys.argv[2])

    status = json.loads(status_path.read_text())

    # Identity: reuse the cached fingerprint (same file -> same cache key).
    identity = None
    cached = load_cached_fingerprint(video_path)
    if cached:
        duration, phash_samples = cached
        identity = FilmIdentity(
            title=Path(video_path).stem,
            year=None,
            duration_seconds=duration,
            phash_samples=[s.to_dict() for s in phash_samples],
        )
        print(f"identity: cached fingerprint ({len(phash_samples)} samples, {duration:.0f}s)")
    else:
        identity = FilmIdentity(
            title=Path(video_path).stem,
            year=None,
            duration_seconds=float(status.get("duration_seconds", 0.0)),
            phash_samples=[],
        )
        print("identity: no cached fingerprint — phash_samples empty (still renderable)")

    visual = []
    for d in status.get("visual_detections", []):
        visual.append(VisualDetection(
            category=Category.from_legacy(d["category"]),
            start=float(d["start"]),
            end=float(d["end"]),
            description=d.get("description", ""),
            confidence=float(d.get("confidence", 0.5)),
            level=SeverityLevel.from_any(d.get("level")),
        ))

    lang = []
    for d in status.get("language_detections", []):
        lang.append(LanguageDetection(
            start=float(d["start"]),
            end=float(d["end"]),
            utterance_start=float(d.get("utterance_start", d["start"])),
            utterance_end=float(d.get("utterance_end", d["end"])),
            word=d.get("word", ""),
            transcript_source="whisper",
            llm_confirmed=True,
            level=SeverityLevel.from_any(d.get("level")),
        ))

    result = ScanResult(
        schema_version=1,
        identity=identity,
        visual_detections=visual,
        language_detections=lang,
        generator={
            "note": "reconstructed from live status file",
            "status": str(status_path),
        },
    )

    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else video_path.with_suffix(".nuclearcutter.json")
    try:
        result.save(out_path)
        print(f"Wrote {out_path}")
    except OSError as exc:
        fallback = Path.cwd() / out_path.name
        print(f"warning: cannot write {out_path.parent} ({exc}); falling back to {fallback}", file=sys.stderr)
        result.save(fallback)
        out_path = fallback
        print(f"Wrote {fallback}")

    print(f"Reconstructed: {len(visual)} visual detections, {len(lang)} language detections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
