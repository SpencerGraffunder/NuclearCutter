"""
Fingerprint a video file so a shared ScanResult (possibly generated from a
different encode/rip/container of the same film) can be matched against a
local file with confidence — without trusting filename or file size.

Approach (see docs/SPEC.md section 5):
  1. Duration match (cheap first filter).
  2. Perceptual hash (pHash) of frames sampled at fixed *percentages* of
     total runtime (not fixed timestamps) — this makes it resilient to
     different container/frame-rate metadata, intro/outro differences, etc,
     as long as the underlying cut of the film is the same.
  3. (Optional, done by the caller, not here) VLM spot-check of a few
     flagged timestamp ranges for extra confidence before trusting a match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from nuclearcutter.utils.ffmpeg import extract_frame_at, probe_duration

# Sample points as a fraction of total runtime. Avoiding the very start/end
# since those are the most likely to differ between releases (studio logos,
# credits, different intro cuts).
DEFAULT_SAMPLE_POINTS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

# Hamming distance threshold below which two phashes are considered a match.
# imagehash phashes are 64-bit (8x8) by default; empirically <=8 is a loose
# "probably the same shot" match, <=4 is tight. We use a slightly looser bound
# here because we're comparing across different encodes/compression, not
# identical frames.
PHASH_MATCH_THRESHOLD = 10

# Fraction of sample points that must match for the whole file to be
# considered a match.
PHASH_MATCH_RATIO = 0.7

# Duration tolerance in seconds. Different container overhead / trailing
# black frames can cause small differences even for the same cut.
DURATION_TOLERANCE_SECONDS = 3.0


@dataclass
class PhashSample:
    pct: float
    phash: str  # hex string representation

    def to_dict(self) -> dict:
        return {"pct": self.pct, "phash": self.phash}

    @staticmethod
    def from_dict(d: dict) -> "PhashSample":
        return PhashSample(pct=d["pct"], phash=d["phash"])


def compute_fingerprint(video_path: Path, sample_points: list[float] = None) -> tuple[float, list[PhashSample]]:
    """Compute (duration_seconds, [PhashSample, ...]) for a video file."""
    sample_points = sample_points or DEFAULT_SAMPLE_POINTS
    duration = probe_duration(video_path)

    samples = []
    for pct in sample_points:
        timestamp = duration * pct
        frame_path = extract_frame_at(video_path, timestamp)
        try:
            img = Image.open(frame_path)
            phash = imagehash.phash(img)
            samples.append(PhashSample(pct=pct, phash=str(phash)))
        finally:
            frame_path.unlink(missing_ok=True)

    return duration, samples


def compare_fingerprints(
    duration_a: float,
    samples_a: list[PhashSample],
    duration_b: float,
    samples_b: list[PhashSample],
) -> tuple[bool, dict]:
    """Compare two fingerprints. Returns (is_match, details_dict)."""
    duration_diff = abs(duration_a - duration_b)
    if duration_diff > DURATION_TOLERANCE_SECONDS:
        return False, {
            "reason": "duration_mismatch",
            "duration_diff_seconds": duration_diff,
        }

    # Match samples by nearest pct (in case sample point sets differ).
    matches = 0
    comparisons = []
    for sa in samples_a:
        closest = min(samples_b, key=lambda sb: abs(sb.pct - sa.pct))
        hash_a = imagehash.hex_to_hash(sa.phash)
        hash_b = imagehash.hex_to_hash(closest.phash)
        distance = hash_a - hash_b
        is_match = distance <= PHASH_MATCH_THRESHOLD
        if is_match:
            matches += 1
        comparisons.append({
            "pct": sa.pct,
            "distance": distance,
            "matched": is_match,
        })

    match_ratio = matches / len(samples_a) if samples_a else 0.0
    is_match = match_ratio >= PHASH_MATCH_RATIO

    return is_match, {
        "reason": "phash_comparison",
        "match_ratio": match_ratio,
        "duration_diff_seconds": duration_diff,
        "comparisons": comparisons,
    }
