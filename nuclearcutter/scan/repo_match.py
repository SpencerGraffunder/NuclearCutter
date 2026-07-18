"""
Finds and verifies a matching ScanResult from the shared timestamps/
repo directory for a local video file, so a full rescan can be skipped
when someone else has already scanned the same film (docs/SPEC.md
section 5-6).
"""

from __future__ import annotations

from pathlib import Path

from nuclearcutter.detection.vlm_confirm import VlmConfirmer
from nuclearcutter.fingerprint.fingerprint import compare_fingerprints, compute_fingerprint, PhashSample
from nuclearcutter.schema import ScanResult
from nuclearcutter.utils.ffmpeg import extract_frames_in_range
from nuclearcutter.utils.llm_client import LLMClient

# How many existing flagged ranges to spot-check with the VLM before
# trusting a fingerprint match, per docs/SPEC.md section 5 step 3.
SPOT_CHECK_COUNT = 3

SPOT_CHECK_PROMPT = """You are verifying whether a timestamp file generated for one release \
of a movie also applies to a different video file that is supposedly the same movie.

The timestamp file says this scene should show: "{description}"

Look at the attached frames sampled from the corresponding time range in the file being \
checked. Does the content roughly match this description (same scene, same general content)? \
Minor differences in color grading/cropping/subtitles are fine — you're checking whether this \
is fundamentally the same scene, not a pixel-perfect match.

Respond with ONLY a JSON object: {{"matches": true or false, "reasoning": "one short sentence"}}"""


def find_matching_scan(video_path: Path, timestamps_dir: Path) -> ScanResult | None:
    """Search timestamps_dir for a ScanResult whose fingerprint matches video_path.

    Returns the matching ScanResult (fingerprint-verified, not yet spot-checked)
    or None if nothing matches closely enough.
    """
    if not timestamps_dir.exists():
        return None

    local_duration, local_samples = compute_fingerprint(video_path)

    for json_path in timestamps_dir.glob("*.json"):
        try:
            candidate = ScanResult.load(json_path)
        except Exception:
            continue

        remote_samples = [PhashSample.from_dict(s) for s in candidate.identity.phash_samples]
        is_match, details = compare_fingerprints(
            local_duration, local_samples,
            candidate.identity.duration_seconds, remote_samples,
        )
        if is_match:
            return candidate

    return None


def spot_check_match(video_path: Path, candidate: ScanResult, client: LLMClient) -> bool:
    """VLM spot-check a few flagged ranges from the candidate ScanResult against the
    local file, per docs/SPEC.md section 5 step 3. Returns True if confident enough
    to skip a full rescan."""
    if not candidate.visual_detections:
        # Nothing to spot-check against (a scan with zero visual detections) —
        # duration+phash match is all we have; treat as sufficient since
        # there's no richer signal available.
        return True

    confirmer_client = client
    to_check = candidate.visual_detections[:SPOT_CHECK_COUNT]

    confirmations = 0
    for detection in to_check:
        frames_dir_files = extract_frames_in_range(
            video_path, detection.start, detection.end,
            fps=2.0, out_dir=video_path.parent / ".cleancut_spotcheck_tmp",
        )
        try:
            if not frames_dir_files:
                continue
            sample_frames = frames_dir_files[:4]
            prompt = SPOT_CHECK_PROMPT.format(description=detection.description)
            result = confirmer_client.vision_query_json(prompt, sample_frames)
            if result.get("matches"):
                confirmations += 1
        finally:
            for f in frames_dir_files:
                f.unlink(missing_ok=True)

    # Require a majority of spot-checked ranges to confirm.
    return confirmations >= (len(to_check) / 2)
