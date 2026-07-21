"""
Integration tests that actually invoke ffmpeg. Skipped automatically if
ffmpeg isn't on PATH. These are slower than the pure-logic unit tests but
catch real correctness issues (A/V sync, filter syntax errors) that
mocked-out tests would miss.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from nuclearcutter.fingerprint.fingerprint import compare_fingerprints, compute_fingerprint
from nuclearcutter.render.renderer import build_output_path, render
from nuclearcutter.schema import (
    Action, Category, FilmIdentity, LanguageDetection, Preferences, ScanResult, VisualDetection,
)
from nuclearcutter.utils.ffmpeg import probe_duration

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


@pytest.fixture
def synthetic_video(tmp_path):
    video_path = tmp_path / "test_movie.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=20",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(video_path),
        ],
        capture_output=True, check=True,
    )
    return video_path


@requires_ffmpeg
def test_render_blur_preserves_duration(synthetic_video, tmp_path):
    identity = FilmIdentity(title="T", year=2024, duration_seconds=20.0, phash_samples=[])
    vd_blur = VisualDetection(category=Category.INTIMATE_SCENES, start=8.0, end=12.0, description="A short scene happens here.", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd_blur], language_detections=[])
    prefs = Preferences(intimate_scenes_action=Action.BLUR)

    out_path = tmp_path / "out.mp4"
    result_path = render(synthetic_video, scan, prefs, output_path=out_path)

    assert result_path.exists()
    output_duration = probe_duration(result_path)
    assert abs(output_duration - 20.0) < 0.5  # blur doesn't change runtime


@requires_ffmpeg
def test_render_blur_preserves_duration(synthetic_video, tmp_path):
    identity = FilmIdentity(title="T", year=2024, duration_seconds=20.0, phash_samples=[])
    vd_blur = VisualDetection(category=Category.NUDITY, start=5.0, end=8.0, description="d", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd_blur], language_detections=[])
    prefs = Preferences(nudity_action=Action.BLUR)

    out_path = tmp_path / "out_blur.mp4"
    result_path = render(synthetic_video, scan, prefs, output_path=out_path)

    output_duration = probe_duration(result_path)
    assert abs(output_duration - 20.0) < 0.5  # blur doesn't change runtime


@requires_ffmpeg
def test_render_with_no_detections_passes_through(synthetic_video, tmp_path):
    identity = FilmIdentity(title="T", year=2024, duration_seconds=20.0, phash_samples=[])
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[])
    prefs = Preferences()

    out_path = tmp_path / "out_passthrough.mp4"
    result_path = render(synthetic_video, scan, prefs, output_path=out_path)

    output_duration = probe_duration(result_path)
    assert abs(output_duration - 20.0) < 0.5


@requires_ffmpeg
def test_build_output_path_appends_cleaned_suffix():
    p = Path("/movies/My Movie (2024).mkv")
    out = build_output_path(p)
    assert out.name == "My Movie (2024)_cleaned.mkv"
    assert out.parent == p.parent


@requires_ffmpeg
def test_fingerprint_identical_file_matches_itself(synthetic_video):
    duration_a, samples_a = compute_fingerprint(synthetic_video)
    duration_b, samples_b = compute_fingerprint(synthetic_video)

    is_match, details = compare_fingerprints(duration_a, samples_a, duration_b, samples_b)
    assert is_match is True
    assert details["match_ratio"] == 1.0


@requires_ffmpeg
def test_fingerprint_different_duration_rejects_match(synthetic_video, tmp_path):
    duration_a, samples_a = compute_fingerprint(synthetic_video)

    # Build a second video with a very different duration -- should reject
    # on duration mismatch alone.
    other_path = tmp_path / "other.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(other_path),
        ],
        capture_output=True, check=True,
    )
    duration_b, samples_b = compute_fingerprint(other_path)

    is_match, details = compare_fingerprints(duration_a, samples_a, duration_b, samples_b)
    assert is_match is False
    assert details["reason"] == "duration_mismatch"
