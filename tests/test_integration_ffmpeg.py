"""
Integration tests that actually invoke ffmpeg. Skipped automatically if
ffmpeg isn't on PATH. These are slower than the pure-logic unit tests but
catch real correctness issues (A/V sync, filter syntax errors) that
mocked-out tests would miss.
"""

import array
import shutil
import subprocess
from pathlib import Path

import pytest

from nuclearcutter.fingerprint.fingerprint import compare_fingerprints, compute_fingerprint
from nuclearcutter.render.renderer import build_output_path, render
from nuclearcutter.schema import (
    SCHEMA_VERSION, AudioAction, FilmIdentity, LanguageDetection, Preferences,
    ScanResult, Selection, VisualDetection, VisualAction,
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
    vd_blur = VisualDetection(start=8.0, end=12.0, tags=["gore.blood"],
                              description="A short scene happens here.", confidence=0.9)
    scan = ScanResult(schema_version=SCHEMA_VERSION, identity=identity,
                      visual_detections=[vd_blur], language_detections=[])
    prefs = Preferences(selection=Selection(tags=["gore.blood"]))

    out_path = tmp_path / "out.mp4"
    result_path = render(synthetic_video, scan, prefs, output_path=out_path)

    assert result_path.exists()
    output_duration = probe_duration(result_path)
    assert abs(output_duration - 20.0) < 0.5  # blur doesn't change runtime


@requires_ffmpeg
def test_render_nudity_blur_preserves_duration(synthetic_video, tmp_path):
    identity = FilmIdentity(title="T", year=2024, duration_seconds=20.0, phash_samples=[])
    vd_blur = VisualDetection(start=5.0, end=8.0, tags=["nudity.shirtless_male"],
                              description="d", confidence=0.9)
    scan = ScanResult(schema_version=SCHEMA_VERSION, identity=identity,
                      visual_detections=[vd_blur], language_detections=[])
    prefs = Preferences(selection=Selection(tags=["nudity.shirtless_male"]))

    out_path = tmp_path / "out_blur.mp4"
    result_path = render(synthetic_video, scan, prefs, output_path=out_path)

    output_duration = probe_duration(result_path)
    assert abs(output_duration - 20.0) < 0.5  # blur doesn't change runtime


@requires_ffmpeg
def test_render_with_no_detections_passes_through(synthetic_video, tmp_path):
    identity = FilmIdentity(title="T", year=2024, duration_seconds=20.0, phash_samples=[])
    scan = ScanResult(schema_version=SCHEMA_VERSION, identity=identity,
                      visual_detections=[], language_detections=[])
    # Nothing selected: the default. A render with no selection must be a
    # pass-through, not a blank screen.
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


def test_build_output_path_multi_suffix_places_cleaned_before_video_ext():
    """For 'Movie.mkv.iso' the _cleaned goes BEFORE the real (.mkv) extension."""
    from nuclearcutter.render.renderer import full_video_suffix

    assert full_video_suffix(Path("Movie.mkv")) == ".mkv"
    assert full_video_suffix(Path("Movie.mkv.iso")) == ".mkv.iso"
    assert build_output_path(Path("Movie.mkv.iso")).name == "Movie_cleaned.mkv.iso"
    long = Path("The.Martian.2015.1080p.HMAX.WEB-DL.DDP5.1.H.264-BLOOM.mkv.iso")
    assert build_output_path(long).name == \
        "The.Martian.2015.1080p.HMAX.WEB-DL.DDP5.1.H.264-BLOOM_cleaned.mkv.iso"


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


@requires_ffmpeg
def test_render_mutes_the_selected_word_and_keeps_the_timeline(synthetic_video, tmp_path):
    """Per-word muting is the feature the old whole-utterance mute could not
    offer: the sentence stays intelligible, the word disappears.

    The assertion here is the TIMELINE, not the audio waveform — a naive mute
    that drops samples shortens the file and desynchronises every later lip
    movement, which is far worse than leaving one swear in.
    """
    identity = FilmIdentity(title="T", year=2024, duration_seconds=20.0, phash_samples=[])
    scan = ScanResult(schema_version=SCHEMA_VERSION, identity=identity,
                      visual_detections=[], language_detections=[
                          LanguageDetection(start=2.0, end=2.6, utterance_start=1.0,
                                            utterance_end=4.0, text="damn it all",
                                            tags=["language.profanity"]),
                      ])
    prefs = Preferences(selection=Selection(tags=["language.profanity"]),
                        audio_actions={"language": AudioAction.MUTE_WORD.value},
                        mute_padding=0.0)

    out_path = tmp_path / "out_muted.mp4"
    result_path = render(synthetic_video, scan, prefs, output_path=out_path)
    assert result_path.exists()
    assert abs(probe_duration(result_path) - 20.0) < 0.5, \
        "muting silences samples, it must not cut the timeline"


@requires_ffmpeg
def test_render_muted_window_is_silent_and_everywhere_else_is_not(synthetic_video, tmp_path):
    """The mute has to be a hole in the right place: silence over the word, tone
    still audible a second later. Otherwise 'mute the slur' silently becomes
    'mute the film', which no test below catches."""
    identity = FilmIdentity(title="T", year=2024, duration_seconds=20.0, phash_samples=[])
    scan = ScanResult(schema_version=SCHEMA_VERSION, identity=identity,
                      visual_detections=[], language_detections=[
                          LanguageDetection(start=2.0, end=2.6, utterance_start=1.0,
                                            utterance_end=4.0, text="damn",
                                            tags=["language.profanity"]),
                      ])
    prefs = Preferences(selection=Selection(tags=["language.profanity"]),
                        audio_actions={"language": AudioAction.MUTE_WORD.value},
                        mute_padding=0.0)
    out = tmp_path / "levels.mp4"
    render(synthetic_video, scan, prefs, output_path=out)

    def rms_at(t: float) -> float:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(t), "-t", "0.2", "-i", str(out),
             "-map", "a", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
            capture_output=True, check=True).stdout
        samples = array.array("h")
        samples.frombytes(raw[: len(raw) // 2 * 2])
        if not samples:
            return 0.0
        return (sum(v * v for v in samples) / len(samples)) ** 0.5 / 32768.0

    during, after = rms_at(2.2), rms_at(5.0)
    assert during < 0.01, f"expected silence over the word, got rms {during:.3f}"
    assert after > 0.05, f"audio outside the muted word vanished (rms {after:.3f})"


@requires_ffmpeg
def test_render_unselected_language_tags_are_not_muted(synthetic_video, tmp_path):
    """A detection the viewer did not ask for must survive untouched — the
    default is nothing censored, per category, every time."""
    identity = FilmIdentity(title="T", year=2024, duration_seconds=20.0, phash_samples=[])
    scan = ScanResult(schema_version=SCHEMA_VERSION, identity=identity,
                      visual_detections=[], language_detections=[
                          LanguageDetection(start=2.0, end=2.6, utterance_start=1.0,
                                            utterance_end=4.0, text="damn it all",
                                            tags=["language.profanity"]),
                      ])
    # The viewer censors slurs; this detection is ordinary profanity.
    prefs = Preferences(selection=Selection(tags=["language.slur"]),
                        audio_actions={"language": AudioAction.MUTE_WORD.value})
    out_path = tmp_path / "out_untouched.mp4"
    result_path = render(synthetic_video, scan, prefs, output_path=out_path)

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "2.2", "-t", "0.2", "-i", str(result_path),
         "-map", "a", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) // 2 * 2])
    rms = ((sum(v * v for v in pcm) / len(pcm)) ** 0.5 / 32768.0) if pcm else 0.0
    assert rms > 0.05, f"an unselected category was muted (rms {rms:.3f})"
