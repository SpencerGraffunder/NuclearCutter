"""Tests for the scan orchestrator, focusing on Stage A caching behavior."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nuclearcutter.detection.nsfw_classifier import CandidateRange
from nuclearcutter.schema import FilmIdentity


@pytest.fixture
def mock_video(tmp_path: Path) -> Path:
    """Create a fake video file so path resolution works."""
    video = tmp_path / "test_movie.mkv"
    video.write_text("fake video content")
    return video


@pytest.fixture
def fake_cache_dir() -> Path:
    """Return a temp dir that will be used as the cache root."""
    return Path(tempfile.mkdtemp(prefix="nuclearcutter_test_cache_"))


def _fake_fingerprint_data(video_path: Path, duration: float = 100.0):
    """Simulate load_cached_fingerprint returning a value."""
    return (duration, [])


def test_stage_a_results_loaded_from_cache_on_second_run(monkeypatch, mock_video, fake_cache_dir):
    """Verify that scan() loads cached Stage A results on re-run instead of
    re-running the NsfwClassifier.

    Regression test for the bug where stage_a_results_file was deleted
    after a completed scan, forcing Stage A to re-run from scratch on the
    next invocation.
    """
    # Redirect all cache paths to our temp dir.
    from nuclearcutter.utils.cache import cache_path_for as real_cache_path_for

    def fake_cache_path_for(video_path, suffix, subdir=""):
        # Use the real implementation but swap the cache root.
        original_root = Path.home() / ".cache" / "nuclearcutter"
        path = real_cache_path_for(video_path, suffix, subdir)
        # Rewrite to our temp dir.
        relative = path.relative_to(original_root)
        result = fake_cache_dir / relative
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.cache_path_for",
        fake_cache_path_for,
    )

    # Mock fingerprint so it doesn't need a real video.
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.load_cached_fingerprint",
        lambda _: (100.0, []),
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.compute_fingerprint",
        lambda _: (100.0, []),
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.cache_fingerprint",
        lambda _a, _b, _c: None,
    )

    # Mock transcription to return nothing.
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.transcribe",
        lambda _v, model=None: [],
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.find_subtitle_file",
        lambda _: None,
    )

    # Mock profanity detection.
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.load_wordlist",
        lambda: [],
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.detect_foul_language",
        lambda _u, _c, _w, _s: [],
    )

    # Mock VLM — confirm first candidate (start=10.0), reject second (start=50.0).
    def _make_fake_confirmer():
        def _fake_confirm(video_path, candidate, dialogue):
            if candidate.start == 10.0:
                from nuclearcutter.schema import Category
                from nuclearcutter.schema import VisualDetection
                return VisualDetection(
                    category=Category.NUDITY,
                    start=candidate.start,
                    end=candidate.end,
                    description="test",
                    confidence=0.9,
                    stage_a_score=candidate.peak_score,
                )
            return None
        return MagicMock(confirm_and_describe=_fake_confirm)

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.VlmConfirmer",
        lambda client: _make_fake_confirmer(),
    )

    # Mock gc.collect to be a no-op.
    monkeypatch.setattr("nuclearcutter.scan.scanner.gc.collect", lambda: None)

    # Create a spy on NsfwClassifier.scan to track calls.
    original_scan = None

    class NsfwClassifierSpy:
        def __init__(self):
            pass

        def scan(self, video_path, sample_interval=1.0, threshold=0.5, progress_callback=None):
            nonlocal original_scan
            # Track that scan was called
            original_scan = True
            return [
                CandidateRange(start=10.0, end=20.0, peak_score=0.9),
                CandidateRange(start=50.0, end=60.0, peak_score=0.8),
            ]

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.NsfwClassifier",
        lambda: NsfwClassifierSpy(),
    )

    from nuclearcutter.scan.scanner import scan
    from nuclearcutter.utils.llm_client import LLMConfig

    llm_config = LLMConfig(
        base_url="http://localhost:9999/v1",
        vlm_model="test-model",
        text_model="test-model",
    )

    # --- First call: no cache, should call NsfwClassifier.scan() ---
    original_scan = False
    result1 = scan(mock_video, llm_config=llm_config)
    assert original_scan is True, "First scan should have called NsfwClassifier.scan()"
    assert len(result1.visual_detections) == 1  # VLM confirms first candidate

    # Verify Stage A results were cached to disk.
    results_path = fake_cache_path_for(mock_video, ".stage_a_results.json", subdir="results")
    assert results_path.exists(), "Stage A results file should exist after first scan"

    # --- Second call: cache exists, should skip NsfwClassifier.scan() ---
    original_scan = False
    result2 = scan(mock_video, llm_config=llm_config)
    assert original_scan is False, "Second scan should NOT have called NsfwClassifier.scan() (cache hit)"
    assert len(result2.visual_detections) == 1

    # Verify the cache file is still there (regression: it should NOT have been deleted).
    assert results_path.exists(), "Stage A results file should still exist after second scan"


def test_stage_a_results_loaded_from_cache_content(monkeypatch, mock_video, fake_cache_dir):
    """Verify that cached Stage A results are correctly parsed and used."""
    from nuclearcutter.utils.cache import cache_path_for as real_cache_path_for

    def fake_cache_path_for(video_path, suffix, subdir=""):
        original_root = Path.home() / ".cache" / "nuclearcutter"
        path = real_cache_path_for(video_path, suffix, subdir)
        relative = path.relative_to(original_root)
        return fake_cache_dir / relative

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.cache_path_for",
        fake_cache_path_for,
    )

    # Pre-populate Stage A results with known data.
    results_path = fake_cache_path_for(mock_video, ".stage_a_results.json", subdir="results")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "ranges": [
            {"start": 15.0, "end": 25.0, "peak_score": 0.95},
            {"start": 55.0, "end": 65.0, "peak_score": 0.85},
        ]
    }))

    # Mock fingerprint.
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.load_cached_fingerprint",
        lambda _: (100.0, []),
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.compute_fingerprint",
        lambda _: (100.0, []),
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.cache_fingerprint",
        lambda _a, _b, _c: None,
    )

    # Mock transcription.
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.transcribe",
        lambda _v, model=None: [],
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.find_subtitle_file",
        lambda _: None,
    )

    # Mock profanity.
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.load_wordlist",
        lambda: [],
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.detect_foul_language",
        lambda _u, _c, _w, _s: [],
    )

    # Mock VLM to confirm one candidate.
    confirm_results = iter([
        None,  # reject first
        MagicMock(category=None, start=55.0, end=65.0, description="test", confidence=0.85, stage_a_score=0.85),
    ])

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.VlmConfirmer",
        lambda client: MagicMock(
            confirm_and_describe=lambda _v, candidate, _d: (
                None if candidate.start == 15.0
                else MagicMock(
                    category=None, start=55.0, end=65.0,
                    description="test", confidence=0.85,
                    stage_a_score=0.85,
                )
            )
        ),
    )

    monkeypatch.setattr("nuclearcutter.scan.scanner.gc.collect", lambda: None)

    # Make sure NsfwClassifier.scan() raises if called (cache should be used).
    class ShouldNotBeCalled:
        def __init__(self):
            pass
        def scan(self, **kwargs):
            raise AssertionError("NsfwClassifier.scan() should not be called when cache exists")

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.NsfwClassifier",
        lambda: ShouldNotBeCalled(),
    )

    from nuclearcutter.scan.scanner import scan
    from nuclearcutter.utils.llm_client import LLMConfig

    llm_config = LLMConfig(
        base_url="http://localhost:9999/v1",
        vlm_model="test-model",
        text_model="test-model",
    )

    result = scan(mock_video, llm_config=llm_config)
    # VLM should have processed both candidates from cache.
    # Actually our mock above only confirms 1 of 2, so we get 1 detection.
    assert len(result.visual_detections) == 1
