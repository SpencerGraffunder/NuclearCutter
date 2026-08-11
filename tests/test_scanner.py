"""Tests for the scan orchestrator, focusing on the unified VLM sweep pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nuclearcutter.detection.vlm_confirm import SweepRange
from nuclearcutter.schema import Category


@pytest.fixture
def mock_video(tmp_path: Path) -> Path:
    """Create a fake video file so path resolution works."""
    video = tmp_path / "test_movie.mkv"
    video.write_text("fake video content")
    return video


def _fake_identity_patches(monkeypatch):
    """Mock out fingerprinting so it doesn't need a real video."""
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


def _mock_transcription(monkeypatch):
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.transcribe",
        lambda _v, model=None: [],
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.find_subtitle_file",
        lambda _: None,
    )


def _mock_profanity(monkeypatch):
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.load_wordlist",
        lambda: [],
    )
    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.detect_foul_language",
        lambda _u, _c, _w, _s, foul_language_prompt=None: [],
    )


def test_scan_runs_sweep_and_confirms_ranges(monkeypatch, mock_video):
    """The scanner should drive the unified VisualSweepDetector: sweep, then
    confirm each returned range into a VisualDetection."""
    _fake_identity_patches(monkeypatch)
    _mock_transcription(monkeypatch)
    _mock_profanity(monkeypatch)
    monkeypatch.setattr("nuclearcutter.scan.scanner.gc.collect", lambda: None)

    captured = {"interval": None}

    class FakeDetector:
        def __init__(self, client):
            pass

        def sweep(self, video_path, sample_interval=5.0, on_flagged_window=None, on_progress=None,
                  resume_from=0, existing_windows=None):
            captured["interval"] = sample_interval
            return [
                SweepRange(start=10.0, end=20.0, category=Category.NUDITY,
                           description="sweep desc", confidence=0.9),
                SweepRange(start=50.0, end=60.0, category=Category.GORE,
                           description="gore", confidence=0.8),
            ]

        def confirm_and_describe(self, video_path, candidate, dialogue_text=""):
            from nuclearcutter.schema import VisualDetection
            return VisualDetection(
                category=candidate.category,
                start=candidate.start,
                end=candidate.end,
                description=candidate.description,
                confidence=candidate.confidence,
            )

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.VisualSweepDetector",
        lambda client, prompts=None: FakeDetector(client),
    )

    from nuclearcutter.scan.scanner import scan
    from nuclearcutter.utils.llm_client import LLMConfig

    llm_config = LLMConfig(
        base_url="http://localhost:9999/v1",
        vlm_model="test-model",
        text_model="test-model",
    )

    result = scan(mock_video, llm_config=llm_config)
    assert captured["interval"] == 2.0, "scanner should use the default sweep interval"
    assert len(result.visual_detections) == 2
    assert result.visual_detections[0].category == Category.NUDITY
    assert result.visual_detections[1].category == Category.GORE


def test_scan_forwards_custom_sweep_interval(monkeypatch, mock_video):
    """A caller-supplied sweep interval must reach the detector."""
    _fake_identity_patches(monkeypatch)
    _mock_transcription(monkeypatch)
    _mock_profanity(monkeypatch)
    monkeypatch.setattr("nuclearcutter.scan.scanner.gc.collect", lambda: None)

    captured = {"interval": None}

    class FakeDetector:
        def __init__(self, client):
            pass

        def sweep(self, video_path, sample_interval=5.0, on_flagged_window=None, on_progress=None,
                  resume_from=0, existing_windows=None):
            captured["interval"] = sample_interval
            return []

        def confirm_and_describe(self, video_path, candidate, dialogue_text=""):
            return None

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.VisualSweepDetector",
        lambda client, prompts=None: FakeDetector(client),
    )

    from nuclearcutter.scan.scanner import scan
    from nuclearcutter.utils.llm_client import LLMConfig

    llm_config = LLMConfig(
        base_url="http://localhost:9999/v1",
        vlm_model="test-model",
        text_model="test-model",
    )

    result = scan(mock_video, llm_config=llm_config, sweep_interval=2.5)
    assert captured["interval"] == 2.5, "scanner should forward sweep_interval to the detector"
    assert result.visual_detections == []


def test_scan_raises_when_all_confirmations_fail(monkeypatch, mock_video):
    """If every sweep range's confirmation fails, the scan should error loudly
    rather than silently emit an unreliable (empty) scan."""
    _fake_identity_patches(monkeypatch)
    _mock_transcription(monkeypatch)
    _mock_profanity(monkeypatch)
    monkeypatch.setattr("nuclearcutter.scan.scanner.gc.collect", lambda: None)

    class FailingDetector:
        def __init__(self, client):
            pass

        def sweep(self, video_path, sample_interval=5.0, on_flagged_window=None, on_progress=None,
                  resume_from=0, existing_windows=None):
            return [SweepRange(start=1.0, end=2.0, category=Category.NUDITY,
                               description="x", confidence=0.5)]

        def confirm_and_describe(self, video_path, candidate, dialogue_text=""):
            return None  # every confirmation "fails" (returns no detection)

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.VisualSweepDetector",
        lambda client, prompts=None: FailingDetector(client),
    )

    from nuclearcutter.scan.scanner import scan
    from nuclearcutter.utils.llm_client import LLMConfig

    llm_config = LLMConfig(
        base_url="http://localhost:9999/v1",
        vlm_model="test-model",
        text_model="test-model",
    )

    with pytest.raises(RuntimeError, match="confirmation queries failed"):
        scan(mock_video, llm_config=llm_config)


def test_scan_resumes_from_prior_status(monkeypatch, mock_video, tmp_path):
    """An interrupted scan resumes: the sweep gets resume_from + existing
    windows, and already-confirmed candidates are not re-confirmed."""
    _fake_identity_patches(monkeypatch)
    _mock_transcription(monkeypatch)
    _mock_profanity(monkeypatch)
    monkeypatch.setattr("nuclearcutter.scan.scanner.gc.collect", lambda: None)

    status_path = tmp_path / "test_movie.nuclearcutter.status.json"
    # Simulate a prior run that swept 8 frames, found one candidate (already
    # confirmed) and one unconfirmed candidate.
    prior = {
        "schema": 1, "video": mock_video.name, "duration_seconds": 100.0,
        "sweep_interval": 2.0, "pid": 1, "started_at": "2026-01-01T00:00:00",
        "phase": "visual_confirm", "frames_done": 8, "frames_total": 50,
        "position_seconds": 16.0,
        "visual_candidates": [
            {"category": "nudity", "start": 10.0, "end": 12.0, "confidence": 0.9, "level": "low"},
            {"category": "gore", "start": 50.0, "end": 52.0, "confidence": 0.8, "level": "low"},
        ],
        "visual_detections": [
            {"category": "nudity", "start": 8.0, "end": 14.0, "description": "already done",
             "confidence": 0.9, "level": "high"},
        ],
        "language_detections": [],
    }
    import json as _json
    status_path.write_text(_json.dumps(prior))

    captured = {"resume_from": None, "existing": None}

    class FakeDetector:
        def __init__(self, client):
            pass

        def sweep(self, video_path, sample_interval=5.0, on_flagged_window=None, on_progress=None,
                  resume_from=0, existing_windows=None):
            captured["resume_from"] = resume_from
            captured["existing"] = existing_windows
            # Return the two candidates from the prior status (reconstructed
            # from the seeded windows by the merge).
            return [
                SweepRange(start=8.0, end=14.0, category=Category.NUDITY,
                           description="already done", confidence=0.9, level="high"),
                SweepRange(start=50.0, end=54.0, category=Category.GORE,
                           description="new", confidence=0.8),
            ]

        def confirm_and_describe(self, video_path, candidate, dialogue_text=""):
            from nuclearcutter.schema import VisualDetection
            # Only the GORE candidate is new — the NUDITY one was confirmed.
            if candidate.category == Category.GORE:
                return VisualDetection(
                    category=Category.GORE, start=50.0, end=54.0,
                    description="confirmed now", confidence=0.8,
                )
            raise AssertionError("already-confirmed candidate was re-confirmed!")

    monkeypatch.setattr(
        "nuclearcutter.scan.scanner.VisualSweepDetector",
        lambda client, prompts=None: FakeDetector(client),
    )

    from nuclearcutter.scan.scanner import scan
    from nuclearcutter.utils.llm_client import LLMConfig

    llm_config = LLMConfig(
        base_url="http://localhost:9999/v1",
        vlm_model="test-model",
        text_model="test-model",
    )

    result = scan(mock_video, llm_config=llm_config, status_path=status_path)

    # Sweep resumed from frame 8 with the two prior raw windows.
    assert captured["resume_from"] == 8
    assert len(captured["existing"]) == 2
    # Both confirmed detections present (prior NUDITY + newly confirmed GORE).
    assert len(result.visual_detections) == 2
    cats = {d.category for d in result.visual_detections}
    assert cats == {Category.NUDITY, Category.GORE}
