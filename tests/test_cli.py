"""Tests for the CLI save path — a completed scan must never be lost to a
failed write (e.g. a disconnected SMB share) at the end of a long scan."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nuclearcutter.cli import _save_result_with_fallback
from nuclearcutter.schema import (
    Category,
    FilmIdentity,
    ScanResult,
    SeverityLevel,
    VisualDetection,
)


def _fake_result() -> ScanResult:
    return ScanResult(
        schema_version=1,
        identity=FilmIdentity(
            title="TestMovie",
            year=None,
            duration_seconds=120.0,
            phash_samples=[],
        ),
        visual_detections=[
            VisualDetection(
                category=Category.NUDITY,
                start=10.0,
                end=20.0,
                description="test",
                confidence=0.9,
                level=SeverityLevel.HIGH,
            )
        ],
        language_detections=[],
        generator={},
    )


def test_save_writes_to_intended_path_when_reachable(tmp_path: Path):
    result = _fake_result()
    out = tmp_path / "movie.nuclearcutter.json"

    saved = _save_result_with_fallback(result, out)

    assert saved == out
    assert out.exists()
    # Recovery copy also exists, next to the movie (no temp files).
    assert (tmp_path / "movie.nuclearcutter.json.recovery.json").exists()


def test_save_falls_back_to_cwd_when_share_disconnected(monkeypatch, tmp_path: Path):
    """A disconnected SMB share raises OSError/FileNotFoundError, NOT
    PermissionError — the old code only caught PermissionError and crashed,
    losing the 15-hour scan. It must fall back to CWD instead."""
    result = _fake_result()
    share_dir = tmp_path / "share"  # pretend this is the mounted SMB share
    share_dir.mkdir(exist_ok=True)
    out = share_dir / "movie.nuclearcutter.json"

    # Make writing ANY path inside the dead share fail (recovery copy AND the
    # intended output), the way a vanished mount does. CWD (mocked to tmp_path)
    # is the fallback.
    monkeypatch.chdir(tmp_path)

    original_save = ScanResult.save

    def _failing_share_save(self, path):
        if str(path).startswith(str(share_dir)):
            raise OSError(2, "No such file or directory")
        return original_save(self, path)

    monkeypatch.setattr(ScanResult, "save", _failing_share_save)

    saved = _save_result_with_fallback(result, out)

    # Fell back to CWD, not the dead share.
    assert saved == tmp_path / out.name
    assert saved.exists()
    assert not out.exists()
    assert not (share_dir / "movie.nuclearcutter.json.recovery.json").exists()


def test_save_raises_when_everywhere_fails(monkeypatch, tmp_path: Path):
    """The recovery copy falls back to CWD if the movie folder is dead, so the
    only way to raise is if EVERY target (movie folder + CWD) is unwritable."""
    result = _fake_result()
    out = tmp_path / "movie.nuclearcutter.json"

    # Everything fails: the movie-folder recovery copy, the CWD recovery copy,
    # the intended output, and the CWD fallback.
    monkeypatch.chdir(tmp_path)

    original_save = ScanResult.save

    def _everything_fails(self, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ScanResult, "save", _everything_fails)

    with pytest.raises(RuntimeError, match="could not save scan result locally"):
        _save_result_with_fallback(result, out)
