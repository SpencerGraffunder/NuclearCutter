"""Local cache directory for NuclearCutter sidecar files.

When a video file is on a read-only volume (network drive, external disk),
we can't write `.fingerprint.json`, `.stage_a_checkpoint.json`, etc. next to
it.  This module maps those paths to `~/.cache/nuclearcutter/` instead, using
a hash of the video's absolute path to keep filenames unique and predictable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _cache_root() -> Path:
    return Path.home() / ".cache" / "nuclearcutter"


def _path_hash(video_path: Path) -> str:
    resolved = video_path.resolve()
    return hashlib.sha256(str(resolved).encode()).hexdigest()[:16]


def cache_path_for(video_path: Path, suffix: str, subdir: str = "") -> Path:
    """Return a writable local path for a sidecar file associated with *video_path*.

    Parameters
    ----------
    video_path : Path
        Path to the video file (may be on a read-only volume).
    suffix : str
        File suffix, e.g. ``".fingerprint.json"``.
    subdir : str
        Optional subdirectory under ``~/.cache/nuclearcutter/``, e.g. ``"checkpoint"``.

    Returns
    -------
    Path
        A path like ``~/.cache/nuclearcutter/checkpoint/abc123...json`` whose
        directory is guaranteed to exist (created if needed).
    """
    parts = [_cache_root()]
    if subdir:
        parts.append(subdir)
    filename = f"{_path_hash(video_path)}{suffix}"
    parts.append(filename)
    p = Path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
