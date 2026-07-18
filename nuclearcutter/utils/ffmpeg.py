"""
Thin wrappers around ffmpeg/ffprobe. All video/audio manipulation in
NuclearCutter goes through here so there's one place that knows how to talk to
ffmpeg.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def probe_duration(video_path: Path) -> float:
    """Return the duration of a media file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def probe_streams(video_path: Path) -> dict:
    """Return full ffprobe stream info (codecs, resolution, audio tracks, subtitle tracks)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_streams", "-show_format",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def extract_frame_at(video_path: Path, timestamp_seconds: float) -> Path:
    """Extract a single frame at the given timestamp to a temp PNG file.

    Caller is responsible for deleting the returned path when done.
    """
    fd, out_path = tempfile.mkstemp(suffix=".png")
    out_path = Path(out_path)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(timestamp_seconds),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ],
        capture_output=True, check=True,
    )
    return out_path


def extract_audio_track(video_path: Path, out_path: Path, start: float = None, end: float = None) -> Path:
    """Extract audio (as 16kHz mono WAV, suitable for Whisper) optionally trimmed to [start, end]."""
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video_path)]
    if end is not None:
        duration = end - (start or 0)
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_path)]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


def extract_frames_in_range(video_path: Path, start: float, end: float, fps: float, out_dir: Path) -> list[Path]:
    """Extract frames from [start, end] at the given fps into out_dir. Returns sorted list of frame paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.png"
    duration = end - start
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(pattern),
        ],
        capture_output=True, check=True,
    )
    return sorted(out_dir.glob("frame_*.png"))
