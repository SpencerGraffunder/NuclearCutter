"""
Thin wrappers around ffmpeg/ffprobe. All video/audio manipulation in
NuclearCutter goes through here so there's one place that knows how to talk to
ffmpeg.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

# Shared kwargs for ffmpeg/ffprobe subprocess calls.
# We capture both stdout and stderr (necessary for error diagnostics), but
# each call is synchronous and the result goes out of scope immediately, so
# this does NOT accumulate memory across repeated calls.  The real memory
# fix for long-running scans is in NsfwClassifier (periodic ONNX session
# recreation), not here.
_FFMPEG_RUN_KWARGS = {"capture_output": True, "text": True, "check": True}


def probe_duration(video_path: Path) -> float:
    """Return the duration of a media file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ],
        **_FFMPEG_RUN_KWARGS,
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
        **_FFMPEG_RUN_KWARGS,
    )
    return json.loads(result.stdout)


def extract_frame_at(video_path: Path, timestamp_seconds: float) -> Path:
    """Extract a single frame at the given timestamp to a temp PNG file.

    Caller is responsible for deleting the returned path when done.
    """
    fd, out_path = tempfile.mkstemp(suffix=".png")
    # Close the low-level fd so ffmpeg can safely write to the path on all
    # platforms. Not closing it can leave an empty file descriptor that some
    # imaging libraries (cv2) cannot read even after ffmpeg writes the file.
    os.close(fd)
    out_path = Path(out_path)

    def _run_ffmpeg(ts: float, path: Path) -> tuple[bool, str]:
        """Run ffmpeg extraction. Returns (success, stderr_text)."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(ts),
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(path),
                ],
                capture_output=True, text=True, check=True,
            )
            return True, result.stderr or ""
        except subprocess.CalledProcessError as e:
            return False, e.stderr or "(no stderr)"

    ok, stderr = _run_ffmpeg(timestamp_seconds, out_path)
    if not ok or not out_path.exists() or out_path.stat().st_size == 0:
        # Retry once, backing off 0.1s. This handles edge cases where the
        # requested timestamp lands right at the end of the file.
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        fd2, out_path = tempfile.mkstemp(suffix=".png")
        os.close(fd2)
        out_path = Path(out_path)
        retry_ts = max(0.0, timestamp_seconds - 0.1)
        ok2, stderr2 = _run_ffmpeg(retry_ts, out_path)
        if not ok2 or not out_path.exists() or out_path.stat().st_size == 0:
            try:
                out_path.unlink(missing_ok=True)
            finally:
                raise RuntimeError(
                    f"ffmpeg failed to extract frame at {timestamp_seconds}s "
                    f"(retry at {retry_ts}s) for {video_path}\n"
                    f"ffmpeg stderr:\n{stderr2}"
                )

        if not ok:
            print(
                f"Warning: ffmpeg returned exit code on first attempt at "
                f"{timestamp_seconds}s but succeeded on retry at {retry_ts}s.\n"
                f"First attempt stderr:\n{stderr}",
                file=__import__("sys").stderr,
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
    subprocess.run(cmd, **_FFMPEG_RUN_KWARGS)
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
        **_FFMPEG_RUN_KWARGS,
    )
    return sorted(out_dir.glob("frame_*.png"))
