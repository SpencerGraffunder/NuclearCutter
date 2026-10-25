"""
Thin wrappers around ffmpeg/ffprobe. All video/audio manipulation in
NuclearCutter goes through here so there's one place that knows how to talk to
ffmpeg.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

# Shared kwargs for ffmpeg/ffprobe subprocess calls.
# We capture both stdout and stderr (necessary for error diagnostics), but
# each call is synchronous and the result goes out of scope immediately, so
# this does NOT accumulate memory across repeated calls.
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


def extract_frame_at(video_path: Path, timestamp_seconds: float, scale_height: int | None = None) -> Path:
    """Extract a single frame at the given timestamp to a temp PNG file.

    `scale_height`, if given, downscales the frame to that output height
    (width auto to preserve aspect ratio). This dramatically cuts VLM vision
    tokens — e.g. 1920x818 -> 480p is ~3.6x fewer tokens per frame — while
    keeping enough detail for coarse flagging.

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
        vf = f"scale=-2:{scale_height}" if scale_height else None
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-frames:v", "1",
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-q:v", "2", str(path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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


def extract_frames_chunk(
    video_path: Path,
    timestamps: list[float],
    out_dir: Path,
    scale_height: int | None = None,
    prefix: str = "chunk_%05d.jpg",
) -> dict[float, Path]:
    """Extract a run of evenly-spaced frames with ONE ffmpeg decode pass.

    `timestamps` must be sorted and evenly spaced (the sweep's samples).
    Instead of one seek+decode per frame — which re-pays keyframe seeking and
    codec startup for EVERY sample (~1-4s each on 1080p HEVC, the dominant
    cost of a full-film sweep, ~3x the VLM call itself) — a single
    `ffmpeg -ss <first> -vf fps=1/step` pass seeks once and decodes forward,
    emitting one frame per step. Frames are written as JPEG (the VLM request
    re-encodes JPEG anyway, so PNG encode+decode per sample is pure waste).

    Returns {timestamp: Path} for the frames ffmpeg actually produced, in
    order — if it stops early (end of file), the trailing timestamps are
    simply absent and the caller can fall back to per-frame extraction for
    those. Never raises on an ffmpeg error: an empty dict means "use the
    per-frame path", which carries the retry/diagnostics.
    """
    ts = sorted(timestamps)
    if not ts:
        return {}
    step = ts[1] - ts[0] if len(ts) > 1 else 1.0
    if step <= 0:
        raise ValueError("timestamps must be strictly increasing")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={1.0 / step:.6f}" + (f",scale=-2:{scale_height}" if scale_height else "")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{ts[0]:.3f}",
            "-i", str(video_path),
            "-frames:v", str(len(ts)),
            "-vf", vf,
            "-q:v", "3",
            str(out_dir / prefix),
        ],
        capture_output=True, text=True, check=False,
    )
    frames = sorted(out_dir.glob(prefix.replace("%05d", "*")))
    if not frames and result.returncode != 0:
        # Total failure (corrupt range, codec error, tail past EOF) — the
        # per-frame fallback path retries and raises with full stderr.
        return {}
    return dict(zip(ts, frames))


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


def extract_frames_in_range(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    out_dir: Path,
    scale_height: int | None = None,
) -> list[Path]:
    """Extract frames from [start, end] at the given fps into out_dir. Returns sorted list of frame paths.

    `scale_height`, if given, downscales frames to that output height (width
    auto, aspect preserved) to cut VLM vision tokens.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.png"
    duration = end - start
    vf = f"fps={fps}" + (f",scale=-2:{scale_height}" if scale_height else "")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", vf,
            "-q:v", "2",
            str(pattern),
        ],
        **_FFMPEG_RUN_KWARGS,
    )
    return sorted(out_dir.glob("frame_*.png"))


def extract_frame_jpeg_bytes(
    video_path: Path, timestamp_seconds: float, scale_height: int | None = None,
) -> bytes:
    """Extract one frame at `timestamp_seconds` and return it as JPEG bytes.

    Unlike `extract_frame_at` (which writes a temp PNG the caller must clean
    up), this streams straight to an in-memory JPEG — the right shape for
    serving a frame over HTTP. `scale_height` downscales to that output height
    (width auto, aspect preserved), e.g. 800px tall for a postage-stamp
    preview. On ffmpeg failure (e.g. timestamp past the end of the file) it
    retries once 0.1s earlier, then raises RuntimeError.
    """
    out = Path(tempfile.mktemp(suffix=".jpg"))

    def _run(ts: float) -> tuple[bool, str]:
        cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path), "-frames:v", "1"]
        if scale_height:
            cmd += ["-vf", f"scale=-2:{scale_height}"]
        cmd += ["-q:v", "2", str(out)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, result.stderr or ""
        except subprocess.CalledProcessError as e:
            return False, e.stderr or "(no stderr)"

    ok, stderr = _run(timestamp_seconds)
    if not ok or not out.exists() or out.stat().st_size == 0:
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
        retry_ts = max(0.0, timestamp_seconds - 0.1)
        ok2, stderr2 = _run(retry_ts)
        if not ok2 or not out.exists() or out.stat().st_size == 0:
            with contextlib.suppress(OSError):
                out.unlink(missing_ok=True)
            raise RuntimeError(
                f"ffmpeg failed to extract frame at {timestamp_seconds}s "
                f"(retry at {retry_ts}s) for {video_path}\n"
                f"ffmpeg stderr:\n{stderr2}"
            )
    try:
        return out.read_bytes()
    finally:
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
