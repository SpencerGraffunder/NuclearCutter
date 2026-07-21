"""
Pass 2 of the two-pass architecture (docs/SPEC.md section 2): reads a
ScanResult + Preferences, and produces the final `_cleaned` output file.

Strategy: split the source into segments at every action boundary, and
build BOTH the video and audio timelines segment-by-segment in lockstep,
then concat each track and mux them together. Blur segments preserve the
original timing while applying an intense blur and overlaying a short
summary of the scene. Mute segments silence audio only.

Untouched segments are re-encoded (not stream-copied) to keep the concat
step reliable across arbitrary cut points — see the note in
_render_track_segments. This trades a bit of extra encode time for
correctness; a stream-copy fast path for untouched segments is a
reasonable future optimization (see README known-limitations).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nuclearcutter.schema import Action, Category, Preferences, ScanResult
from nuclearcutter.utils.ffmpeg import probe_streams

# Discard ffmpeg stderr (progress lines) to avoid memory accumulation.
_FFMPEG_KW = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, "check": True}


@dataclass
class TimelineSegment:
    """One span of the ORIGINAL source timeline, and what to do with it.

    For blur and mute segments, output duration equals the source duration.
    """
    start: float
    end: float
    action: Action | None  # None = untouched passthrough
    category: Category | None = None
    description: str = ""
    audio_muted: bool = False  # for blur segments with blur_mute_audio enabled

    @property
    def source_duration(self) -> float:
        return self.end - self.start


def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")


def plan_timeline(scan: ScanResult, prefs: Preferences, duration: float) -> list[TimelineSegment]:
    """Build the full ordered, non-overlapping timeline covering the whole film.

    Visual detections (nudity/intimate_scenes) drive segment boundaries.
    Foul-language detections never create a new segment boundary on their
    own (video is untouched for pure language mutes) — they're applied as
    an audio-only mute layered on top of whichever segment(s) they fall
    within, via _language_mute_ranges_within.
    """
    active = [
        d for d in scan.visual_detections
        if prefs.action_for(d.category) == Action.BLUR
    ]
    active.sort(key=lambda d: d.start)

    segments: list[TimelineSegment] = []
    cursor = 0.0
    for d in active:
        start = max(d.start, cursor)
        if start >= d.end:
            continue  # fully overlapped by a previous (already-planned) detection
        if start > cursor:
            segments.append(TimelineSegment(start=cursor, end=start, action=None))

        action = prefs.action_for(d.category)
        audio_muted = False
        if action == Action.BLUR:
            audio_muted = (
                prefs.nudity_blur_mute_audio if d.category == Category.NUDITY
                else prefs.intimate_scenes_blur_mute_audio
            )

        segments.append(TimelineSegment(
            start=start, end=d.end, action=action, category=d.category,
            description=d.description, audio_muted=audio_muted,
        ))
        cursor = d.end

    if cursor < duration:
        segments.append(TimelineSegment(start=cursor, end=duration, action=None))

    return segments


def render(
    input_path: Path,
    scan: ScanResult,
    prefs: Preferences,
    output_path: Path = None,
    font_path: str = None,
) -> Path:
    output_path = output_path or build_output_path(input_path)
    stream_info = probe_streams(input_path)
    video_stream = next(s for s in stream_info["streams"] if s["codec_type"] == "video")
    width, height = int(video_stream["width"]), int(video_stream["height"])
    fps = _parse_fps(video_stream.get("r_frame_rate", "24/1"))
    duration = float(stream_info["format"]["duration"])

    timeline = plan_timeline(scan, prefs, duration)

    with tempfile.TemporaryDirectory(prefix="cleancut_render_") as tmp:
        tmp_dir = Path(tmp)

        video_files, audio_files = _render_track_segments(
            input_path, timeline, scan, prefs, width, height, fps, tmp_dir, font_path,
        )

        final_video = _concat_track(video_files, tmp_dir, "video_concat.mp4")
        final_audio = _concat_track(audio_files, tmp_dir, "audio_concat.m4a")
        _mux_final_output(final_video, final_audio, output_path)

    return output_path


_FFMPEG_FILTERS: set[str] | None = None


def _parse_fps(r_frame_rate: str) -> float:
    num, denom = r_frame_rate.split("/")
    return float(num) / float(denom)


def _ffmpeg_has_filter(filter_name: str) -> bool:
    global _FFMPEG_FILTERS
    if _FFMPEG_FILTERS is None:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        filters = set()
        for line in proc.stdout.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            filters.add(parts[-1])
        _FFMPEG_FILTERS = filters
    return filter_name in _FFMPEG_FILTERS


def _build_blur_filter(text: str, font_path: str | None) -> str:
    filters = ["boxblur=luma_radius=30:luma_power=3:chroma_radius=30:chroma_power=3"]
    if text.strip() and _ffmpeg_has_filter("drawtext"):
        wrapped_text = _wrap_for_display(text)
        escaped_text = _escape_for_drawtext(wrapped_text)
        fontfile_arg = f":fontfile='{font_path}'" if font_path else ""
        drawtext = (
            f"drawtext=text='{escaped_text}'{fontfile_arg}:"
            "fontcolor=white:fontsize=36:"
            "box=1:boxcolor=black@0.6:boxborderw=10:"
            "x=(w-text_w)/2:y=h-(text_h)-40:"
            "line_spacing=10"
        )
        filters.append(drawtext)
    return ",".join(filters)


def _render_track_segments(
    input_path: Path,
    timeline: list[TimelineSegment],
    scan: ScanResult,
    prefs: Preferences,
    width: int,
    height: int,
    fps: float,
    tmp_dir: Path,
    font_path: str,
) -> tuple[list[Path], list[Path]]:
    video_files: list[Path] = []
    audio_files: list[Path] = []

    for i, seg in enumerate(timeline):
        v_out = tmp_dir / f"v_{i:04d}.mp4"
        a_out = tmp_dir / f"a_{i:04d}.m4a"

        if seg.action is None:
            output_duration = seg.source_duration
            _extract_video_segment(input_path, seg.start, output_duration, v_out, video_filter=None)
            _extract_audio_segment(
                input_path, seg.start, output_duration, a_out,
                mute_ranges=_language_mute_ranges_within(scan, prefs, seg.start, seg.end),
            )

        elif seg.action == Action.BLUR:
            output_duration = seg.source_duration
            blur_filter = _build_blur_filter(seg.description, font_path)
            _extract_video_segment(input_path, seg.start, output_duration, v_out, video_filter=blur_filter)
            if seg.audio_muted:
                _silent_audio(output_duration, a_out)
            else:
                _extract_audio_segment(
                    input_path, seg.start, output_duration, a_out,
                    mute_ranges=_language_mute_ranges_within(scan, prefs, seg.start, seg.end),
                )

        video_files.append(v_out)
        audio_files.append(a_out)

    return video_files, audio_files


def _language_mute_ranges_within(scan: ScanResult, prefs: Preferences, seg_start: float, seg_end: float) -> list[tuple[float, float]]:
    """Foul-language mute windows that fall within [seg_start, seg_end), converted to
    times relative to the segment's own start (0-based) for use in a per-segment audio filter."""
    if prefs.foul_language_action != Action.MUTE:
        return []

    ranges = []
    for d in scan.language_detections:
        if not d.llm_confirmed:
            continue
        if prefs.foul_language_mute_scope == "utterance":
            m_start, m_end = d.utterance_start, d.utterance_end
        else:
            m_start, m_end = d.start, d.end

        overlap_start = max(m_start, seg_start)
        overlap_end = min(m_end, seg_end)
        if overlap_start < overlap_end:
            ranges.append((overlap_start - seg_start, overlap_end - seg_start))
    return ranges


def _wrap_for_display(text: str, max_chars_per_line: int = 40) -> str:
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > max_chars_per_line and current:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _escape_for_drawtext(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\u2019")
    text = text.replace("\n", "\r")
    return text


def _extract_video_segment(input_path: Path, start: float, duration: float, out_path: Path, video_filter: str | None) -> None:
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(input_path), "-t", str(duration)]
    if video_filter:
        cmd += ["-vf", video_filter]
    cmd += ["-an", "-c:v", "libx264", "-crf", "17", "-preset", "medium", str(out_path)]
    subprocess.run(cmd, **_FFMPEG_KW)


def _extract_audio_segment(input_path: Path, start: float, duration: float, out_path: Path, mute_ranges: list[tuple[float, float]]) -> None:
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(input_path), "-t", str(duration), "-vn"]
    if mute_ranges:
        clauses = "+".join(f"between(t,{r0},{r1})" for r0, r1 in mute_ranges)
        cmd += ["-af", f"volume=0:enable='{clauses}'"]
    cmd += ["-c:a", "aac", "-b:a", "192k", str(out_path)]
    subprocess.run(cmd, **_FFMPEG_KW)


def _silent_audio(duration: float, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", str(duration),
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    )


def _concat_track(files: list[Path], tmp_dir: Path, out_name: str) -> Path:
    concat_list = tmp_dir / f"concat_{out_name}.txt"
    concat_list.write_text("\n".join(f"file '{f}'" for f in files))
    out_path = tmp_dir / out_name
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy",
            str(out_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    )
    return out_path


def _mux_final_output(video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy",
            str(output_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    )
