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

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nuclearcutter.schema import AudioAction, Category, Preferences, ScanResult, VisualAction
from nuclearcutter.utils.ffmpeg import probe_streams

# Discard ffmpeg stderr (progress lines) to avoid memory accumulation.
_FFMPEG_KW = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, "check": True}


@dataclass
class TimelineSegment:
    """One span of the ORIGINAL source timeline, and what to do with it.

    Each segment carries a VISUAL action (what to do to the video: none/blur/
    black) and an AUDIO action (what to do to the audio: none/mute/replace).
    For blur/black/mute segments, output duration equals the source duration.
    """
    start: float
    end: float
    visual: VisualAction  # what to do to the VIDEO
    audio: AudioAction  # what to do to the AUDIO (usually for language)
    category: Category | None = None
    description: str = ""  # clean summary shown as text overlay for blur/black

    @property
    def source_duration(self) -> float:
        return self.end - self.start


# Video container extensions — used to place "_cleaned" BEFORE the real video
# extension, even when the file has a second suffix (e.g. "Movie.mkv.iso").
VIDEO_EXTS = {
    ".mkv", ".mp4", ".m4v", ".avi", ".mov", ".wmv", ".webm", ".ts", ".m2ts",
    ".mpg", ".mpeg", ".vob", ".flv", ".ogv",
}


def full_video_suffix(path: Path) -> str:
    """The complete extension chain of a video file.

    "Movie.mkv" -> ".mkv"; "Movie.mkv.iso" -> ".mkv.iso". Python's Path only
    treats the LAST suffix as the extension, so for multi-suffix files we
    re-attach the inner video extension to keep the real format in the name.
    """
    stem = path.stem
    inner = Path(stem).suffix
    if inner and inner.lower() in VIDEO_EXTS:
        return inner + path.suffix
    return path.suffix


def build_output_path(input_path: Path) -> Path:
    """Movie.mkv -> Movie_cleaned.mkv; Movie.mkv.iso -> Movie_cleaned.mkv.iso.

    "_cleaned" always goes BEFORE the real video extension (the inner one, for
    multi-suffix files like an mkv named with an .iso extension), never after.
    """
    ext = full_video_suffix(input_path)
    stem = input_path.stem
    if ext.startswith(".") and Path(stem).suffix and Path(stem).suffix.lower() in VIDEO_EXTS:
        stem = Path(stem).stem  # strip the inner video ext; it's part of `ext`
    return input_path.with_name(f"{stem}_cleaned{ext}")


def plan_timeline(scan: ScanResult, prefs: Preferences, duration: float) -> list[TimelineSegment]:
    """Build the full ordered, non-overlapping timeline covering the whole film.

    Visual detections (nudity/gore/violence) drive segment boundaries whenever
    their category's VISUAL action is not "none" AND the detection's severity
    level meets the user's threshold for that category (prefs.level_for). So a
    user who sets `nudity_level = "high"` leaves low/med nudity untouched.

    Foul-language detections never create a new segment boundary on their own
    (video is untouched for pure language mutes) — they're applied as an
    audio-only mute layered on top of whichever segment(s) they fall within,
    via _language_mute_ranges_within.

    Exception: if the user sets foul_language_visual to a non-"none" action,
    confirmed language detections DO drive visual segments (e.g. blur the video
    while someone is swearing), over the whole utterance window — still subject
    to the foul_language_level threshold.
    """
    events: list[tuple[float, float, Category, VisualAction, AudioAction, str]] = []

    # Extra seconds blurred/blacked on each side of a visual segment.
    blur_pad = max(0.0, float(getattr(prefs, "blur_padding", 0.0)))

    for d in scan.visual_detections:
        va = prefs.visual_for(d.category)
        if va != VisualAction.NONE and d.level.is_corrected_by(prefs.level_for(d.category)):
            events.append((
                max(0.0, d.start - blur_pad),
                min(duration, d.end + blur_pad), d.category, va,
                prefs.audio_for(d.category), d.description,
            ))

    lang_va = prefs.visual_for(Category.FOUL_LANGUAGE)
    lang_threshold = prefs.level_for(Category.FOUL_LANGUAGE)
    if lang_va != VisualAction.NONE:
        for d in scan.language_detections:
            if not d.llm_confirmed or not d.level.is_corrected_by(lang_threshold):
                continue
            events.append((
                max(0.0, d.utterance_start - blur_pad),
                min(duration, d.utterance_end + blur_pad), Category.FOUL_LANGUAGE,
                lang_va, prefs.audio_for(Category.FOUL_LANGUAGE),
                f'Speaker says "{d.word}".',
            ))

    events.sort(key=lambda e: e[0])

    segments: list[TimelineSegment] = []
    cursor = 0.0
    for start, end, cat, va, aa, desc in events:
        start = max(start, cursor)
        if start >= end:
            continue  # fully overlapped by a previous (already-planned) detection
        if start > cursor:
            segments.append(TimelineSegment(start=cursor, end=start,
                                            visual=VisualAction.NONE, audio=AudioAction.NONE))
        segments.append(TimelineSegment(
            start=start, end=end,
            visual=va,
            audio=aa,
            category=cat,
            description=desc,
        ))
        cursor = end

    if cursor < duration:
        segments.append(TimelineSegment(start=cursor, end=duration,
                                        visual=VisualAction.NONE, audio=AudioAction.NONE))

    return segments


class RenderStopped(RuntimeError):
    """Raised when the user hits Stop during a render. No resume is supported
    for a half-done render; the partial output is discarded by the caller."""


def render(
    input_path: Path,
    scan: ScanResult,
    prefs: Preferences,
    output_path: Path = None,
    font_path: str = None,
    status_path: Path | str = None,
    progress_callback=None,
    stop_event=None,
    workers: int | None = None,
    encode_threads: int | None = None,
) -> Path:
    output_path = output_path or build_output_path(input_path)
    stream_info = probe_streams(input_path)
    video_stream = next(s for s in stream_info["streams"] if s["codec_type"] == "video")
    width, height = int(video_stream["width"]), int(video_stream["height"])
    fps = _parse_fps(video_stream.get("r_frame_rate", "24/1"))
    duration = float(stream_info["format"]["duration"])
    codec_name = video_stream.get("codec_name", "h264")
    # The source's CONTAINER format. Some files have unusual extensions (e.g.
    # "Movie.mkv.iso" holding matroska content) — ffmpeg can't guess a muxer
    # from the output extension, so the final mux must name the format
    # explicitly or it fails with "Unable to choose an output format".
    mux_format = str(stream_info.get("format", {}).get("format_name", "")).split(",")[0].strip()

    # Preserve the source codec family instead of forcing libx264. This keeps
    # x265/HEVC sources at x265-sized files instead of ballooning them into
    # much larger H.264 output (a known complaint with re-encoding).
    video_encoder = _encoder_for_codec(codec_name)

    timeline = plan_timeline(scan, prefs, duration)

    # Optional live status for the web GUI (see utils/scan_status.py).
    status = None
    if status_path:
        from nuclearcutter.utils.scan_status import ScanStatus, _now_iso

        status = ScanStatus(
            video=Path(input_path).name,
            duration_seconds=duration,
            sweep_interval=None,
            pid=os.getpid(),
            started_at=_now_iso(),
            phase="render",
            frames_total=len(timeline),
        )
        # Seed the timeline markers with the detections that will be applied.
        for d in scan.visual_detections:
            status.add_visual_detection(d.category, d.start, d.end, d.description or "", d.confidence)
        for d in scan.language_detections:
            if d.llm_confirmed:
                status.add_language_detection(d.word, d.start, d.end, d.utterance_start, d.utterance_end)
        status_path = Path(status_path)

    def _write_status():
        if status is not None:
            try:
                status.write(status_path)
            except Exception as exc:
                print(f"warning: status write failed: {exc}", file=sys.stderr)

    def _progress(phase: str, detail=None):
        if progress_callback:
            try:
                progress_callback(phase, detail)
            except Exception:
                pass  # never let a UI callback crash the render

    def _on_segment(i, total, seg_start):
        if status is not None:
            status.set_phase("render")
            status.frames_done = i
            status.position_seconds = seg_start
            _write_status()
        _progress("render", (i, total, seg_start))

    # Parallel segment encoding: run several ffmpeg processes at once, each
    # limited to a share of the cores (instead of one encode hogging all of
    # them while the rest of the film waits).
    cpus = os.cpu_count() or 4
    if workers is None:
        workers = max(1, min(cpus // 2, 4))
    if workers > 1 and encode_threads is None:
        encode_threads = max(1, cpus // workers)
    elif workers <= 1:
        encode_threads = None

    with tempfile.TemporaryDirectory(prefix="cleancut_render_") as tmp:
        tmp_dir = Path(tmp)

        video_files, audio_files = _render_track_segments(
            input_path, timeline, scan, prefs, width, height, fps, tmp_dir, font_path, video_encoder,
            on_segment=_on_segment,
            stop_event=stop_event,
            workers=workers,
            encode_threads=encode_threads,
        )
        if status is not None:
            status.set_phase("concat"); _write_status()
        _progress("concat", None)
        final_video = _concat_track(video_files, tmp_dir, "video_concat.mp4")
        final_audio = _concat_track(audio_files, tmp_dir, "audio_concat.m4a")
        if status is not None:
            status.set_phase("mux"); _write_status()
        _progress("mux", None)
        _mux_final_output(final_video, final_audio, output_path, mux_format=mux_format or None)
        if status is not None:
            status.set_phase("done")
            status.frames_done = len(timeline)
            _write_status()
        _progress("done", (len(timeline), len(timeline), duration))

    return output_path


def _encoder_for_codec(codec_name: str) -> str:
    """Map a source video codec to the ffmpeg encoder that preserves its family.

    This keeps re-encoded output sizes roughly in line with the source (x265 stays
    x265, AV1 stays AV1) rather than always falling back to H.264, which is much
    larger for the same quality.
    """
    family = {
        "h264": "libx264",
        "hevc": "libx265",
        "h265": "libx265",
        "av1": "libsvtav1",
        "vp9": "libvpx-vp9",
        "vp8": "libvpx",
    }
    return family.get(codec_name, "libx264")


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
            # ffmpeg `-filters` lines look like: " T. boxblur  V->V  Blur..."
            # parts[0] is the flag column (e.g. "T."), parts[1] is the filter name.
            if len(parts) >= 2 and not parts[1].startswith("="):
                filters.add(parts[1])
        _FFMPEG_FILTERS = filters
    return filter_name in _FFMPEG_FILTERS


def _build_blur_filter(text: str, font_path: str | None, strength: float = 1.0,
                       width: int = 1920, height: int = 1080) -> str:
    """Return the -vf filter chain for a blur segment.

    We always apply the intense boxblur. The summary-text overlay is NOT drawn
    here: this ffmpeg build lacks the drawtext filter (no freetype), so text is
    composited separately via a PIL-generated PNG + the overlay filter (see
    _make_overlay_png / _extract_video_segment's overlay_path).

    `strength` scales the blur intensity: boxblur's luma_radius (how far each
    pass spreads) and luma_power (how many passes) are both multiplied. 1.0 =
    the standard radius 30 / 3 passes; 2.0 = twice as extreme (radius 60 /
    6 passes). The radius is clamped so it never exceeds the frame plane size
    (boxblur's hard limit is min(luma_w/2, luma_h/2) — exceeding it makes
    ffmpeg fail outright).
    """
    strength = max(0.1, float(strength))
    radius = int(round(30 * strength))
    power = max(1, int(round(3 * strength)))
    radius = min(radius, max(1, min(width, height) // 2))
    filters = [
        f"boxblur=luma_radius={radius}:luma_power={power}:"
        f"chroma_radius={radius}:chroma_power={power}"
    ]
    return ",".join(filters)


def _make_overlay_png(text: str, width: int, height: int, font_path: str | None, out_path: Path) -> None:
    """Render the summary text onto a transparent full-frame PNG.

    The text sits in a semi-opaque black box near the bottom-center, matching
    the look of the old drawtext overlay. This works with ffmpeg's universal
    `overlay` filter, so it doesn't depend on a drawtext-capable build.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    wrapped = _wrap_for_display(text, max_chars_per_line=45)

    font = None
    candidates = [font_path] if font_path else []
    candidates += [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        if not path:
            continue
        try:
            font = ImageFont.truetype(path, 32)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Measure the wrapped text so the black box hugs it.
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=8)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = 24, 16
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    x0 = (width - box_w) // 2
    y0 = height - box_h - 48

    draw.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        radius=12,
        fill=(0, 0, 0, 178),
    )
    draw.multiline_text(
        (x0 + pad_x, y0 + pad_y),
        wrapped,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=8,
    )
    img.save(str(out_path))


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
    video_encoder: str,
    on_segment=None,
    stop_event=None,
    workers: int = 2,
    encode_threads: int | None = None,
) -> tuple[list[Path], list[Path]]:
    """Render every segment's video + audio, running up to `workers` segment
    encodes in PARALLEL (each ffmpeg process pinned to `encode_threads`).

    On multi-core machines this is the biggest render speedup: instead of one
    encode using all cores while the others wait, several short segments encode
    at once. `on_segment(done, total, seg_start)` fires as each segment
    COMPLETES (done is monotonic). Results are reassembled in timeline order.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(timeline)
    results: dict[int, tuple[Path, Path]] = {}
    starts = [seg.start for seg in timeline]
    done = 0
    lock = threading.Lock()

    def _work(i: int, seg: TimelineSegment):
        v_out = tmp_dir / f"v_{i:04d}.mp4"
        a_out = tmp_dir / f"a_{i:04d}.m4a"
        output_duration = seg.source_duration

        if seg.visual == VisualAction.NONE:
            _extract_video_segment(input_path, seg.start, output_duration, v_out,
                                   video_encoder, video_filter=None, fps=fps,
                                   threads=encode_threads)
        elif seg.visual in (VisualAction.BLUR, VisualAction.BLACK):
            overlay_path: Path | None = None
            if seg.description.strip() and _ffmpeg_has_filter("overlay"):
                overlay_path = tmp_dir / f"overlay_{i:04d}.png"
                _make_overlay_png(seg.description, width, height, font_path, overlay_path)
            if seg.visual == VisualAction.BLUR:
                video_filter = _build_blur_filter(seg.description, font_path,
                                                  strength=getattr(prefs, "blur_strength", 1.0),
                                                  width=width, height=height)
                _extract_video_segment(
                    input_path, seg.start, output_duration, v_out, video_encoder,
                    video_filter=video_filter, fps=fps, overlay_path=overlay_path,
                    threads=encode_threads,
                )
            else:  # BLACK: solid black frames + text overlay
                _extract_black_segment(output_duration, width, height, fps, v_out,
                                       video_encoder, overlay_path,
                                       threads=encode_threads)

        # ---- Audio correction ----
        # Visual categories (nudity/gore/violence): the only muting option is
        # MUTE_SCENE, which silences the WHOLE flagged segment (no per-sound
        # recognition). Foul-language windows layer on top of any segment
        # otherwise (word/phrase granularity from whisper timestamps).
        if (seg.audio.mutes_audio
                and seg.category is not None
                and seg.category != Category.FOUL_LANGUAGE):
            _silent_audio(output_duration, a_out)
        else:
            mute_ranges = _language_mute_ranges_within(scan, prefs, seg)
            if mute_ranges:
                _extract_audio_segment(input_path, seg.start, output_duration, a_out, mute_ranges=mute_ranges)
            else:
                _extract_audio_segment(input_path, seg.start, output_duration, a_out, mute_ranges=None)

        return i, v_out, a_out

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {}
        for i, seg in enumerate(timeline):
            if stop_event is not None and stop_event.is_set():
                raise RenderStopped("render stopped by user")
            futures[pool.submit(_work, i, seg)] = i
        for fut in as_completed(futures):
            i, v_out, a_out = fut.result()
            with lock:
                done += 1
            results[i] = (v_out, a_out)
            if on_segment:
                on_segment(done, total, starts[i])

    video_files = [results[i][0] for i in range(total)]
    audio_files = [results[i][1] for i in range(total)]
    return video_files, audio_files


def _language_mute_ranges_within(scan: ScanResult, prefs: Preferences, seg: TimelineSegment) -> list[tuple[float, float]]:
    """Foul-language mute windows that fall within this segment, as times relative
    to the segment's own start (0-based).

    Word vs phrase scope is taken from `prefs.foul_language_audio`. Each window is
    padded by `prefs.mute_padding` on both sides so word onset/offset audio doesn't
    leak through (whisper word timestamps can be tight). Clamped to the segment.
    """
    lang_action = prefs.audio_for(Category.FOUL_LANGUAGE)
    if not lang_action.mutes_audio:
        return []
    if lang_action in (AudioAction.REPLACE_WORD, AudioAction.REPLACE_PHRASE):
        print("warning: replace_* not implemented yet — muting instead", file=sys.stderr)

    pad = max(0.0, float(getattr(prefs, "mute_padding", 0.5)))
    phrase_scope = lang_action in (AudioAction.MUTE_PHRASE, AudioAction.REPLACE_PHRASE)
    threshold = prefs.level_for(Category.FOUL_LANGUAGE)
    ranges: list[tuple[float, float]] = []
    for d in scan.language_detections:
        if not d.llm_confirmed or not d.level.is_corrected_by(threshold):
            continue
        if phrase_scope:
            m_start, m_end = d.utterance_start, d.utterance_end
        else:
            m_start, m_end = d.start, d.end
        ranges.extend(_clamped_window(m_start, m_end, pad, seg.start, seg.end, seg.start))
    return ranges


def _clamped_window(start: float, end: float, pad: float, seg_start: float, seg_end: float, base: float) -> list[tuple[float, float]]:
    """Return the padded [start,end) window clamped to [seg_start,seg_end), relative to `base`."""
    overlap_start = max(start - pad, seg_start)
    overlap_end = min(end + pad, seg_end)
    if overlap_start < overlap_end:
        return [(overlap_start - base, overlap_end - base)]
    return []


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


def _extract_video_segment(
    input_path: Path, start: float, duration: float, out_path: Path,
    video_encoder: str, video_filter: str | None, fps: float,
    overlay_path: Path | None = None, threads: int | None = None,
) -> None:
    cmd = ["ffmpeg", "-y", "-fflags", "+genpts", "-ss", str(start), "-i", str(input_path)]
    if overlay_path is not None:
        # Composite a pre-rendered text overlay (PIL PNG) on top of the blurred
        # video. -loop 1 turns the still PNG into a stream; -t is given as an
        # OUTPUT option (after all inputs) so the encode stops at `duration`
        # regardless of the never-ending looped image input.
        cmd += ["-loop", "1", "-i", str(overlay_path)]
        fc = (
            f"[0:v]{video_filter}[bg];"
            "[1:v]format=rgba,scale=iw:ih[ovr];"
            "[bg][ovr]overlay=0:0:format=auto[v]"
        )
        cmd += ["-filter_complex", fc, "-map", "[v]"]
    elif video_filter:
        cmd += ["-vf", video_filter]
    # Encode at a fixed frame rate and a shared, whole-number timebase so every
    # segment lands on the same timestamp grid. Without this, independently
    # encoded segments have slightly different timebases/timings, and the concat
    # produces timestamp discontinuities at each boundary — which VLC reads as a
    # "new file" (title + black flash). -r + -vsync cfr force constant frames;
    # -video_track_timescale pins the MP4 timebase so concat joins cleanly.
    cmd += [
        "-t", str(duration), "-an",
        "-r", f"{fps:.6f}", "-vsync", "cfr",
        "-video_track_timescale", "15360",
    ]
    if threads is not None:
        # When several segments encode in parallel, each gets a share of the
        # cores instead of all of them fighting for every core.
        cmd += ["-threads", str(threads)]
    cmd += ["-c:v", video_encoder, "-crf", "17", "-preset", "medium"]
    if video_encoder == "libx265":
        cmd += ["-tag:v", "hvc1"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, **_FFMPEG_KW)


def _extract_black_segment(
    duration: float, width: int, height: int, fps: float, out_path: Path,
    video_encoder: str, overlay_path: Path | None = None,
    threads: int | None = None,
) -> None:
    """Encode a segment of solid BLACK video (optionally with a text overlay).

    Used for the `black` visual action: replaces the flagged footage entirely
    with a black screen, so the viewer sees nothing but the clean summary text.
    Uses the same fixed-framerate/timescale discipline as _extract_video_segment
    so it concatenates cleanly with the other segments.
    """
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps:.6f}"]
    if overlay_path is not None:
        cmd += ["-loop", "1", "-i", str(overlay_path)]
        fc = (
            "[0:v]format=rgba[bg];"
            "[1:v]format=rgba,scale=iw:ih[ovr];"
            "[bg][ovr]overlay=0:0:format=auto[v]"
        )
        cmd += ["-filter_complex", fc, "-map", "[v]"]
    cmd += [
        "-t", str(duration), "-an",
        "-r", f"{fps:.6f}", "-vsync", "cfr",
        "-video_track_timescale", "15360",
    ]
    if threads is not None:
        cmd += ["-threads", str(threads)]
    cmd += ["-c:v", video_encoder, "-crf", "17", "-preset", "medium"]
    if video_encoder == "libx265":
        cmd += ["-tag:v", "hvc1"]
    cmd += ["-movflags", "+faststart", str(out_path)]
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
            "-fflags", "+genpts",
            "-c", "copy",
            str(out_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    )
    return out_path


def _mux_final_output(video_path: Path, audio_path: Path, output_path: Path,
                      mux_format: str | None = None) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy",
        # Normalize timestamps so there are no gaps/jumps across the concat
        # boundaries. `-avoid_negative_ts make_zero` shifts the earliest
        # timestamp to zero, and `+faststart` moves the moov atom to the
        # front so players open immediately. Together these stop VLC from
        # re-detecting the stream (title + black flash) at each blur cut.
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
    ]
    if mux_format:
        # Explicit muxer: the output extension may not reveal the container
        # (e.g. an .iso-named file holding matroska content), and extension
        # guessing fails with "Unable to choose an output format".
        cmd += ["-f", mux_format]
    cmd += [str(output_path)]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
