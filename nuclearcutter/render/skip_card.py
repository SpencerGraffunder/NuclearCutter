"""
Generates the black-background text card used for `skip` actions
(docs/SPEC.md section 4.3). Card duration scales with how long the
description takes to read (~200-250 wpm), not a fixed length — a short
visual-only beat renders briefly, a dialogue-heavy scene gets more time.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

WORDS_PER_MINUTE = 220
MIN_CARD_SECONDS = 2.0
MAX_CARD_SECONDS = 15.0
# Small buffer added on top of pure reading-speed math so the card doesn't
# feel like it's yanked away the instant an average reader finishes.
READING_BUFFER_SECONDS = 1.0


def card_duration_for_text(text: str) -> float:
    word_count = max(len(text.split()), 1)
    reading_seconds = (word_count / WORDS_PER_MINUTE) * 60.0
    duration = reading_seconds + READING_BUFFER_SECONDS
    return min(max(duration, MIN_CARD_SECONDS), MAX_CARD_SECONDS)


def render_skip_card_clip(
    text: str,
    duration: float,
    width: int,
    height: int,
    fps: float,
    out_path: Path,
    font_path: str = None,
) -> Path:
    """Render a black background video clip with the given text centered, for `duration` seconds.

    Produces a silent video-only clip (audio is handled separately by the
    caller, since skip actions mute audio for the segment). Uses ffmpeg's
    drawtext filter with automatic wrapping via a fixed font size chosen to
    fit typical description lengths.
    """
    wrapped_text = _wrap_for_display(text)
    escaped_text = _escape_for_drawtext(wrapped_text)

    font_size = 42
    fontfile_arg = f":fontfile='{font_path}'" if font_path else ""

    drawtext = (
        f"drawtext=text='{escaped_text}'{fontfile_arg}:"
        f"fontcolor=white:fontsize={font_size}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2:"
        f"line_spacing=12"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration}",
            "-vf", drawtext,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        capture_output=True, check=True,
    )
    return out_path


def _wrap_for_display(text: str, max_chars_per_line: int = 50) -> str:
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
    # ffmpeg drawtext requires escaping these characters within the text arg.
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\u2019")  # sidestep quote-escaping entirely, use a typographic apostrophe
    text = text.replace("\n", "\r")  # drawtext uses \r as its line-break token in the filter string
    return text
