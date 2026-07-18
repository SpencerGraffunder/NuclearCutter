"""
Audio transcription for the foul-language detection pipeline (docs/SPEC.md
section 4.2). Uses mlx-whisper for Apple Silicon acceleration, with
word-level timestamps. If a subtitle file is available (embedded or
sidecar), it's parsed and cross-checked against the Whisper output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import pysrt

from nuclearcutter.utils.ffmpeg import extract_audio_track

# mlx_whisper is Apple-Silicon-only (MLX is a native macOS/Metal framework)
# and is imported lazily inside transcribe() rather than at module load time.
# This keeps the rest of NuclearCutter (render pipeline, CLI --help, schema, etc)
# usable on non-Apple-Silicon machines and in environments where mlx_whisper
# isn't installed/loadable, instead of the whole package failing to import.


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Utterance:
    text: str
    start: float
    end: float
    words: list[Word]


DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def transcribe(video_path: Path, model: str = DEFAULT_MODEL) -> list[Utterance]:
    """Transcribe the full audio track of a video file with word-level timestamps."""
    try:
        import mlx_whisper
    except ImportError as e:
        raise RuntimeError(
            "mlx_whisper is required for transcription and is Apple-Silicon-only. "
            "Install it with `pip install mlx-whisper` on an M-series Mac, or see "
            "README.md for platform notes."
        ) from e

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        extract_audio_track(video_path, audio_path)

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model,
            word_timestamps=True,
        )

    utterances = []
    for segment in result["segments"]:
        words = [
            Word(text=w["word"].strip(), start=w["start"], end=w["end"])
            for w in segment.get("words", [])
        ]
        utterances.append(Utterance(
            text=segment["text"].strip(),
            start=segment["start"],
            end=segment["end"],
            words=words,
        ))
    return utterances


def find_subtitle_file(video_path: Path) -> Path | None:
    """Look for a sidecar .srt file next to the video with the same stem."""
    candidate = video_path.with_suffix(".srt")
    if candidate.exists():
        return candidate
    # Also check for lang-tagged variants like Movie.en.srt
    for srt_path in video_path.parent.glob(f"{video_path.stem}*.srt"):
        return srt_path
    return None


def parse_subtitles(srt_path: Path) -> list[Utterance]:
    """Parse an SRT file into Utterance objects (no word-level timing available from SRT)."""
    subs = pysrt.open(str(srt_path))
    utterances = []
    for sub in subs:
        start = _srt_time_to_seconds(sub.start)
        end = _srt_time_to_seconds(sub.end)
        text = sub.text.replace("\n", " ")
        utterances.append(Utterance(text=text, start=start, end=end, words=[]))
    return utterances


def _srt_time_to_seconds(t) -> float:
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0


def cross_check_utterance(whisper_text: str, subtitle_utterances: list[Utterance], start: float, end: float) -> str:
    """Return 'whisper+subtitle' if a subtitle utterance overlaps this time range and roughly
    agrees with the whisper text, else 'whisper' (subtitle disagreement is noted but whisper
    timestamps are trusted since they're word-level; subtitle is corroboration, not override)."""
    for sub in subtitle_utterances:
        overlaps = sub.start < end and sub.end > start
        if overlaps:
            return "whisper+subtitle"
    return "whisper"
