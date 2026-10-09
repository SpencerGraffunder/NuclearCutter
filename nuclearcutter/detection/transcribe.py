"""
Audio transcription for the foul-language detection pipeline (docs/SPEC.md
section 4.2). Transcription is delegated to an already-running **whisper.cpp
server** (`whisper-server`): the audio track is extracted to a 16 kHz mono WAV
and sent to the server's `/inference` endpoint, which returns `verbose_json`
with word-level timestamps. That JSON has the same `segments[].words[]` shape
as mlx-whisper, so `segments_to_utterances` converts it unchanged.

NuclearCutter never runs whisper itself — the model is loaded on the server
(hot-swappable via its `/load` endpoint; see utils/model_server.py). If a
subtitle file is available (sidecar), it's parsed and cross-checked against
the transcription.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pysrt
import requests

from nuclearcutter.utils.ffmpeg import extract_audio_track
from nuclearcutter.utils.model_server import WhisperConfig

# How long a single transcription request may take. A 3-hour film is one
# multipart request; on a GPU this is minutes, on a slow CPU it can be long.
_TRANSCRIBE_TIMEOUT = 3600


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


def transcribe(video_path: Path, cfg: WhisperConfig | None,
               progress_callback=None) -> list[Utterance]:
    """Transcribe the full audio track of a video file with word-level
    timestamps, using the whisper.cpp server at `cfg.base_url`.

    `progress_callback(frac)`, if given, is called with the fraction of the
    transcription complete (0.0..1.0). The whisper.cpp server does the whole
    film in one request and exposes no progress API, so we report 0.0 while
    in flight and 1.0 on completion (the GUI renders 0.0 as an indeterminate
    bar). The whisper model is whatever the server currently has loaded —
    select it with the GUI's whisper model dropdown (which hot-swaps it).
    """
    if not cfg or not cfg.base_url:
        raise RuntimeError(
            "No whisper server configured for transcription. Set the whisper "
            "server IP/URL in the web GUI (or --whisper-base-url on the CLI). "
            "See README.md."
        )
    _require_whisper_up(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        extract_audio_track(video_path, audio_path)

        if progress_callback is not None:
            progress_callback(0.0)
        result = _remote_transcribe(cfg, audio_path)
        if progress_callback is not None:
            progress_callback(1.0)

    return segments_to_utterances(result)


def _require_whisper_up(cfg: WhisperConfig) -> None:
    try:
        r = requests.get(f"{cfg.base_url.rstrip('/')}/health", timeout=5)
        if r.status_code == 200:
            return
        if r.status_code == 503:
            raise RuntimeError(
                "whisper server is still loading its model "
                f"({cfg.base_url}). Try again in a few seconds."
            )
        raise RuntimeError(
            f"whisper server at {cfg.base_url} answered {r.status_code} to /health "
            f"({r.text[:120]!r}). Is it running? (GET /health should be 200.)"
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"cannot reach the whisper server at {cfg.base_url}: {exc}. "
            "Start whisper-server or fix the whisper IP/URL in the GUI."
        ) from exc


def _remote_transcribe(cfg: WhisperConfig, audio_path: Path) -> dict:
    """POST the audio to the whisper server's inference endpoint and return the
    parsed verbose_json dict (with `segments`)."""
    url = f"{cfg.base_url.rstrip('/')}{cfg.inference_path}"
    with open(audio_path, "rb") as fh:
        resp = requests.post(
            url,
            files={"file": (audio_path.name, fh, "audio/wav")},
            data={
                "response_format": "verbose_json",
                "temperature": "0.0",
                "word_timestamps": "1",
            },
            timeout=_TRANSCRIBE_TIMEOUT,
        )
    if not resp.ok:
        raise RuntimeError(
            f"whisper server returned {resp.status_code}: {resp.text[:400]}"
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"whisper server returned non-JSON: {resp.text[:200]!r}"
        ) from exc
    if "segments" not in data:
        raise RuntimeError(
            f"whisper server response missing 'segments': {str(data)[:200]}"
        )
    return data


def segments_to_utterances(result: dict) -> list[Utterance]:
    """Convert a whisper (mlx-whisper or whisper.cpp verbose_json) result dict
    into Utterance objects. Both produce `segments[].{text,start,end,words[]}`
    where each word is `{word, start, end}`."""
    utterances = []
    for segment in result.get("segments", []):
        words = [
            Word(text=w["word"].strip(), start=w["start"], end=w["end"])
            for w in segment.get("words", [])
        ]
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        utterances.append(Utterance(
            text=text,
            start=segment["start"],
            end=segment["end"],
            words=words,
        ))
    return utterances


class TranscriptionStopped(RuntimeError):
    """Raised when the user stops mid-transcription — the whisper request
    child was killed and nothing was saved (the transcript cache only covers a
    COMPLETED transcription; a stopped one re-transcribes on resume)."""


def transcribe_killable(video_path: Path, cfg: WhisperConfig | None,
                        progress_callback=None, stop_event=None) -> list[Utterance]:
    """Like transcribe(), but runs the whisper HTTP request in a CHILD PROCESS
    that the caller can hard-kill via `stop_event` (the GUI's Stop button).

    Stopping mid-transcription terminates the child immediately — that partial
    transcription is discarded (it re-runs on resume). If transcription
    COMPLETES, the utterances are returned normally so the caller can save the
    transcript cache. When `stop_event` is None, falls back to the in-process
    transcribe() (e.g. the headless CLI, where Ctrl-C kills everything anyway).
    """
    if stop_event is None:
        return transcribe(video_path, cfg, progress_callback=progress_callback)

    if not cfg or not cfg.base_url:
        raise RuntimeError(
            "No whisper server configured for transcription. Set the whisper "
            "server IP/URL in the web GUI (or --whisper-base-url on the CLI)."
        )

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        extract_audio_track(video_path, audio_path)
        if stop_event is not None and stop_event.is_set():
            raise TranscriptionStopped("transcription stopped before starting")

        out_path = Path(tmp) / "transcript.json"
        progress_path = Path(tmp) / "progress.txt"
        err_path = Path(tmp) / "child.err"

        code = _child_code()
        proc = subprocess.Popen(
            [sys.executable, "-c", code,
             str(audio_path), cfg.base_url, cfg.inference_path,
             str(out_path), str(progress_path)],
            stdout=subprocess.DEVNULL,
            stderr=open(err_path, "w"),
        )

        last_pct = -1.0
        try:
            while proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    proc.terminate()  # SIGTERM — kills the whisper request
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise TranscriptionStopped("transcription killed by stop")
                try:
                    lines = progress_path.read_text().split()
                    if lines:
                        frac = float(lines[-1])
                        if frac > last_pct and progress_callback:
                            last_pct = frac
                            progress_callback(frac)
                except (OSError, ValueError):
                    pass
                time.sleep(0.5)
        finally:
            if proc.poll() is None:
                proc.terminate()

        if proc.returncode != 0:
            err = ""
            try:
                err = err_path.read_text()[-500:]
            except OSError:
                pass
            raise RuntimeError(
                f"transcription process exited with code {proc.returncode}"
                + (f": {err.strip()}" if err.strip() else "")
            )

        data = json.loads(out_path.read_text())
        return utterances_from_dict(data["utterances"])


def _child_code() -> str:
    """The code run in the transcription child process: does the single whisper
    HTTP request (so the parent can kill it on Stop), writes progress 0.0 at
    start / 1.0 on completion, and dumps the finished utterances JSON."""
    return (
        "import sys, json, time, tempfile\n"
        "from pathlib import Path\n"
        "import requests\n"
        "from nuclearcutter.detection.transcribe import _remote_transcribe, segments_to_utterances, utterances_to_dict\n"
        "from nuclearcutter.utils.model_server import WhisperConfig\n"
        "audio, base, inpath, out, prog = sys.argv[1:6]\n"
        "def report(f):\n"
        "    with open(prog, 'a') as fh: fh.write(f'{f}\\n')\n"
        "report(0.0)\n"
        "cfg = WhisperConfig(base_url=base, inference_path=inpath)\n"
        "result = _remote_transcribe(cfg, Path(audio))\n"
        "utt = segments_to_utterances(result)\n"
        "with open(out, 'w') as fh: json.dump({'utterances': utterances_to_dict(utt)}, fh)\n"
        "report(1.0)\n"
    )


def utterances_to_dict(utterances: list[Utterance]) -> list[dict]:
    return [
        {
            "text": u.text,
            "start": u.start,
            "end": u.end,
            "words": [{"text": w.text, "start": w.start, "end": w.end} for w in u.words],
        }
        for u in utterances
    ]


def utterances_from_dict(data: list[dict]) -> list[Utterance]:
    return [
        Utterance(
            text=u["text"],
            start=u["start"],
            end=u["end"],
            words=[Word(text=w["text"], start=w["start"], end=w["end"]) for w in u.get("words", [])],
        )
        for u in data
    ]


def write_transcript_cache(path: Path, video_path: Path, utterances: list[Utterance]) -> None:
    """Write the transcript cache (validated against the video's size/mtime)."""
    import os

    try:
        st = os.stat(video_path)
        data = {
            "video_size": st.st_size,
            "video_mtime": st.st_mtime,
            "utterances": utterances_to_dict(utterances),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
    except OSError as exc:
        print(f"warning: could not write transcript cache {path}: {exc}",
              file=__import__("sys").stderr)


def read_transcript_cache(path: Path, video_path: Path) -> list[Utterance] | None:
    """Load a transcript cache if it exists and still matches the video.

    Returns None when there is no cache, it's corrupt, or the video changed —
    the caller then re-transcribes.
    """
    import os

    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        st = os.stat(video_path)
        if data.get("video_size") != st.st_size or data.get("video_mtime") != st.st_mtime:
            return None
        return utterances_from_dict(data.get("utterances", []))
    except (OSError, ValueError, TypeError):
        return None


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
    """Return 'whisper+subtitle' if a subtitle utterance overlaps this time range, else
    'whisper' (whisper timestamps are trusted since they're word-level; subtitle is
    corroboration, not override)."""
    for sub in subtitle_utterances:
        overlaps = sub.start < end and sub.end > start
        if overlaps:
            return "whisper+subtitle"
    return "whisper"
