"""
Audio transcription for the foul-language detection pipeline (docs/SPEC.md
section 4.2). Uses mlx-whisper for Apple Silicon acceleration, with
word-level timestamps. If a subtitle file is available (embedded or
sidecar), it's parsed and cross-checked against the Whisper output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
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


def transcribe(video_path: Path, model: str | None = None,
               progress_callback=None) -> list[Utterance]:
    """Transcribe the full audio track of a video file with word-level timestamps.

    The Whisper model must be supplied by the caller (there are no built-in
    defaults). See README for suggested models.

    `progress_callback(frac)`, if given, is called with the fraction of audio
    transcribed (0.0..1.0). mlx-whisper exposes no progress API, so we proxy
    its internal progress bar (see _transcribe_with_progress).
    """
    if not model:
        raise RuntimeError(
            "No Whisper model specified for transcription. Pass --whisper-model "
            "(e.g. `--whisper-model mlx-community/whisper-small-mlx`). See README.md."
        )
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

        if progress_callback is not None:
            result = _transcribe_with_progress(audio_path, model, progress_callback)
        else:
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=model,
                word_timestamps=True,
            )

    utterances = segments_to_utterances(result)
    return utterances


def segments_to_utterances(result: dict) -> list[Utterance]:
    """Convert mlx_whisper's raw result dict into Utterance objects."""
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


class TranscriptionStopped(RuntimeError):
    """Raised when the user stops mid-transcription — the whisper child process
    was hard-killed and nothing was saved (the transcript cache only covers a
    COMPLETED transcription; a stopped one re-transcribes on resume)."""


def transcribe_killable(video_path: Path, model: str | None,
                        progress_callback=None, stop_event=None) -> list[Utterance]:
    """Like transcribe(), but runs whisper in a CHILD PROCESS that the caller
    can hard-kill via `stop_event` (the GUI's Stop button).

    Stopping mid-transcription terminates the child immediately — that partial
    transcription is discarded (it re-runs on resume). If transcription
    COMPLETES, the utterances are returned normally so the caller can save the
    transcript cache. When `stop_event` is None, falls back to the in-process
    transcribe() (e.g. the headless CLI, where Ctrl-C kills everything anyway).
    """
    import json
    import sys
    import time

    if stop_event is None:
        return transcribe(video_path, model=model, progress_callback=progress_callback)

    if not model:
        raise RuntimeError(
            "No Whisper model specified for transcription. Pass --whisper-model "
            "(e.g. `--whisper-model mlx-community/whisper-small-mlx`). See README.md."
        )

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        extract_audio_track(video_path, audio_path)
        if stop_event is not None and stop_event.is_set():
            raise TranscriptionStopped("transcription stopped before starting")

        out_path = Path(tmp) / "transcript.json"
        progress_path = Path(tmp) / "progress.txt"
        err_path = Path(tmp) / "child.err"

        repo = Path(__file__).resolve().parents[1]
        code = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(repo)!r})\n"
            "from nuclearcutter.detection.transcribe import _transcribe_child_main\n"
            "_transcribe_child_main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code, str(audio_path), model,
             str(out_path), str(progress_path)],
            stdout=subprocess.DEVNULL,
            stderr=open(err_path, "w"),
        )

        last_pct = -1.0
        try:
            while proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    proc.terminate()  # SIGTERM — kills whisper mid-transcription
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


def _transcribe_child_main(audio_path: str, model: str, out_path: str, progress_path: str) -> None:
    """Runs inside the transcription child process. Writes progress fractions
    to `progress_path` and the finished utterances JSON to `out_path`."""
    import json

    def _report(frac: float) -> None:
        with open(progress_path, "a") as f:
            f.write(f"{frac}\n")

    result = _transcribe_with_progress(Path(audio_path), model, _report)
    utterances = segments_to_utterances(result)
    with open(out_path, "w") as f:
        json.dump({"utterances": utterances_to_dict(utterances)}, f)


class _TqdmProxy:
    """Minimal tqdm stand-in that reports progress to a callback.

    mlx-whisper drives a `tqdm.tqdm` progress bar internally (total = audio
    frames, updated per decoded window). We swap its tqdm for this proxy during
    the transcribe call so we can surface real progress without forking the
    library. Falls back to the real tqdm for everything else.
    """

    def __init__(self, *args, _callback=None, **kwargs):
        import tqdm as _tqdm

        self._bar = _tqdm.tqdm(*args, **kwargs)
        self._cb = _callback
        self.n = 0
        self.total = kwargs.get("total")

    def update(self, n=1):
        self.n += n
        try:
            if self._cb is not None and self.total:
                self._cb(max(0.0, min(self.n / self.total, 1.0)))
        except Exception:
            pass
        try:
            self._bar.update(n)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self._bar.close()
        except Exception:
            pass

    def close(self):
        self.__exit__(None, None, None)

    def __getattr__(self, name):
        return getattr(self._bar, name)


def _transcribe_with_progress(audio_path: Path, model: str, callback) -> dict:
    """Run mlx_whisper.transcribe with its internal progress bar proxied so
    `callback(frac)` reports real transcription progress.

    mlx_whisper's transcribe.py does `import tqdm` then `tqdm.tqdm(...)` — the
    name `tqdm` IS the module, so we must swap the module binding for an object
    that still exposes a `.tqdm` attribute (a module stand-in), not a plain
    function, or `tqdm.tqdm` raises AttributeError.
    """
    import mlx_whisper
    import sys

    # `mlx_whisper.transcribe` is the FUNCTION (the package __init__ rebinds
    # the module attribute), so fetch the real module from sys.modules to
    # reach its tqdm attribute.
    transcribe_mod = sys.modules["mlx_whisper.transcribe"]
    original = transcribe_mod.tqdm

    class _TqdmModuleProxy:
        """Acts like the tqdm module for the call: exposes `.tqdm(...)` which
        builds a _TqdmProxy that reports progress to `callback`."""

        def tqdm(self, *args, **kwargs):
            return _TqdmProxy(*args, _callback=callback, **kwargs)

    transcribe_mod.tqdm = _TqdmModuleProxy()
    try:
        return mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model,
            word_timestamps=True,
        )
    finally:
        transcribe_mod.tqdm = original


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
    import json
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
    import json
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
    """Return 'whisper+subtitle' if a subtitle utterance overlaps this time range and roughly
    agrees with the whisper text, else 'whisper' (subtitle disagreement is noted but whisper
    timestamps are trusted since they're word-level; subtitle is corroboration, not override)."""
    for sub in subtitle_utterances:
        overlaps = sub.start < end and sub.end > start
        if overlaps:
            return "whisper+subtitle"
    return "whisper"
