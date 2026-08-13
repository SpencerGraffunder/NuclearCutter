"""Tests for the transcript cache (nuclearcutter/detection/transcribe.py).

A resumed scan must skip re-transcribing when the SAME file was already
transcribed, but must never reuse a cache for a changed video.
"""

import os

from nuclearcutter.detection.transcribe import (
    Utterance,
    Word,
    read_transcript_cache,
    write_transcript_cache,
)


def _utterances():
    return [
        Utterance(text="hello world", start=0.0, end=1.2,
                  words=[Word(text="hello", start=0.0, end=0.4),
                         Word(text="world", start=0.5, end=1.2)]),
        Utterance(text="goodbye", start=2.0, end=2.5, words=[Word(text="goodbye", start=2.0, end=2.5)]),
    ]


def test_transcript_cache_roundtrip(tmp_path):
    video = tmp_path / "m.mkv"
    video.write_bytes(b"video-data")
    cache = tmp_path / "m.nuclearcutter.transcript.json"
    write_transcript_cache(cache, video, _utterances())

    loaded = read_transcript_cache(cache, video)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].text == "hello world"
    assert loaded[0].words[1].text == "world"
    assert loaded[1].end == 2.5


def test_transcript_cache_invalidated_when_video_changes(tmp_path):
    video = tmp_path / "m.mkv"
    video.write_bytes(b"v1")
    cache = tmp_path / "m.nuclearcutter.transcript.json"
    write_transcript_cache(cache, video, _utterances())

    # Same path, different content -> size/mtime mismatch -> cache rejected.
    video.write_bytes(b"a-different-video")
    assert read_transcript_cache(cache, video) is None


def test_transcript_cache_missing_returns_none(tmp_path):
    video = tmp_path / "m.mkv"
    video.write_bytes(b"v")
    assert read_transcript_cache(tmp_path / "nope.json", video) is None


def test_transcript_cache_corrupt_returns_none(tmp_path):
    video = tmp_path / "m.mkv"
    video.write_bytes(b"v")
    cache = tmp_path / "m.nuclearcutter.transcript.json"
    cache.write_text("{ not json")
    assert read_transcript_cache(cache, video) is None


def test_transcribe_killable_stops_before_start(monkeypatch, tmp_path):
    """A pre-set stop event aborts before any whisper process is spawned."""
    import threading

    import pytest

    from nuclearcutter.detection.transcribe import (
        TranscriptionStopped,
        transcribe_killable,
    )

    video = tmp_path / "t.mkv"
    video.write_bytes(b"x")
    monkeypatch.setattr("nuclearcutter.detection.transcribe.extract_audio_track",
                        lambda *a, **k: tmp_path / "a.wav")
    stop = threading.Event()
    stop.set()
    with pytest.raises(TranscriptionStopped):
        transcribe_killable(video, "some-model", stop_event=stop)


def test_transcribe_killable_terminates_child_on_stop(monkeypatch, tmp_path):
    """Stop during transcription terminates (SIGTERMs) the whisper child."""
    import threading
    import time

    import pytest

    from nuclearcutter.detection.transcribe import TranscriptionStopped, transcribe_killable

    video = tmp_path / "t.mkv"
    video.write_bytes(b"x")
    monkeypatch.setattr("nuclearcutter.detection.transcribe.extract_audio_track",
                        lambda *a, **k: tmp_path / "a.wav")

    class FakeProc:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.returncode = None

        def poll(self):
            return None  # always "running"

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return None

    fake = FakeProc()
    monkeypatch.setattr("nuclearcutter.detection.transcribe.subprocess.Popen",
                        lambda *a, **k: fake)

    stop = threading.Event()

    def _set_stop():
        time.sleep(0.6)  # let the poll loop run at least once
        stop.set()

    threading.Thread(target=_set_stop, daemon=True).start()
    with pytest.raises(TranscriptionStopped):
        transcribe_killable(video, "some-model", stop_event=stop)
    assert fake.terminated
