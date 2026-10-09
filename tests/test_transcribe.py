"""Tests for transcription (nuclearcutter/detection/transcribe.py).

Transcription is now a remote client of a whisper.cpp server. These tests mock
the HTTP layer (no real server, no model, no ffmpeg) and cover the result
conversion plus the request/health/progress/error paths.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nuclearcutter.detection.transcribe import (
    TranscriptionStopped,
    _remote_transcribe,
    segments_to_utterances,
    transcribe,
    transcribe_killable,
    utterances_from_dict,
    utterances_to_dict,
)
from nuclearcutter.utils.model_server import WhisperConfig

SAMPLE = {
    "segments": [
        {
            "start": 0.5,
            "end": 2.0,
            "text": "  Hello world ",
            "words": [
                {"word": " Hello", "start": 0.5, "end": 1.0},
                {"word": " world", "start": 1.0, "end": 2.0},
            ],
        },
        {"start": 3.0, "end": 4.0, "text": "", "words": []},  # empty -> skipped
    ],
}


def test_segments_to_utterances_converts_words():
    u = segments_to_utterances(SAMPLE)
    assert len(u) == 1  # the empty-text segment is dropped
    assert u[0].text == "Hello world"
    assert u[0].start == 0.5 and u[0].end == 2.0
    assert [w.text for w in u[0].words] == ["Hello", "world"]
    assert u[0].words[0].start == 0.5


def test_segments_to_utterances_empty():
    assert segments_to_utterances({"segments": []}) == []
    assert segments_to_utterances({}) == []


def test_utterances_roundtrip():
    u = segments_to_utterances(SAMPLE)
    back = utterances_from_dict(utterances_to_dict(u))
    assert back == u


def _fake_audio(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    return audio


def _extract_side_effect(video_path, out_path, *a, **k):
    """Stand-in for extract_audio_track: just materialize the out file."""
    Path(out_path).write_bytes(b"RIFF")
    return Path(out_path)


class TestTranscribe:
    def test_requires_config(self, tmp_path):
        with pytest.raises(RuntimeError, match="No whisper server configured"):
            transcribe(tmp_path / "movie.mp4", WhisperConfig(base_url=""))
        with pytest.raises(RuntimeError):
            transcribe(tmp_path / "movie.mp4", None)

    def test_happy_path_reports_progress(self, tmp_path):
        import requests as real_requests

        with patch("nuclearcutter.detection.transcribe.extract_audio_track",
                   side_effect=_extract_side_effect), \
             patch("nuclearcutter.detection.transcribe.requests.get",
                   return_value=real_requests.Response()) as mock_get, \
             patch("nuclearcutter.detection.transcribe.requests.post") as mock_post:
            mock_get.return_value.status_code = 200
            mock_post.return_value.ok = True
            mock_post.return_value.json.return_value = SAMPLE
            seen = []
            result = transcribe(tmp_path / "movie.mp4", WhisperConfig(),
                                progress_callback=seen.append)
        assert len(result) == 1
        assert seen == [0.0, 1.0]
        args, kwargs = mock_post.call_args
        assert args[0] == "http://127.0.0.1:8081/audio/transcriptions"
        assert kwargs["data"]["response_format"] == "verbose_json"
        assert kwargs["data"]["word_timestamps"] == "1"

    def test_health_503_loading(self, tmp_path):
        import requests as real_requests

        with patch("nuclearcutter.detection.transcribe.extract_audio_track",
                   side_effect=_extract_side_effect), \
             patch("nuclearcutter.detection.transcribe.requests.get") as mock_get:
            mock_get.return_value.status_code = 503
            with pytest.raises(RuntimeError, match="still loading its model"):
                transcribe(tmp_path / "movie.mp4", WhisperConfig())

    def test_unreachable_server(self, tmp_path):
        import requests as real_requests

        with patch("nuclearcutter.detection.transcribe.extract_audio_track",
                   side_effect=_extract_side_effect), \
             patch("nuclearcutter.detection.transcribe.requests.get",
                   side_effect=real_requests.RequestException("refused")):
            with pytest.raises(RuntimeError, match="cannot reach the whisper server"):
                transcribe(tmp_path / "movie.mp4", WhisperConfig())

    def test_non_ok_response_raises(self, tmp_path):
        import requests as real_requests

        with patch("nuclearcutter.detection.transcribe.extract_audio_track",
                   side_effect=_extract_side_effect), \
             patch("nuclearcutter.detection.transcribe.requests.get") as mock_get, \
             patch("nuclearcutter.detection.transcribe.requests.post") as mock_post:
            mock_get.return_value.status_code = 200
            mock_post.return_value.ok = False
            mock_post.return_value.status_code = 500
            mock_post.return_value.text = "boom"
            with pytest.raises(RuntimeError, match="returned 500"):
                transcribe(tmp_path / "movie.mp4", WhisperConfig())

    def test_missing_segments_raises(self, tmp_path):
        import requests as real_requests

        with patch("nuclearcutter.detection.transcribe.extract_audio_track",
                   side_effect=_extract_side_effect), \
             patch("nuclearcutter.detection.transcribe.requests.get") as mock_get, \
             patch("nuclearcutter.detection.transcribe.requests.post") as mock_post:
            mock_get.return_value.status_code = 200
            mock_post.return_value.ok = True
            mock_post.return_value.json.return_value = {"no_segments": True}
            with pytest.raises(RuntimeError, match="missing 'segments'"):
                transcribe(tmp_path / "movie.mp4", WhisperConfig())


class TestRemoteTranscribe:
    @patch("nuclearcutter.detection.transcribe.requests.post")
    def test_posts_multipart_file(self, mock_post, tmp_path):
        audio = _fake_audio(tmp_path)
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = {"segments": []}
        data = _remote_transcribe(WhisperConfig(), audio)
        assert data == {"segments": []}
        _url, kwargs = mock_post.call_args
        assert "file" in kwargs["files"]
        name, fh, ctype = kwargs["files"]["file"]
        assert name == "audio.wav" and ctype == "audio/wav"


class TestTranscribeKillable:
    def test_none_stop_event_falls_back(self, tmp_path):
        # stop_event=None -> delegates to transcribe() (mocked out).
        with patch("nuclearcutter.detection.transcribe.transcribe", return_value=[]) as mock_t:
            assert transcribe_killable(tmp_path / "m.mp4", WhisperConfig(), stop_event=None) == []
            mock_t.assert_called_once()

    def test_set_stop_event_stops_before_start(self, tmp_path):
        import threading

        ev = threading.Event()
        ev.set()
        with patch("nuclearcutter.detection.transcribe.extract_audio_track"):
            with pytest.raises(TranscriptionStopped):
                transcribe_killable(tmp_path / "m.mp4", WhisperConfig(), stop_event=ev)

    def test_no_config_raises(self, tmp_path):
        import threading

        with pytest.raises(RuntimeError, match="No whisper server configured"):
            transcribe_killable(tmp_path / "m.mp4", WhisperConfig(base_url=""), stop_event=threading.Event())
