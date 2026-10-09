"""Tests for the remote model-server client (nuclearcutter/utils/model_server.py).

No servers are spawned and no model weights / GPU are needed: all HTTP is
mocked and the whisper model directory is a tmp_path.
"""

from unittest.mock import patch

import pytest

from nuclearcutter.utils.model_server import (
    ModelServerError,
    WhisperConfig,
    is_server_up,
    list_models,
    require_server_up,
    wait_for_server,
    whisper_is_up,
    whisper_list_models,
    whisper_load_model,
)


class TestIsServerUp:
    @patch("nuclearcutter.utils.model_server.requests.get")
    def test_up_when_ok(self, mock_get):
        mock_get.return_value.ok = True
        assert is_server_up("http://localhost:1234/v1") is True

    @patch("nuclearcutter.utils.model_server.requests.get")
    def test_down_on_exception(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("refused")
        assert is_server_up("http://localhost:1234/v1") is False


class TestListModels:
    @patch("nuclearcutter.utils.model_server.requests.get")
    def test_parses_data_ids(self, mock_get):
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {
            "data": [{"id": "a"}, {"id": "b"}, {"no_id": 1}],
        }
        assert list_models("http://x/v1") == ["a", "b"]

    @patch("nuclearcutter.utils.model_server.requests.get")
    def test_unreachable_returns_empty(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("refused")
        assert list_models("http://x/v1") == []


class TestRequireServerUp:
    @patch("nuclearcutter.utils.model_server.is_server_up", return_value=False)
    def test_raises_when_down(self, _mock):
        with pytest.raises(ModelServerError, match="No model server answering"):
            require_server_up("http://localhost:9999/v1")

    @patch("nuclearcutter.utils.model_server.is_server_up", return_value=True)
    def test_ok_when_up(self, _mock):
        require_server_up("http://localhost:9999/v1")  # must not raise


class TestWaitForServer:
    @patch("nuclearcutter.utils.model_server.is_server_up")
    def test_returns_false_on_timeout(self, mock_up):
        mock_up.return_value = False
        assert wait_for_server("http://x/v1", timeout=0.1, interval=0.01) is False

    @patch("nuclearcutter.utils.model_server.is_server_up")
    def test_returns_true_when_up(self, mock_up):
        mock_up.side_effect = [False, False, True]
        assert wait_for_server("http://x/v1", timeout=5, interval=0.01) is True


class TestWhisperIsUp:
    @patch("nuclearcutter.utils.model_server.requests.get")
    def test_ok_when_health_200(self, mock_get):
        mock_get.return_value.status_code = 200
        assert whisper_is_up("http://localhost:8081") is True

    @patch("nuclearcutter.utils.model_server.requests.get")
    def test_loading_503_counts_as_up(self, mock_get):
        mock_get.return_value.status_code = 503
        assert whisper_is_up(WhisperConfig(base_url="http://localhost:8081")) is True

    @patch("nuclearcutter.utils.model_server.requests.get")
    def test_down_on_exception(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("refused")
        assert whisper_is_up("http://localhost:8081") is False


class TestWhisperListModels:
    def test_lists_bin_stems_sorted(self, tmp_path):
        (tmp_path / "ggml-small.bin").write_bytes(b"x")
        (tmp_path / "ggml-base.en.bin").write_bytes(b"x")
        (tmp_path / "not-a-model.txt").write_text("x")
        (tmp_path / "ggml-med.bin").write_bytes(b"x")
        cfg = WhisperConfig(models_dir=str(tmp_path))
        assert whisper_list_models(cfg) == [
            "ggml-base.en", "ggml-med", "ggml-small",
        ]

    def test_missing_dir_returns_empty(self):
        cfg = WhisperConfig(models_dir="/nonexistent/whisper/models")
        assert whisper_list_models(cfg) == []


class TestWhisperLoadModel:
    def test_resolves_dropdown_id_to_absolute_path(self, tmp_path):
        (tmp_path / "ggml-small.bin").write_bytes(b"x")
        cfg = WhisperConfig(models_dir=str(tmp_path))
        with patch("nuclearcutter.utils.model_server.requests.post") as mock_post:
            mock_post.return_value.ok = True
            mock_post.return_value.text = "Load was successful!"
            whisper_load_model(cfg, "ggml-small")
            args, kwargs = mock_post.call_args
            assert args[0] == "http://127.0.0.1:8081/load"
            # The `model` field is a file whose CONTENT is the absolute path.
            _name, content, _ctype = kwargs["files"]["model"]
            assert content == str(tmp_path / "ggml-small.bin").encode()

    def test_explicit_path_sent_verbatim(self, tmp_path):
        cfg = WhisperConfig(models_dir=str(tmp_path))
        with patch("nuclearcutter.utils.model_server.requests.post") as mock_post:
            mock_post.return_value.ok = True
            mock_post.return_value.text = "ok"
            whisper_load_model(cfg, "/abs/path/ggml-tiny.bin")
            _name, content, _ctype = mock_post.call_args[1]["files"]["model"]
            assert content == b"/abs/path/ggml-tiny.bin"

    def test_unknown_dropdown_id_raises(self, tmp_path):
        cfg = WhisperConfig(models_dir=str(tmp_path))
        with pytest.raises(ModelServerError, match="not found"):
            whisper_load_model(cfg, "ggml-does-not-exist")

    def test_non_ok_response_raises(self, tmp_path):
        (tmp_path / "ggml-small.bin").write_bytes(b"x")
        cfg = WhisperConfig(models_dir=str(tmp_path))
        with patch("nuclearcutter.utils.model_server.requests.post") as mock_post:
            mock_post.return_value.ok = False
            mock_post.return_value.status_code = 500
            mock_post.return_value.text = "bad model"
            with pytest.raises(ModelServerError, match="/load failed"):
                whisper_load_model(cfg, "ggml-small")
