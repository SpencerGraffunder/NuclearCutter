"""Tests for the model backend manager (nuclearcutter/utils/model_server.py),
without spawning a real server (no model weights / GPU needed in CI)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nuclearcutter.utils.model_server import (
    DEFAULT_MLX_MODEL_PATH,
    MLX_BASE_URL,
    ModelServerConfig,
    ModelServerError,
    ensure_backend,
    is_server_up,
    wait_for_server,
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


class TestWaitForServer:
    @patch("nuclearcutter.utils.model_server.is_server_up")
    def test_returns_false_on_timeout(self, mock_up):
        mock_up.return_value = False
        assert wait_for_server("http://x/v1", timeout=0.1, interval=0.01) is False

    @patch("nuclearcutter.utils.model_server.is_server_up")
    def test_returns_true_when_up(self, mock_up):
        mock_up.side_effect = [False, False, True]
        assert wait_for_server("http://x/v1", timeout=5, interval=0.01) is True


class TestEnsureBackend:
    def test_standalone_requires_running_server(self):
        cfg = ModelServerConfig(backend="standalone", base_url="http://localhost:9999/v1")
        with patch("nuclearcutter.utils.model_server.is_server_up", return_value=False):
            with pytest.raises(ModelServerError, match="No server answering"):
                ensure_backend(cfg)

    def test_standalone_ok_when_server_up(self):
        cfg = ModelServerConfig(backend="standalone", base_url="http://localhost:9999/v1")
        with patch("nuclearcutter.utils.model_server.is_server_up", return_value=True):
            assert ensure_backend(cfg) is None

    def test_unknown_backend_raises(self):
        cfg = ModelServerConfig(backend="bogus")
        with pytest.raises(ModelServerError, match="Unknown model_backend"):
            ensure_backend(cfg)

    def test_mlx_vlm_reuses_existing_server(self):
        cfg = ModelServerConfig(backend="mlx-vlm", base_url="http://localhost:1234/v1")
        with patch("nuclearcutter.utils.model_server.is_server_up", return_value=True):
            assert ensure_backend(cfg) is None

    def test_mlx_vlm_missing_model_path(self):
        cfg = ModelServerConfig(backend="mlx-vlm", model_path="/nonexistent/xyz")
        with patch("nuclearcutter.utils.model_server.is_server_up", return_value=False):
            with pytest.raises(ModelServerError, match="model_path not found"):
                ensure_backend(cfg)
