"""Tests for the LLM client, especially model-name validation.

Model-name typos (e.g. "qwen3.5:4b-mlx" vs "qwen3.5-4b-mlx") are a common
footgun — LM Studio uses hyphens while Ollama uses colons.  test_connection
should catch these before a multi-hour scan starts.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from nuclearcutter.utils.llm_client import LLMClient, LLMConfig


def _fake_models_response(model_ids: list[str], status: int = 200):
    """Return a mock requests.Response that mimics a /v1/models endpoint."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.ok = status < 400
    resp.json.return_value = {"data": [{"id": m} for m in model_ids]}
    resp.raise_for_status = lambda: (
        None if resp.ok else (_ for _ in ()).throw(
            requests.HTTPError(f"{status}", response=resp)
        )
    )
    return resp


class TestModelValidation:
    """test_connection should validate model names against the server's model list."""

    def _client(self, vlm: str = "qwen3.5-4b-mlx", text: str = "qwen3.5-4b-mlx") -> LLMClient:
        cfg = LLMConfig(
            base_url="http://localhost:9999/v1",
            vlm_model=vlm,
            text_model=text,
        )
        return LLMClient(cfg)

    @patch("nuclearcutter.utils.llm_client.requests.get")
    def test_valid_model_passes(self, mock_get: MagicMock):
        """A model name that matches the server's list should not raise."""
        mock_get.return_value = _fake_models_response(["qwen3.5-4b-mlx"])
        self._client().test_connection()  # no exception

    @patch("nuclearcutter.utils.llm_client.requests.get")
    def test_invalid_model_raises(self, mock_get: MagicMock):
        """A model name not in the server's list should raise RuntimeError."""
        mock_get.return_value = _fake_models_response(["qwen3.5-4b-mlx"])
        client = self._client(vlm="qwenvl:wrong")
        with pytest.raises(RuntimeError, match="VLM model.*not found"):
            client.test_connection()

    @patch("nuclearcutter.utils.llm_client.requests.get")
    def test_colon_model_caught(self, mock_get: MagicMock):
        """Ollama-style 'model:tag' should be caught when server has hyphenated names."""
        mock_get.return_value = _fake_models_response(["qwen3.5-4b-mlx"])
        client = self._client(vlm="qwen3.5:4b-mlx")
        with pytest.raises(RuntimeError):
            client.test_connection()

    @patch("nuclearcutter.utils.llm_client.requests.get")
    def test_text_model_also_validated(self, mock_get: MagicMock):
        """Both VLM and text models should be checked against the server list."""
        mock_get.return_value = _fake_models_response(["qwen3.5-4b-mlx"])
        client = self._client(vlm="qwen3.5-4b-mlx", text="wrong-model-name")
        with pytest.raises(RuntimeError, match="text model.*not found"):
            client.test_connection()

    @patch("nuclearcutter.utils.llm_client.requests.get")
    def test_skips_validation_when_server_unreachable(self, mock_get: MagicMock):
        """If the server doesn't respond to /v1/models, validation is skipped."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        client = self._client()
        client.test_connection()  # should not raise

    @patch("nuclearcutter.utils.llm_client.requests.get")
    def test_skips_validation_when_list_empty(self, mock_get: MagicMock):
        """If /v1/models returns an empty list, validation is skipped."""
        mock_get.return_value = _fake_models_response([])
        self._client().test_connection()  # no exception
