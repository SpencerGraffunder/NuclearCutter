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


class TestThinkingDisabled:
    """Qwen3-family models 'think' by default — every request (scan, confirm,
    benchmark) must explicitly disable the reasoning chain, or a multi-hour
    scan balloons into multi-hour thinking."""

    @staticmethod
    def _completion(content: str = "ok"):
        resp = MagicMock(spec=requests.Response)
        resp.ok = True
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return resp

    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_vision_query_disables_thinking(self, mock_post: MagicMock, tmp_path):
        """The VLM sweep/confirm/benchmark path sends enable_thinking=False."""
        from PIL import Image

        png = tmp_path / "f.png"
        Image.new("RGB", (4, 4), "red").save(png)
        mock_post.return_value = self._completion()
        client = LLMClient(LLMConfig(base_url="http://x/v1", vlm_model="m", text_model="m"))
        client.vision_query("prompt", [png])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["enable_thinking"] is False
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_vision_thinking_off_even_when_config_enabled(self, mock_post: MagicMock, tmp_path):
        """Vision requests disable thinking UNCONDITIONALLY — even if a caller
        sets enable_thinking=True on the config (that knob only governs text)."""
        from PIL import Image

        png = tmp_path / "f.png"
        Image.new("RGB", (4, 4), "red").save(png)
        mock_post.return_value = self._completion()
        cfg = LLMConfig(base_url="http://x/v1", vlm_model="m", text_model="m",
                        enable_thinking=True)
        client = LLMClient(cfg)
        client.vision_query("prompt", [png])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["enable_thinking"] is False
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_text_query_disables_thinking(self, mock_post: MagicMock):
        """The foul-language context check path sends enable_thinking=False."""
        mock_post.return_value = self._completion()
        client = LLMClient(LLMConfig(base_url="http://x/v1", vlm_model="m", text_model="m"))
        client.text_query("prompt")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["enable_thinking"] is False
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_request_log_callback_fires_with_payload(self, mock_post: MagicMock):
        """The show-prompts hook receives every request's payload + response."""
        mock_post.return_value = self._completion("some answer")
        client = LLMClient(LLMConfig(base_url="http://x/v1", vlm_model="m", text_model="m"))
        seen = []
        client.request_log_callback = lambda payload, data: seen.append((payload, data))
        client.text_query("hello")
        assert len(seen) == 1
        payload, data = seen[0]
        assert payload["messages"][-1]["content"] == "hello"
        assert data["choices"][0]["message"]["content"] == "some answer"


def _models_response(entries: list[dict]):
    """A mock /v1/models response whose entries carry metadata like LM Studio."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.ok = True
    resp.json.return_value = {"data": entries}
    resp.raise_for_status = lambda: None
    return resp


def _client(base="http://localhost:9999/v1", **kw):
    return LLMClient(LLMConfig(base_url=base, vlm_model="m", text_model="m", **kw))


class TestModelContextLength:
    """model_context_length should read the server-reported LOADED context
    window for a model (LM Studio's max_model_len, etc.), or return None."""

    @patch("nuclearcutter.utils.llm_client.requests.get")
    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_reads_max_model_len(self, mock_post, mock_get):
        mock_get.return_value = _models_response([{"id": "m", "max_model_len": 32768}])
        assert _client().model_context_length("m") == 32768
        mock_post.assert_not_called()  # never reached the Ollama fallback

    @patch("nuclearcutter.utils.llm_client.requests.get")
    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_reads_alternate_keys(self, mock_post, mock_get):
        mock_get.return_value = _models_response([{"id": "m", "context_length": 8192}])
        assert _client().model_context_length("m") == 8192
        mock_get.return_value = _models_response([{"id": "m", "max_context_length": 4096}])
        assert _client().model_context_length("m") == 4096

    @patch("nuclearcutter.utils.llm_client.requests.get")
    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_matches_model_by_suffix(self, mock_post, mock_get):
        """mlx-vlm serves the model id as a full filesystem path."""
        mock_get.return_value = _models_response([{"id": "/Users/me/models/MyModel", "max_model_len": 16384}])
        assert _client().model_context_length("MyModel") == 16384

    @patch("nuclearcutter.utils.llm_client.requests.get")
    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_unknown_returns_none(self, mock_post, mock_get):
        mock_get.return_value = _models_response([{"id": "m"}])
        # Ollama fallback also reports nothing.
        resp = MagicMock(spec=requests.Response)
        resp.ok = True
        resp.status_code = 200
        resp.json.return_value = {"model_info": {}}
        resp.raise_for_status = lambda: None
        mock_post.return_value = resp
        assert _client().model_context_length("m") is None

    @patch("nuclearcutter.utils.llm_client.requests.get")
    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_unreachable_returns_none(self, mock_post, mock_get):
        mock_get.side_effect = requests.ConnectionError("down")
        mock_post.side_effect = requests.ConnectionError("down")
        assert _client().model_context_length("m") is None

    @patch("nuclearcutter.utils.llm_client.requests.get")
    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_ollama_show_fallback(self, mock_post, mock_get):
        """Ollama reports context via /api/show on the native base (no /v1)."""
        mock_get.side_effect = requests.ConnectionError("no /v1/models")
        resp = MagicMock(spec=requests.Response)
        resp.ok = True
        resp.status_code = 200
        resp.json.return_value = {"model_info": {"context_length": 24576}}
        resp.raise_for_status = lambda: None
        mock_post.return_value = resp
        assert _client(base="http://localhost:11434/v1").model_context_length("llama3") == 24576
        # The native base must not keep the /v1 suffix.
        assert mock_post.call_args[0][0] == "http://localhost:11434/api/show"


class TestSummaryQuery:
    """summary_query tries the frame-based vision path first and retries
    text-only on failure — so a text-only summary model still yields text."""

    @staticmethod
    def _png(tmp_path):
        from PIL import Image

        png = tmp_path / "f.png"
        Image.new("RGB", (4, 4), "red").save(png)
        return png

    def test_vision_success_no_fallback(self, tmp_path):
        client = _client(summary_model="s")
        png = self._png(tmp_path)
        with patch.object(client, "_summary_vision", return_value="vision answer") as mv, \
             patch.object(client, "_summary_text") as mt:
            raw, used_vision = client.summary_query("prompt", [png])
        assert raw == "vision answer"
        assert used_vision is True
        mt.assert_not_called()

    def test_vision_failure_falls_back_to_text(self, tmp_path):
        client = _client(summary_model="s")
        png = self._png(tmp_path)
        with patch.object(client, "_summary_vision", side_effect=RuntimeError("no vision")) as mv, \
             patch.object(client, "_summary_text", return_value="text answer") as mt:
            raw, used_vision = client.summary_query("prompt", [png])
        assert raw == "text answer"
        assert used_vision is False
        mv.assert_called_once()
        mt.assert_called_once()

    def test_no_images_is_text_only(self):
        client = _client(summary_model="s")
        with patch.object(client, "_summary_text", return_value="text answer") as mt:
            raw, used_vision = client.summary_query("prompt", [])
        assert raw == "text answer"
        assert used_vision is False
        mt.assert_called_once()

    @patch("nuclearcutter.utils.llm_client.requests.post")
    def test_vision_uses_summary_model_not_vlm(self, mock_post, tmp_path):
        mock_post.return_value = TestThinkingDisabled._completion("ok")
        client = LLMClient(LLMConfig(base_url="http://x/v1", vlm_model="small-vlm",
                                     text_model="t", summary_model="big-summary"))
        png = self._png(tmp_path)
        raw, used_vision = client.summary_query("prompt", [png])
        assert used_vision is True
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "big-summary"
        # Frames are attached as image parts.
        assert any(p.get("type") == "image_url" for p in payload["messages"][-1]["content"])
