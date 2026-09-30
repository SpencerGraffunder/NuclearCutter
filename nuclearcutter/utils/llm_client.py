"""
Thin client for talking to a local OpenAI-compatible inference server
(Ollama's /v1 endpoint, LM Studio's local server, or anything else that
speaks the same API). NuclearCutter deliberately does not hard-depend on any
specific inference backend — see docs/SPEC.md section 4.1.

Both the vision-language calls (nudity/intimate scene confirmation +
description) and the text-only calls (foul language context check) go
through this same client, just with different models configured.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from nuclearcutter.prompts import get_prompt


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"  # Ollama default
    vlm_model: str | None = None  # no default — user must pass --vlm-model
    text_model: str | None = None  # no default — user must pass --text-model
    # The SUMMARY model is a SEPARATE, usually larger model used only at RENDER
    # time (not the every-frame sweep/confirm pass), to write the on-screen text
    # that replaces blurred/blacked footage and the small captions on muted
    # audio. Because it runs once per segment rather than every frame, a big
    # model is affordable. When left None, the VLM is used for frame-based
    # summaries and the text model for text-only ones (graceful fallback).
    summary_model: str | None = None
    # How many frames to sample across each blurred/blacked segment for the
    # summary pass. More frames = more accurate, more vision tokens. 12 is the
    # default; 0 disables frames (text-only summaries).
    summary_frames: int = 12
    summary_max_tokens: int = 6000  # generation cap for summary-model requests
    # Fallback LOADED context window (tokens) assumed for the summary model when
    # the server doesn't report one. The summary prompt must fit this: we warn
    # when frames + dialogue + template risk exceeding it. The user mentioned
    # ~20-30k because VRAM limits how large a model can be loaded.
    summary_max_context: int = 30000
    # Try the frame-based (vision) summary first, falling back to text-only.
    summary_vision: bool = True
    api_key: str = "not-needed"  # most local servers ignore this but the client requires *something*
    timeout: int = 90  # text-only requests; kept short so slow LLM checks fall back to the wordlist quickly
    vision_timeout: int = 8000  # VLM requests with images can be much slower
    # Bounds on generation length. Reasoning models will happily "think" for
    # thousands of tokens (or forever, if unbounded) before emitting an answer;
    # a firm cap guarantees every request terminates so a full scan can't hang
    # indefinitely. Tokens are free locally, but time isn't.
    text_max_tokens: int = 6000
    vision_max_tokens: int = 9000
    # Qwen3-family models are reasoning models that "think" before answering by
    # default, emitting thousands of tokens (and taking minutes per request)
    # even for simple yes/no classification. `enable_thinking=False` is sent in
    # the request so servers that support it (LM Studio, mlx-vlm) skip the
    # reasoning chain entirely — a huge latency win for our short JSON answers.
    enable_thinking: bool = False
    # Vision requests send frames as JPEG; downscaling them is the single
    # biggest speed lever for local VLM inference (~6x faster on an M-series Mac
    # at ~480px with no measurable accuracy loss for scene classification).
    # Frames larger than this many pixels (per image) are resized down.
    vision_max_pixels: int = 480 * 360
    # Reasoning models spend most of their budget "thinking" before answering.
    # Instructing them to answer directly cuts latency dramatically and
    # prevents empty responses when the thinking chain hits the token cap.
    # None = use the system_prompt from prompts.json (the shared prompt file).
    system_prompt: str | None = None


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        # Optional hook: called after every successful request with
        # (response_json, elapsed_seconds). The web GUI uses it to track
        # tokens-per-prompt, generation speed, and pp speed.
        self.usage_callback = None
        # Optional hook: called with (payload, response_json) for every
        # request. The web GUI uses it to stream prompts/responses to the
        # terminal when "show prompts and responses" is enabled.
        self.request_log_callback = None

    def _post(self, payload: dict, timeout: int = None) -> dict:
        timeout = timeout if timeout is not None else self.config.timeout
        t0 = time.monotonic()
        resp = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        if not resp.ok:
            body = resp.text[:2000]  # server error details
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} from {resp.url}\n"
                f"Response body: {body}",
                response=resp,
            )
        data = resp.json()
        if self.usage_callback is not None:
            try:
                self.usage_callback(data, time.monotonic() - t0)
            except Exception:
                pass  # never let stats tracking break a request
        if self.request_log_callback is not None:
            try:
                self.request_log_callback(payload, data)
            except Exception:
                pass  # never let prompt logging break a request
        return data

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _system_prompt(self) -> str:
        """The system prompt for model requests.

        Resolved from prompts.json (the shared prompt file — the single source
        of truth for what the models are asked), unless the caller overrode it
        on the LLMConfig.
        """
        return self.config.system_prompt or get_prompt("system_prompt")

    def list_models(self) -> list[str]:
        """Return the model ids advertised by the configured server's /v1/models.

        Used by the web GUI's "use existing model server" flow: after the user
        types an IP/base URL, the GUI calls this (via the server endpoint) to
        populate the model dropdown. Returns [] when unreachable.
        """
        try:
            resp = requests.get(
                f"{self.config.base_url}/models",
                headers=self._headers(),
                timeout=min(self.config.timeout, 10),
            )
            resp.raise_for_status()
        except requests.RequestException:
            return []
        data = resp.json().get("data", [])
        return [m["id"] for m in data if "id" in m]

    def model_context_length(self, model: str | None) -> int | None:
        """Return the server-reported context length (tokens) for `model`, or None.

        This is what the user asked about: not the model's theoretical max, but
        the window it is LOADED with — the thing that actually limits a prompt.
        OpenAI-compatible backends expose it in different places, so we probe a
        few:

          * GET /v1/models entries: LM Studio and vLLM include `max_model_len`;
            some servers use `context_length` / `max_context_length`. LM Studio
            reports the model's max context (the largest it *can* be loaded at),
            which is a safe upper bound when we can't read the actual loaded size.
          * Ollama: `context_length` from POST /api/show on the native base
            (the `/v1` suffix is stripped).

        Returns None when nothing usable is reported — callers fall back to a
        safe default (LLMConfig.summary_max_context) and warn conservatively.
        """
        if not model:
            return None

        def _as_int(val) -> int | None:
            if val is None or isinstance(val, bool):
                return None
            if isinstance(val, str):
                return int(val) if val.isdigit() else None
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
            return None

        def _id_matches(avail_id: str) -> bool:
            if model == avail_id:
                return True
            return avail_id.rstrip("/").endswith("/" + model) or avail_id.endswith(model)

        # 1) GET /v1/models entry metadata (LM Studio / vLLM / most OpenAI-compatible).
        try:
            resp = requests.get(
                f"{self.config.base_url}/models",
                headers=self._headers(),
                timeout=min(self.config.timeout, 10),
            )
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                if not _id_matches(str(entry.get("id", ""))):
                    continue
                for key in ("max_model_len", "context_length", "max_context_length",
                            "llama.context_length"):
                    val = _as_int(entry.get(key))
                    if val:
                        return val
        except (requests.RequestException, ValueError):
            pass

        # 2) Ollama's native /api/show (base_url minus any trailing /v1).
        native = self.config.base_url.rstrip("/")
        if native.endswith("/v1"):
            native = native[: -len("/v1")]
        try:
            resp = requests.post(
                f"{native}/api/show",
                json={"model": model},
                headers=self._headers(),
                timeout=min(self.config.timeout, 10),
            )
            resp.raise_for_status()
            val = _as_int((resp.json().get("model_info") or {}).get("context_length"))
            if val:
                return val
        except (requests.RequestException, ValueError):
            pass

        return None

    def test_connection(self) -> None:
        """Verify the local server is reachable and the configured models exist.

        Raises RuntimeError with a helpful message listing available models
        if either the VLM or text model is not found on the server. This
        catches typos (e.g. "qwen3.5:4b-mlx" instead of "qwen3.5-4b-mlx")
        before the scan wastes hours.
        """
        try:
            resp = requests.get(
                f"{self.config.base_url}/models",
                headers=self._headers(),
                timeout=min(self.config.timeout, 10),
            )
            resp.raise_for_status()
        except requests.RequestException:
            # Server unreachable or doesn't support /v1/models — skip validation.
            return

        available = [m["id"] for m in resp.json().get("data", []) if "id" in m]
        if not available:
            return  # empty list, can't validate

        for label, model in [("VLM", self.config.vlm_model), ("text", self.config.text_model)]:
            if model is None:
                # No default models in the code — if the caller didn't supply
                # one, there's nothing to validate here; a clear error is raised
                # by the CLI before any work is attempted.
                continue
            # Match by exact id OR by "the model id ends with our model name".
            # The mlx-vlm server serves the model id as its full filesystem path,
            # so `--vlm-model Qwen3.5-9B-MLX-4bit` must match an id of
            # `/Users/.../Qwen3.5-9B-MLX-4bit`.
            def _matches(avail_id: str) -> bool:
                if model == avail_id:
                    return True
                return avail_id.rstrip("/").endswith("/" + model) or avail_id.endswith(model)

            if not any(_matches(a) for a in available):
                suggestions = [a for a in available if model.replace(":", "-") in a or model.split(":")[0] in a]
                hint = (
                    f"\n  Did you mean one of these?\n    " + "\n    ".join(suggestions[:5])
                    if suggestions else ""
                )
                raise RuntimeError(
                    f"{label} model {model!r} not found on server at {self.config.base_url}.\n"
                    f"Available models:\n    " + "\n    ".join(available)
                    + hint
                )

    def _encode_image(self, image_path: Path) -> str:
        """Read an image, downscale if needed, and return as a base64 JPEG data URI.

        Downscaling to ~480px is the biggest speed lever for local VLM inference
        (see `vision_max_pixels`): it cuts HTTP body size and prefill cost ~6x
        without meaningfully impacting scene-classification accuracy.
        """
        max_pixels = self.config.vision_max_pixels
        img = Image.open(image_path)
        # Resize if the image is larger than max_pixels (keeping aspect ratio).
        if img.size[0] * img.size[1] > max_pixels:
            ratio = (max_pixels / (img.size[0] * img.size[1])) ** 0.5
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def vision_query(self, prompt: str, image_paths: list[Path], json_mode: bool = False) -> str:
        """Send a prompt + one or more images to the configured VLM. Returns raw text response."""
        content = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            b64 = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        payload = {
            "model": self.config.vlm_model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": self.config.vision_max_tokens,
            # Vision classification (sweep / confirm / summary) never needs a
            # reasoning chain — thinking just adds latency and can blow the token
            # cap. ALWAYS disable it here, independent of config.enable_thinking
            # (which only governs the text path). LM Studio honours
            # `chat_template_kwargs` for Qwen3-family models; the top-level
            # `enable_thinking` is ignored on some setups, so send BOTH to
            # actually skip the reasoning chain.
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # Most local inference servers (LM Studio, Ollama, etc.) do NOT support
        # response_format/json_mode for multimodal (vision) requests, even when
        # they support it for text-only. We skip json_mode for vision calls and
        # rely on _parse_json_loose to extract JSON from the text response.

        result = self._post(payload, timeout=self.config.vision_timeout)
        return result["choices"][0]["message"]["content"]

    def text_query(self, prompt: str, json_mode: bool = False) -> str:
        """Send a text-only prompt to the configured text LLM. Returns raw text response."""
        payload = {
            "model": self.config.text_model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": self.config.text_max_tokens,
            "enable_thinking": self.config.enable_thinking,
            # LM Studio honours `chat_template_kwargs` for Qwen3-family models;
            # the top-level `enable_thinking` is ignored on some setups, so send
            # both to actually skip the reasoning chain.
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
        }
        if json_mode:
            # Some servers (OpenAI-compatible) accept json_object; LM Studio
            # requires json_schema or text and rejects json_object with 400.
            # Try json_object first, then fall back to plain text (the prompt
            # already demands strict JSON and callers parse loosely).
            payload["response_format"] = {"type": "json_object"}
            try:
                result = self._post(payload)
            except requests.HTTPError as exc:
                if "response_format" in str(exc) and exc.response is not None:
                    payload.pop("response_format", None)
                    result = self._post(payload)
                else:
                    raise
            raw = result["choices"][0]["message"]["content"]
            # mlx-vlm's server ACCEPTS json_object (HTTP 200) but returns an
            # empty {} instead of generating — so no HTTPError fires and the
            # retry above never triggers. Detect that and retry as plain text
            # (the prompt demands strict JSON; _parse_json_loose handles it).
            if not raw.strip() or raw.strip() == "{}":
                payload.pop("response_format", None)
                result = self._post(payload)
            return result["choices"][0]["message"]["content"]

        result = self._post(payload)
        return result["choices"][0]["message"]["content"]

    def vision_query_json(self, prompt: str, image_paths: list[Path]) -> dict:
        raw = self.vision_query(prompt, image_paths, json_mode=True)
        return _parse_json_loose(raw)

    def text_query_json(self, prompt: str) -> dict:
        raw = self.text_query(prompt, json_mode=True)
        return _parse_json_loose(raw)

    # ------------------------------------------------------------------
    # Summary-model requests (render-time per-segment descriptions + captions)
    # ------------------------------------------------------------------

    def summary_query(self, prompt: str, image_paths: list[Path]) -> tuple[str, bool]:
        """Run the SUMMARY model. Returns (raw_text, used_vision).

        The summary model is a separate, larger model used once per blurred/
        blacked/muted segment at render time (see `summary_model` on LLMConfig).
        It is NOT the every-frame VLM.

        When images are provided and vision is enabled, the frames are attached
        and the request is sent to the summary model (falling back to the VLM
        if no summary model is configured). If the model/backend rejects or
        fails the vision request — e.g. the user picked a text-only model as
        the summary model — the SAME prompt is retried text-only so a
        description/caption can still be produced from the transcript. Returns
        the raw response text; callers parse it leniently.
        """
        if image_paths and self.config.summary_vision:
            try:
                return self._summary_vision(prompt, image_paths), True
            except Exception:
                # Vision failed for ANY reason (text-only summary model, HTTP
                # error, timeout, malformed response, server hiccup) — retry
                # the same prompt text-only so a caption/description still
                # appears. This pass is best-effort; the caller degrades too.
                pass
        return self._summary_text(prompt), False

    def _summary_model(self, vision: bool) -> str:
        """The model to use for a summary request: the configured summary model,
        else the VLM for vision requests / the text model for text-only ones."""
        if self.config.summary_model:
            return self.config.summary_model
        return self.config.vlm_model if vision else self.config.text_model

    def _summary_payload(self, prompt: str, model: str, content) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": self.config.summary_max_tokens,
            # Summary descriptions (vision or text-only) are short, direct
            # outputs — never enable thinking here either.
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _summary_vision(self, prompt: str, image_paths: list[Path]) -> str:
        content = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            b64 = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        payload = self._summary_payload(prompt, self._summary_model(vision=True), content)
        result = self._post(payload, timeout=self.config.vision_timeout)
        return result["choices"][0]["message"]["content"]

    def _summary_text(self, prompt: str) -> str:
        payload = self._summary_payload(prompt, self._summary_model(vision=False), prompt)
        result = self._post(payload, timeout=self.config.timeout)
        return result["choices"][0]["message"]["content"]


def _parse_json_loose(raw: str) -> dict:
    """Some local models wrap JSON in markdown fences despite json_mode. Strip and parse defensively."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
