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
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"  # Ollama default
    vlm_model: str | None = None  # no default — user must pass --vlm-model
    text_model: str | None = None  # no default — user must pass --text-model
    api_key: str = "not-needed"  # most local servers ignore this but the client requires *something*
    timeout: int = 90  # text-only requests; kept short so slow LLM checks fall back to the wordlist quickly
    vision_timeout: int = 8000  # VLM requests with images can be much slower
    # Bounds on generation length. Reasoning models will happily "think" for
    # thousands of tokens (or forever, if unbounded) before emitting an answer;
    # a firm cap guarantees every request terminates so a full scan can't hang
    # indefinitely. Tokens are free locally, but time isn't.
    text_max_tokens: int = 6000
    vision_max_tokens: int = 9000
    # Vision requests send frames as JPEG; downscaling them is the single
    # biggest speed lever for local VLM inference (~6x faster on an M-series Mac
    # at ~480px with no measurable accuracy loss for scene classification).
    # Frames larger than this many pixels (per image) are resized down.
    vision_max_pixels: int = 480 * 360
    # Reasoning models spend most of their budget "thinking" before answering.
    # Instructing them to answer directly cuts latency dramatically and
    # prevents empty responses when the thinking chain hits the token cap.
    system_prompt: str = (
        "Answer directly and concisely. Do not provide lengthy reasoning or "
        "chain-of-thought; output only the requested result."
    )


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def _post(self, payload: dict, timeout: int = None) -> dict:
        timeout = timeout if timeout is not None else self.config.timeout
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
        return resp.json()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

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
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": self.config.vision_max_tokens,
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
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": self.config.text_max_tokens,
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


def _parse_json_loose(raw: str) -> dict:
    """Some local models wrap JSON in markdown fences despite json_mode. Strip and parse defensively."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
