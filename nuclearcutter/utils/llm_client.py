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
    base_url: str = "http://localhost:11434/v1"  # Ollama default
    vlm_model: str = "qwen2.5-vl:7b"
    text_model: str = "qwen2.5:7b"
    api_key: str = "not-needed"  # most local servers ignore this but the client requires *something*
    timeout: int = 120


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def _post(self, payload: dict) -> dict:
        resp = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout,
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
            if model not in available:
                suggestions = [m for m in available if model.replace(":", "-") in m or model.split(":")[0] in m]
                hint = (
                    f"\n  Did you mean one of these?\n    " + "\n    ".join(suggestions[:5])
                    if suggestions else ""
                )
                raise RuntimeError(
                    f"{label} model {model!r} not found on server at {self.config.base_url}.\n"
                    f"Available models:\n    " + "\n    ".join(available)
                    + hint
                )

    @staticmethod
    def _encode_image(image_path: Path, max_pixels: int = 1920 * 1080 // 2) -> str:
        """Read an image, resize if needed, and return as a base64 JPEG data URI.

        Resizing to ~half resolution reduces the HTTP body size drastically
        (PNG frames can be 2-4 MB each; JPEG at half-res is ~100-300 KB)
        without meaningfully impacting VLM classification accuracy.
        """
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
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
        }
        # Most local inference servers (LM Studio, Ollama, etc.) do NOT support
        # response_format/json_mode for multimodal (vision) requests, even when
        # they support it for text-only. We skip json_mode for vision calls and
        # rely on _parse_json_loose to extract JSON from the text response.

        result = self._post(payload)
        return result["choices"][0]["message"]["content"]

    def text_query(self, prompt: str, json_mode: bool = False) -> str:
        """Send a text-only prompt to the configured text LLM. Returns raw text response."""
        payload = {
            "model": self.config.text_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

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
