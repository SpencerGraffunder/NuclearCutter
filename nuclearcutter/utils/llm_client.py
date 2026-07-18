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
import json
from dataclasses import dataclass
from pathlib import Path

import requests


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
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

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
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

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
