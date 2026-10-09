"""
Talks to the inference backends NuclearCutter uses — which are ALWAYS
already-running, remote (or local-but-external) servers. NuclearCutter never
spawns its own inference server:

- OpenAI-compatible /v1 server (llama.cpp, LM Studio, Ollama, vLLM, ...) —
  one server can serve the VLM, the text model, and the summary model at once
  (each selected by model id from the /v1/models dropdown).
- whisper.cpp transcription server (`whisper-server`) — one model loaded at a
  time, hot-swappable via its /load endpoint; the model dropdown lists the
  .bin files in a directory that lives on the same machine as that server.

All configuration is a base URL + model id (see docs/SPEC.md section 4.1).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

import requests

# OpenAI-compatible base URL default (llama.cpp/LM Studio/Ollama /v1).
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"

# whisper.cpp server defaults for this deployment.
WHISPER_DEFAULT_BASE_URL = "http://127.0.0.1:8081"
WHISPER_DEFAULT_INFERENCE_PATH = "/audio/transcriptions"
# Where the whisper-server's .bin model files live (on the machine running the
# server). The dropdown lists these; /load hot-swaps the loaded model.
WHISPER_DEFAULT_MODELS_DIR = "/home/graffunder/whisper-server/src/models"

GGML_BIN_RE = re.compile(r"ggml-[A-Za-z0-9._\-]+\.bin$")


@dataclass
class WhisperConfig:
    base_url: str = WHISPER_DEFAULT_BASE_URL  # e.g. http://192.168.4.164:8081
    inference_path: str = WHISPER_DEFAULT_INFERENCE_PATH  # the server's --inference-path
    model: str = ""  # model id (file stem, e.g. "ggml-base.en")
    models_dir: str = WHISPER_DEFAULT_MODELS_DIR  # for the model dropdown listing
    timeout: int = 3600  # a 3-hour film is one multipart request; be generous


class ModelServerError(RuntimeError):
    pass


# ------------------------------------------------------------------ OpenAI /v1

def is_server_up(base_url: str) -> bool:
    """Return True if an OpenAI-compatible server is answering on base_url."""
    try:
        r = requests.get(f"{base_url}/models", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


def list_models(base_url: str) -> list[str]:
    """Model ids advertised by base_url's /v1/models ([] when unreachable)."""
    try:
        r = requests.get(f"{base_url}/models", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        return []
    try:
        data = r.json().get("data", [])
        return [m["id"] for m in data if "id" in m]
    except ValueError:
        return []


def wait_for_server(base_url: str, timeout: float = 120.0, interval: float = 1.0) -> bool:
    """Poll base_url until the server answers or timeout elapses."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_up(base_url):
            return True
        time.sleep(interval)
    return False


def require_server_up(base_url: str, label: str = "model server") -> None:
    """Raise a clear error when the configured OpenAI-compatible server is
    unreachable — the one preflight check NuclearCutter does instead of
    starting the server itself."""
    if not is_server_up(base_url):
        raise ModelServerError(
            f"No {label} answering at {base_url} (expected an OpenAI-compatible "
            f"/v1 server — llama.cpp, LM Studio, Ollama, vLLM, ...). "
            f"Start it or fix the base URL."
        )


# ------------------------------------------------------------------ whisper.cpp

def whisper_is_up(cfg: WhisperConfig | str) -> bool:
    """True if the whisper server's /health answers ok (or "loading model").

    Accepts a WhisperConfig or a plain base-url string.
    """
    base_url = cfg.base_url if isinstance(cfg, WhisperConfig) else cfg
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=3)
        return r.status_code in (200, 503)
    except requests.RequestException:
        return False


def whisper_list_models(cfg: WhisperConfig) -> list[str]:
    """List whisper model files available for the dropdown.

    The whisper.cpp server has no model-list endpoint, so we list the .bin
    files in `cfg.models_dir` — a directory on the machine that runs the
    server (it must be readable by the NuclearCutter service user; on this
    box it is the same machine). Returns model ids (file stems, e.g.
    "ggml-base.en") sorted for stable dropdown order.
    """
    try:
        names = os.listdir(cfg.models_dir)
    except OSError:
        return []
    ids = []
    for name in names:
        m = GGML_BIN_RE.fullmatch(name)
        if m:
            ids.append(name[:-4])  # strip .bin -> "ggml-base.en"
    return sorted(ids)


def whisper_load_model(cfg: WhisperConfig, model_id: str) -> str:
    """Hot-swap the loaded whisper model via the server's /load endpoint.

    `model_id` is a dropdown id (file stem) or an explicit path; it is sent
    as the `model` form field, which the server resolves against its own
    filesystem. Returns the server's response text. Raises ModelServerError
    on failure. NOTE: /load restarts the server's model under its mutex —
    other clients of that server (e.g. Storyteller) see a brief interruption.
    """
    # The whisper-server /load handler reads a multipart FILE field named
    # `model` and loads the file whose CONTENT is the model path string.
    # An absolute path is sent verbatim; a dropdown id (file stem, e.g.
    # "ggml-base.en") is resolved to an absolute path inside models_dir,
    # because the server must be able to find the file on its own filesystem.
    path = model_id
    if not path.startswith("/"):
        candidate = os.path.join(cfg.models_dir, model_id + ".bin")
        if not os.path.exists(candidate):
            raise ModelServerError(
                f"whisper model {model_id!r} not found (looked for {candidate}). "
                f"Check the whisper server's model directory."
            )
        path = candidate
    try:
        r = requests.post(
            f"{cfg.base_url.rstrip('/')}/load",
            files={"model": ("model.bin", path.encode("utf-8"), "text/plain")},
            timeout=600,
        )
    except requests.RequestException as exc:
        raise ModelServerError(f"whisper /load request failed: {exc}") from exc
    if not r.ok:
        raise ModelServerError(
            f"whisper /load failed ({r.status_code}): {r.text[:300]}\n"
            "NOTE: a failed /load can leave the server's internal state stuck; "
            "restart whisper-server if /health stops answering."
        )
    return r.text.strip()
