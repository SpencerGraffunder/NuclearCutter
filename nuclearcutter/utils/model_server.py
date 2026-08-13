"""
Manages the local VLM/text inference backend.

Three backends:
- `mlx-vlm` (default): NuclearCutter spawns its own `mlx_vlm.server` on
  localhost:1234 and serves the configured model in-process. Fast, self-contained,
  and optimized for this workload (downscaled images, KV-cache quantization,
  continuous batching). No external app needed.
- `llama.cpp`: spawns the locally-installed `llama-server` binary serving a
  .gguf model on the same port. Optionally with `--mmproj` for vision.
- `standalone`: talk to an already-running OpenAI-compatible server (LM Studio,
  Ollama, a manually-started mlx-vlm/llama.cpp server, etc.) via `base_url`.

All three speak the same OpenAI-compatible `/v1` API on port 1234 by default,
so switching backends does not require changing any other configuration.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# The model path LM Studio was configured with, used by default for the mlx-vlm
# backend so the exact same model file is served. Overridable via config.
DEFAULT_MLX_MODEL_PATH = "/Users/spencer/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"

# Where the mlx-vlm server listens. Kept identical to the old LM Studio default
# so `base_url` never needs to change between backends.
MLX_SERVER_HOST = "127.0.0.1"
MLX_SERVER_PORT = 1234
MLX_BASE_URL = f"http://{MLX_SERVER_HOST}:{MLX_SERVER_PORT}/v1"

# Speed knobs for the mlx-vlm server (see mlx_vlm.server.cli).
MLX_KV_BITS = 4  # KV-cache quantization (TurboQuant-ish); ~3% faster, low cost
MLX_MAX_TOKENS = 2048  # per-request generation cap on the server side
# Max KV cache (context) in tokens. The model's config advertises 256K, but our
# largest call (confirm: ~10.1K input + 2048 max output) is ~12.2K. Capping here
# keeps VRAM down — each KV token costs meaningful unified memory at 9B scale —
# while leaving comfortable headroom. 16K safely covers the worst case.
MLX_MAX_KV_SIZE = 16384

# llama.cpp server context size (same rationale as MLX_MAX_KV_SIZE).
LLAMA_CONTEXT = 16384


@dataclass
class ModelServerConfig:
    backend: str = "mlx-vlm"  # "mlx-vlm" | "llama.cpp" | "standalone"
    model_path: str = DEFAULT_MLX_MODEL_PATH  # mlx-vlm backend: HF/MLX model dir; llama.cpp: .gguf file
    mmproj_path: str = ""  # llama.cpp vision projector (.gguf), enables image input
    base_url: str = MLX_BASE_URL  # standalone backend (and where mlx-vlm listens)
    auto_install: bool = True  # pip-install mlx-vlm if missing (mlx-vlm backend)


class ModelServerError(RuntimeError):
    pass


def _ensure_mlx_vlm_installed() -> None:
    """Import mlx_vlm, installing it via pip if missing (when allowed)."""
    try:
        import mlx_vlm  # noqa: F401
        return
    except ImportError:
        pass

    raise ModelServerError(
        "mlx-vlm is not installed. Install it into this environment with:\n"
        "    pip install mlx-vlm\n"
        "or set model_backend = \"standalone\" in config.toml to use an "
        "already-running server (LM Studio / Ollama / a manual mlx-vlm server)."
    )


def _llama_model_id(model_path: str) -> str:
    """The model id llama.cpp's /v1/models will report for a .gguf file.

    Uses the file stem (e.g. "Qwen3-VL-8B-Instruct-Q4_K_M.gguf" -> that stem),
    so the GUI can pre-select it as the VLM/text model id.
    """
    return Path(model_path).stem


def _command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def is_server_up(base_url: str) -> bool:
    """Return True if an OpenAI-compatible server is answering on base_url."""
    try:
        r = requests.get(f"{base_url}/models", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


def wait_for_server(base_url: str, timeout: float = 120.0, interval: float = 1.0) -> bool:
    """Poll base_url until the server answers or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_up(base_url):
            return True
        time.sleep(interval)
    return False


def _python() -> str:
    """Path to the current Python interpreter (venv-aware)."""
    return sys.executable


def start_mlx_vlm_server(config: ModelServerConfig, log_path: Path | None = None) -> subprocess.Popen:
    """Spawn `python -m mlx_vlm.server` serving the configured model on the
    shared port, and wait until it answers. Returns the Popen handle so the
    caller can terminate it when done."""
    if not config.model_path:
        raise ModelServerError("model_path is required for the mlx-vlm backend (config.toml: model_path)")
    if not Path(config.model_path).exists():
        raise ModelServerError(
            f"model_path not found: {config.model_path}\n"
            "Set model_path in config.toml to the directory containing the MLX "
            "model (e.g. the LM Studio MLX folder you already use)."
        )

    _ensure_mlx_vlm_installed()

    log_target = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        [
            _python(), "-m", "mlx_vlm.server",
            "--host", MLX_SERVER_HOST,
            "--port", str(MLX_SERVER_PORT),
            "--model", config.model_path,
            "--kv-bits", str(MLX_KV_BITS),
            "--max-tokens", str(MLX_MAX_TOKENS),
            "--max-kv-size", str(MLX_MAX_KV_SIZE),
        ],
        stdout=log_target,
        stderr=subprocess.STDOUT,
    )

    if not wait_for_server(config.base_url, timeout=180):
        proc.terminate()
        raise ModelServerError(
            f"mlx-vlm server did not start on {config.base_url}. Check the log: {log_path}"
        )
    return proc


def start_llama_cpp_server(config: ModelServerConfig, log_path: Path | None = None) -> subprocess.Popen:
    """Spawn the local `llama-server` binary serving a .gguf model, and wait
    until it answers on the shared OpenAI-compatible port.

    Returns the Popen handle so the caller can terminate it when done. The
    model id the server reports is the .gguf file's stem (see _llama_model_id),
    which is also what the GUI pre-selects as vlm/text model.
    """
    binary = shutil.which("llama-server") or shutil.which("llama-cli")
    if not binary:
        raise ModelServerError(
            "llama-server not found on PATH. Install llama.cpp first, e.g.\n"
            "    brew install llama.cpp\n"
            "or pick another backend in the web GUI."
        )
    if not config.model_path or not Path(config.model_path).exists():
        raise ModelServerError(
            f"model_path not found: {config.model_path!r}\n"
            "Set model_path to a .gguf file for the llama.cpp backend."
        )

    cmd = [
        binary, "-m", config.model_path,
        "--host", MLX_SERVER_HOST,
        "--port", str(MLX_SERVER_PORT),
        "--alias", _llama_model_id(config.model_path),
        "-c", str(LLAMA_CONTEXT),
        "-ngl", "99",  # offload all layers to Metal (Apple Silicon)
        "--no-webui",
    ]
    if config.mmproj_path:
        if not Path(config.mmproj_path).exists():
            raise ModelServerError(f"mmproj_path not found: {config.mmproj_path!r}")
        cmd += ["--mmproj", config.mmproj_path]

    log_target = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=log_target, stderr=subprocess.STDOUT)

    if not wait_for_server(config.base_url, timeout=180):
        proc.terminate()
        raise ModelServerError(
            f"llama-server did not start on {config.base_url}. Check the log: {log_path}"
        )
    return proc


def ensure_backend(config: ModelServerConfig, log_path: Path | None = None):
    """Ensure the configured backend is running.

    For `mlx-vlm` / `llama.cpp`: start (and keep a handle on) our own server,
    returning it. For `standalone`: just verify something answers on base_url,
    raising a clear error if not. Returns a cleanup callable (or None if
    nothing to manage).
    """
    if config.backend == "standalone":
        if not is_server_up(config.base_url):
            raise ModelServerError(
                f"No server answering at {config.base_url}.\n"
                "Start one (e.g. LM Studio, Ollama, or `python -m mlx_vlm.server "
                f"--model <path> --port {MLX_SERVER_PORT}`) or set "
                'model_backend = "mlx-vlm" in config.toml to auto-start it.'
            )
        return None

    if config.backend not in ("mlx-vlm", "llama.cpp"):
        raise ModelServerError(f"Unknown model_backend: {config.backend!r}")

    if is_server_up(config.base_url):
        # Something is already on the port (e.g. a leftover, or the user started
        # one manually). Reuse it rather than fighting over the port.
        logging.warning(
            "A server is already answering on %s — reusing it for backend '%s'.",
            config.base_url, config.backend,
        )
        return None

    if config.backend == "llama.cpp":
        return start_llama_cpp_server(config, log_path=log_path)

    return start_mlx_vlm_server(config, log_path=log_path)
