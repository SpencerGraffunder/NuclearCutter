"""
Talks to the inference backends NuclearCutter uses — which are ALWAYS
already-running, remote (or local-but-external) servers. NuclearCutter never
spawns its own inference server:

- OpenAI-compatible /v1 server (llama.cpp, LM Studio, Ollama, vLLM, ...) —
  one server can serve the VLM, the text model, and the summary model at once
  (each selected by model id from the /v1/models dropdown).
- whisper.cpp transcription server (`whisper-server`) — one model loaded at a
  time, hot-swappable via its /load endpoint; the model dropdown lists the
  .bin files in the server's model folder, auto-detected from the
  whisper-server process's own `-m` argument when both run on one host.

One server ADDRESS (the machine) configures everything: the LLM/VLM server is
expected on port 8080 and the whisper server on port 8081 of that address
(see `derive_server_urls`). All configuration is that one address + model ids
(see docs/SPEC.md section 4.1).
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass

import requests

# OpenAI-compatible base URL default (llama.cpp/LM Studio/Ollama /v1).
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"

# The two inference servers of one deployment live on the SAME machine, on
# FIXED ports: the OpenAI-compatible LLM/VLM server on :8080 and the
# whisper.cpp server on :8081. The GUI therefore asks for ONE address (the
# machine) and derives both URLs from it — see `derive_server_urls`.
LLM_DEFAULT_PORT = 8080
WHISPER_DEFAULT_PORT = 8081

# whisper.cpp server defaults for this deployment.
WHISPER_DEFAULT_BASE_URL = "http://127.0.0.1:8081"
WHISPER_DEFAULT_INFERENCE_PATH = "/audio/transcriptions"
# Where the whisper-server's .bin model files live (on the machine running the
# server). Kept as the CLI's explicit default; the web GUI auto-detects the
# directory from the whisper-server process instead (see
# detect_whisper_models_dir).
WHISPER_DEFAULT_MODELS_DIR = "/home/graffunder/whisper-server/src/models"

GGML_BIN_RE = re.compile(r"ggml-[A-Za-z0-9._\-]+\.bin$")


@dataclass
class WhisperConfig:
    base_url: str = WHISPER_DEFAULT_BASE_URL  # e.g. http://192.168.4.164:8081
    inference_path: str = WHISPER_DEFAULT_INFERENCE_PATH  # the server's --inference-path
    model: str = ""  # model id (file stem, e.g. "ggml-base.en")
    # For the model dropdown listing / /load resolution. Empty = AUTO-DETECT
    # from the whisper-server process on this host (its -m argument already
    # points at the model folder — the server knows its own models dir).
    models_dir: str = ""
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


# ------------------------------------------------------------------ one address

def normalize_server_address(address: str) -> str:
    """Normalize the single GUI server address (scheme://host[:port]).

    Accepts "127.0.0.1", "http://192.168.4.164", "http://host:9000/v1" (a
    leftover path is dropped) and returns a clean "http://host[:port]".
    An empty/missing scheme defaults to http.
    """
    from urllib.parse import urlsplit

    addr = (address or "").strip().strip("/")
    if not addr:
        return ""
    if "://" not in addr:
        addr = "http://" + addr
    parts = urlsplit(addr)
    host = parts.netloc or addr
    return f"{parts.scheme or 'http'}://{host}"


def derive_server_urls(address: str) -> tuple[str, str]:
    """Derive the LLM and whisper base URLs from ONE server address.

    Both servers run on the same machine: the OpenAI-compatible LLM/VLM server
    on port 8080 (…/v1) and the whisper.cpp server on port 8081. When the
    address carries an explicit port, it is used for the LLM server and the
    whisper server is assumed on the NEXT port (port+1).

    Returns (llm_base_url, whisper_base_url).
    """
    from urllib.parse import urlsplit

    base = normalize_server_address(address)
    parts = urlsplit(base)
    host = parts.hostname or "127.0.0.1"
    port = parts.port
    if port is None:
        llm_port, whisper_port = LLM_DEFAULT_PORT, WHISPER_DEFAULT_PORT
    else:
        llm_port, whisper_port = port, port + 1
    return (
        f"{parts.scheme}://{host}:{llm_port}/v1",
        f"{parts.scheme}://{host}:{whisper_port}",
    )


# ------------------------------------------------------------------ whisper.cpp

# Cached (path, models_dir, loaded_model, fetched_at) of the local
# whisper-server process probe — /proc scanning on every dropdown fill would
# be wasteful; 60s is plenty for a process that rarely changes.
_WHISPER_PROC_CACHE: dict = {"path": "", "dir": "", "loaded": "", "at": 0.0}
_WHISPER_PROC_TTL = 60.0


def _probe_whisper_process() -> tuple[str, str, str]:
    """Find the local whisper-server process and read its model args.

    Returns (model_path, models_dir, loaded_model_id) — the path from its
    `-m`/`--model` argument, its parent directory, and the file stem (the
    model id the dropdown uses). Empty strings when no whisper-server is
    running on this host (or /proc is unavailable, e.g. macOS).
    """
    now = time.monotonic()
    if _WHISPER_PROC_CACHE["at"] and now - _WHISPER_PROC_CACHE["at"] < _WHISPER_PROC_TTL:
        return _WHISPER_PROC_CACHE["path"], _WHISPER_PROC_CACHE["dir"], _WHISPER_PROC_CACHE["loaded"]

    path = ""
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        pids = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                args = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        if not args or not re.search(r"whisper[._-]?server$", os.path.basename(args[0])):
            continue
        for i, arg in enumerate(args):
            if arg in ("-m", "--model") and i + 1 < len(args):
                path = args[i + 1]
                break
            if arg.startswith("--model="):
                path = arg.split("=", 1)[1]
                break
        if path:
            break

    models_dir = os.path.dirname(path) if path else ""
    loaded = ""
    if path:
        stem = os.path.basename(path)
        loaded = stem[:-4] if stem.endswith(".bin") else stem
    _WHISPER_PROC_CACHE.update(path=path, dir=models_dir, loaded=loaded, at=now)
    return path, models_dir, loaded


def detect_whisper_models_dir() -> str:
    """Auto-detect the local whisper-server's model folder.

    The whisper-server is started with `-m <path-to-model.bin>`, so the server
    process itself already knows where its models live — no GUI setting needed.
    Returns "" when no whisper-server is running on this host.
    """
    return _probe_whisper_process()[1]


def whisper_loaded_model_id() -> str:
    """The model id (file stem) the local whisper-server currently has loaded,
    or "" when unknown."""
    return _probe_whisper_process()[2]


def whisper_models_dir_for(cfg: WhisperConfig) -> str:
    """The effective models dir for a config: the explicit one if set, else
    auto-detected from the local whisper-server process."""
    if cfg.models_dir:
        return cfg.models_dir
    return detect_whisper_models_dir()


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


def whisper_list_models(cfg: WhisperConfig) -> list[dict]:
    """List whisper model files available for the dropdown.

    The whisper.cpp server has no model-list endpoint, so we list the .bin
    files in the model folder — either `cfg.models_dir` (explicit) or the
    auto-detected folder of the local whisper-server process (the `-m`
    argument it was started with). Returns [{"id", "label", "loaded"}] sorted
    for stable dropdown order; `loaded` marks the model the server currently
    has loaded.
    """
    models_dir = whisper_models_dir_for(cfg)
    try:
        names = os.listdir(models_dir)
    except OSError:
        return []
    loaded = whisper_loaded_model_id()
    out = []
    for name in names:
        m = GGML_BIN_RE.fullmatch(name)
        if m:
            mid = name[:-4]  # strip .bin -> "ggml-base.en"
            out.append({"id": mid, "label": mid, "loaded": mid == loaded})
    out.sort(key=lambda e: e["id"])
    return out


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
    # "ggml-base.en") is resolved to an absolute path inside the model folder
    # (explicit `models_dir`, else auto-detected from the local process),
    # because the server must be able to find the file on its own filesystem.
    path = model_id
    if not path.startswith("/"):
        models_dir = whisper_models_dir_for(cfg)
        candidate = os.path.join(models_dir, model_id + ".bin")
        if not os.path.exists(candidate):
            raise ModelServerError(
                f"whisper model {model_id!r} not found (looked for {candidate}). "
                f"Check the whisper server's model folder"
                + ("" if models_dir else " — no local whisper-server process was found to auto-detect it")
                + "."
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
