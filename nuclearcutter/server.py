"""
NuclearCutter web GUI server.

Serves the matrix-themed web UI (nuclearcutter/static/index.html) plus a
JSON API, and runs scan / render / benchmark jobs in background threads so
the browser shows live progress.

No login: binds 0.0.0.0 so any device on the LAN can open the UI. The VLM
prompts shown/edited in the GUI are read from the same prompts.json the
scanner uses (see nuclearcutter/prompts.py).

Run with:  python3 nuclearcutter.py serve   (see README.md)
"""

from __future__ import annotations

import atexit
import contextlib
import datetime
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from nuclearcutter.bench import attach_usage_hook, run_benchmark
from nuclearcutter.detection.profanity import DEFAULT_FOUL_LANGUAGE_SCALE
from nuclearcutter.detection.vlm_confirm import (
    DEFAULT_LEVEL_SCALES, _LEVELS_IN_ORDER, vision_max_pixels_for_scale,
)
from nuclearcutter.render.renderer import (
    RenderStopped, build_output_path, full_video_suffix, render as render_pass,
)
from nuclearcutter.scan.scanner import ScanStopped, scan as scan_pass
from nuclearcutter.schema import (
    AudioAction, Preferences, ScanResult, SeverityLevel, VisualAction,
)
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig
from nuclearcutter.utils.model_server import (
    DEFAULT_MLX_MODEL_PATH, MLX_BASE_URL, ModelServerConfig, _llama_model_id,
    ensure_backend, is_server_up,
)
from nuclearcutter.utils.system_stats import SystemStats

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Where GUI settings are persisted (loaded at startup, saved on every change).
# The old config.toml is gone; this is the app's own saved state.
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "settings.json"

CATEGORY_KEYS = ("nudity", "gore", "violence", "foul_language")
VISUAL_ACTIONS = ("none", "blur", "black")
AUDIO_ACTIONS = {
    "nudity": ("none", "mute_scene"),
    "gore": ("none", "mute_scene"),
    "violence": ("none", "mute_scene"),
    "foul_language": ("none", "mute_word", "mute_phrase", "replace_word", "replace_phrase"),
}
LEVELS = ("low", "med", "high", "exhigh")

# Media extensions offered in the GUI file picker.
MEDIA_EXTS = {
    ".mkv", ".mp4", ".m4v", ".avi", ".mov", ".wmv", ".webm", ".ts", ".m2ts",
    ".iso", ".mpg", ".mpeg", ".vob", ".flv", ".ogv",
}

DEFAULT_WHISPER = "mlx-community/whisper-small-mlx"

_LEVEL_RANK = {"low": 0, "med": 1, "high": 2, "exhigh": 3}


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _thumbnail_data_uri(data_uri: str, max_height: int = 16) -> str:
    """Downscale a base64 image data URI to a tiny JPEG thumbnail (for the
    terminal's inline image display). Returns the original URI on any failure."""
    try:
        import base64 as _b64
        import io as _io

        from PIL import Image

        header, _, b64 = data_uri.partition(",")
        img = Image.open(_io.BytesIO(_b64.b64decode(b64)))
        if img.height > max_height:
            ratio = max_height / img.height
            img = img.resize((max(1, int(img.width * ratio)), max_height), Image.LANCZOS)
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=50)
        return "data:image/jpeg;base64," + _b64.b64encode(buf.getvalue()).decode()
    except Exception:
        return data_uri


@dataclass
class JobInfo:
    kind: str
    status: str = "idle"  # idle | running | stopping | done | stopped | error
    phase: str = ""
    message: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    progress: float = 0.0          # 0..1 overall
    position: float = 0.0          # seconds into the film
    duration: float = 0.0
    frames_done: int = 0
    frames_total: int = 0
    eta: float | None = None
    # Per-step progress: transcribe / scan / verify / render. None = in
    # progress (indeterminate), a float 0..1 = determinate.
    steps: dict = field(default_factory=lambda: {
        "transcribe": 0.0, "scan": 0.0, "verify": 0.0, "render": 0.0,
    })
    results: list = field(default_factory=list)  # benchmark rows
    logs: list = field(default_factory=list)  # ring buffer of terminal output
    thread: threading.Thread | None = None
    stop_event: threading.Event | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": round(self.progress, 4),
            "position": round(self.position, 1),
            "duration": round(self.duration, 1),
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "eta": round(self.eta, 1) if self.eta is not None else None,
            "steps": self.steps,
            "results": self.results,
            "logs": self.logs,
        }


class _LogCapture:
    """Captures a job thread's stdout/stderr into a bounded ring buffer.

    The job's print() output (scanner `[sweep] N/M frames`, warnings, etc.)
    lands in `sink` (a list on the JobInfo) and — via `all_sink` — in the
    server's merged terminal stream. Capped so a multi-hour scan can't grow
    memory forever.
    """

    MAX_LINES = 500

    def __init__(self, sink: list, all_sink=None):
        self._sink = sink
        self._all_sink = all_sink
        self._buf = ""

    def write(self, s: str) -> None:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self._sink.append(line)
                if len(self._sink) > self.MAX_LINES:
                    del self._sink[: len(self._sink) - self.MAX_LINES]
                if self._all_sink is not None:
                    self._all_sink(line)

    def flush(self) -> None:
        if self._buf.strip():
            line = self._buf.strip()
            self._sink.append(line)
            if self._all_sink is not None:
                self._all_sink(line)
            self._buf = ""

    def isatty(self) -> bool:
        return False


class AppState:
    """Holds all GUI settings + live job state. A single instance per server."""

    def __init__(self, settings_path: Path | None = None):
        # --- Scan settings -------------------------------------------------
        self.video_path = ""
        self.backend = "mlx-vlm"  # mlx-vlm | llama.cpp | standalone
        # When enabled, every model prompt + response (with downscaled image
        # thumbnails for vision calls) is streamed to the terminal.
        self.show_prompts = False
        # Default to the LM Studio MLX folder so "local mlx-vlm" works out of
        # the box on this machine.
        self.model_path = DEFAULT_MLX_MODEL_PATH
        self.mmproj_path = ""
        self.base_url = MLX_BASE_URL
        self.vlm_model = ""
        self.text_model = ""
        self.whisper_model = DEFAULT_WHISPER
        self.scale = "480p"
        self.sweep_interval = 2.0
        # --- Render settings ----------------------------------------------
        self.output_name = ""  # empty -> <stem>_cleaned
        self.scan_path = ""  # scan file for render; empty -> auto-detect
        self.font_path = ""
        self.blur_strength = 1.0
        self.mute_padding = 0.5
        self.blur_padding = 0.0
        self.levels = {c: "med" for c in CATEGORY_KEYS}
        self.visual_actions = {
            "nudity": "blur", "gore": "blur", "violence": "blur",
            "foul_language": "none",
        }
        self.audio_actions = {
            "nudity": "none", "gore": "none", "violence": "none",
            "foul_language": "mute_phrase",
        }
        # --- Runtime -------------------------------------------------------
        self.scan = JobInfo(kind="scan")
        self.render = JobInfo(kind="render")
        self.bench = JobInfo(kind="benchmark")
        self.scan_result: ScanResult | None = None
        self.scan_result_path = ""
        self.server_proc = None
        self.settings_path: Path | None = None
        self.stats = SystemStats(pid=os.getpid())
        self._backend_up: bool = False
        self._backend_up_at: float = 0.0
        # Merged terminal stream: every log line from every job, as it comes in.
        self.all_logs: list[dict] = []  # [{"t": "HH:MM:SS", "job": kind, "line": str}, ...]
        # Live timeline during a scan: raw sweep candidates + confirmed
        # detections, streamed to the GUI so the timeline marks update in real
        # time (candidates render as generic/unconfirmed marks).
        self.live_timeline: dict = {
            "visual_candidates": [],
            "visual_detections": [],
            "language_detections": [],
        }
        self._model_usage = {
            "requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "elapsed_total": 0.0,
            "last_gen_tok_s": None, "avg_gen_tok_s": None,
            "last_pp_tok_s": None, "last_tokens_per_prompt": None,
        }
        self._usage_lock = threading.Lock()
        self._sweep_started_at: float | None = None

        if settings_path is not None:
            self.settings_path = Path(settings_path)
            self._load_settings()
        self._refresh_loaded_scan()  # load any existing scan for the saved movie

    # ------------------------------------------------------------------
    # Settings persistence (saved on the server, reloaded at startup)
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        """Load previously saved GUI settings from the settings file (if any).

        Applied per-key and leniently: a single bad/corrupt field must not
        reset all the others (e.g. an old value that no longer validates).
        A missing file is ignored — defaults stay in place.
        """
        if self.settings_path is None or not self.settings_path.exists():
            return
        try:
            data = json.loads(self.settings_path.read_text())
        except (OSError, ValueError) as exc:
            print(f"warning: could not read settings file {self.settings_path}: {exc}",
                  file=__import__("sys").stderr)
            return
        if not isinstance(data, dict):
            return
        for key, val in data.items():
            try:
                self.update_settings({key: val})
            except ValueError:
                pass  # skip one bad field; keep the rest

    def _save_settings(self) -> None:
        """Persist current settings atomically (tmp + rename)."""
        if self.settings_path is None:
            return
        try:
            tmp = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.settings_dict(), indent=2))
            os.replace(tmp, self.settings_path)
        except OSError as exc:
            print(f"warning: could not save settings to {self.settings_path}: {exc}",
                  file=__import__("sys").stderr)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def settings_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "backend": self.backend,
            "show_prompts": self.show_prompts,
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "base_url": self.base_url,
            "vlm_model": self.vlm_model,
            "text_model": self.text_model,
            "whisper_model": self.whisper_model,
            "scale": self.scale,
            "sweep_interval": self.sweep_interval,
            "output_name": self.output_name,
            "scan_path": self.scan_path,
            "font_path": self.font_path,
            "blur_strength": self.blur_strength,
            "mute_padding": self.mute_padding,
            "blur_padding": self.blur_padding,
            "levels": dict(self.levels),
            "visual_actions": dict(self.visual_actions),
            "audio_actions": dict(self.audio_actions),
        }

    def update_settings(self, payload: dict) -> dict:
        """Apply validated GUI settings. Raises ValueError on bad values."""
        if "video_path" in payload:
            self.video_path = str(payload["video_path"] or "").strip()
            self._refresh_loaded_scan()
        if "show_prompts" in payload:
            self.show_prompts = bool(payload["show_prompts"])
        if "backend" in payload:
            backend = str(payload["backend"]).strip()
            if backend not in ("mlx-vlm", "llama.cpp", "standalone"):
                raise ValueError(f"backend must be mlx-vlm / llama.cpp / standalone, got {backend!r}")
            self.backend = backend
        if "model_path" in payload:
            self.model_path = str(payload["model_path"] or "").strip()
        if "mmproj_path" in payload:
            self.mmproj_path = str(payload["mmproj_path"] or "").strip()
        if "base_url" in payload:
            self.base_url = str(payload["base_url"] or MLX_BASE_URL).strip()
        if "vlm_model" in payload:
            self.vlm_model = str(payload["vlm_model"] or "").strip()
        if "text_model" in payload:
            self.text_model = str(payload["text_model"] or "").strip()
        if "whisper_model" in payload:
            self.whisper_model = str(payload["whisper_model"] or DEFAULT_WHISPER).strip()
        if "scale" in payload:
            scale = str(payload["scale"]).strip().lower()
            if scale not in ("360p", "480p", "720p", "1080p"):
                raise ValueError(f"scale must be 360p/480p/720p/1080p, got {scale!r}")
            self.scale = scale
        if "sweep_interval" in payload:
            self.sweep_interval = float(payload["sweep_interval"])
            if self.sweep_interval <= 0:
                raise ValueError("sweep_interval must be > 0")
        if "output_name" in payload:
            self.output_name = str(payload["output_name"] or "").strip()
        if "scan_path" in payload:
            self.scan_path = str(payload["scan_path"] or "").strip()
        if "font_path" in payload:
            self.font_path = str(payload["font_path"] or "").strip()
        for f in ("blur_strength", "mute_padding", "blur_padding"):
            if f in payload:
                setattr(self, f, max(0.0, float(payload[f])))
        if "levels" in payload and isinstance(payload["levels"], dict):
            for cat, lvl in payload["levels"].items():
                if cat not in CATEGORY_KEYS:
                    continue
                if str(lvl) not in LEVELS:
                    raise ValueError(f"level for {cat} must be low/med/high/exhigh, got {lvl!r}")
                self.levels[cat] = str(lvl)
        if "visual_actions" in payload and isinstance(payload["visual_actions"], dict):
            for cat, act in payload["visual_actions"].items():
                if cat not in CATEGORY_KEYS:
                    continue
                if str(act) not in VISUAL_ACTIONS:
                    raise ValueError(f"visual action for {cat} must be none/blur/black, got {act!r}")
                self.visual_actions[cat] = str(act)
        if "audio_actions" in payload and isinstance(payload["audio_actions"], dict):
            for cat, act in payload["audio_actions"].items():
                if cat not in CATEGORY_KEYS:
                    continue
                if str(act) not in AUDIO_ACTIONS[cat]:
                    raise ValueError(
                        f"audio action for {cat} must be one of {AUDIO_ACTIONS[cat]}, got {act!r}"
                    )
                self.audio_actions[cat] = str(act)
        self._save_settings()
        return self.settings_dict()

    def prefs(self) -> Preferences:
        return Preferences(
            nudity_visual=VisualAction(self.visual_actions["nudity"]),
            nudity_audio=AudioAction(self.audio_actions["nudity"]),
            nudity_level=SeverityLevel.from_any(self.levels["nudity"]),
            gore_visual=VisualAction(self.visual_actions["gore"]),
            gore_audio=AudioAction(self.audio_actions["gore"]),
            gore_level=SeverityLevel.from_any(self.levels["gore"]),
            violence_visual=VisualAction(self.visual_actions["violence"]),
            violence_audio=AudioAction(self.audio_actions["violence"]),
            violence_level=SeverityLevel.from_any(self.levels["violence"]),
            foul_language_visual=VisualAction(self.visual_actions["foul_language"]),
            foul_language_audio=AudioAction(self.audio_actions["foul_language"]),
            foul_language_level=SeverityLevel.from_any(self.levels["foul_language"]),
            blur_strength=self.blur_strength,
            mute_padding=self.mute_padding,
            blur_padding=self.blur_padding,
        )

    # ------------------------------------------------------------------
    # Backend + client
    # ------------------------------------------------------------------

    def _ensure_backend(self, job: JobInfo):
        """Start/verify the model backend; returns (server_proc, llm_config)."""
        if self.backend == "standalone":
            cfg = ModelServerConfig(backend="standalone", base_url=self.base_url)
            proc = ensure_backend(cfg)
            vlm_model = self.vlm_model
            text_model = self.text_model
        elif self.backend == "llama.cpp":
            cfg = ModelServerConfig(backend="llama.cpp", model_path=self.model_path,
                                    mmproj_path=self.mmproj_path, base_url=self.base_url)
            proc = ensure_backend(cfg, log_path=Path(tempfile.gettempdir()) / "nuclearcutter_llama.log")
            vlm_model = text_model = _llama_model_id(self.model_path)
        else:  # mlx-vlm
            cfg = ModelServerConfig(backend="mlx-vlm", model_path=self.model_path,
                                    base_url=self.base_url)
            proc = ensure_backend(cfg, log_path=Path(tempfile.gettempdir()) / "nuclearcutter_mlx_vlm.log")
            # The mlx-vlm server serves the model id as its full filesystem path.
            vlm_model = text_model = self.model_path

        llm_config = LLMConfig(
            base_url=self.base_url,
            vlm_model=vlm_model,
            text_model=text_model,
        )
        llm_config.vision_max_pixels = vision_max_pixels_for_scale(self.scale)
        return proc, llm_config

    def _new_client(self, llm_config: LLMConfig) -> LLMClient:
        client = LLMClient(llm_config)
        client.usage_callback = self._on_usage
        return client

    def _on_usage(self, data: dict, elapsed: float) -> None:
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        with self._usage_lock:
            m = self._model_usage
            m["requests"] += 1
            prompt_tok = int(usage.get("prompt_tokens", 0))
            completion_tok = int(usage.get("completion_tokens", 0))
            m["prompt_tokens"] += prompt_tok
            m["completion_tokens"] += completion_tok
            m["elapsed_total"] = m.get("elapsed_total", 0.0) + elapsed
            m["last_tokens_per_prompt"] = prompt_tok
            # Prefer the server's own timings (llama.cpp/LM Studio style) when
            # present; otherwise estimate from usage + wall-clock (our sweep
            # requests are prompt-dominated, so prompt_tokens/elapsed is a
            # decent pp-throughput estimate and completion_tokens/elapsed a
            # rough generation rate).
            if timings.get("predicted_ms") and timings.get("predicted_n"):
                m["last_gen_tok_s"] = round(timings["predicted_n"] / (timings["predicted_ms"] / 1000.0), 1)
            elif completion_tok and elapsed > 0:
                m["last_gen_tok_s"] = round(completion_tok / elapsed, 1)
            if timings.get("prompt_ms") and timings.get("prompt_n"):
                m["last_pp_tok_s"] = round(timings["prompt_n"] / (timings["prompt_ms"] / 1000.0), 1)
            elif prompt_tok and elapsed > 0:
                m["last_pp_tok_s"] = round(prompt_tok / elapsed, 1)
            if m.get("elapsed_total", 0.0) > 0 and m["completion_tokens"]:
                m["avg_gen_tok_s"] = round(m["completion_tokens"] / m["elapsed_total"], 1)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def _active_job(self) -> JobInfo | None:
        for j in (self.scan, self.render, self.bench):
            if j.thread is not None and j.thread.is_alive():
                return j
        return None

    def _launch(self, job: JobInfo, fn) -> None:
        """Spawn a daemon thread running fn(); sets running state."""
        if job.thread is not None and job.thread.is_alive():
            raise HTTPException(409, f"{job.kind} is already running")
        active = self._active_job()
        if active is not None:
            raise HTTPException(409, f"{active.kind} is already running — stop it first")
        job.status = "running"
        job.phase = "starting"
        job.message = ""
        job.error = ""
        job.started_at = _now_iso()
        job.finished_at = ""
        job.progress = 0.0
        job.position = 0.0
        job.eta = None
        job.frames_done = 0
        job.frames_total = 0
        job.steps = {"transcribe": 0.0, "scan": 0.0, "verify": 0.0, "render": 0.0}
        job.stop_event = threading.Event()
        job.results = []
        job.logs = []
        job.thread = threading.Thread(
            target=self._wrapped_job(job, fn), daemon=True, name=f"nc-{job.kind}"
        )
        job.thread.start()

    def _log(self, job: JobInfo, line: str) -> None:
        """Append a line to a job's log AND the merged terminal stream."""
        job.logs.append(line)
        self.all_logs.append({
            "t": datetime.datetime.now().strftime("%H:%M:%S"),
            "job": job.kind,
            "line": line,
        })
        if len(self.all_logs) > 2000:
            del self.all_logs[: len(self.all_logs) - 2000]

    def _log_line(self, job_label: str, line: str, imgs: list | None = None) -> None:
        """Append to the merged terminal stream (optionally with inline image
        thumbnails), bounded like the job logs."""
        entry = {
            "t": datetime.datetime.now().strftime("%H:%M:%S"),
            "job": job_label,
            "line": line,
        }
        if imgs:
            entry["imgs"] = imgs
        self.all_logs.append(entry)
        if len(self.all_logs) > 2000:
            del self.all_logs[: len(self.all_logs) - 2000]

    def _log_request(self, job_label: str, payload: dict, data: dict) -> None:
        """Stream a model request/response to the terminal (when enabled).

        Called from the LLM client for every request. Vision payloads include
        downscaled thumbnails of the images being sent, rendered inline at
        text height by the GUI.
        """
        if not self.show_prompts:
            return
        try:
            model = payload.get("model", "?")
            content = (payload.get("messages") or [{}])[-1].get("content", "")
            text_parts: list[str] = []
            imgs: list[str] = []
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url:
                            imgs.append(_thumbnail_data_uri(url, max_height=16))
            else:
                text_parts.append(str(content))
            response = ""
            if data.get("choices"):
                response = data["choices"][0].get("message", {}).get("content", "")
            prompt_text = " ".join(t for t in text_parts if t) or "(no text)"
            self._log_line(job_label, f"{model} → {prompt_text}")
            if imgs:
                self._log_line(job_label, f"  [{len(imgs)} image(s) sent]", imgs=imgs)
            if response:
                self._log_line(job_label, f"{model} ← {response}")
        except Exception:
            pass  # never let prompt logging break a request

    def _refresh_loaded_scan(self) -> None:
        """When a movie is selected (or the server starts), load any existing
        scan JSON for it so the timeline shows its detections and the progress
        bars reflect the already-completed work. Also reflects a preserved
        transcript even when the scan/verify result was cleared (the two are
        separate sections)."""
        self.scan_result = None
        self.scan_result_path = ""
        if not self.video_path:
            return
        video = Path(self.video_path)
        if not video.exists():
            return
        transcript_path = video.with_suffix(".nuclearcutter.transcript.json")
        has_transcript = transcript_path.exists()
        candidate = video.with_suffix(".nuclearcutter.json")
        if self.scan_path and Path(self.scan_path).exists():
            candidate = Path(self.scan_path)
        if not candidate.exists():
            # No scan/verify result, but the whisper transcript may have been
            # preserved by a scan+verify clear — keep its bar accurate.
            if has_transcript and not (self.scan.thread is not None and self.scan.thread.is_alive()):
                self.scan.steps["transcribe"] = 1.0
                self.scan.status = "stopped"
                self.scan.message = ("Transcript cached — scan + verify were cleared "
                                     "(start the scan to re-run the VLM sweep)")
                self._log(self.scan, f"[scan] {self.scan.message}")
            return
        try:
            result = ScanResult.load(candidate)
        except Exception:
            return  # corrupt/unreadable — ignore, not worth failing a scan over
        self.scan_result = result
        self.scan_result_path = str(candidate)
        # If the scan job isn't currently running, mark the already-done work
        # so the GUI shows full transcribe/scan/verify bars for this movie.
        if not (self.scan.thread is not None and self.scan.thread.is_alive()):
            self.scan.status = "done"
            self.scan.phase = "done"
            self.scan.progress = 1.0
            self.scan.duration = result.identity.duration_seconds
            self.scan.position = self.scan.duration
            # Transcription is a separate section: only counts as done if the
            # whisper transcript cache still exists (it can be cleared on its
            # own via the GUI's per-section clear button).
            self.scan.steps = {
                "transcribe": 1.0 if has_transcript else 0.0,
                "scan": 1.0, "verify": 1.0, "render": 0.0,
            }
            pending = "" if has_transcript else " — transcription pending (re-transcribe on next scan)"
            self.scan.message = (
                f"Loaded existing scan: {len(result.visual_detections)} visual, "
                f"{len(result.language_detections)} language — {candidate.name}{pending}"
            )
            self._log(self.scan, f"[scan] {self.scan.message}")

    def _wrapped_job(self, job: JobInfo, fn):
        """Return the job thread body: run fn with stdout/stderr captured into
        the job log and the merged terminal stream."""
        cap = _LogCapture(job.logs, all_sink=lambda line: self._log(job, line))

        def _run():
            try:
                with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
                    fn()
            finally:
                cap.flush()

        return _run

    def start_scan(self) -> JobInfo:
        video = Path(self.video_path)
        if not self.video_path or not video.exists():
            raise HTTPException(400, f"video file not found: {self.video_path!r}")
        if not self.whisper_model:
            raise HTTPException(400, "whisper_model is required")
        self._validate_local_backend()
        self._save_settings()  # ensure the current GUI settings are on disk
        self._launch(self.scan, self._run_scan)
        return self.scan

    def _validate_local_backend(self) -> None:
        if self.backend == "standalone":
            if not self.vlm_model:
                raise HTTPException(400, "vlm_model is required when using an existing server")
            return
        if not self.model_path or not Path(self.model_path).exists():
            raise HTTPException(400, f"model_path not found: {self.model_path!r}")

    def _run_scan(self) -> None:
        job = self.scan
        video = Path(self.video_path)
        status_path = video.with_suffix(".nuclearcutter.status.json")
        partial_path = video.with_suffix(".nuclearcutter.json")
        log_path = video.with_suffix(".nuclearcutter.log")
        try:
            from nuclearcutter.utils.ffmpeg import probe_duration

            job.duration = probe_duration(video)
            job.phase = "starting backend"
            job.message = f"Starting {self.backend} backend..."
            proc, llm_config = self._ensure_backend(job)
            self.server_proc = proc

            client = self._new_client(llm_config)
            client.test_connection()
            client.request_log_callback = (
                lambda payload, data: self._log_request("scan", payload, data)
            )

            # Seed the live timeline from any prior run's status file, so a
            # resumed scan shows its existing candidates/detections immediately.
            self.live_timeline = {"visual_candidates": [], "visual_detections": [],
                                  "language_detections": []}
            seed_status = video.with_suffix(".nuclearcutter.status.json")
            try:
                if seed_status.exists():
                    import json as _json

                    seed = _json.loads(seed_status.read_text())
                    self.live_timeline["visual_candidates"] = list(seed.get("visual_candidates", []))
                    self.live_timeline["visual_detections"] = list(seed.get("visual_detections", []))
                    self.live_timeline["language_detections"] = list(seed.get("language_detections", []))
            except (OSError, ValueError):
                pass

            def _progress(stage, detail):
                if stage == "transcribing":
                    job.phase = "transcribing"
                    if isinstance(detail, (int, float)):
                        job.steps["transcribe"] = max(0.0, min(float(detail), 1.0))
                        pct = int(round(float(detail) * 100))
                        if pct % 10 == 0 and pct != job._last_transcribe_pct:
                            job._last_transcribe_pct = pct
                            self._log(job, f"[transcribing] {pct}%")
                        if float(detail) >= 1.0:
                            self._log(job, "[transcribing] done")
                    else:
                        job.steps["transcribe"] = 0.0
                        job._last_transcribe_pct = -1
                        self._log(job, "[transcribing] starting")
                elif stage == "visual_sweep" and detail:
                    done, total = detail
                    job.phase = "visual_sweep"
                    job.frames_done, job.frames_total = int(done), int(total)
                    job.position = done * self.sweep_interval
                    job.steps["scan"] = (done / total) if total else 0.0
                    if self._sweep_started_at is None:
                        self._sweep_started_at = time.monotonic()
                    elapsed = time.monotonic() - self._sweep_started_at
                    job.eta = None
                    if elapsed > 3 and done > 0:
                        remaining = (total - done) / (done / elapsed)
                        job.eta = remaining
                elif stage == "visual_confirm" and detail:
                    i, total = detail
                    job.phase = "visual_confirm"
                    job.steps["verify"] = (i / total) if total else 0.0
                    if i == 0:
                        self._log(job, f"[visual_confirm] confirming {total} candidate range(s)")
                elif stage == "candidate" and isinstance(detail, dict):
                    # Raw sweep hit — shown as a generic/unconfirmed timeline mark.
                    self.live_timeline["visual_candidates"].append({
                        "category": detail.get("category", ""),
                        "start": detail.get("start", 0.0),
                        "end": detail.get("end", 0.0),
                        "confidence": detail.get("confidence", 0.5),
                        "level": detail.get("level", "low"),
                    })
                elif stage == "visual_detection" and isinstance(detail, dict):
                    # Confirmed + classified detection — render-settings filterable.
                    self.live_timeline["visual_detections"].append({
                        "category": detail.get("category", ""),
                        "start": detail.get("start", 0.0),
                        "end": detail.get("end", 0.0),
                        "level": detail.get("level", "med"),
                        "confidence": detail.get("confidence", 0.5),
                        "description": detail.get("description", ""),
                    })
                elif stage == "language_detections" and isinstance(detail, list):
                    self.live_timeline["language_detections"] = detail
                elif stage == "language_detection":
                    job.phase = "language_detection"
                    self._log(job, "[language_detection]")
                elif stage == "done":
                    job.steps = {"transcribe": 1.0, "scan": 1.0, "verify": 1.0, "render": 0.0}
                    job.position = job.duration
                    self._log(job, "[done]")

            result = scan_pass(
                video,
                llm_config=llm_config,
                title=video.stem,
                year=None,
                progress_callback=_progress,
                whisper_model=self.whisper_model,
                sweep_interval=self.sweep_interval,
                status_path=status_path,
                partial_result_path=partial_path,
                scale=self.scale,
                stop_event=job.stop_event,
            )

            from nuclearcutter.cli import _save_result_with_fallback

            saved = _save_result_with_fallback(result, partial_path)
            self.scan_result = result
            self.scan_result_path = str(saved)
            job.duration = result.identity.duration_seconds
            job.position = job.duration
            job.progress = 1.0
            job.steps = {"transcribe": 1.0, "scan": 1.0, "verify": 1.0, "render": 0.0}
            job.phase = "done"
            job.status = "done"
            job.message = (f"Scan complete: {len(result.visual_detections)} visual, "
                           f"{len(result.language_detections)} language — {saved.name}")
            self._log(job, f"{job.message}")
        except ScanStopped as exc:
            job.status = "stopped"
            job.message = f"Stopped — progress saved, press Start to resume ({exc})"
            self._log(job, f"{job.message}")
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = f"Scan failed: {exc}"
            self._log(job, f"ERROR ({job.phase}): {exc}")
            try:
                log_path.write_text(f"{_now_iso()} scan error: {exc}\n")
            except OSError:
                pass
        finally:
            job.finished_at = _now_iso()
            self._sweep_started_at = None

    def stop_scan(self) -> JobInfo:
        if self.scan.thread is not None and self.scan.thread.is_alive():
            self.scan.status = "stopping"
            self.scan.stop_event.set()
        return self.scan

    def clear_scan_progress(self) -> dict:
        """Clear the visual scan + verify pass for the current movie.

        Deletes the in-progress status file, the (partial/final) result file,
        and the scan log, so the next scan re-runs the VLM sweep + confirm
        pass from scratch. The whisper transcript cache is intentionally
        KEPT — transcription is a separate section, cleared via
        `clear_transcription_progress`. Scan + verify always clear together:
        verify confirms the sweep's candidates, so a cleared sweep invalidates
        the confirm pass too. Render/benchmark job state is untouched.
        """
        if self.scan.thread is not None and self.scan.thread.is_alive():
            raise HTTPException(409, "scan is running — stop it before clearing")
        video = Path(self.video_path) if self.video_path else None
        removed = []
        if video is not None:
            for suffix in (".nuclearcutter.status.json", ".nuclearcutter.json",
                           ".nuclearcutter.log", ".nuclearcutter.json.recovery.json"):
                p = video.with_suffix(suffix)
                if p.exists():
                    try:
                        p.unlink()
                        removed.append(str(p))
                    except OSError:
                        pass
        self.scan = JobInfo(kind="scan")
        # Transcription is a SEPARATE section and was not cleared — keep its
        # progress bar accurate (100% if the transcript cache still exists)
        # instead of zeroing the whole display. Scan/verify stay at 0%.
        if video is not None and video.with_suffix(".nuclearcutter.transcript.json").exists():
            self.scan.steps["transcribe"] = 1.0
        self.scan_result = None
        self.scan_result_path = ""
        self.live_timeline = {"visual_candidates": [], "visual_detections": [],
                              "language_detections": []}
        return {"removed": removed}

    def clear_transcription_progress(self) -> dict:
        """Clear the transcription pass for the current movie.

        Deletes the whisper transcript cache so the next scan re-transcribes,
        while keeping any already-completed scan + verify results intact. If a
        scan was loaded from a saved result file it stays loaded, but the
        transcribe bar drops back to 0% (the restart re-runs whisper and then
        skips the already-done sweep/confirm — see `_load_resume_state`).
        """
        if self.scan.thread is not None and self.scan.thread.is_alive():
            raise HTTPException(409, "scan is running — stop it before clearing")
        video = Path(self.video_path) if self.video_path else None
        removed = []
        if video is not None:
            p = video.with_suffix(".nuclearcutter.transcript.json")
            if p.exists():
                try:
                    p.unlink()
                    removed.append(str(p))
                except OSError:
                    pass
        # Reflect in the GUI: transcription is no longer done, scan/verify are.
        self.scan.steps["transcribe"] = 0.0
        if self.scan_result is not None:
            self.scan.status = "stopped"
            self.scan.message = "Transcript cleared — start the scan to re-transcribe (scan + verify kept)"
            self._log(self.scan, f"[scan] {self.scan.message}")
        return {"removed": removed}

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def start_render(self) -> JobInfo:
        video = Path(self.video_path)
        if not self.video_path or not video.exists():
            raise HTTPException(400, f"video file not found: {self.video_path!r}")
        scan_path = self._resolve_scan_path(video)
        self._save_settings()  # ensure the current GUI settings are on disk
        self._launch(self.render, lambda: self._run_render(video, scan_path))
        return self.render

    def _resolve_scan_path(self, video: Path) -> Path:
        if self.scan_path:
            p = Path(self.scan_path)
            if not p.exists():
                raise HTTPException(400, f"scan file not found: {self.scan_path!r}")
            return p
        if self.scan_result_path and Path(self.scan_result_path).exists():
            return Path(self.scan_result_path)
        auto = video.with_suffix(".nuclearcutter.json")
        if auto.exists():
            return auto
        raise HTTPException(400, "no scan file found — scan the movie first, or set a scan file path")

    def _run_render(self, video: Path, scan_path: Path) -> None:
        job = self.render
        try:
            scan_result = ScanResult.load(scan_path)
            job.duration = scan_result.identity.duration_seconds
            prefs = self.prefs()
            output_path = self._render_output_path(video)

            def _progress(stage, detail):
                if stage == "render" and detail:
                    i, total, seg_start = detail
                    job.phase = "render"
                    job.frames_done, job.frames_total = int(i), int(total)
                    job.position = seg_start
                    job.steps["render"] = (i / total) if total else 0.0
                    job.progress = (i / total) if total else 0.0
                    if i == 0:
                        self._log(job, f"rendering {total} segment(s)")
                    if i > 0:
                        elapsed = (time.monotonic() - self._render_started) or 1e-9
                        job.eta = (total - i) * (elapsed / i)
                    # Intermediate progress lines every 5%, so the terminal
                    # shows the render actually advancing.
                    pct = int(i / total * 100) if total else 0
                    if pct % 5 == 0 and pct != job._last_render_pct:
                        job._last_render_pct = pct
                        self._log(job, f"segment {i}/{total} ({pct}%)")
                elif stage in ("concat", "mux"):
                    job.phase = stage
                    job.steps["render"] = 1.0
                    self._log(job, f"[{stage}]")

            self._render_started = time.monotonic()
            job._last_render_pct = -1
            render_pass(
                video, scan_result, prefs,
                output_path=output_path,
                font_path=self.font_path or None,
                progress_callback=_progress,
                stop_event=job.stop_event,
            )
            job.status = "done"
            job.phase = "done"
            job.progress = 1.0
            job.steps["render"] = 1.0
            job.position = job.duration
            self._log(job, f"done -> {output_path}")
            job.message = f"Rendered to {output_path}"
            self._log(job, f"{job.message}")
        except RenderStopped as exc:
            job.status = "stopped"
            job.message = f"Render stopped ({exc}) — no partial output kept."
            self._log(job, f"{job.message}")
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = f"Render failed: {exc}"
            self._log(job, f"ERROR ({job.phase}): {exc}")
        finally:
            job.finished_at = _now_iso()

    def _render_output_path(self, video: Path) -> Path:
        if not self.output_name:
            return build_output_path(video)
        out = Path(self.output_name).expanduser()
        if not out.suffix:
            # No extension given — append the source's full extension chain
            # (e.g. ".mkv.iso" for a matroska named with an .iso suffix).
            out = out.with_name(out.name + full_video_suffix(video))
        if out.parent == Path("."):
            out = video.parent / out.name
        return out

    def stop_render(self) -> JobInfo:
        if self.render.thread is not None and self.render.thread.is_alive():
            self.render.status = "stopping"
            self.render.stop_event.set()
        return self.render

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def start_benchmark(self) -> JobInfo:
        video = Path(self.video_path)
        if not self.video_path or not video.exists():
            raise HTTPException(400, f"video file not found: {self.video_path!r}")
        self._validate_local_backend()
        self._save_settings()  # ensure the current GUI settings are on disk
        self._launch(self.bench, self._run_benchmark)
        return self.bench

    def _run_benchmark(self) -> None:
        job = self.bench
        video = Path(self.video_path)
        try:
            job.phase = "starting backend"
            job.message = f"Starting {self.backend} backend..."
            self._log(job, f"starting {self.backend} backend...")
            proc, llm_config = self._ensure_backend(job)
            self.server_proc = proc
            client = self._new_client(llm_config)
            client.test_connection()
            attach_usage_hook(client)
            client.request_log_callback = (
                lambda payload, data: self._log_request("benchmark", payload, data)
            )
            self._log(job, "backend ready, connection ok")

            # Load a scan if one exists, so flagged-window frames are included.
            scan_result = None
            try:
                scan_result = ScanResult.load(self._resolve_scan_path(video))
            except HTTPException:
                scan_result = None

            job.phase = "benchmarking"
            job.message = "Extracting 12 frames and running sweep + confirm prompts..."
            self._log(job, "[benchmark] extracting 12 frames and running sweep + confirm prompts")
            result = run_benchmark(
                video, client, scan=scan_result, scale=self.scale,
                stop_event=job.stop_event,
            )
            job.results = [result["summary"], result["batches"], result["confirms"]]
            if job.stop_event is not None and job.stop_event.is_set():
                job.status = "stopped"
                job.phase = "stopped"
                self._log(job, f"stopped after {result['summary']['batches']} batch(es)")
                job.message = (f"Benchmark stopped after {result['summary']['batches']} batch(es) "
                               f"— partial results shown below")
            else:
                job.status = "done"
                job.phase = "done"
                s = result["summary"]
                self._log(job, 
                    f"[benchmark] done: {s['frames']} frames, accuracy {s['accuracy']}, "
                    f"mean sweep {s['mean_sweep_s']}s"
                )
                job.message = (
                    f"Benchmark done: {s['frames']} frames, "
                    f"accuracy {s['accuracy']}, "
                    f"mean sweep {s['mean_sweep_s']}s"
                )
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = f"Benchmark failed: {exc}"
            self._log(job, f"ERROR ({job.phase}): {exc}")
        finally:
            job.finished_at = _now_iso()

    def stop_benchmark(self) -> JobInfo:
        if self.bench.thread is not None and self.bench.thread.is_alive():
            self.bench.status = "stopping"
            self.bench.stop_event.set()
        return self.bench

    # ------------------------------------------------------------------
    # State snapshot for the GUI
    # ------------------------------------------------------------------

    def timeline_dict(self) -> dict | None:
        """Timeline data for the GUI.

        A COMPLETED scan returns its full detections (visual_candidates empty).
        During a scan it returns the LIVE timeline — raw sweep candidates
        (shown as generic/unconfirmed marks) plus confirmed visual/language
        detections as they're found — so the timeline updates in real time.
        """
        if self.scan_result is not None:
            return {
                "duration": self.scan_result.identity.duration_seconds,
                "visual_candidates": [],
                "visual_detections": [
                    {"category": d.category.value, "start": d.start, "end": d.end,
                     "level": d.level.value, "confidence": d.confidence,
                     "description": d.description}
                    for d in self.scan_result.visual_detections
                ],
                "language_detections": [
                    {"category": "foul_language", "start": d.start, "end": d.end,
                     "utterance_start": d.utterance_start, "utterance_end": d.utterance_end,
                     "word": d.word, "level": d.level.value, "llm_confirmed": d.llm_confirmed}
                    for d in self.scan_result.language_detections
                ],
            }
        live = self.live_timeline
        if any(live.values()) or self.scan.duration:
            return {
                "duration": self.scan.duration or 0.0,
                "visual_candidates": list(live["visual_candidates"]),
                "visual_detections": list(live["visual_detections"]),
                "language_detections": list(live["language_detections"]),
            }
        return None

    def model_stats_dict(self) -> dict:
        with self._usage_lock:
            m = dict(self._model_usage)
        return m

    def state_dict(self) -> dict:
        scan_job = self.scan.to_dict()
        # Speed per frame (sweep): frames / elapsed since sweep began.
        if scan_job["status"] == "running" and self._sweep_started_at:
            elapsed = time.monotonic() - self._sweep_started_at
            if elapsed > 0 and scan_job["frames_done"]:
                scan_job["speed_frames_s"] = round(scan_job["frames_done"] / elapsed, 2)
        else:
            scan_job["speed_frames_s"] = None
        # Cache the server-reachability probe (an HTTP GET) — the GUI polls
        # every second and this must never block the status refresh.
        now = time.monotonic()
        if now - self._backend_up_at >= 5.0:
            self._backend_up_at = now
            self._backend_up = is_server_up(self.base_url)
        return {
            "settings": self.settings_dict(),
            "jobs": {
                "scan": scan_job,
                "render": self.render.to_dict(),
                "benchmark": self.bench.to_dict(),
            },
            "all_logs": self.all_logs,
            "timeline": self.timeline_dict(),
            "system": self.stats.sample(),
            "model": self.model_stats_dict(),
            "server": {
                "backend": self.backend,
                "base_url": self.base_url,
                "backend_up": self._backend_up,
                "settings_file": str(self.settings_path) if self.settings_path else "",
            },
        }


STATE = AppState(settings_path=DEFAULT_SETTINGS_PATH)


def _state() -> AppState:
    return STATE


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class SettingsBody(BaseModel):
    settings: dict


class LogBody(BaseModel):
    line: str


class ClearScanBody(BaseModel):
    section: str = "scan_verify"  # "transcribe" | "scan_verify"


def create_app(state: AppState | None = None) -> FastAPI:
    app = FastAPI(title="NuclearCutter", version="0.2.0")
    st = state if state is not None else _state()

    @app.get("/")
    def index():
        if not INDEX_HTML.exists():
            raise HTTPException(500, f"UI file missing: {INDEX_HTML}")
        return FileResponse(INDEX_HTML)

    @app.get("/api/state")
    def api_state():
        return st.state_dict()

    @app.get("/api/models")
    def api_models(base_url: str = "", vlm: str = ""):
        """List models advertised by a server (the 'use existing server' flow).

        `vlm` (optional) filters the list to ids containing that string.
        """
        url = (base_url or st.base_url).strip()
        client = LLMClient(LLMConfig(base_url=url))
        models = client.list_models()
        return {"base_url": url, "reachable": bool(models), "models": models}

    @app.post("/api/settings")
    def api_settings(body: SettingsBody):
        try:
            return {"settings": st.update_settings(body.settings)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/scan/start")
    def api_scan_start():
        return {"job": st.start_scan().to_dict()}

    @app.post("/api/scan/stop")
    def api_scan_stop():
        return {"job": st.stop_scan().to_dict()}

    @app.post("/api/scan/clear")
    def api_scan_clear(body: ClearScanBody | None = None):
        """Clear per-section scan progress.

        `section` is either "transcribe" (delete the whisper transcript cache
        so the next scan re-transcribes, keeping scan/verify results) or
        "scan_verify" (delete the VLM sweep + confirm results — the transcript
        is kept). Scan + verify always clear together. Defaults to
        "scan_verify" (matches the old full-clear behavior minus the
        transcript, which now has its own button).
        """
        section = (body.section if body is not None else "scan_verify").strip().lower()
        if section == "transcribe":
            return st.clear_transcription_progress()
        if section in ("scan", "verify", "scan_verify"):
            return st.clear_scan_progress()
        raise HTTPException(400, f"unknown clear section: {section!r}")

    @app.post("/api/render/start")
    def api_render_start():
        return {"job": st.start_render().to_dict()}

    @app.post("/api/render/stop")
    def api_render_stop():
        return {"job": st.stop_render().to_dict()}

    @app.post("/api/benchmark/start")
    def api_benchmark_start():
        return {"job": st.start_benchmark().to_dict()}

    @app.post("/api/benchmark/stop")
    def api_benchmark_stop():
        return {"job": st.stop_benchmark().to_dict()}

    @app.get("/api/definitions")
    def api_definitions():
        """The per-category severity level definitions, shown as hover tooltips
        in the render settings. Same source the scanner's prompts are built from."""
        out = {}
        for cat, levels in DEFAULT_LEVEL_SCALES.items():
            out[cat.value] = {lv.value: levels[lv] for lv in _LEVELS_IN_ORDER}
        out["foul_language"] = DEFAULT_FOUL_LANGUAGE_SCALE
        return out

    @app.get("/api/browse")
    def api_browse(path: str = "", exts: str = ""):
        """Directory listing for the GUI's file pickers.

        Browses the SERVER's filesystem (the server reads the files). `path`
        empty starts at the user's home; a path to a file resolves to its
        parent directory.

        `exts` filters files by extension (comma-separated, e.g. ".gguf,.bin").
        "all" lists every file (for picking e.g. a model file); omitted/empty
        lists only media extensions (the movie picker).
        """
        p = Path(path).expanduser() if path else Path.home()
        if not p.is_absolute():
            p = Path.home()
        p = p.resolve()
        if p.is_file():
            p = p.parent
        if not p.is_dir():
            raise HTTPException(400, f"not a directory: {p}")
        try:
            entries = list(p.iterdir())
        except PermissionError as exc:
            raise HTTPException(403, f"permission denied: {p}") from exc
        dirs = sorted(
            (e.name for e in entries if e.is_dir() and not e.name.startswith(".")),
            key=str.lower,
        )
        if exts == "all":
            names = (e.name for e in entries if e.is_file() and not e.name.startswith("."))
        elif exts:
            allowed = {x.strip().lower() for x in exts.split(",") if x.strip()}
            names = (e.name for e in entries if e.is_file() and e.suffix.lower() in allowed)
        else:
            names = (e.name for e in entries if e.is_file() and e.suffix.lower() in MEDIA_EXTS)
        files = sorted(names, key=str.lower)
        return {
            "path": str(p),
            "parent": str(p.parent) if p != p.parent else None,
            "home": str(Path.home()),
            "dirs": dirs,
            "files": files,
        }

    @app.post("/api/log")
    def api_log(body: LogBody):
        """Client-side messages (UI errors, button feedback) go into the
        merged terminal stream so ALL status/errors live in one place."""
        if body.line:
            st.all_logs.append({
                "t": datetime.datetime.now().strftime("%H:%M:%S"),
                "job": "ui",
                "line": body.line,
            })
            if len(st.all_logs) > 2000:
                del st.all_logs[: len(st.all_logs) - 2000]
        return {"ok": True}

    @app.get("/api/health")
    def api_health():
        return {"ok": True}

    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch the web GUI server (blocking)."""
    import uvicorn

    if not INDEX_HTML.exists():
        print(f"warning: UI file missing at {INDEX_HTML}", file=__import__("sys").stderr)

    def _cleanup():
        proc = STATE.server_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                pass

    atexit.register(_cleanup)
    local_url = f"http://localhost:{port}"
    net_url = f"http://<this-machine-ip>:{port}"
    print("=" * 64)
    print(" NuclearCutter web GUI")
    print(f"   Open in a browser on this machine:  {local_url}")
    print(f"   From any device on the network:     {net_url}")
    if host not in ("0.0.0.0", ""):
        print(f"   (bound to {host} — only this machine can reach it)")
    print("   (no login; anyone on the network can reach it)")
    print(f"   Settings saved to: {STATE.settings_path}")
    print("=" * 64)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
