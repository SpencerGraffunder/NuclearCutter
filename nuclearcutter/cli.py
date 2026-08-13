"""
NuclearCutter CLI.

Primary interface is the web GUI (this repo's default):
    python3 nuclearcutter.py serve          # or just: python3 nuclearcutter.py
    -> open http://localhost:8000 in a browser (any device on the network
       can reach it at http://<this-machine-ip>:8000, no login)

Headless scan/render are still available for scripting:
    python3 nuclearcutter.py scan MOVIE.mkv [--scale 480p ...]
    python3 nuclearcutter.py render MOVIE.mkv [--nudity-level high ...]

There is no config file any more — everything is set in the web GUI (or via
CLI flags for the headless commands). VLM prompts come from prompts.json
(see nuclearcutter/prompts.py), shared by the GUI and the scanner.
"""

from __future__ import annotations

import argparse
import atexit
import sys
import tempfile
from pathlib import Path

from nuclearcutter.render.renderer import build_output_path, render as render_pass
from nuclearcutter.schema import AudioAction, Preferences, ScanResult, SeverityLevel, VisualAction
from nuclearcutter.utils.llm_client import LLMConfig
from nuclearcutter.utils.model_server import DEFAULT_MLX_MODEL_PATH, MLX_BASE_URL, ModelServerConfig, ensure_backend

DEFAULT_WHISPER = "mlx-community/whisper-small-mlx"
DEFAULT_SCALE = "480p"


def _default_scan_path(video_path: Path) -> Path:
    return video_path.with_suffix(".nuclearcutter.json")


def _save_result_with_fallback(result: ScanResult, out_path: Path) -> Path:
    """Persist a completed scan result, never losing it to a failed write.

    The movie's folder is the working dir — the recovery copy lives right next
    to the intended output (no temp files). Strategy (in order of durability):
      1. Write a recovery copy next to the movie FIRST.
      2. If the movie folder itself is unwritable (e.g. the SMB share dropped
         mid-scan), the recovery copy falls back to CWD — still no temp files.
      3. Best-effort write to the intended destination (often the SMB share).
         If that fails, fall back to CWD too.
      4. Keep whichever copy succeeded as the answer.

    Returns the path the result was actually saved to.
    """
    recovery_path = out_path.with_name(out_path.name + ".recovery.json")
    try:
        result.save(recovery_path)
    except OSError:
        # Movie folder dead — fall the recovery copy back to CWD.
        recovery_path = Path.cwd() / recovery_path.name
        try:
            result.save(recovery_path)
        except OSError as exc:
            raise RuntimeError(f"could not save scan result locally: {exc}") from exc

    saved_to = recovery_path
    try:
        result.save(out_path)
        saved_to = out_path
    except OSError as exc:
        fallback = Path.cwd() / out_path.name
        print(f"warning: could not write {out_path} ({exc}); falling back to {fallback}")
        try:
            result.save(fallback)
            saved_to = fallback
        except OSError as exc2:
            print(f"warning: could not write fallback {fallback} ({exc2}); "
                  f"recovery copy kept at {recovery_path}", file=sys.stderr)
    return saved_to


def _ensure_server(cfg: ModelServerConfig):
    """Start (or verify) the inference backend. Returns a cleanup callable or None."""
    log_path = Path(tempfile.gettempdir()) / "nuclearcutter_mlx_vlm.log"
    proc = ensure_backend(cfg, log_path=log_path)
    if proc is not None:
        atexit.register(lambda p=proc: (p.terminate(), p.wait(timeout=10)))
    return proc


def cmd_scan(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 1

    server_cfg = ModelServerConfig(
        backend=args.backend,
        model_path=args.model_path or DEFAULT_MLX_MODEL_PATH,
        base_url=args.base_url or MLX_BASE_URL,
    )
    print(f"Starting model backend: {args.backend} ...")
    try:
        _ensure_server(server_cfg)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    llm_config = LLMConfig(base_url=server_cfg.base_url, vlm_model=None, text_model=None)
    if server_cfg.backend == "mlx-vlm":
        llm_config.vlm_model = llm_config.text_model = server_cfg.model_path
    elif server_cfg.backend == "llama.cpp":
        from nuclearcutter.utils.model_server import _llama_model_id

        llm_config.vlm_model = llm_config.text_model = _llama_model_id(server_cfg.model_path)
    else:  # standalone
        if not args.vlm_model or not args.text_model:
            print("error: --vlm-model and --text-model are required with backend=standalone",
                  file=sys.stderr)
            return 1
        llm_config.vlm_model = args.vlm_model
        llm_config.text_model = args.text_model

    def progress(stage: str, detail):
        if detail is None:
            print(f"[{stage}]")
        elif stage == "visual_sweep" and detail:
            done, total = detail
            print(f"\r[{stage}] {done} / {total} frames", end="", flush=True)
        elif stage == "visual_confirm" and detail:
            i, total = detail
            print(f"\r[{stage}] {i} / {total} candidates", end="", flush=True)

    print(f"Scanning {video_path.name}...")
    status_path = video_path.with_suffix(".nuclearcutter.status.json")
    log_path = video_path.with_suffix(".nuclearcutter.log")
    out_path = Path(args.output) if args.output else _default_scan_path(video_path)

    from nuclearcutter.scan.scanner import scan as scan_pass

    result: ScanResult = scan_pass(
        video_path,
        llm_config=llm_config,
        title=args.title or Path(video_path).stem,
        year=args.year,
        progress_callback=progress,
        whisper_model=args.whisper_model or DEFAULT_WHISPER,
        sweep_interval=args.sweep_interval,
        status_path=status_path,
        partial_result_path=out_path,
        scale=args.scale or DEFAULT_SCALE,
    )
    print()

    try:
        saved_to = _save_result_with_fallback(result, out_path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    recovery_path = out_path.with_name(out_path.name + ".recovery.json")

    def _safe_print(msg: str = "", *, err: bool = False) -> None:
        try:
            print(msg, file=sys.stderr if err else sys.stdout)
        except (OSError, ValueError):
            pass

    _safe_print(f"Scan complete: {len(result.visual_detections)} visual detections, "
                f"{len(result.language_detections)} language detections.")
    _safe_print(f"Wrote {saved_to}")
    if saved_to != recovery_path:
        _safe_print(f"Recovery copy: {recovery_path}")
    _safe_print(f"Live status written to {status_path}")
    return 0


def _prefs_from_args(args) -> Preferences:
    return Preferences(
        nudity_visual=VisualAction(args.nudity_visual),
        nudity_audio=AudioAction(args.nudity_audio),
        nudity_level=SeverityLevel.from_any(args.nudity_level),
        gore_visual=VisualAction(args.gore_visual),
        gore_audio=AudioAction(args.gore_audio),
        gore_level=SeverityLevel.from_any(args.gore_level),
        violence_visual=VisualAction(args.violence_visual),
        violence_audio=AudioAction(args.violence_audio),
        violence_level=SeverityLevel.from_any(args.violence_level),
        foul_language_visual=VisualAction(args.foul_language_visual),
        foul_language_audio=AudioAction(args.foul_language_audio),
        foul_language_level=SeverityLevel.from_any(args.foul_language_level),
        blur_strength=args.blur_strength,
        mute_padding=args.mute_padding,
        blur_padding=args.blur_padding,
    )


def cmd_render(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 1

    scan_path = Path(args.scan) if args.scan else _default_scan_path(video_path)
    if not scan_path.exists():
        print(f"error: scan file not found: {scan_path} (run `nuclearcutter scan` first)",
              file=sys.stderr)
        return 1

    scan_result = ScanResult.load(scan_path)
    prefs = _prefs_from_args(args)
    output_path = Path(args.output) if args.output else build_output_path(video_path)
    print(f"Rendering {video_path.name} -> {output_path.name}...")

    def progress(stage: str, detail):
        if detail and stage == "render":
            i, total, _seg = detail
            print(f"\r[{stage}] segment {i}/{total}", end="", flush=True)
        elif detail is None:
            print(f"[{stage}]")

    render_pass(
        video_path, scan_result, prefs,
        output_path=output_path,
        font_path=args.font or None,
        progress_callback=progress,
    )
    print(f"\nDone: {output_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from nuclearcutter.server import run_server

    run_server(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nuclearcutter",
        description="Detect and censor content in local movie files. "
                    "Default: start the web GUI server (open http://localhost:8000).",
    )
    parser.set_defaults(func=lambda a: cmd_serve(a))  # serve is the default command
    parser.set_defaults(host="0.0.0.0", port=8000)  # defaults for the default (serve) command
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start the web GUI server (default when no command is given).")
    p_serve.add_argument("--host", default="0.0.0.0", help="Interface to bind (default 0.0.0.0 = all devices on the network).")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    p_serve.set_defaults(func=cmd_serve)

    p_scan = sub.add_parser("scan", help="Scan a movie file (headless) and produce a timestamp JSON.")
    p_scan.add_argument("video")
    p_scan.add_argument("--output", "-o", help="Output path for scan JSON (default: MOVIE.nuclearcutter.json)")
    p_scan.add_argument("--title", help="Movie title, stored in the scan file for reference")
    p_scan.add_argument("--year", type=int, help="Release year, stored in the scan file for reference")
    p_scan.add_argument("--backend", choices=["mlx-vlm", "llama.cpp", "standalone"], default="mlx-vlm")
    p_scan.add_argument("--model-path", default=None, help="Local model path (mlx-vlm dir or llama.cpp .gguf)")
    p_scan.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL (standalone)")
    p_scan.add_argument("--vlm-model", default=None, help="VLM model id (required with --backend standalone)")
    p_scan.add_argument("--text-model", default=None, help="Text model id (required with --backend standalone)")
    p_scan.add_argument("--whisper-model", default=None, help=f"Whisper model (default: {DEFAULT_WHISPER})")
    p_scan.add_argument("--scale", choices=["360p", "480p", "720p", "1080p"], default=None,
                        help=f"Scale frames before VLM (default: {DEFAULT_SCALE})")
    p_scan.add_argument("--sweep-interval", type=float, default=2.0,
                        help="Seconds between sampled frames in the VLM sweep (default: 2.0)")
    p_scan.set_defaults(func=cmd_scan)

    p_render = sub.add_parser("render", help="Render a cleaned copy of a movie file (headless).")
    p_render.add_argument("video")
    p_render.add_argument("--scan", "-s", help="Path to scan JSON (default: MOVIE.nuclearcutter.json)")
    p_render.add_argument("--output", "-o", help="Output path (default: MOVIE_cleaned.ext)")
    p_render.add_argument("--nudity-visual", choices=["none", "blur", "black"], default="blur")
    p_render.add_argument("--nudity-audio", choices=["none", "mute_scene"], default="none")
    p_render.add_argument("--nudity-level", choices=["low", "med", "high", "exhigh"], default="med")
    p_render.add_argument("--gore-visual", choices=["none", "blur", "black"], default="blur")
    p_render.add_argument("--gore-audio", choices=["none", "mute_scene"], default="none")
    p_render.add_argument("--gore-level", choices=["low", "med", "high", "exhigh"], default="med")
    p_render.add_argument("--violence-visual", choices=["none", "blur", "black"], default="blur")
    p_render.add_argument("--violence-audio", choices=["none", "mute_scene"], default="none")
    p_render.add_argument("--violence-level", choices=["low", "med", "high", "exhigh"], default="med")
    p_render.add_argument("--foul-language-visual", choices=["none", "blur", "black"], default="none")
    p_render.add_argument("--foul-language-audio",
                          choices=["none", "mute_word", "mute_phrase", "replace_word", "replace_phrase"],
                          default="mute_phrase")
    p_render.add_argument("--foul-language-level", choices=["low", "med", "high", "exhigh"], default="med")
    p_render.add_argument("--blur-strength", type=float, default=1.0,
                          help="Blur intensity multiplier (1.0 = standard, 2.0 = twice as extreme)")
    p_render.add_argument("--mute-padding", type=float, default=0.5,
                          help="Extra seconds muted before/after each flagged word (default: 0.5)")
    p_render.add_argument("--blur-padding", type=float, default=0.0,
                          help="Extra seconds blurred/blacked before/after each flagged segment (default: 0.0)")
    p_render.add_argument("--font", default=None, help="Path to a .ttf font file for blur overlay text")
    p_render.set_defaults(func=cmd_render)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
