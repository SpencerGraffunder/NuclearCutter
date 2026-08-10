"""
CLI entrypoint. See docs/SPEC.md section 7 — CLI-only for v1, two primary
commands mapping to the two-pass architecture.

Settings come from config.toml (see nuclearcutter/utils/config.py); the only
required argument is the movie path. CLI flags override the config when given.

Usage:
    nuclearcutter scan MOVIE.mkv
    nuclearcutter render MOVIE.mkv
    nuclearcutter render MOVIE.mkv --nudity none   # override a config preference
"""

from __future__ import annotations

import argparse
import atexit
import sys
import tempfile
from pathlib import Path

from nuclearcutter.render.renderer import build_output_path, render as render_pass
from nuclearcutter.scan.repo_match import find_matching_scan, spot_check_match
from nuclearcutter.scan.scanner import scan as scan_pass
from nuclearcutter.schema import AudioAction, Preferences, ScanResult, SeverityLevel, VisualAction
from nuclearcutter.utils.config import AppConfig, default_config_path, load_config
from nuclearcutter.utils.llm_client import LLMClient, LLMConfig
from nuclearcutter.utils.model_server import ModelServerConfig, ensure_backend


def _default_scan_path(video_path: Path) -> Path:
    return video_path.with_suffix(".nuclearcutter.json")


def _tui_enabled(args) -> bool:
    """Whether the inline TUI should run for this invocation.

    On by default unless the user passes `--no-tui` or stdout is not a TTY
    (e.g. piped/redirected output), where a full-screen dashboard would be
    useless and could corrupt a log file.
    """
    if getattr(args, "no_tui", False):
        return False
    if not sys.stdout.isatty():
        return False
    return True


def _resolve_config(args) -> AppConfig:
    cfg_path = Path(args.config) if getattr(args, "config", None) else None
    cfg = load_config(cfg_path)

    # CLI flags override config when explicitly provided.
    if getattr(args, "backend", None):
        cfg.model_backend = args.backend
    if getattr(args, "base_url", None):
        cfg.base_url = args.base_url
    if getattr(args, "vlm_model", None):
        cfg.vlm_model = args.vlm_model
    if getattr(args, "text_model", None):
        cfg.text_model = args.text_model
    if getattr(args, "whisper_model", None):
        cfg.whisper_model = args.whisper_model
    if getattr(args, "sweep_interval", None) is not None:
        cfg.sweep_interval = args.sweep_interval
    if getattr(args, "timestamps_dir", None):
        cfg.timestamps_dir = args.timestamps_dir
    if getattr(args, "vision_timeout", None) is not None:
        cfg.vision_timeout = args.vision_timeout
    return cfg


def _ensure_server(cfg: AppConfig):
    """Start (or verify) the inference backend. Returns a cleanup callable or None."""
    server_cfg = ModelServerConfig(
        backend=cfg.model_backend,
        model_path=cfg.model_path,
        base_url=cfg.base_url,
    )
    log_path = Path(tempfile.gettempdir()) / "nuclearcutter_mlx_vlm.log"
    proc = ensure_backend(server_cfg, log_path=log_path)
    if proc is not None:
        atexit.register(lambda p=proc: (p.terminate(), p.wait(timeout=10)))
    return proc


def cmd_scan(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 1

    cfg = _resolve_config(args)

    if not cfg.vlm_model or not cfg.text_model or not cfg.whisper_model:
        print(
            "error: scan needs vlm_model, text_model, and whisper_model.\n"
            "  Set them in config.toml, or pass --vlm-model/--text-model/--whisper-model.",
            file=sys.stderr,
        )
        return 1

    # Start/verify the inference backend before doing any work.
    print(f"Starting model backend: {cfg.model_backend} ...")
    try:
        _ensure_server(cfg)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    llm_config = LLMConfig(
        base_url=cfg.base_url,
        vlm_model=cfg.vlm_model,
        text_model=cfg.text_model,
    )
    llm_config.vision_timeout = cfg.vision_timeout
    # The mlx-vlm server pre-loads the model under its full filesystem path and
    # caches it keyed by that exact string — requests must name that full path
    # as the model id or the server tries to fetch the short name from HF.
    # Both VLM and text checks go through the same model, per the design.
    if cfg.model_backend == "mlx-vlm":
        llm_config.vlm_model = cfg.model_path
        llm_config.text_model = cfg.model_path

    if cfg.timestamps_dir and not args.force_rescan:
        timestamps_dir = Path(cfg.timestamps_dir)
        print(f"Checking {timestamps_dir} for an existing scan...")
        candidate = find_matching_scan(video_path, timestamps_dir)
        if candidate:
            print("Fingerprint match found. Spot-checking with VLM before trusting it...")
            client = LLMClient(llm_config)
            if spot_check_match(video_path, candidate, client):
                out_path = Path(args.output) if args.output else _default_scan_path(video_path)
                try:
                    candidate.save(out_path)
                    print(f"Verified match. Wrote scan result to {out_path} (skipped full rescan).")
                except PermissionError:
                    fallback = Path.cwd() / out_path.name
                    print(f"warning: no write permission at {out_path.parent}, falling back to {fallback}")
                    candidate.save(fallback)
                    print(f"Verified match. Wrote scan result to {fallback} (skipped full rescan).")
                return 0
            else:
                print("Spot-check failed to confirm match closely enough — falling back to full scan.")

    def progress(stage: str, detail):
        if detail is None:
            print(f"[{stage}]")
        elif stage in ("visual_sweep",) and detail:
            t, total = detail
            print(f"\r[{stage}] {t:.0f}s / {total:.0f}s", end="", flush=True)
        elif stage == "visual_confirm" and detail:
            i, total = detail
            print(f"\r[{stage}] {i} / {total} candidates", end="", flush=True)

    print(f"Scanning {video_path.name}...")
    status_path = Path(args.status_file) if args.status_file else None
    log_path = Path(tempfile.gettempdir()) / f"nuclearcutter_scan_{video_path.stem}.log"

    def _do_scan():
        return scan_pass(
            video_path,
            llm_config=llm_config,
            title=args.title or Path(video_path).stem,
            year=args.year,
            progress_callback=progress,
            whisper_model=cfg.whisper_model,
            sweep_interval=cfg.sweep_interval,
            status_path=status_path,
            category_prompts=_category_prompts_from_config(cfg),
        )

    use_tui = _tui_enabled(args)
    # Always write a live status file (unless the user explicitly disabled it),
    # so `nuclearcutter tui --status ...` can attach to this scan even when it
    # runs with piped/redirected stdout (where the inline TUI is auto-disabled).
    if status_path is None:
        status_path = Path(tempfile.gettempdir()) / f"nuclearcutter_scan_{video_path.stem}.status.json"

    try:
        if use_tui:
            # Inline dashboard: run the scan in a worker thread while the TUI
            # renders from the live status file.
            from nuclearcutter.tui import run_with_tui

            result: ScanResult = run_with_tui(_do_scan, status_path, log_path=log_path)
        else:
            result: ScanResult = _do_scan()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print()

    out_path = Path(args.output) if args.output else _default_scan_path(video_path)
    try:
        result.save(out_path)
    except PermissionError:
        fallback = Path.cwd() / out_path.name
        print(f"warning: no write permission at {out_path.parent}, falling back to {fallback}")
        result.save(fallback)
        out_path = fallback
    print(f"Scan complete: {len(result.visual_detections)} visual detections, "
          f"{len(result.language_detections)} language detections.")
    print(f"Wrote {out_path}")
    if status_path:
        print(f"Live status written to {status_path} (watch with `nuclearcutter tui --status {status_path}`)")

    if cfg.timestamps_dir:
        print(f"\nTo share this scan, copy {out_path.name} into {cfg.timestamps_dir} and open a PR.")

    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    from nuclearcutter.tui import run_tui

    run_tui(
        status_path=args.status,
        log_path=args.log,
        interval=args.interval,
        sweep_interval=args.sweep_interval,
    )
    return 0


def _prefs_from_config(cfg: AppConfig, args) -> Preferences:
    """Build Preferences from config.toml, with CLI flags overriding."""
    return Preferences(
        nudity_visual=VisualAction(_arg_or_cfg(args, "nudity_visual", cfg.nudity_visual)),
        nudity_audio=AudioAction(_arg_or_cfg(args, "nudity_audio", cfg.nudity_audio)),
        nudity_level=SeverityLevel.from_any(_arg_or_cfg(args, "nudity_level", cfg.nudity_level)),
        gore_visual=VisualAction(_arg_or_cfg(args, "gore_visual", cfg.gore_visual)),
        gore_audio=AudioAction(_arg_or_cfg(args, "gore_audio", cfg.gore_audio)),
        gore_level=SeverityLevel.from_any(_arg_or_cfg(args, "gore_level", cfg.gore_level)),
        violence_visual=VisualAction(_arg_or_cfg(args, "violence_visual", cfg.violence_visual)),
        violence_audio=AudioAction(_arg_or_cfg(args, "violence_audio", cfg.violence_audio)),
        violence_level=SeverityLevel.from_any(_arg_or_cfg(args, "violence_level", cfg.violence_level)),
        foul_language_visual=VisualAction(_arg_or_cfg(args, "foul_language_visual", cfg.foul_language_visual)),
        foul_language_audio=AudioAction(_arg_or_cfg(args, "foul_language_audio", cfg.foul_language_audio)),
        foul_language_level=SeverityLevel.from_any(_arg_or_cfg(args, "foul_language_level", cfg.foul_language_level)),
        blur_strength=_float_or_cfg(args, "blur_strength", cfg.blur_strength),
        mute_padding=_float_or_cfg(args, "mute_padding", cfg.mute_padding),
    )


def _category_prompts_from_config(cfg: AppConfig) -> dict:
    """Per-category CUSTOM prompts from config (empty = use the built-in fixed
    level scale). Only passed when the user explicitly overrides."""
    return {
        "nudity": cfg.nudity_prompt,
        "gore": cfg.gore_prompt,
        "violence": cfg.violence_prompt,
        "foul_language": cfg.foul_language_prompt,
    }


def _float_or_cfg(args, name: str, cfg_value: float) -> float:
    val = getattr(args, name, None)
    return float(val) if val is not None else float(cfg_value)


def _arg_or_cfg(args, name: str, cfg_value: str) -> str:
    val = getattr(args, name, None)
    return val if val else cfg_value


def cmd_render(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 1

    cfg = _resolve_config(args)
    scan_path = Path(args.scan) if args.scan else _default_scan_path(video_path)
    if not scan_path.exists():
        print(f"error: scan file not found: {scan_path} (run `nuclearcutter scan` first)", file=sys.stderr)
        return 1

    scan_result = ScanResult.load(scan_path)

    if args.prefs:
        prefs = Preferences.load(Path(args.prefs))
    else:
        prefs = _prefs_from_config(cfg, args)

    output_path = Path(args.output) if args.output else build_output_path(video_path)
    print(f"Rendering {video_path.name} -> {output_path.name}...")

    def _do_render():
        return render_pass(
            video_path, scan_result, prefs,
            output_path=output_path,
            font_path=args.font or (cfg.font or None),
            status_path=status_path,
        )

    status_path = Path(tempfile.gettempdir()) / f"nuclearcutter_render_{video_path.stem}.status.json"
    log_path = Path(tempfile.gettempdir()) / f"nuclearcutter_render_{video_path.stem}.log"
    if _tui_enabled(args):
        from nuclearcutter.tui import run_with_tui

        run_with_tui(_do_render, status_path, log_path=log_path)
    else:
        _do_render()
    print(f"Done: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nuclearcutter", description="Detect and censor content in local movie files.")
    parser.add_argument("--config", default=None, help=f"Path to config.toml (default: {default_config_path()})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan a movie file and produce a timestamp JSON.")
    p_scan.add_argument("video")
    p_scan.add_argument("--output", "-o", help="Output path for scan JSON (default: MOVIE.nuclearcutter.json)")
    p_scan.add_argument("--status-file", default=None,
                        help="Write a live JSON status file while scanning (watch with `nuclearcutter tui`)")
    p_scan.add_argument("--no-tui", action="store_true", help="Disable the inline live dashboard")
    p_scan.add_argument("--force-rescan", action="store_true", help="Skip the shared-repo match check")
    p_scan.add_argument("--title", help="Movie title, stored in the scan file for reference")
    p_scan.add_argument("--year", type=int, help="Release year, stored in the scan file for reference")
    p_scan.add_argument("--backend", choices=["mlx-vlm", "standalone"], default=None,
                        help="Inference backend (default: from config, mlx-vlm)")
    p_scan.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL (default: from config)")
    p_scan.add_argument("--vlm-model", default=None, help="Vision model name (default: from config)")
    p_scan.add_argument("--text-model", default=None, help="Text model name (default: from config)")
    p_scan.add_argument("--whisper-model", default=None, help="Whisper model (default: from config)")
    p_scan.add_argument("--timestamps-dir", default=None, help="Directory of shared scan JSONs (default: from config)")
    p_scan.add_argument("--vision-timeout", type=int, default=None,
                        help="Timeout in seconds for VLM vision requests (default: from config)")
    p_scan.add_argument("--sweep-interval", type=float, default=None,
                        help="Seconds between sampled frames in the VLM sweep (default: from config)")
    p_scan.set_defaults(func=cmd_scan)

    p_render = sub.add_parser("render", help="Render a cleaned copy of a movie file using a scan JSON + preferences.")
    p_render.add_argument("video")
    p_render.add_argument("--scan", "-s", help="Path to scan JSON (default: MOVIE.nuclearcutter.json)")
    p_render.add_argument("--output", "-o", help="Output path (default: MOVIE_cleaned.ext)")
    p_render.add_argument("--prefs", help="Path to a saved Preferences JSON, overrides config below")
    p_render.add_argument("--nudity-visual", choices=["none", "blur", "black"], default=None)
    p_render.add_argument("--nudity-audio", choices=["none", "mute_scene"], default=None)
    p_render.add_argument("--nudity-level", choices=["low", "med", "high", "exhigh"], default=None,
                          help="Correct nudity at/above this severity level")
    p_render.add_argument("--gore-visual", choices=["none", "blur", "black"], default=None)
    p_render.add_argument("--gore-audio", choices=["none", "mute_scene"], default=None)
    p_render.add_argument("--gore-level", choices=["low", "med", "high", "exhigh"], default=None)
    p_render.add_argument("--violence-visual", choices=["none", "blur", "black"], default=None)
    p_render.add_argument("--violence-audio", choices=["none", "mute_scene"], default=None)
    p_render.add_argument("--violence-level", choices=["low", "med", "high", "exhigh"], default=None)
    p_render.add_argument("--foul-language-visual", choices=["none", "blur", "black"], default=None)
    p_render.add_argument("--foul-language-audio", choices=["none", "mute_word", "mute_phrase", "replace_word", "replace_phrase"], default=None)
    p_render.add_argument("--foul-language-level", choices=["low", "med", "high", "exhigh"], default=None)
    p_render.add_argument("--blur-strength", type=float, default=None,
                          help="Blur intensity multiplier (1.0 = standard, 2.0 = twice as extreme)")
    p_render.add_argument("--mute-padding", type=float, default=None,
                          help="Extra seconds muted before/after each flagged word (default: from config, 0.5)")
    p_render.add_argument("--no-tui", action="store_true", help="Disable the inline live dashboard")
    p_render.add_argument("--font", default=None, help="Path to a .ttf font file for blur overlay text (default: from config)")
    p_render.set_defaults(func=cmd_render)

    p_tui = sub.add_parser("tui", help="Live scan dashboard (attach to a running scan or a status file).")
    p_tui.add_argument("--status", default=None, help="Path to a live scan status JSON (from `scan --status-file`).")
    p_tui.add_argument("--log", default=None, help="Path to a scan log file to tail (attach mode, works on a running scan).")
    p_tui.add_argument("--interval", type=float, default=1.0, help="UI refresh interval in seconds.")
    p_tui.add_argument("--sweep-interval", type=float, default=2.0,
                       help="Sweep sample interval (s) used to map frames to seconds in log-attach mode.")
    p_tui.set_defaults(func=cmd_tui)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
