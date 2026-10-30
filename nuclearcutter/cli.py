"""
NuclearCutter CLI.

Primary interface is the web GUI (this repo's default):
    python3 nuclearcutter.py serve          # or just: python3 nuclearcutter.py
    -> open http://localhost:8000 in a browser (any device on the network
       can reach it at http://<this-machine-ip>:8000, no login)

Headless scan/render are still available for scripting:
    python3 nuclearcutter.py scan  MOVIE.mkv --vlm-model MODEL [--scale 480p ...]
    python3 nuclearcutter.py tags   --search bikini        # what can be censored
    python3 nuclearcutter.py render MOVIE.mkv --preset graphic --allow gore
    python3 nuclearcutter.py render MOVIE.mkv --censor nudity.full_nude_female \
                                  --censor language.slur_racial --visual gore=black

A render censors NOTHING unless you select something with --preset/--censor, so
headless runs never surprise you with someone else's idea of what is offensive.
The scan file holds facts (tag ids); the selection is yours, every time.

There is no per-user config file for what to censor — the scan file records what
is in the film, and the choice of what to hide is made at render time in the GUI
(or with flags here). VLM prompts come from prompts.json
(see nuclearcutter/prompts.py), shared by the GUI and the scanner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from nuclearcutter.render.renderer import build_output_path, render as render_pass
from nuclearcutter.schema import (
    AudioAction, Preferences, ScanResult, Selection, VisualAction,
)
from nuclearcutter.taxonomy import (
    PRESET_NAMES, TAGS, Family, Minors, Rendering, audio_options,
    selection_named,
)
from nuclearcutter.utils.llm_client import LLMConfig
from nuclearcutter.utils.model_server import (
    DEFAULT_BASE_URL, WHISPER_DEFAULT_BASE_URL, WHISPER_DEFAULT_MODELS_DIR,
    WhisperConfig, require_server_up,
)

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


def cmd_scan(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 1

    base_url = args.base_url or DEFAULT_BASE_URL
    print(f"Connecting to model server: {base_url} ...")
    try:
        require_server_up(base_url, api_key=args.api_key)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    vlm_model = args.vlm_model
    text_model = args.text_model or vlm_model
    if not vlm_model:
        print("error: --vlm-model is required (the VLM model id served by --base-url)",
              file=sys.stderr)
        return 1

    llm_config = LLMConfig(base_url=base_url, vlm_model=vlm_model, text_model=text_model,
                           api_key=args.api_key or "not-needed")

    def progress(stage: str, detail):
        if detail is None:
            print(f"[{stage}]")
        elif stage == "visual_sweep" and detail:
            done, total = detail
            print(f"\r[{stage}] {done} / {total} frames", end="", flush=True)
        elif stage == "visual_confirm" and detail:
            i, total = detail
            print(f"\r[{stage}] {i} / {total} candidates", end="", flush=True)
        elif stage == "summary":
            if detail:
                i, total = detail
                print(f"\r[{stage}] {i} / {total} descriptions", end="", flush=True)
            else:
                print(f"\r[{stage}] rewriting descriptions with the summary model", end="", flush=True)

    print(f"Scanning {video_path.name}...")
    status_path = video_path.with_suffix(".nuclearcutter.status.json")
    log_path = video_path.with_suffix(".nuclearcutter.log")
    out_path = Path(args.output) if args.output else _default_scan_path(video_path)

    from nuclearcutter.scan.scanner import scan as scan_pass

    whisper_cfg = WhisperConfig(
        base_url=args.whisper_base_url or WHISPER_DEFAULT_BASE_URL,
        models_dir=args.whisper_models_dir or WHISPER_DEFAULT_MODELS_DIR,
        model=args.whisper_model or "",
    )

    result: ScanResult = scan_pass(
        video_path,
        llm_config=llm_config,
        title=args.title or Path(video_path).stem,
        year=args.year,
        progress_callback=progress,
        whisper_cfg=whisper_cfg,
        sweep_interval=args.sweep_interval,
        sweep_resolution=args.sweep_resolution,
        status_path=status_path,
        partial_result_path=out_path,
        scale=args.scale or DEFAULT_SCALE,
        run_dialogue_sweep=not args.no_dialogue_sweep,
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


def _split_ids(value: str) -> list[str]:
    """`--censor a,b` and `--censor a --censor b` both work."""
    out: list[str] = []
    for part in (value or "").split(","):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return out


def _expand_id(raw: str) -> list[str]:
    """Resolve one `--censor` argument: a tag id, or a whole family.

    `--censor gore` meaning "every gore tag" is what people expect from a flag,
    and it is unambiguous because family names never contain a dot.
    """
    tid = raw.strip().lower()
    if tid in TAGS:
        return [tid]
    try:
        fam = Family(tid)
    except ValueError:
        raise SystemExit(
            f"unknown category or tag: {raw!r}\n"
            f"  run `nuclearcutter tags` to list the vocabulary, or\n"
            f"  `nuclearcutter tags --search {tid}` to look for one")
    return [t.id for t in TAGS.values() if t.family is fam]


def _selection_from_args(args) -> Selection:
    """Build the viewer's selection from --preset / --censor / --allow / --grep.

    Order matters and is documented in --help: preset, then adds, then removals.
    That way `--preset everything --allow gore` reads the way it sounds, and a
    script that only says `--censor nudity.full_nude_female` censors exactly that
    and nothing else (the default is still nothing).
    """
    sel = selection_named(getattr(args, "preset", None) or "none")
    for raw in getattr(args, "censor", None) or []:
        for part in _split_ids(raw):
            for tid in _expand_id(part):
                sel.toggle(tid, True)
    for raw in getattr(args, "allow", None) or []:
        for part in _split_ids(raw):
            for tid in _expand_id(part):
                sel.toggle(tid, False)
    for pattern in getattr(args, "grep", None) or []:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise SystemExit(f"bad --grep pattern {pattern!r}: {exc}")
        matched = [t.id for t in TAGS.values()
                   if rx.search(t.id) or rx.search(t.label)]
        if not matched:
            raise SystemExit(f"--grep {pattern!r} matched no tags "
                             f"(try `nuclearcutter tags --search {pattern}`)")
        for tid in matched:
            sel.toggle(tid, True)
    if getattr(args, "tag_file", None):
        path = Path(args.tag_file)
        try:
            raw = path.read_text()
        except OSError as exc:
            raise SystemExit(f"cannot read --tag-file {path}: {exc}")
        ids: list[str] = []
        if raw.lstrip().startswith("{") or raw.lstrip().startswith("["):
            try:
                data = json.loads(raw)
            except ValueError as exc:
                raise SystemExit(f"--tag-file {path} is not valid JSON: {exc}")
            if isinstance(data, dict):
                data = data.get("tags", [])
            ids = [str(x) for x in data]
        else:  # one id per line, # comments allowed — hand-editable on purpose
            ids = [ln.strip() for ln in raw.splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
        for tid in ids:
            if tid not in TAGS:
                raise SystemExit(f"--tag-file {path}: unknown tag id {tid!r}")
            sel.toggle(tid, True)
    if getattr(args, "rendering", None):
        wanted = {r.strip().lower() for r in _split_ids(args.rendering)}
        known = {r.value for r in Rendering}
        bad = wanted - known
        if bad:
            raise SystemExit(f"unknown rendering style(s): {', '.join(sorted(bad))} "
                             f"(known: {', '.join(sorted(known))})")
        sel.renderings = wanted
    return sel


def _family_action_overrides(args, kind: str) -> dict[str, str]:
    """Parse repeatable `--visual gore=black` / `--audio language=mute_word`."""
    out: dict[str, str] = {}
    allowed_visual = {a.value for a in VisualAction}
    for raw in getattr(args, kind, None) or []:
        fam_text, sep, act_text = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"--{kind} expects FAMILY=ACTION, got {raw!r}")
        fam_text = fam_text.strip().lower()
        act_text = act_text.strip().lower()
        try:
            fam = Family(fam_text)
        except ValueError:
            raise SystemExit(f"--{kind}: unknown family {fam_text!r} "
                             f"(known: {', '.join(f.value for f in Family)})")
        if kind == "visual":
            if act_text not in allowed_visual:
                raise SystemExit(f"--visual: unknown action {act_text!r} "
                                 f"(known: {', '.join(sorted(allowed_visual))})")
        else:
            options = audio_options(fam)
            if act_text not in options:
                raise SystemExit(f"--audio {fam.value}: {act_text!r} is not available "
                                 f"for this family (options: {', '.join(options)})")
        out[fam.value] = act_text
    return out


def _prefs_from_args(args) -> Preferences:
    """Assemble the render's settings from the command line.

    Unmentioned families keep the taxonomy defaults, so `--preset nudity` alone
    still means something sensible for language (it does not censor it).
    """
    return Preferences(
        selection=_selection_from_args(args),
        visual_actions=_family_action_overrides(args, "visual"),
        audio_actions=_family_action_overrides(args, "audio"),
        blur_strength=args.blur_strength,
        mute_padding=args.mute_padding,
        blur_padding=args.blur_padding,
    )


def cmd_tags(args: argparse.Namespace) -> int:
    """Print the vocabulary the --censor/--allow flags take.

    Exists because a list of ~120 ids nobody can see is a list nobody uses, and
    because the alternative — hardcoding ids in someone's script from reading a
    doc that might be stale — fails silently (an unknown id selects nothing).
    """
    if args.presets:
        for name in PRESET_NAMES:
            ids = selection_named(name).tags
            print(f"{name:10s} {len(ids):3d} tags")
        return 0

    wanted = list(TAGS.values())
    if args.family:
        try:
            fam = Family(args.family.strip().lower())
        except ValueError:
            print(f"error: unknown family {args.family!r} "
                  f"(known: {', '.join(f.value for f in Family)})", file=sys.stderr)
            return 2
        wanted = [t for t in wanted if t.family is fam]
    if args.audio:
        wanted = [t for t in wanted if t.audio]
    if args.visual:
        wanted = [t for t in wanted if t.visual]
    if args.search:
        try:
            rx = re.compile(args.search, re.IGNORECASE)
            wanted = [t for t in wanted if rx.search(t.id) or rx.search(t.label)]
        except re.error as exc:
            print(f"error: bad pattern {args.search!r}: {exc}", file=sys.stderr)
            return 2

    if args.json:
        print(json.dumps([{"id": t.id, "family": t.family.value, "label": t.label,
                           "audio": t.audio, "visual": t.visual,
                           "always_censor": t.always_censor} for t in wanted],
                         indent=2))
        return 0

    current_family = None
    for t in wanted:
        if t.family is not current_family:
            current_family = t.family
            print(f"\n{current_family.label}  ({current_family.value})")
        flags = "".join([" audio" if t.audio else "", " VISUAL" if t.visual else "",
                         " ALWAYS-ON" if t.always_censor else ""])
        print(f"  {t.id:38s} {t.label}{flags}")
    print(f"\n{len(wanted)} tag(s). Pick with --censor ID  or  --censor FAMILY.")
    return 0


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

    # Say what is about to be hidden, before an hour of encoding. A headless run
    # has no timeline to look at, and discovering afterwards that --allow gore did
    # not do what you meant means re-encoding the film.
    hits = [d for d in scan_result.visual_detections if d.censorable(prefs.selection)]
    lang_hits = [d for d in scan_result.language_detections
                 if d.censorable(prefs.selection)]
    kinds: dict[str, int] = {}
    for d in hits:
        for t in d.censorable(prefs.selection):
            tag = TAGS.get(t)
            key = tag.label if tag else t
            kinds[key] = kinds.get(key, 0) + 1
    print(f"Rendering {video_path.name} -> {output_path.name}")
    if not prefs.selection.tags:
        print("  nothing is selected to censor — the picture and sound will be "
              "unchanged. Use --preset or --censor (see `nuclearcutter tags`).")
    else:
        print(f"  censoring {len(hits)} of {len(scan_result.visual_detections)} visual "
              f"and {len(lang_hits)} of {len(scan_result.language_detections)} language "
              f"detection(s), from {len(prefs.selection.tags)} selected categorie(s)")
        top = sorted(kinds.items(), key=lambda kv: -kv[1])[:8]
        for label, n in top:
            print(f"    {n:4d}  {label}")
        if len(kinds) > 8:
            print(f"    ...and {len(kinds) - 8} more categorie(s)")

    # Optional: a separate, larger model writes nicer on-screen text for
    # blurred/blacked segments (--polish-descriptions) and small captions over
    # audio-only mutes. Any failure degrades gracefully; the scan file already
    # holds a one-line description of each segment.
    polish = bool(getattr(args, "polish_descriptions", False))
    if polish and not args.summary_model:
        print("error: --polish-descriptions needs --summary-model", file=sys.stderr)
        return 2
    summarizer = None
    if args.summary_model:
        try:
            from nuclearcutter.render.summarize import SegmentSummarizer, SummaryConfig
            from nuclearcutter.utils.llm_client import LLMClient, LLMConfig

            base_url = args.base_url or DEFAULT_BASE_URL
            cfg = LLMConfig(
                base_url=base_url,
                vlm_model=args.summary_model,
                text_model=args.summary_model,
                summary_model=args.summary_model,
                summary_frames=args.summary_frames,
                summary_max_context=args.summary_max_context,
                api_key=args.api_key or "not-needed",
            )
            client = LLMClient(cfg)
            transcript_path = video_path.with_suffix(".nuclearcutter.transcript.json")
            summarizer = SegmentSummarizer(SummaryConfig(
                client=client,
                video_path=video_path,
                transcript_path=transcript_path if transcript_path.exists() else None,
                frames=args.summary_frames,
                max_context=args.summary_max_context,
            ))
            warn = summarizer.preflight_context_warning()
            if warn:
                print(f"[summary] WARNING: {warn}", file=sys.stderr)
            print(f"[summary] summary model {args.summary_model!r} enabled "
                  f"({args.summary_frames} frame(s)/segment)", file=sys.stderr)
        except Exception as exc:
            print(f"[summary] summary pass disabled: {exc}", file=sys.stderr)
            summarizer = None

    def progress(stage: str, detail):
        if detail and stage == "render":
            i, total, _seg = detail
            print(f"\r[{stage}] segment {i}/{total}", end="", flush=True)
        elif detail is None:
            print(f"[{stage}]")

    render_pass(
        video_path, scan_result, prefs,
        output_path=output_path,
        progress_callback=progress,
        summarizer=summarizer,
        polish_descriptions=polish,
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
    p_scan.add_argument("--base-url", default=None,
                        help=f"OpenAI-compatible /v1 model server (default: {DEFAULT_BASE_URL})")
    p_scan.add_argument("--api-key", default="",
                        help="Bearer token if the model server requires one (Strata does).")
    p_scan.add_argument("--vlm-model", required=True,
                        help="VLM model id served by --base-url (required)")
    p_scan.add_argument("--text-model", default=None,
                        help="Text model id (default: same as --vlm-model)")
    p_scan.add_argument("--whisper-base-url", default=None,
                        help=f"whisper.cpp transcription server (default: {WHISPER_DEFAULT_BASE_URL})")
    p_scan.add_argument("--whisper-model", default=None,
                        help="whisper model id (file stem) or absolute path; empty = use whatever the server has loaded")
    p_scan.add_argument("--whisper-models-dir", default=None,
                        help=f"dir of the whisper server's .bin models (default: {WHISPER_DEFAULT_MODELS_DIR})")
    p_scan.add_argument("--scale", choices=["360p", "480p", "720p", "1080p"], default=None,
                        help=f"Scale frames before VLM (default: {DEFAULT_SCALE})")
    p_scan.add_argument("--sweep-interval", type=float, default=2.0,
                        help="Seconds between sampled frames in the VLM sweep (default: 2.0)")
    p_scan.add_argument("--sweep-resolution", choices=["full", "360p", "240p"], default="full",
                        help="Per-image pixel cap for the sweep pass only — lower is much faster "
                             "(the vision server's preprocessing cost scales with pixels); "
                             "confirm/summary passes always run at full --scale (default: full)")
    p_scan.add_argument("--no-dialogue-sweep", action="store_true",
                        help="Skip the dialogue sweep: the pass that reads the whole transcript "
                             "for slurs, threats and sexual talk that a wordlist cannot see. "
                             "Saves time; costs recall (its detections are the only source of "
                             "language.slur_*, language.degradation_insult, ...).")
    p_scan.add_argument("--identify-scale", choices=["360p", "480p", "720p", "1080p"], default=None,
                        help="Resolution for the classify pass only. Default: --scale. "
                             "Gendered modesty tags need enough pixels to tell a man from a woman.")
    p_scan.set_defaults(func=cmd_scan)

    p_tags = sub.add_parser("tags", help="List or search the tag vocabulary.")
    p_tags.add_argument("--search", "-q", help="Substring or regex to match against tag ids and labels.")
    p_tags.add_argument("--family", help=f"One of: {', '.join(f.value for f in Family)}")
    p_tags.add_argument("--audio", action="store_true", help="Only audio-actionable tags.")
    p_tags.add_argument("--visual", action="store_true", help="Only picture-affecting tags.")
    p_tags.add_argument("--json", action="store_true", help="Machine-readable output.")
    p_tags.add_argument("--presets", action="store_true", help="List the --preset names instead of tags.")
    p_tags.set_defaults(func=cmd_tags)

    p_render = sub.add_parser("render", help="Render a cleaned copy of a movie file (headless).")
    p_render.add_argument("video")
    p_render.add_argument("--scan", "-s", help="Path to scan JSON (default: MOVIE.nuclearcutter.json)")
    p_render.add_argument("--output", "-o", help="Output path (default: MOVIE_cleaned.ext)")
    p_render.add_argument(
        "--preset", choices=list(PRESET_NAMES), default=None,
        help="Start from a group of categories, then refine with --censor/--allow. "
             "Default: nothing is censored.")
    p_render.add_argument("--censor", action="append", metavar="TAG|FAMILY",
                          help="Add a tag id, or a whole family (gore, nudity, ...). "
                               "Repeatable and comma-separated. Applied after --preset.")
    p_render.add_argument("--allow", action="append", metavar="TAG|FAMILY",
                          help="Remove a tag id or family from the selection. "
                               "Applied after --censor, so `--preset everything "
                               "--allow gore` means what it says.")
    p_render.add_argument("--grep", action="append", metavar="REGEX",
                          help="Select every tag whose id or label matches a regex "
                               "(e.g. --grep 'female|bikini').")
    p_render.add_argument("--tag-file", metavar="FILE",
                          help="File of tag ids to add: one per line (# comments ok) "
                               "or JSON. Lets you keep a household profile.")
    p_render.add_argument("--rendering",
                          help="Comma-separated rendering styles to include "
                               f"({', '.join(r.value for r in Rendering)}). "
                               "Default: all of them.")
    p_render.add_argument("--visual", action="append", metavar="FAMILY=ACTION",
                          help="How to hide a family: none|blur|black "
                               "(e.g. --visual gore=black). Repeatable.")
    p_render.add_argument("--audio", action="append", metavar="FAMILY=ACTION",
                          help="What to do with a family's audio "
                               f"(e.g. --audio language=mute_word). Repeatable; "
                               f"options differ per family ({', '.join(audio_options(Family.LANGUAGE))}).")
    p_render.add_argument("--polish-descriptions", action="store_true",
                          help="Spend one call per CENSORED segment rewriting its blur/black card "
                               "text with --summary-model. Off: the scan file's own short "
                               "description is used (it is written by the classify pass).")
    p_render.add_argument("--blur-strength", type=float, default=1.0,
                          help="Blur intensity multiplier (1.0 = standard, 2.0 = twice as extreme)")
    p_render.add_argument("--mute-padding", type=float, default=0.5,
                          help="Extra seconds muted before/after each flagged word (default: 0.5)")
    p_render.add_argument("--blur-padding", type=float, default=0.0,
                          help="Extra seconds blurred/blacked before/after each flagged segment (default: 0.0)")
    p_render.add_argument("--summary-model", default=None,
                          help="Model id for the render-time summary pass (improves blur/black descriptions + adds "
                               "muted-audio captions). Uses the server at --base-url. Empty = disabled.")
    p_render.add_argument("--summary-frames", type=int, default=12,
                          help="Frames sampled per segment for the summary pass (default: 12, 0 = text-only)")
    p_render.add_argument("--summary-max-context", type=int, default=30000,
                          help="Assumed loaded context window (tokens) for the summary model when the server "
                               "reports none (default: 30000)")
    p_render.add_argument("--base-url", default=DEFAULT_BASE_URL,
                          help=f"OpenAI-compatible /v1 model server for the summary model (default: {DEFAULT_BASE_URL})")
    p_render.add_argument("--api-key", default="",
                          help="Bearer token if the summary model's server requires one (Strata does).")
    p_render.set_defaults(func=cmd_render)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
