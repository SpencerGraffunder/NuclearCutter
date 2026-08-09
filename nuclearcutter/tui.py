"""
Live TUI dashboard for a running scan.

Two data sources:
- `--status PATH`  read a live status JSON written by `nuclearcutter scan
                    --status-file PATH` (rich data: phase, position, markers,
                    detections). The default for future scans.
- `--log PATH`     tail a scan log (attach mode). Works against an
                    already-running scan started before the status file
                    existed — parses `[sweep] N / M frames` lines, phase
                    markers, and the fingerprint line for duration.

If neither is given, the TUI auto-detects a recent `*.status.json` or scan log
in the current directory / repo root.

Shows: a film timeline with markers for visual and language cut locations, a
current scan-position marker, a time estimate, and CPU/GPU/RAM usage + temps
(CPU/RAM via psutil; GPU/temps best-effort via `sudo -n powermetrics`, which
silently degrades to "n/a" when passwordless sudo is unavailable).

Usage:
    nuclearcutter tui --status Movie.nuclearcutter.status.json
    nuclearcutter tui --log martian_scan.log
    nuclearcutter tui
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nuclearcutter.utils.scan_status import ScanStatus

# ---------------------------------------------------------------------------
# Data model (unified view built from either source)
# ---------------------------------------------------------------------------


@dataclass
class View:
    phase: str = "starting"
    video: str = ""
    duration: float | None = None
    position: float | None = None
    frames_done: int = 0
    frames_total: int = 0
    candidates: list = field(default_factory=list)   # raw sweep hits
    detections: list = field(default_factory=list)   # confirmed visual
    lang_detections: list = field(default_factory=list)
    eta_seconds: float | None = None
    source: str = "status"
    started_at: str = ""
    pid: int | None = None

    @property
    def pct(self) -> float:
        if self.duration:
            return max(0.0, min(100.0, (self.position or 0) / self.duration * 100))
        if self.frames_total:
            return self.frames_done / self.frames_total * 100
        return 0.0


# ---------------------------------------------------------------------------
# Status-file source
# ---------------------------------------------------------------------------


def view_from_status(path: Path, now: float | None = None) -> View:
    now = now if now is not None else time.time()
    st = ScanStatus.load(path)
    v = View(
        phase=st.phase,
        video=st.video,
        duration=st.duration_seconds,
        position=st.position_seconds,
        frames_done=st.frames_done,
        frames_total=st.frames_total,
        candidates=st.visual_candidates,
        detections=st.visual_detections,
        lang_detections=st.language_detections,
        source="status",
        started_at=st.started_at,
        pid=st.pid,
    )
    # ETA: rate measured from when the sweep began.
    if st.started_at and v.duration and st.position_seconds:
        started = _dt.datetime.fromisoformat(st.started_at)
        elapsed = now - started.timestamp()
        if elapsed > 0 and st.position_seconds > 0:
            rate = st.position_seconds / elapsed
            remaining = v.duration - st.position_seconds
            if rate > 0:
                v.eta_seconds = remaining / rate
    return v


# ---------------------------------------------------------------------------
# Log-attach source (works against a scan started before status files existed)
# ---------------------------------------------------------------------------


_SWEEP_RE = re.compile(r"\[sweep\] (\d+) / (\d+) frames")
_FP_RE = re.compile(r"loaded cached fingerprint \(\d+ samples, ([\d.]+)s\)")


def _phase_from_log(text: str) -> str:
    """Detect the current phase from a scan log.

    Priority order matters: phase-change lines (e.g. `[visual_sweep]`) go to
    stdout and can be block-buffered behind `tee`, so they may not have flushed
    yet — but the `[sweep] N frames` progress lines go to stderr (unbuffered).
    So treat the presence of `[sweep]` as evidence the visual sweep is running.
    """
    if "Scan complete" in text or "[done]" in text:
        return "done"
    if "[language_detection]" in text:
        return "language_detection"
    if "[visual_confirm]" in text:
        return "visual_confirm"
    if "[visual_sweep]" in text or "[sweep]" in text:
        return "visual_sweep"
    if "[transcribing]" in text:
        return "transcribing"
    if "[fingerprinting]" in text:
        return "fingerprinting"
    return "starting"


def view_from_log(path: Path, interval: float = 2.0, now: float | None = None) -> View:
    """Tail a scan log into a View. `interval` = sweep sample interval (s)."""
    now = now if now is not None else time.time()
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return View(source="log", phase="starting")

    v = View(source="log", video=path.name, phase=_phase_from_log(text))

    # Duration from the fingerprint line.
    m = _FP_RE.search(text)
    if m:
        v.duration = float(m.group(1))

    # Sweep frames: the most recent progress line.
    frames = [(int(a), int(b)) for a, b in _SWEEP_RE.findall(text)]
    if frames:
        v.frames_done, v.frames_total = frames[-1]
        v.position = v.frames_done * interval

    # If sweep is underway, estimate ETA from frame throughput.
    if v.frames_total and v.frames_done:
        v.eta_seconds = estimate_eta(v, now)
    return v


def estimate_eta(v: View, now: float | None = None) -> float | None:
    """ETA from sweep progress: assume frame rate is constant."""
    if not v.frames_total or not v.frames_done:
        return None
    # We can't know when the sweep started from the log alone; use a rough
    # assumption of ~5.5s per 4-frame batch (480px downscale) as a fallback
    # when we have no timing anchor. Callers with a started_at override this.
    remaining = v.frames_total - v.frames_done
    rate_fps = 4 / 26.0  # ~4 frames per ~26s batch, from the mlx-vlm benchmark
    if remaining > 0:
        return remaining / rate_fps
    return 0.0


# ---------------------------------------------------------------------------
# System stats (CPU/RAM via psutil, GPU/temp best-effort via powermetrics)
# ---------------------------------------------------------------------------


class SystemStats:
    """Collects CPU/RAM (psutil) and, if possible, GPU/temps (powermetrics)."""

    def __init__(self, pid: int | None = None):
        self._psutil = None
        try:
            import psutil

            self._psutil = psutil
            psutil.cpu_percent(interval=None)  # warm up the first-read 0.0
        except Exception:
            pass
        self._proc = None
        self.pid = None
        if pid:
            self.set_pid(pid)
        self._powermetrics_ok: bool | None = None

    def set_pid(self, pid: int | None) -> None:
        """(Re)bind the monitored scan process (for per-process CPU/RSS)."""
        self.pid = pid
        self._proc = None
        if pid and self._psutil:
            try:
                self._proc = self._psutil.Process(pid)
                self._proc.cpu_percent(interval=None)
            except Exception:
                self._proc = None

    def _powermetrics_available(self) -> bool:
        if self._powermetrics_ok is not None:
            return self._powermetrics_ok
        # `sudo -n` fails fast if a password is required (no prompt, no hang).
        if not shutil.which("powermetrics"):
            self._powermetrics_ok = False
            return False
        try:
            r = subprocess.run(
                ["sudo", "-n", "powermetrics", "--samplers", "smc", "-n", "1", "-i", "100"],
                capture_output=True, timeout=5,
            )
            self._powermetrics_ok = r.returncode == 0 and b"temperature" in r.stdout.lower()
        except Exception:
            self._powermetrics_ok = False
        return self._powermetrics_ok

    def sample(self) -> dict:
        out = {"cpu": None, "mem_pct": None, "mem_used": None, "mem_total": None,
               "proc_cpu": None, "proc_mem": None, "gpu": None, "temp": None}
        if self._psutil:
            try:
                out["cpu"] = self._psutil.cpu_percent(interval=None)
                vm = self._psutil.virtual_memory()
                out["mem_pct"] = vm.percent
                out["mem_used"] = vm.used
                out["mem_total"] = vm.total
            except Exception:
                pass
            if self._proc:
                try:
                    out["proc_cpu"] = self._proc.cpu_percent(interval=None)
                    out["proc_mem"] = self._proc.memory_info().rss
                except Exception:
                    pass
        if self._powermetrics_available():
            try:
                r = subprocess.run(
                    ["sudo", "-n", "powermetrics", "--samplers", "smc,gpu_power",
                     "-n", "1", "-i", "100"],
                    capture_output=True, timeout=5,
                )
                txt = r.stdout.decode(errors="replace").lower()
                m = re.search(r"cpu die temperature:\s*([\d.]+) c", txt)
                if m:
                    out["temp"] = float(m.group(1))
                m = re.search(r"gpu (?:0 )?active residency:\s*([\d.]+)%", txt)
                if m:
                    out["gpu"] = float(m.group(1))
            except Exception:
                pass
        return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_CAT_COLORS = {
    "NUDITY": "red",
    "INTIMATE_SCENES": "bright_red",
    "GORE_VIOLENCE": "yellow",
    "FOUL_LANGUAGE": "magenta",
}



def _cat_key(c) -> str:
    """Normalize a category (enum value like 'nudity' or string) to the
    uppercase keys used by the marker/color maps."""
    return str(c).upper()


def _fmt_secs(s: float | None) -> str:
    if s is None or s < 0:
        return "--:--"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _fmt_dur(s: float | None) -> str:
    if not s:
        return "?"
    return _fmt_secs(s)


def _timeline(v: View, width: int = 70) -> Text:
    """A single-line timeline: markers + current-position cursor."""
    duration = v.duration or 1.0
    cells = ["·"] * width

    def place(start, end, ch):
        if start is None or end is None or duration <= 0:
            return
        s = max(0, int(start / duration * width))
        e = min(width - 1, int(end / duration * width))
        for i in range(s, e + 1):
            cells[i] = ch

    for c in v.candidates:
        place(c.get("start"), c.get("end"), "v")
    for d in v.detections:
        ch = {"NUDITY": "V", "INTIMATE_SCENES": "I", "GORE_VIOLENCE": "G"}.get(
            _cat_key(d.get("category")), "V"
        )
        place(d.get("start"), d.get("end"), ch)
    for d in v.lang_detections:
        place(d.get("start"), d.get("end"), "L")

    pos = v.position
    if pos is not None and duration > 0:
        idx = max(0, min(width - 1, int(pos / duration * width)))
        cells[idx] = "█"

    t = Text()
    for i, ch in enumerate(cells):
        style = "default"
        if ch == "v":
            style = "dim yellow"
        elif ch == "V":
            style = _CAT_COLORS["NUDITY"]
        elif ch == "I":
            style = _CAT_COLORS["INTIMATE_SCENES"]
        elif ch == "G":
            style = _CAT_COLORS["GORE_VIOLENCE"]
        elif ch == "L":
            style = "magenta"
        elif ch == "█":
            style = "bold white on blue"
        t.append(ch, style=style)
    return t


def _detection_rows(v: View) -> list[str]:
    rows = []
    for d in v.detections[:8]:
        cat = _cat_key(d.get("category"))
        rows.append(
            f"[{_fmt_secs(d.get('start'))}–{_fmt_secs(d.get('end'))}] "
            f"{cat} ({(d.get('confidence') or 0):.2f}) — {d.get('description','')[:60]}"
        )
    for d in v.lang_detections[:5]:
        rows.append(
            f"[{_fmt_secs(d.get('start'))}–{_fmt_secs(d.get('end'))}] "
            f"FOUL_LANGUAGE “{d.get('word','')}”"
        )
    if not rows:
        rows.append("(none found so far)")
    return rows


def build_panel(v: View, stats: dict) -> Panel:
    title = Text(" NuclearCutter — scan monitor ", style="bold cyan")
    phase = Text(f" phase: {v.phase} ", style="bold white on dark_green")

    timeline = _timeline(v)
    pct = v.pct
    pos_str = _fmt_secs(v.position) if v.position is not None else "--:--"
    dur_str = _fmt_dur(v.duration)

    prog = Text()
    prog.append("Timeline ", style="bold")
    prog.append(f"({dur_str})", style="dim")
    prog.append("\n")
    prog.append(timeline)
    prog.append("\n")
    bar_w = 40
    filled = int(bar_w * pct / 100)
    prog.append(" " * filled, style="on bright_blue")
    prog.append(" " * (bar_w - filled), style="on grey19")
    prog.append(f"  {pct:5.1f}%  pos {pos_str} / {dur_str}\n")

    meta = Text()
    meta.append(f"Frames: {v.frames_done}/{v.frames_total}   ", style="bold")
    if v.eta_seconds is not None:
        meta.append(f"ETA {_fmt_secs(v.eta_seconds)}   ", style="yellow")
    meta.append(f"source: {v.source}\n", style="dim")

    # Detection summary + rows.
    det = Text()
    det.append(
        f"Visual: {len(v.detections)} confirmed · {len(v.candidates)} candidate(s) · "
        f"Language: {len(v.lang_detections)}\n",
        style="bold",
    )
    for row in _detection_rows(v):
        det.append("  • " + row + "\n", style="dim")

    # System stats.
    sys_ = Text("System  ", style="bold")
    sys_.append(f"CPU {stats.get('cpu') if stats.get('cpu') is not None else 'n/a'}%   ")
    if stats.get("proc_cpu") is not None:
        sys_.append(f"(scan {stats['proc_cpu']:.0f}%)   ")
    mem = stats.get("mem_pct")
    sys_.append(f"RAM {mem if mem is not None else 'n/a'}%   ")
    sys_.append(f"GPU {stats.get('gpu') if stats.get('gpu') is not None else 'n/a'}   ")
    sys_.append(f"Temp {stats.get('temp') if stats.get('temp') is not None else 'n/a'}°C\n")
    if stats.get("proc_mem"):
        sys_.append(f"scan RSS {stats['proc_mem']/1e9:.1f} GB", style="dim")

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left")
    table.add_row(title + phase)
    table.add_row(prog)
    table.add_row(meta)
    table.add_row(det)
    table.add_row(sys_)
    return Panel(table, border_style="cyan")


# ---------------------------------------------------------------------------
# Auto-detection + main loop
# ---------------------------------------------------------------------------


def _auto_source(cwd: Path) -> tuple[str, Path]:
    """Find a recent status file or scan log to attach to."""
    repo = Path(__file__).resolve().parents[1]
    for base in (cwd, repo):
        statuses = sorted(base.glob("*.nuclearcutter.status.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if statuses:
            return "status", statuses[0]
        logs = sorted(base.glob("*_scan_*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            return "log", logs[0]
    return "log", cwd / "scan.log"


def run_with_tui(worker, status_path: Path, interval: float = 1.0,
                 log_path: Path | None = None, title: str = "NuclearCutter"):
    """Run `worker()` in a background thread while rendering the TUI dashboard.

    The worker is expected to write progress to `status_path` (a live status
    JSON). Its stdout/stderr are redirected to `log_path` (or devnull) so its
    progress prints don't corrupt the dashboard. Returns the worker's return
    value, or re-raises its exception.
    """
    import contextlib
    import threading

    status_path = Path(status_path)
    result: dict = {"value": None, "error": None}

    def _worker():
        target = open(log_path, "w") if log_path else open("/dev/null", "w")
        try:
            with contextlib.redirect_stdout(target), contextlib.redirect_stderr(target):
                result["value"] = worker()
        except BaseException as exc:  # capture & re-raise after TUI closes
            result["error"] = exc
        finally:
            target.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    stats = SystemStats()
    console = Console()
    try:
        with Live(console=console, refresh_per_second=1 / interval, screen=True) as live:
            while t.is_alive():
                try:
                    v = view_from_status(status_path)
                except Exception:
                    v = View(phase="starting", source="status", video=status_path.name)
                stats.set_pid(v.pid)
                live.update(build_panel(v, stats.sample()))
                time.sleep(interval)
            # Final frame so the user sees where things ended.
            try:
                v = view_from_status(status_path)
            except Exception:
                v = View(phase="done", source="status")
            stats.set_pid(v.pid)
            live.update(build_panel(v, stats.sample()))
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    if result["error"] is not None:
        raise result["error"]
    return result["value"]



def run_tui(status_path: Path | None = None, log_path: Path | None = None,
            interval: float = 1.0, sweep_interval: float = 2.0) -> None:
    """Run the dashboard until Ctrl-C. `interval` = UI refresh rate (s)."""
    source = None
    path = None
    if status_path:
        source, path = "status", Path(status_path)
    elif log_path:
        source, path = "log", Path(log_path)
    else:
        source, path = _auto_source(Path.cwd())
        print(f"auto-detected source: {source} -> {path}", file=sys.stderr)

    stats = SystemStats()
    try:
        from rich.console import Console
        console = Console()
        with Live(console=console, refresh_per_second=1 / interval, screen=False) as live:
            while True:
                try:
                    if source == "status":
                        v = view_from_status(path)
                        stats.set_pid(v.pid)
                    else:
                        v = view_from_log(path, interval=sweep_interval)
                    s = stats.sample()
                except Exception as exc:  # keep the TUI alive through transient errors
                    v = View(source=source, phase=f"error: {exc}")
                    s = {}
                live.update(build_panel(v, s))
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        # Final frame so the user sees where things ended before we exit.
        try:
            console = Console()
            console.print(build_panel(view_from_status(path) if source == "status" else view_from_log(path, interval=sweep_interval), stats.sample() if stats else {}))
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="nuclearcutter tui",
                                     description="Live scan dashboard.")
    parser.add_argument("--status", default=None, help="Path to a live scan status JSON.")
    parser.add_argument("--log", default=None, help="Path to a scan log file to tail (attach mode).")
    parser.add_argument("--interval", type=float, default=1.0, help="UI refresh interval (s).")
    parser.add_argument("--sweep-interval", type=float, default=2.0,
                        help="Sweep sample interval (s) used to map frames->seconds in log-attach mode.")
    args = parser.parse_args(argv)
    run_tui(status_path=args.status, log_path=args.log,
            interval=args.interval, sweep_interval=args.sweep_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
