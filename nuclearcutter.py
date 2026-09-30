#!/usr/bin/env python3
"""NuclearCutter self-bootstrapping launcher.

Runs the real CLI inside the project's own virtual environment (`.venv`),
creating the venv and installing dependencies automatically on first run.
This is what makes the simple workflow work:

    git clone https://github.com/SpencerGraffunder/NuclearCutter.git
    cd NuclearCutter
    python3 nuclearcutter.py            # starts the web GUI server
    # open http://localhost:8000 in a browser (any device on the network
    # can reach it at http://<this-machine-ip>:8000 — no login)

No `pip install`, no `source .venv/bin/activate` needed. The first run
creates `.venv` and installs everything (this needs network + a moment);
after that it just launches the venv's Python directly.

You can also call it as `./nuclearcutter.py` (it has a shebang), or pass a
command: `python3 nuclearcutter.py serve`, `scan MOVIE.mkv`, `render MOVIE.mkv`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "bin" / "python"  # POSIX; "Scripts/python.exe" on Windows
MARKER = VENV / ".deps_ok"  # bumped whenever deps are (re)installed
PYPROJECT = ROOT / "pyproject.toml"


def _need_install() -> bool:
    """Install if the venv is missing, or if pyproject.toml changed since the
    last successful install (e.g. after a `git pull` added a dependency)."""
    if not VENV_PY.exists():
        return True
    if not MARKER.exists():
        return True
    try:
        return MARKER.stat().st_mtime < PYPROJECT.stat().st_mtime
    except OSError:
        return True


def _ensure_venv() -> None:
    if not _need_install():
        return
    print("NuclearCutter: setting up virtual environment (.venv) ...")
    if not VENV_PY.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    print("NuclearCutter: installing dependencies (first run / deps changed) ...")
    subprocess.run([str(VENV_PY), "-m", "pip", "install", "-e", str(ROOT)], check=True)
    MARKER.write_text("ok")


def main() -> int:
    _ensure_venv()
    # Resolve the package regardless of the current working directory, and keep
    # the user's CWD so relative movie paths still work. Replaces this process.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    os.execvpe(str(VENV_PY), [str(VENV_PY), "-m", "nuclearcutter.cli", *sys.argv[1:]], env)


if __name__ == "__main__":
    sys.exit(main())
