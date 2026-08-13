"""
System stats for the web GUI status section: CPU/RAM via psutil — including
per-process usage for NuclearCutter itself (NOT system totals).
"""

from __future__ import annotations

import time


class SystemStats:
    """Collects CPU/RAM (psutil) — system totals plus per-process usage for the
    NuclearCutter process tree (this server + any spawned model server)."""

    def __init__(self, pid: int | None = None):
        self._time = time
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

    def set_pid(self, pid: int | None) -> None:
        """(Re)bind the monitored process (for per-process CPU/RSS)."""
        self.pid = pid
        self._proc = None
        if pid and self._psutil:
            try:
                self._proc = self._psutil.Process(pid)
                self._proc.cpu_percent(interval=None)
            except Exception:
                self._proc = None

    def sample(self) -> dict:
        """Return a stats snapshot.

        `mem_used`/`mem_pct` are SYSTEM totals; `proc_mem_total` is the memory
        used by the NuclearCutter process tree (this server + any spawned
        model server), which is what the GUI shows as "RAM used by
        NuclearCutter".
        """
        out = {"cpu": None, "mem_pct": None, "mem_used": None, "mem_total": None,
               "proc_cpu": None, "proc_mem": None, "proc_mem_total": None}
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
                    # Also include the mlx-vlm/llama.cpp server child process
                    # (where the heavy model memory actually lives), so
                    # "NuclearCutter RAM" reflects the real memory in use.
                    out["proc_mem_total"] = out["proc_mem"]
                    for child in self._proc.children(recursive=True):
                        try:
                            out["proc_mem_total"] += child.memory_info().rss
                        except Exception:
                            pass
                except Exception:
                    pass
        return out
