"""Pre-flight checks for the nightly training orchestrator.

Before backfill / RL training we:

1. Stop stale Fortuna Streamlit processes that hold ``fortuna.duckdb``.
2. Recommend safer ``n_envs`` / ``use_subproc`` settings on low-RAM laptops.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

# Nightly defaults in ``scripts/nightly_train.py`` — used for auto-tune only.
DEFAULT_N_ENVS = 8
DEFAULT_USE_SUBPROC = True


@dataclass
class PreflightResult:
    killed_pids: list[int] = field(default_factory=list)
    n_envs: int = DEFAULT_N_ENVS
    use_subproc: bool = DEFAULT_USE_SUBPROC
    physical_ram_gb: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def as_detail(self) -> dict[str, object]:
        return {
            "killed_pids": self.killed_pids,
            "killed_count": len(self.killed_pids),
            "n_envs": self.n_envs,
            "use_subproc": self.use_subproc,
            "physical_ram_gb": self.physical_ram_gb,
            "notes": self.notes,
        }


def physical_ram_gb() -> Optional[float]:
    """Return installed physical RAM in GiB, or ``None`` if unknown."""
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return None
            return float(stat.ullTotalPhys) / (1024**3)
        except Exception:  # noqa: BLE001
            return None
    return None


def recommend_rl_env(
    requested_n_envs: int,
    requested_use_subproc: bool,
    *,
    physical_ram: Optional[float] = None,
) -> tuple[int, bool, list[str]]:
    """Cap parallel RL env count on low-RAM machines."""
    ram = physical_ram if physical_ram is not None else physical_ram_gb()
    n_envs = max(1, int(requested_n_envs))
    use_subproc = bool(requested_use_subproc)
    notes: list[str] = []

    if ram is None:
        return n_envs, use_subproc, notes

    if ram < 12:
        if n_envs > 2:
            notes.append(f"physical RAM {ram:.1f} GiB -> n_envs capped 2 (was {n_envs})")
            n_envs = 2
        if use_subproc:
            notes.append("physical RAM < 12 GiB -> disabling SubprocVecEnv")
            use_subproc = False
    elif ram < 20:
        if n_envs > 4:
            notes.append(f"physical RAM {ram:.1f} GiB -> n_envs capped 4 (was {n_envs})")
            n_envs = 4
    elif ram < 28 and n_envs > 6:
        notes.append(f"physical RAM {ram:.1f} GiB -> n_envs capped 6 (was {n_envs})")
        n_envs = 6

    return n_envs, use_subproc, notes


def _powershell_json(command: str) -> str:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "powershell failed")
    return proc.stdout.strip()


def find_stale_fortuna_streamlit_pids(repo_root: Path) -> list[int]:
    """Return PIDs of Streamlit processes serving this repo's dashboard."""
    if sys.platform != "win32":
        return []

    repo = str(repo_root.resolve())
    # PowerShell single-quoted literal: double internal single quotes.
    repo_literal = repo.replace("'", "''")
    script = (
        "$repo = '" + repo_literal + "'; "
        "Get-CimInstance Win32_Process | "
        "Where-Object { "
        "$_.CommandLine -and "
        "($_.CommandLine -like '*streamlit*') -and "
        "($_.CommandLine -like '*streamlit_app.py*') -and "
        "($_.CommandLine -like ('*' + $repo + '*')) "
        "} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = _powershell_json(script)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[preflight] could not enumerate Streamlit PIDs: %s", exc)
        return []

    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return sorted(set(pids))


def kill_stale_fortuna_streamlit(repo_root: Path) -> list[int]:
    """Stop Fortuna Streamlit processes so DuckDB can open exclusively."""
    pids = find_stale_fortuna_streamlit_pids(repo_root)
    if not pids:
        return []

    killed: list[int] = []
    for pid in pids:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            killed.append(pid)
            logger.info("[preflight] stopped stale Streamlit PID %d", pid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[preflight] failed to stop PID %d: %s", pid, exc)
    return killed


def run_nightly_preflight(
    repo_root: Path,
    *,
    requested_n_envs: int = DEFAULT_N_ENVS,
    requested_use_subproc: bool = DEFAULT_USE_SUBPROC,
    kill_streamlit: bool = True,
    auto_tune_envs: bool = True,
) -> PreflightResult:
    """Run all pre-flight steps and return the resolved training settings."""
    result = PreflightResult(
        n_envs=max(1, int(requested_n_envs)),
        use_subproc=bool(requested_use_subproc),
    )
    result.physical_ram_gb = physical_ram_gb()

    if kill_streamlit:
        result.killed_pids = kill_stale_fortuna_streamlit(repo_root)
        if result.killed_pids:
            result.notes.append(
                f"stopped {len(result.killed_pids)} stale Streamlit process(es): "
                f"{result.killed_pids}"
            )
        else:
            result.notes.append("no stale Fortuna Streamlit processes found")

    if auto_tune_envs:
        tuned_n, tuned_subproc, tune_notes = recommend_rl_env(
            result.n_envs,
            result.use_subproc,
            physical_ram=result.physical_ram_gb,
        )
        result.n_envs = tuned_n
        result.use_subproc = tuned_subproc
        result.notes.extend(tune_notes)

    if result.physical_ram_gb is not None:
        result.notes.insert(
            0,
            f"physical RAM {result.physical_ram_gb:.1f} GiB",
        )

    logger.info(
        "[preflight] killed=%s n_envs=%d use_subproc=%s",
        result.killed_pids or "none",
        result.n_envs,
        result.use_subproc,
    )
    return result


__all__ = [
    "DEFAULT_N_ENVS",
    "DEFAULT_USE_SUBPROC",
    "PreflightResult",
    "find_stale_fortuna_streamlit_pids",
    "kill_stale_fortuna_streamlit",
    "physical_ram_gb",
    "recommend_rl_env",
    "run_nightly_preflight",
]
