"""Pre-flight checks for the nightly training orchestrator.

Before backfill / RL training we:

1. Stop stale Fortuna Streamlit processes that hold ``fortuna.duckdb``.
2. Recommend safer ``n_envs`` / ``use_subproc`` settings on low-RAM laptops.
3. Evaluate typed lane readiness (backfill / ML / RL / promotion) before
   expensive per-symbol work begins.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from fortuna.utils.logging import get_logger

if TYPE_CHECKING:
    from fortuna.config.settings import Settings

logger = get_logger(__name__)

# Nightly defaults in ``scripts/nightly_train.py`` — used for auto-tune only.
DEFAULT_N_ENVS = 8
DEFAULT_USE_SUBPROC = True

_SKIP_CLASS_NONE = "none"
_SKIP_CLASS_POLICY = "policy"
_SKIP_CLASS_MISSING = "missing_prerequisites"
_SKIP_CLASS_GLOBAL = "global_blocker"


@dataclass(frozen=True)
class NightlyLaneDecision:
    lane: str
    allowed: bool
    disposition: str
    skip_class: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "allowed": self.allowed,
            "disposition": self.disposition,
            "skip_class": self.skip_class,
            "detail": self.detail,
        }


@dataclass
class PreflightResult:
    killed_pids: list[int] = field(default_factory=list)
    n_envs: int = DEFAULT_N_ENVS
    use_subproc: bool = DEFAULT_USE_SUBPROC
    physical_ram_gb: Optional[float] = None
    notes: list[str] = field(default_factory=list)
    lane_decisions: list[NightlyLaneDecision] = field(default_factory=list)
    should_abort: bool = False
    global_blockers: list[str] = field(default_factory=list)
    runtime_readiness: Optional[dict[str, object]] = None
    scaling_posture: Optional[dict[str, object]] = None

    def as_detail(self) -> dict[str, object]:
        out: dict[str, object] = {
            "killed_pids": self.killed_pids,
            "killed_count": len(self.killed_pids),
            "n_envs": self.n_envs,
            "use_subproc": self.use_subproc,
            "physical_ram_gb": self.physical_ram_gb,
            "notes": self.notes,
        }
        if self.lane_decisions:
            out["lane_decisions"] = [row.to_dict() for row in self.lane_decisions]
            out["should_abort"] = self.should_abort
            out["global_blockers"] = list(self.global_blockers)
            if self.runtime_readiness is not None:
                out["runtime_readiness"] = self.runtime_readiness
        if self.scaling_posture is not None:
            out["scaling_posture"] = self.scaling_posture
        return out

    def lane_readiness_detail(self) -> dict[str, object]:
        detail = {
            "lane_decisions": [row.to_dict() for row in self.lane_decisions],
            "should_abort": self.should_abort,
            "global_blockers": list(self.global_blockers),
            "runtime_readiness": self.runtime_readiness,
        }
        if self.scaling_posture is not None:
            detail["scaling_posture"] = self.scaling_posture
        return detail

    def lane_decision(self, lane: str) -> Optional[NightlyLaneDecision]:
        for row in self.lane_decisions:
            if row.lane == lane:
                return row
        return None


def classify_data_blocker(error: str) -> Optional[str]:
    """Map a failure string to a single global data blocker code, if any."""
    text = str(error or "").strip()
    if not text:
        return None
    lower = text.lower()
    if "certificate_verify_failed" in lower or ("ssl" in lower and "cert" in lower):
        return "smartapi_ssl_blocked"
    if "name resolution" in lower or "getaddrinfo failed" in lower or "failed to resolve" in lower:
        return "smartapi_unreachable"
    if "connectionerror" in lower or "max retries exceeded" in lower:
        return "smartapi_unreachable"
    if "401" in lower or "unauthorized" in lower or "invalid token" in lower:
        return "smartapi_auth_failed"
    if "403" in lower or "forbidden" in lower:
        return "smartapi_auth_failed"
    if re.search(r"\btimeout\b", lower):
        return "smartapi_unreachable"
    return None


def _smartapi_configured() -> bool:
    try:
        from fortuna.config.smartapi_settings import get_smartapi_settings

        cfg = get_smartapi_settings()
        return bool(getattr(cfg, "configured", False))
    except Exception:  # noqa: BLE001
        return False


def _cache_dirs_ok(settings: Settings) -> bool:
    try:
        cache = settings.resolve_path(settings.data_cache_dir)
        return cache.parent.exists() or cache.exists()
    except Exception:  # noqa: BLE001
        return False


def _decision(
    lane: str,
    *,
    disposition: str,
    skip_class: str,
    detail: str,
) -> NightlyLaneDecision:
    return NightlyLaneDecision(
        lane=lane,
        allowed=disposition == "run",
        disposition=disposition,
        skip_class=skip_class,
        detail=detail,
    )


def build_lane_readiness(
    settings: Settings,
    *,
    execution_plan: dict[str, Any],
    data_source: str,
    symbols: list[str],
    physical_ram_gb: Optional[float] = None,
    n_envs: int = DEFAULT_N_ENVS,
    use_subproc: bool = DEFAULT_USE_SUBPROC,
) -> tuple[
    list[NightlyLaneDecision],
    list[str],
    bool,
    Optional[dict[str, object]],
    Optional[dict[str, object]],
]:
    from fortuna.app.runtime_readiness import build_runtime_readiness_summary
    from fortuna.app.scaling_posture import build_scaling_posture_summary

    global_blockers: list[str] = []
    basket_count = len(symbols)
    scaling = build_scaling_posture_summary(
        settings,
        basket_count=basket_count,
        physical_ram_gb=physical_ram_gb,
        n_envs=n_envs,
        use_subproc=use_subproc,
    )
    scaling_dict = scaling.to_dict()
    enforce_cap = bool(getattr(settings, "scaling_enforce_basket_cap", False))
    if enforce_cap and not scaling.within_recommended:
        if basket_count > 0:
            global_blockers.append("basket_exceeds_compute_budget")

    readiness = build_runtime_readiness_summary(
        settings,
        basket_count=basket_count,
        physical_ram_gb=physical_ram_gb,
        n_envs=n_envs,
        use_subproc=use_subproc,
    )
    readiness_dict = readiness.to_dict()

    if bool(getattr(settings, "nightly_automation_enabled", False)):
        if readiness.nightly_posture == "blocked":
            global_blockers.append("nightly_automation_blocked")

    source = str(data_source or "").strip().lower()
    if source == "smartapi" and not _smartapi_configured():
        global_blockers.append("smartapi_credentials_missing")
    if not _cache_dirs_ok(settings):
        global_blockers.append("cache_dirs_missing")

    run_ml = bool(execution_plan.get("run_ml"))
    run_rl = bool(execution_plan.get("run_rl"))
    allow_ml_promotion = bool(execution_plan.get("allow_ml_promotion"))
    allow_rl_promotion = bool(execution_plan.get("allow_rl_promotion"))
    target = str(execution_plan.get("target") or "all")

    backfill_blocked = bool(global_blockers)
    if backfill_blocked:
        backfill = _decision(
            "backfill",
            disposition="block",
            skip_class=_SKIP_CLASS_MISSING,
            detail="global blockers: " + ", ".join(global_blockers),
        )
    else:
        backfill = _decision(
            "backfill",
            disposition="run",
            skip_class=_SKIP_CLASS_NONE,
            detail=f"data_source={source} symbols={len(symbols)}",
        )

    if not run_ml:
        ml = _decision(
            "ml",
            disposition="skip",
            skip_class=_SKIP_CLASS_POLICY,
            detail=f"target={target} run_ml=False",
        )
    elif backfill_blocked:
        ml = _decision(
            "ml",
            disposition="block",
            skip_class=_SKIP_CLASS_MISSING,
            detail="backfill lane blocked",
        )
    elif not bool(settings.agentic_ml_scorer_enabled):
        ml = _decision(
            "ml",
            disposition="block",
            skip_class=_SKIP_CLASS_MISSING,
            detail="FORTUNA_AGENTIC_ML_SCORER_ENABLED=0",
        )
    else:
        ml = _decision(
            "ml",
            disposition="run",
            skip_class=_SKIP_CLASS_NONE,
            detail=f"cash_symbols={execution_plan.get('ml_symbol_count', 0)}",
        )

    if not run_rl:
        rl = _decision(
            "rl",
            disposition="skip",
            skip_class=_SKIP_CLASS_POLICY,
            detail=f"target={target} run_rl=False",
        )
    elif backfill_blocked:
        rl = _decision(
            "rl",
            disposition="block",
            skip_class=_SKIP_CLASS_MISSING,
            detail="backfill lane blocked",
        )
    else:
        rl = _decision(
            "rl",
            disposition="run",
            skip_class=_SKIP_CLASS_NONE,
            detail=f"rl_symbols={execution_plan.get('rl_symbol_count', 0)}",
        )

    promotion_wanted = allow_ml_promotion or allow_rl_promotion or target == "all"
    ml_blocked = ml.disposition == "block"
    rl_blocked = rl.disposition == "block"
    if not promotion_wanted:
        promotion = _decision(
            "promotion",
            disposition="skip",
            skip_class=_SKIP_CLASS_POLICY,
            detail=f"target={target} promotion not requested for lane",
        )
    elif ml_blocked or rl_blocked or backfill_blocked:
        promotion = _decision(
            "promotion",
            disposition="block",
            skip_class=_SKIP_CLASS_MISSING,
            detail="upstream lane blocked",
        )
    elif target == "all":
        promotion = _decision(
            "promotion",
            disposition="skip",
            skip_class=_SKIP_CLASS_POLICY,
            detail="mixed_target_requires_manual_review",
        )
    else:
        promotion = _decision(
            "promotion",
            disposition="run",
            skip_class=_SKIP_CLASS_NONE,
            detail=f"allow_ml={allow_ml_promotion} allow_rl={allow_rl_promotion}",
        )

    decisions = [backfill, ml, rl, promotion]
    should_abort = resolve_nightly_abort(decisions, execution_plan)
    return decisions, global_blockers, should_abort, readiness_dict, scaling_dict


def resolve_nightly_abort(
    lane_decisions: list[NightlyLaneDecision],
    execution_plan: dict[str, Any],
) -> bool:
    """Return True only when every requested training lane is blocked."""
    by_lane = {row.lane: row for row in lane_decisions}
    run_ml = bool(execution_plan.get("run_ml"))
    run_rl = bool(execution_plan.get("run_rl"))
    if not run_ml and not run_rl:
        return False
    ml = by_lane.get("ml")
    rl = by_lane.get("rl")
    ml_blocked = run_ml and ml is not None and ml.disposition == "block"
    rl_blocked = run_rl and rl is not None and rl.disposition == "block"
    if run_ml and run_rl:
        return ml_blocked and rl_blocked
    if run_ml:
        return ml_blocked
    return rl_blocked


def attach_lane_readiness(
    result: PreflightResult,
    settings: Settings,
    *,
    execution_plan: dict[str, Any],
    data_source: str,
    symbols: list[str],
) -> PreflightResult:
    decisions, blockers, should_abort, readiness, scaling = build_lane_readiness(
        settings,
        execution_plan=execution_plan,
        data_source=data_source,
        symbols=symbols,
        physical_ram_gb=result.physical_ram_gb,
        n_envs=result.n_envs,
        use_subproc=result.use_subproc,
    )
    result.lane_decisions = decisions
    result.global_blockers = blockers
    result.should_abort = should_abort
    result.runtime_readiness = readiness
    result.scaling_posture = scaling
    return result


def lane_readiness_for_step(
    settings: Settings,
    *,
    execution_plan: dict[str, Any],
    data_source: str,
    symbols: list[str],
) -> dict[str, object]:
    """Build typed lane-readiness payload for the nightly report step."""
    pf = PreflightResult()
    attach_lane_readiness(
        pf,
        settings,
        execution_plan=execution_plan,
        data_source=data_source,
        symbols=symbols,
    )
    return pf.lane_readiness_detail()


def lane_disposition(
    lane_decisions: list[NightlyLaneDecision],
    lane: str,
) -> Optional[NightlyLaneDecision]:
    for row in lane_decisions:
        if row.lane == lane:
            return row
    return None


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
    settings: Optional[Settings] = None,
    execution_plan: Optional[dict[str, Any]] = None,
    data_source: str = "",
    symbols: Optional[list[str]] = None,
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

    if settings is not None and execution_plan is not None:
        attach_lane_readiness(
            result,
            settings,
            execution_plan=execution_plan,
            data_source=data_source,
            symbols=list(symbols or []),
        )
    elif settings is not None:
        from fortuna.app.scaling_posture import build_scaling_posture_summary

        baseline = build_scaling_posture_summary(
            settings,
            basket_count=0,
            physical_ram_gb=result.physical_ram_gb,
            n_envs=result.n_envs,
            use_subproc=result.use_subproc,
        )
        result.scaling_posture = baseline.to_dict()

    logger.info(
        "[preflight] killed=%s n_envs=%d use_subproc=%s lanes=%d",
        result.killed_pids or "none",
        result.n_envs,
        result.use_subproc,
        len(result.lane_decisions),
    )
    return result


__all__ = [
    "DEFAULT_N_ENVS",
    "DEFAULT_USE_SUBPROC",
    "NightlyLaneDecision",
    "PreflightResult",
    "attach_lane_readiness",
    "build_lane_readiness",
    "classify_data_blocker",
    "find_stale_fortuna_streamlit_pids",
    "kill_stale_fortuna_streamlit",
    "lane_disposition",
    "lane_readiness_for_step",
    "physical_ram_gb",
    "recommend_rl_env",
    "resolve_nightly_abort",
    "run_nightly_preflight",
]
