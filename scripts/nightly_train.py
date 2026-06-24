#!/usr/bin/env python
"""Nightly orchestrator for Fortuna training jobs.

Schedule (all sequential, each isolated; failures in one step don't block the next):
    1. Backfill 180d of 5m OHLCV for every symbol in the basket (SmartAPI).
    2. Train ``RegimeDetector`` on the basket's combined session-feature corpus.
    3. For each symbol:
        a. Train CNN policy (500k steps, CUDA, SubprocVecEnv-8).
        b. (Optional) Train MLP baseline (cheaper, only on the first pass per week).
        c. Save checkpoint, write run summary.
    4. Run ``AdaptiveOptimizerAgent`` walk-forward parameter sweep on every
       deterministic JSON strategy in ``strategies/``.
    5. Write a markdown summary report to ``reports/nightly/<DATE>.md``.
    6. (Optional, ``--sleep-after``) Suspend the host.

Wall-clock budget for a 10-symbol basket on the current rig:
    - Backfill        ~3 min (cached after night 1)
    - Regime          ~5 min
    - RL training     ~12 min × 10 symbols = 2 hours
    - Strategy tuning ~20 min
    - Reporting       ~1 min
    Total: ~2.5 hours (well within the 02:00 -> 06:30 window)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- Warm-up circular import before any fortuna imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Force line-buffered stdout/stderr so long-running phases stream into the
# log file in real time. Without this, Windows' file-redirection makes
# Python use 8KB block buffering and a hang becomes invisible until the
# buffer flushes (we lost ~3.5h of telemetry on the first nightly run).
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 — older interpreters / non-TextIO wrappers.
    pass

from fortuna.app.nightly_basket import (  # noqa: E402
    emit_training_candidate_manifest,
    emit_training_research_plan,
    emit_workflow_snapshot,
    resolve_symbol_basket,
    resolve_training_candidate_response,
)
from fortuna.app.promotion_review import (  # noqa: E402
    build_promotion_review,
    export_promotion_review,
)
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401,E402
from fortuna.utils.keep_awake import activate as _keep_awake_activate  # noqa: E402
from fortuna.utils.keep_awake import release as _keep_awake_release  # noqa: E402
from fortuna.utils.nightly_preflight import (  # noqa: E402
    attach_lane_readiness,
    classify_data_blocker,
    lane_readiness_for_step,
    run_nightly_preflight,
)
from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults  # noqa: E402

apply_low_spec_gpu_defaults()

# Hold a Windows power-state lock for the *entire* lifetime of this
# orchestrator so the OS cannot sleep mid-job after WakeToRun fires.
# This is the actual fix for the silent 3.5h gap we observed: Task
# Scheduler woke the box at 02:00, the script started, then the OS
# went back to sleep because nothing was holding ES_SYSTEM_REQUIRED.
_KEEP_AWAKE_OK = _keep_awake_activate()

import pandas as pd  # noqa: E402

from fortuna.config.settings import get_settings  # noqa: E402
from fortuna.data.manager import MarketDataManager  # noqa: E402
from fortuna.models.metadata import ModelKind  # noqa: E402
from fortuna.observability.recorder import (  # noqa: E402
    emit_workflow_step,
    workflow_boundary,
    workflow_run_context,
)
from fortuna.paper import PaperLeague, PaperLeagueConfig  # noqa: E402
from fortuna.rl.env.reward import RewardConfig  # noqa: E402
from fortuna.rl.inference.regime_detector import RegimeDetector  # noqa: E402
from fortuna.rl.training.checkpoint import PolicyCheckpoint  # noqa: E402
from fortuna.rl.training.trainer import FortunaRLTrainer, TrainerConfig  # noqa: E402
from fortuna.utils.logging import get_logger, setup_logging  # noqa: E402

# Wire the root logger to stdout so ``logger.info(...)`` lines (phase
# banners, per-symbol "starting" markers, "report ->" footers, etc.)
# actually surface in the redirected nightly log file. Without this,
# ``get_logger("nightly")`` returns a logger with no handlers and every
# ``logger.info`` call is silently dropped — the orchestrator was running
# blind for those telemetry hooks until now.
setup_logging(level="INFO")
logger = get_logger("nightly")


DEFAULT_BASKET: list[str] = [
    "ICICIBANK.NS",
    "HDFCBANK.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "TATASTEEL.NS",
]

# Mirror of ``DEFAULT_BASKET`` on the NFO segment. ``BASE.FUT`` resolves to
# the rolling front-month contract — SmartAPI returns each token's full
# listing-to-expiry history (typically ~120 days of 5m bars), which is
# enough for an RL training run with the same fold structure as cash.
DEFAULT_FUTURES_BASKET: list[str] = [
    s.replace(".NS", ".FUT") for s in DEFAULT_BASKET
]


# ---------------------------------------------------------------------- types


@dataclass
class StepResult:
    name: str
    status: str  # "ok" | "skip" | "fail"
    started_at: str
    ended_at: str
    duration_s: float
    detail: dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class NightlyReport:
    started_at: str
    ended_at: str = ""
    total_duration_s: float = 0.0
    basket: list[str] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    overall_status: str = "running"

    def add(self, step: StepResult) -> None:
        self.steps.append(step)
        logger.info(
            "[nightly] %-30s %-5s %.1fs %s",
            step.name,
            step.status,
            step.duration_s,
            step.detail,
        )


# ------------------------------------------------------------------- runners


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _timed(name: str, fn, *args, **kwargs) -> StepResult:
    """Run ``fn`` and capture timing/status into a ``StepResult``."""
    t0 = time.perf_counter()
    started = _now_iso()
    try:
        detail = fn(*args, **kwargs) or {}
        return StepResult(
            name=name,
            status="ok",
            started_at=started,
            ended_at=_now_iso(),
            duration_s=time.perf_counter() - t0,
            detail=detail,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] failed: %s\n%s", name, exc, traceback.format_exc())
        return StepResult(
            name=name,
            status="fail",
            started_at=started,
            ended_at=_now_iso(),
            duration_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _cash_equity_symbols(symbols: list[str]) -> list[str]:
    return [s for s in symbols if not _is_rolling_future(s) and ".FUT" not in s.upper()]


def _effective_training_target(
    basket_meta: dict[str, object] | None,
    training_research_meta: dict[str, object] | None,
) -> str:
    research_target = ""
    if isinstance(training_research_meta, dict):
        research_target = str(
            training_research_meta.get("effective_refresh_target")
            or training_research_meta.get("refresh_target")
            or ""
        ).strip().lower()
    basket_target = ""
    if isinstance(basket_meta, dict):
        basket_target = str(
            basket_meta.get("target")
            or basket_meta.get("requested_target")
            or ""
        ).strip().lower()
    target = research_target or basket_target or "all"
    return target if target in {"all", "ml", "rl"} else "all"


def _build_training_execution_plan(
    symbols: list[str],
    basket_meta: dict[str, object] | None,
    training_research_meta: dict[str, object] | None,
) -> dict[str, object]:
    target = _effective_training_target(basket_meta, training_research_meta)
    cash_symbols = _cash_equity_symbols(symbols)
    selection_source = ""
    discovery_summary = ""
    discovery_target = ""
    if isinstance(basket_meta, dict):
        selection_source = str(basket_meta.get("selection_source") or "").strip().lower()
        discovery_summary = str(basket_meta.get("discovery_summary") or "").strip()
        discovery_target = str(
            basket_meta.get("discovery_recommended_refresh_target") or ""
        ).strip().lower()
    if isinstance(training_research_meta, dict):
        if not selection_source:
            selection_source = str(
                training_research_meta.get("selection_source") or ""
            ).strip().lower()
        if not discovery_summary:
            discovery_summary = str(
                training_research_meta.get("discovery_summary") or ""
            ).strip()
        if not discovery_target:
            discovery_target = str(
                training_research_meta.get("discovery_recommended_refresh_target") or ""
            ).strip().lower()
    run_ml = target in {"ml", "all"}
    run_rl = target in {"rl", "all"}
    return {
        "target": target,
        "run_ml": run_ml,
        "run_rl": run_rl,
        "allow_ml_promotion": target == "ml",
        "allow_rl_promotion": target == "rl",
        "cash_symbol_count": len(cash_symbols),
        "ml_symbol_count": len(cash_symbols) if run_ml else 0,
        "rl_symbol_count": len(symbols) if run_rl else 0,
        "selection_source": selection_source or None,
        "discovery_summary": discovery_summary or None,
        "discovery_recommended_refresh_target": (
            discovery_target if discovery_target in {"ml", "rl", "all"} else None
        ),
    }


def _lane_disposition(lane_detail: dict[str, object] | None, lane: str) -> str:
    if not isinstance(lane_detail, dict):
        return "run"
    rows = lane_detail.get("lane_decisions")
    if not isinstance(rows, list):
        return "run"
    for row in rows:
        if isinstance(row, dict) and str(row.get("lane", "")).strip() == lane:
            return str(row.get("disposition", "run") or "run")
    return "run"


def _lane_row(lane_detail: dict[str, object] | None, lane: str) -> dict[str, object]:
    if not isinstance(lane_detail, dict):
        return {}
    rows = lane_detail.get("lane_decisions")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("lane", "")).strip() == lane:
            return row
    return {}


def _skip_step(name: str, detail: dict[str, object]) -> StepResult:
    return StepResult(
        name=name,
        status="skip",
        started_at=_now_iso(),
        ended_at=_now_iso(),
        duration_s=0.0,
        detail=detail,
    )


def _should_abort_after_lane_readiness(lane_detail: dict[str, object] | None) -> bool:
    if not isinstance(lane_detail, dict):
        return False
    return bool(lane_detail.get("should_abort"))


def finalize_overall_status(report: NightlyReport) -> str:
    n_fail = sum(1 for s in report.steps if s.status == "fail")
    if n_fail > 0:
        return f"partial ({n_fail} failures)"

    meaningful_ok = any(
        s.status == "ok"
        and (
            s.name.startswith(("backfill:", "ml_train:", "rl_train:"))
            or s.name in {"backfill", "ml_train", "rl_train", "regime_detector"}
        )
        for s in report.steps
    )
    if meaningful_ok:
        return "ok"

    lane_step = next((s for s in report.steps if s.name == "lane_readiness"), None)
    lane_detail = lane_step.detail if lane_step is not None else None
    global_blockers: list[str] = []
    if isinstance(lane_detail, dict):
        raw = lane_detail.get("global_blockers")
        if isinstance(raw, list):
            global_blockers = [str(row) for row in raw if str(row or "").strip()]

    blocked_skip = any(
        s.status == "skip"
        and isinstance(s.detail, dict)
        and str(s.detail.get("skip_class", "")).strip()
        in {"global_blocker", "missing_prerequisites"}
        for s in report.steps
        if s.name in {"backfill", "ml_train", "rl_train", "backfill:global_blocker"}
        or s.name.startswith("backfill:")
    )
    if global_blockers or blocked_skip:
        return "blocked"
    return "skipped"


def _run_backfill_phase(
    report: NightlyReport,
    symbols: list[str],
    *,
    timeframe: str,
    days: int,
    data_source: str,
    lane_gating: bool,
    lane_detail: dict[str, object] | None,
    fail_fast: bool,
) -> None:
    disposition = _lane_disposition(lane_detail, "backfill")
    if lane_gating and disposition != "run":
        row = _lane_row(lane_detail, "backfill")
        report.add(_skip_step(
            "backfill",
            {
                "skip_class": str(row.get("skip_class", "missing_prerequisites")),
                "reason": str(row.get("detail", "backfill lane not allowed")),
                "global_blockers": list(
                    lane_detail.get("global_blockers", [])
                    if isinstance(lane_detail, dict)
                    else []
                ),
            },
        ))
        return

    for idx, sym in enumerate(symbols):
        step = _timed(
            f"backfill:{sym}",
            backfill_symbol,
            sym,
            timeframe,
            days,
            data_source,
        )
        report.add(step)
        if step.status == "fail" and fail_fast:
            code = classify_data_blocker(step.error or "")
            if code:
                remaining = symbols[idx + 1 :]
                if remaining:
                    report.add(_skip_step(
                        "backfill:global_blocker",
                        {
                            "skip_class": "global_blocker",
                            "blocker": code,
                            "failed_symbol": sym,
                            "remaining_symbols": len(remaining),
                            "symbols": remaining,
                        },
                    ))
                break


# --------------------------------------------------------------- data layer


def _is_rolling_future(symbol: str) -> bool:
    """Return True for ``BASE.FUT`` (rolling front-month) symbols.

    Explicit ``BASE.FUT.DDMMMYYYY`` pinned-expiry symbols are NOT rolling
    and are safe to cache like any cash equity.
    """
    s = symbol.upper()
    return s.endswith(".FUT") or (".FUT." not in s and ".FUT" in s and not s.split(".FUT", 1)[1])


def backfill_symbol(symbol: str, timeframe: str, days: int, data_source: str) -> dict:
    settings = get_settings()
    mdm = MarketDataManager(settings, data_source=data_source)
    # Rolling ``.FUT`` symbols silently change tradingsymbol/token at every
    # monthly expiry. The cache file is keyed by the bare ``BASE.FUT`` name
    # so a stale cache from before the roll would return last-month's data.
    # Force a refresh on every nightly backfill to guarantee the cached
    # parquet always contains data for the *currently*-resolved front-month.
    force = _is_rolling_future(symbol)
    df = mdm.get_ohlcv(symbol, timeframe, days=days, force_refresh=force)
    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "first": str(df.index.min()),
        "last": str(df.index.max()),
        "force_refresh": force,
    }


def _load_all_sessions(
    symbols: list[str], timeframe: str, days: int, data_source: str
) -> list[pd.DataFrame]:
    """Build a per-session corpus across all symbols.

    Each symbol's OHLCV is grouped by calendar date; sessions with <20 bars
    are dropped to align with ``RegimeDetector``'s minimum.
    """
    settings = get_settings()
    mdm = MarketDataManager(settings, data_source=data_source)
    out: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            # The backfill phase already force-refreshed every rolling
            # .FUT cache earlier in this orchestrator run, so reading from
            # cache here is safe — and avoids a second SmartAPI round-trip
            # that would burn through the broker's rate budget.
            df = mdm.get_ohlcv(sym, timeframe, days=days)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[regime] %s skipped: %s", sym, exc)
            continue
        if df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        for _, grp in df.groupby(idx.normalize()):
            if len(grp) >= 20:
                out.append(grp)
    return out


def train_regime_multi(
    symbols: list[str], timeframe: str, days: int, data_source: str, output: Path
) -> dict:
    """Train RegimeDetector on session-grouped multi-symbol corpus.

    Concatenates sessions from every symbol into a synthetic long-form series
    that bypasses RegimeDetector's calendar-date groupby — each symbol-day pair
    keeps its own row in the feature matrix.
    """
    sessions = _load_all_sessions(symbols, timeframe, days, data_source)
    if not sessions:
        raise RuntimeError("no sessions found for regime detector training")

    # Concatenate with **distinct timestamp shifts** per session so the calendar
    # groupby inside RegimeDetector keeps them apart. We do this by offsetting
    # each subsequent session by 1 day past the last.
    pieces: list[pd.DataFrame] = []
    cursor = pd.Timestamp("2000-01-01")
    for s in sessions:
        shifted = s.copy()
        shifted.index = shifted.index + (cursor - shifted.index.normalize().min())
        pieces.append(shifted)
        cursor = shifted.index.max().normalize() + pd.Timedelta(days=1)
    combined = pd.concat(pieces).sort_index()

    det = RegimeDetector()
    meta = det.train(combined, test_fraction=0.2, random_state=42)
    output.parent.mkdir(parents=True, exist_ok=True)
    det.save(output)
    return {
        "sessions": len(sessions),
        "accuracy": meta.accuracy,
        "n_train": meta.n_train,
        "n_test": meta.n_test,
        "output": str(output),
    }


# --------------------------------------------------------------- RL trainer


def _inherited_reward(symbol: str, timeframe: str, models_root: Path) -> RewardConfig:
    """Pick the best (most-recent-by-created_at) reward config seen for this symbol."""
    best: Optional[PolicyCheckpoint] = None
    for sub in ("validated", "rejected"):
        d = models_root / sub
        if not d.exists():
            continue
        for run in d.iterdir():
            meta = run / "metadata.json"
            if not meta.exists():
                continue
            try:
                cp = PolicyCheckpoint.read(meta)
            except Exception:  # noqa: BLE001
                continue
            if cp.symbol != symbol or cp.timeframe != timeframe:
                continue
            if best is None or cp.created_at > best.created_at:
                best = cp
    if best is not None:
        return RewardConfig(**best.reward_config)
    return RewardConfig()


def train_one_symbol(
    symbol: str,
    timeframe: str,
    policy: str,
    timesteps: int,
    days: int,
    data_source: str,
    n_envs: int,
    device: str,
    use_subproc: bool,
    checkpoint_root: Path,
) -> dict:
    from fortuna.config.settings import get_settings
    from fortuna.rl.evaluation.baseline import baseline_sharpe_for_symbol

    reward = _inherited_reward(symbol, timeframe, checkpoint_root)
    baseline = baseline_sharpe_for_symbol(symbol, timeframe, settings=get_settings())
    # The backfill phase already wrote a fresh .FUT cache for the current
    # front-month contract earlier in this orchestrator run. The trainer
    # reads from that cache — issuing a third force_refresh here just
    # burns the SmartAPI rate budget and risks a 403 (as happened with
    # SBIN.FUT on the first attempt).
    cfg = TrainerConfig(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        data_source=data_source,
        train_bars=750,
        test_bars=375,
        step_bars=375,
        total_timesteps=timesteps,
        n_envs=n_envs,
        n_bars=20,
        warmup_bars=30,
        use_subproc=use_subproc,
        ppo_n_steps=4096 if policy == "CnnPolicy" else 4096,
        batch_size=512 if policy == "CnnPolicy" else 256,
        n_epochs=10,
        learning_rate=3e-4,
        policy_type=policy,
        device=device,
        reward_config=reward,
        checkpoint_dir=checkpoint_root,
        eval_freq=25_000,
        early_stop_patience=8,
        seed=44,
        baseline_sharpe=baseline,
    )
    trainer = FortunaRLTrainer(cfg)
    meta = trainer.train()
    return {
        "symbol": symbol,
        "policy": policy,
        "run_id": meta.run_id,
        "passed": bool(meta.verdict_passed),
        "sharpe": float(meta.oos_metrics.sharpe_ratio),
        "pf": float(meta.oos_metrics.profit_factor),
        "trades": int(meta.oos_metrics.total_trades),
        "win_rate": float(meta.oos_metrics.win_rate_pct),
        "reward_config": cfg.reward_config.to_dict(),
        "advisory_ready": bool(meta.advisory_ready),
        "baseline_sharpe": meta.baseline_sharpe,
        "beats_baseline": meta.beats_baseline,
    }


def train_ml_scorer_symbol(
    symbol: str,
    *,
    timeframe: str,
    days: int,
    repo_root: Path,
    learning_dir: str,
    output_dir: Path,
    model_kind: str = "logistic",
    horizon_bars: int = 3,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    cmd = [
        "uv",
        "run",
        "--group",
        "ml",
        "python",
        "scripts/train_ml_signal_scorer.py",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--learning-dir",
        learning_dir,
        "--output-dir",
        str(output_dir),
        "--days",
        str(days),
        "--horizon-bars",
        str(horizon_bars),
        "--eval-fraction",
        str(eval_fraction),
        "--seed",
        str(seed),
        "--model-kind",
        model_kind,
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(message or f"ml scorer training failed for {symbol}")
    return {
        "symbol": symbol,
        "model_kind": model_kind,
        "output_dir": str(output_dir),
    }


def _ml_promotion_rank(meta) -> tuple[float, float, float, int]:
    oos = getattr(meta, "oos_metrics", None)
    return (
        float(getattr(oos, "roc_auc", 0.0) or 0.0),
        float(getattr(oos, "precision", 0.0) or 0.0),
        float(getattr(oos, "accuracy", 0.0) or 0.0),
        int(getattr(oos, "n_samples", 0) or 0),
    )


def promote_best_ml_scorer(
    models_root: Path,
    ml_base: Path,
    *,
    include_rejected: bool = False,
    eligible_symbols: list[str] | None = None,
    timeframe: str | None = None,
    workflow_snapshot_path: str | None = None,
) -> dict:
    from fortuna.ml.artifacts import load_metadata
    from fortuna.models.metadata import PromotionRecord, PromotionStatus
    from fortuna.models.promotion import compute_ml_advisory_ready, promote

    eligible = {
        str(symbol).upper().strip() for symbol in (eligible_symbols or []) if str(symbol).strip()
    }
    best_meta = None
    best_dir = None
    for sub in ("validated", "rejected"):
        bucket = ml_base / sub
        if not bucket.exists():
            continue
        for run in bucket.iterdir():
            model_path = run / "model.joblib"
            if not model_path.is_file():
                continue
            if sub == "rejected" and not include_rejected:
                continue
            meta = load_metadata(run)
            if meta is None:
                continue
            if eligible and str(meta.symbol or "").upper().strip() not in eligible:
                continue
            if (
                timeframe
                and str(meta.timeframe or "").strip()
                and str(meta.timeframe) != str(timeframe)
            ):
                continue
            ready, _ = compute_ml_advisory_ready(meta)
            if not include_rejected and not ready:
                continue
            if best_meta is None or _ml_promotion_rank(meta) > _ml_promotion_rank(best_meta):
                best_meta = meta
                best_dir = run

    promoted: list[dict] = []
    if best_meta is None or best_dir is None:
        return {
            "promoted": promoted,
            "count": 0,
            "model_kind": ModelKind.ML_SCORER.value,
            "workflow_snapshot_path": workflow_snapshot_path,
        }

    ready, reasons = compute_ml_advisory_ready(best_meta)
    record = PromotionRecord.from_scorer_metadata(
        best_meta,
        best_dir,
        status=PromotionStatus.VALIDATED,
        advisory_ready=ready,
    )
    record.reasons = list(reasons or getattr(best_meta, "verdict_reasons", []) or [])
    record.workflow_snapshot_path = workflow_snapshot_path
    try:
        ptr = promote(record, models_root=models_root, promoted_by="nightly", ml_base=ml_base)
        promoted.append(
            {
                "run_id": best_meta.run_id,
                "symbol": best_meta.symbol,
                "timeframe": best_meta.timeframe,
                "pointer": str(ptr),
                "artifact_dir": str(best_dir),
                "roc_auc": float(getattr(best_meta.oos_metrics, "roc_auc", 0.0)),
                "precision": float(getattr(best_meta.oos_metrics, "precision", 0.0)),
                "advisory_ready": ready,
            }
        )
    except Exception as exc:  # noqa: BLE001
        promoted.append(
            {
                "run_id": best_meta.run_id,
                "symbol": best_meta.symbol,
                "timeframe": best_meta.timeframe,
                "artifact_dir": str(best_dir),
                "advisory_ready": ready,
                "error": str(exc),
            }
        )
    return {
        "promoted": promoted,
        "count": len([row for row in promoted if "pointer" in row]),
        "model_kind": ModelKind.ML_SCORER.value,
        "workflow_snapshot_path": workflow_snapshot_path,
    }


# ------------------------------------------- deterministic strategy tuning


def optimize_deterministic_strategies(
    symbol: str,
    timeframe: str,
    days: int,
    data_source: str,
    strategy_dir: Path,
    output_dir: Path,
    param_grid_path: Optional[Path],
) -> dict:
    """Run a ``PaperLeague`` walk-forward sweep with adaptive grid refinement.

    Each call appends to ``logs/learner/<strategy_stem>_learning.json`` via
    ``AdaptiveLearner``, so successive nightly runs converge the param grid.
    """
    cfg = PaperLeagueConfig(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        data_source=data_source,
        strategy_dir=strategy_dir,
        param_grid_path=param_grid_path,
        train_bars=156,
        test_bars=78,
        fold_step_bars=78,
        min_folds=2,
        max_candidates_per_strategy=24,
        max_workers=2,
        time_budget_sec=90.0,
        init_cash=100_000.0,
        rank_by="profit_pct",
        enable_learning=True,
        output_dir=output_dir,
    )
    result = PaperLeague(cfg).run()
    champ = result.champion
    return {
        "symbol": symbol,
        "strategy_dir": str(strategy_dir),
        "run_dir": str(result.run_dir),
        "competitors": len(result.competitors),
        "champion": (champ.strategy_name if champ else None),
        "champ_profit_pct": (float(champ.profit_pct) if champ else 0.0),
        "champ_win_pct": (float(champ.win_ratio_pct) if champ else 0.0),
    }


# ----------------------------------------------------------- promotion / sleep


def promote_best_per_symbol(
    models_root: Path,
    *,
    include_rejected: bool = False,
    workflow_snapshot_path: str | None = None,
) -> dict:
    """For each symbol, promote the best checkpoint via live.json pointers.

    By default only ``advisory_ready`` checkpoints are promoted.
    """
    from fortuna.models.metadata import PromotionRecord, PromotionStatus
    from fortuna.models.promotion import promote

    best: dict[str, PolicyCheckpoint] = {}
    sources: dict[str, Path] = {}
    for sub in ("validated", "rejected"):
        d = models_root / sub
        if not d.exists():
            continue
        for run in d.iterdir():
            meta = run / "metadata.json"
            if not meta.exists():
                continue
            try:
                cp = PolicyCheckpoint.read(meta)
            except Exception:  # noqa: BLE001
                continue
            if sub == "rejected" and not include_rejected:
                continue
            if not include_rejected and not cp.advisory_ready:
                continue
            key = f"{cp.symbol}|{cp.timeframe}"
            if key not in best or cp.verdict_score > best[key].verdict_score:
                best[key] = cp
                sources[key] = run

    promoted: list[dict] = []
    for key, cp in best.items():
        src = sources[key]
        record = PromotionRecord.from_rl_checkpoint(
            cp,
            src,
            status=PromotionStatus.VALIDATED,
        )
        record.workflow_snapshot_path = workflow_snapshot_path
        try:
            ptr = promote(record, models_root=models_root, promoted_by="nightly")
            promoted.append({
                "symbol": cp.symbol,
                "run_id": cp.run_id,
                "score": cp.verdict_score,
                "passed": cp.verdict_passed,
                "pointer": str(ptr),
            })
        except ValueError as exc:
            promoted.append({
                "symbol": cp.symbol,
                "run_id": cp.run_id,
                "score": cp.verdict_score,
                "passed": cp.verdict_passed,
                "error": str(exc),
            })
    return {
        "promoted": promoted,
        "count": len([p for p in promoted if "pointer" in p]),
        "model_kind": ModelKind.RL_POLICY.value,
        "workflow_snapshot_path": workflow_snapshot_path,
    }


def emit_promotion_review_artifacts(
    promote_result: dict,
    *,
    models_root: Path,
    report_dir: Path,
    fmt: str = "md",
    out_dir: Optional[Path] = None,
) -> dict[str, object]:
    target_dir = out_dir or (report_dir / "promotions")
    written: list[str] = []
    kind_values = {row.value for row in ModelKind}
    raw_kind = str(promote_result.get("model_kind", ModelKind.RL_POLICY.value) or "")
    default_kind = ModelKind(raw_kind) if raw_kind in kind_values else ModelKind.RL_POLICY
    model_kinds: list[str] = []
    for row in promote_result.get("promoted", []):
        if not isinstance(row, dict) or "pointer" not in row:
            continue
        symbol = str(row.get("symbol", "") or "").strip()
        run_id = str(row.get("run_id", "") or "").strip()
        row_raw_kind = str(row.get("model_kind", "") or "").strip()
        row_kind = ModelKind(row_raw_kind) if row_raw_kind in kind_values else default_kind
        if row_kind.value not in model_kinds:
            model_kinds.append(row_kind.value)
        review = build_promotion_review(
            kind=row_kind,
            models_root=models_root,
            symbol=symbol or None,
            run_id=run_id or None,
            settings=get_settings(),
        )
        if review is None:
            continue
        stem_parts = [part for part in (symbol.replace(".", "_"), run_id) if part]
        stem = "__".join(stem_parts) if stem_parts else "promotion_review"
        path = export_promotion_review(review, target_dir / f"{stem}.{fmt}", fmt=fmt)
        written.append(str(path))
    reported_kind = raw_kind or default_kind.value
    if len(model_kinds) > 1:
        reported_kind = "mixed"
    return {
        "count": len(written),
        "paths": written,
        "format": fmt,
        "model_kind": reported_kind,
        "model_kinds": model_kinds or [default_kind.value],
    }


def _merge_promotion_results(*results: dict) -> dict[str, object]:
    promoted: list[dict[str, object]] = []
    workflow_snapshot_path = None
    model_kinds: list[str] = []
    counts_by_kind: dict[str, int] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        raw_kind = str(result.get("model_kind", "") or "").strip().lower()
        if raw_kind and raw_kind not in model_kinds:
            model_kinds.append(raw_kind)
        if workflow_snapshot_path is None:
            workflow_snapshot_path = result.get("workflow_snapshot_path")
        for row in result.get("promoted", []):
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item_kind = str(item.get("model_kind", "") or raw_kind).strip().lower()
            if item_kind:
                item["model_kind"] = item_kind
                counts_by_kind[item_kind] = counts_by_kind.get(item_kind, 0) + (
                    1 if "pointer" in item else 0
                )
            promoted.append(item)
    successful = [row for row in promoted if "pointer" in row]
    return {
        "promoted": promoted,
        "count": len(successful),
        "model_kind": "mixed" if len(model_kinds) > 1 else (model_kinds[0] if model_kinds else ""),
        "model_kinds": model_kinds,
        "counts_by_kind": counts_by_kind,
        "workflow_snapshot_path": workflow_snapshot_path,
    }


def write_report(report: NightlyReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = out_dir / f"{date}.json"
    md_path = out_dir / f"{date}.md"

    json_path.write_text(
        json.dumps({**asdict(report)}, indent=2, default=str),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append(f"# Nightly training report — {date}\n")
    lines.append(f"- Started: `{report.started_at}`")
    lines.append(f"- Ended:   `{report.ended_at}`")
    lines.append(f"- Total:   `{report.total_duration_s:.0f}s`")
    lines.append(f"- Status:  **{report.overall_status}**")
    lines.append(f"- Basket:  `{', '.join(report.basket)}`\n")
    scale_step = next((s for s in report.steps if s.name == "scaling_posture"), None)
    scale_detail = scale_step.detail if scale_step is not None else None
    lane_step = next((s for s in report.steps if s.name == "lane_readiness"), None)
    lane_detail = lane_step.detail if lane_step is not None else None
    if scale_detail is None and isinstance(lane_detail, dict):
        scale_detail = lane_detail.get("scaling_posture")
    if isinstance(scale_detail, dict):
        lines.append("## Scaling posture\n")
        lines.append(f"- Basket posture: `{scale_detail.get('basket_posture', '—')}`")
        lines.append(f"- Basket count: `{scale_detail.get('basket_count', 0)}`")
        lines.append(
            f"- Recommended max symbols: `{scale_detail.get('recommended_max_symbols', '—')}`"
        )
        lines.append(f"- Within recommended: `{scale_detail.get('within_recommended', '—')}`")
        compute = scale_detail.get("compute") or {}
        if isinstance(compute, dict):
            lines.append(
                f"- Machine: `{compute.get('machine_class', '—')}` "
                f"RAM `{compute.get('physical_ram_gb', '—')}` GiB "
                f"RL n_envs `{compute.get('rl_n_envs', '—')}`"
            )
        warnings = scale_detail.get("warnings") or []
        for row in warnings[:3]:
            if isinstance(row, dict):
                lines.append(f"- Warning: {row.get('name')}: {row.get('detail')}")
        lines.append("")
    if isinstance(lane_detail, dict) and lane_detail.get("lane_decisions"):
        lines.append("## Lane readiness\n")
        blockers = lane_detail.get("global_blockers") or []
        if blockers:
            lines.append(f"- Global blockers: `{', '.join(str(b) for b in blockers)}`")
        for row in lane_detail.get("lane_decisions", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('lane', '—')}`: "
                f"**{row.get('disposition', '—')}** "
                f"({row.get('skip_class', 'none')}) — {row.get('detail', '')}"
            )
        lines.append("")
    retention_step = next((s for s in report.steps if s.name == "artifact_retention"), None)
    retention_detail = retention_step.detail if retention_step is not None else None
    if isinstance(retention_detail, dict):
        lines.append("## Artifact retention\n")
        lines.append(f"- Enabled: `{retention_detail.get('enabled', False)}`")
        lines.append(f"- Policy: `{retention_detail.get('policy', '—')}`")
        lines.append(f"- Retained count: `{retention_detail.get('retained_count', 0)}`")
        lines.append(f"- Pruned count: `{retention_detail.get('pruned_count', 0)}`")
        detail_text = str(retention_detail.get("detail", "") or "").strip()
        if detail_text:
            lines.append(f"- Detail: {detail_text}")
        lines.append("")
    workflow_step = next((s for s in report.steps if s.name == "workflow_snapshot"), None)
    workflow_detail = workflow_step.detail if workflow_step is not None else None
    if isinstance(workflow_detail, dict):
        lines.append("## Workflow Snapshot\n")
        lines.append(f"- Artifact: `{workflow_detail.get('path', '')}`")
        lines.append(
            "- Summary: "
            f"universe={workflow_detail.get('universe_count', 0)} "
            f"shortlist={workflow_detail.get('shortlist_count', 0)} "
            f"briefing_candidates={workflow_detail.get('briefing_candidates', 0)} "
            f"ml={workflow_detail.get('ml_count', 0)} "
            f"rl={workflow_detail.get('rl_count', 0)}\n"
        )
        if workflow_detail.get("team_role_count", 0):
            lines.append(
                "- Team posture: "
                f"roles={workflow_detail.get('team_ok_role_count', 0)}/"
                f"{workflow_detail.get('team_role_count', 0)} "
                f"nightly={workflow_detail.get('team_nightly_aligned_reports', 0)}/"
                f"{workflow_detail.get('team_nightly_enabled_reports', 0)} "
                f"target={workflow_detail.get('team_nightly_recommended_target', '—')}"
                + (
                    " force_refresh=1"
                    if workflow_detail.get("team_nightly_recommended_force_refresh")
                    else ""
                )
                + "\n"
            )
        latest_team_mix = str(
            workflow_detail.get("team_nightly_latest_target_mix", "") or ""
        ).strip()
        if latest_team_mix:
            lines.append(f"- Team nightly target mix: `{latest_team_mix}`\n")
        latest_refreshed_team_mix = str(
            workflow_detail.get("team_nightly_latest_refreshed_target_mix", "") or ""
        ).strip()
        if latest_refreshed_team_mix:
            lines.append(
                f"- Team refreshed nightly target mix: `{latest_refreshed_team_mix}`\n"
            )
        team_research_headline = str(
            workflow_detail.get("team_research_headline", "") or ""
        ).strip()
        if team_research_headline:
            lines.append(
                "- Team research posture: "
                f"{team_research_headline} "
                f"(ml={workflow_detail.get('team_research_ml_count', 0)} "
                f"rl={workflow_detail.get('team_research_rl_count', 0)} "
                f"policy={workflow_detail.get('team_research_selection_policy', 'ranked')} "
                f"target={workflow_detail.get('team_research_refresh_target', 'all')}"
                + (
                    " "
                    "discovery_target="
                    f"{workflow_detail.get('team_research_discovery_recommended_target')}"
                    if workflow_detail.get("team_research_discovery_recommended_target")
                    else ""
                )
                + (
                    " "
                    f"effective_target={workflow_detail.get('team_research_effective_target')}"
                    if workflow_detail.get("team_research_effective_target")
                    else ""
                )
                + (
                    f" refreshed={workflow_detail.get('team_research_refreshed_count', 0)}"
                    if workflow_detail.get("team_research_refresh_requested")
                    else ""
                )
                + (" mismatch=1" if workflow_detail.get("team_research_target_mismatch") else "")
                + ")\n"
            )
        if workflow_detail.get("research_count", 0) or workflow_detail.get(
            "research_refresh_requested"
        ):
            lines.append(
                "- Training research: "
                f"count={workflow_detail.get('research_count', 0)} "
                f"ml={workflow_detail.get('research_ml_count', 0)} "
                f"rl={workflow_detail.get('research_rl_count', 0)} "
                f"selection_policy={workflow_detail.get('research_selection_policy', 'ranked')} "
                f"refresh_target={workflow_detail.get('research_refresh_target', 'all')}"
                + (
                    " "
                    "discovery_target="
                    f"{workflow_detail.get('research_discovery_recommended_refresh_target')}"
                    if workflow_detail.get("research_discovery_recommended_refresh_target")
                    else ""
                )
                + (
                    " "
                    f"effective_target={workflow_detail.get('research_effective_refresh_target')}"
                    if workflow_detail.get("research_effective_refresh_target")
                    else ""
                )
                + (
                    f" refreshed={workflow_detail.get('research_refreshed_count', 0)}"
                    if workflow_detail.get("research_refresh_requested")
                    else ""
                )
                + (" mismatch=1" if workflow_detail.get("research_target_mismatch") else "")
                + "\n"
            )
        if workflow_detail.get("allocation_research_alignment_enabled"):
            mix = str(workflow_detail.get("allocation_research_target_mix", "") or "").strip()
            if mix:
                lines.append(f"- Allocation research alignment: `{mix}`\n")
            else:
                lines.append("- Allocation research alignment: `enabled`\n")
        refreshed_mix = str(
            workflow_detail.get("allocation_refreshed_research_target_mix", "") or ""
        ).strip()
        if refreshed_mix:
            lines.append(f"- Refreshed allocation research alignment: `{refreshed_mix}`\n")
    candidate_step = next((s for s in report.steps if s.name == "training_candidates"), None)
    candidate_detail = candidate_step.detail if candidate_step is not None else None
    if isinstance(candidate_detail, dict):
        lines.append("## Training Candidates\n")
        lines.append(f"- Artifact: `{candidate_detail.get('path', '')}`")
        lines.append(
            "- Summary: "
            f"count={candidate_detail.get('count', 0)} "
            f"ml={candidate_detail.get('ml_count', 0)} "
            f"rl={candidate_detail.get('rl_count', 0)} "
            f"selection_policy={candidate_detail.get('selection_policy', 'ranked')}\n"
        )
    research_step = next((s for s in report.steps if s.name == "training_research"), None)
    research_detail = research_step.detail if research_step is not None else None
    if isinstance(research_detail, dict):
        lines.append("## Training Research Plan\n")
        lines.append(f"- Artifact: `{research_detail.get('path', '')}`")
        lines.append(
            "- Summary: "
            f"count={research_detail.get('count', 0)} "
            f"ml={research_detail.get('ml_count', 0)} "
            f"rl={research_detail.get('rl_count', 0)} "
            f"selection_policy={research_detail.get('selection_policy', 'ranked')} "
            f"refresh_target={research_detail.get('refresh_target', 'all')}"
            + (
                " "
                f"discovery_target={research_detail.get('discovery_recommended_refresh_target')}"
                if research_detail.get("discovery_recommended_refresh_target")
                else ""
            )
            + (
                " "
                f"effective_target={research_detail.get('effective_refresh_target')}"
                if research_detail.get("effective_refresh_target")
                else ""
            )
            + (
                f" refreshed={research_detail.get('refreshed_count', 0)}"
                if research_detail.get("refresh_requested")
                else ""
            )
            + (" mismatch=1" if research_detail.get("target_mismatch") else "")
            + "\n"
        )
        discovery_summary = str(research_detail.get("discovery_summary", "") or "").strip()
        discovery_regimes = str(research_detail.get("discovery_regime_mix", "") or "").strip()
        preferred_count = int(research_detail.get("discovery_preferred_count", 0) or 0)
        if discovery_summary or discovery_regimes or preferred_count > 0:
            lines.append(
                "- Discovery context: "
                f"preferred={preferred_count} "
                f"regimes={discovery_regimes or '—'}"
                + (f" summary={discovery_summary}" if discovery_summary else "")
                + "\n"
            )
    review_step = next((s for s in report.steps if s.name == "promotion_reviews"), None)
    review_detail = review_step.detail if review_step is not None else None
    if isinstance(review_detail, dict) and review_detail.get("count", 0):
        lines.append("## Promotion Reviews\n")
        lines.append(f"- Format: `{review_detail.get('format', 'md')}`")
        model_kinds = review_detail.get("model_kinds", ())
        if isinstance(model_kinds, list) and model_kinds:
            lines.append(f"- Model kinds: `{', '.join(str(kind) for kind in model_kinds)}`")
        elif review_detail.get("model_kind"):
            lines.append(f"- Model kind: `{review_detail.get('model_kind')}`")
        for path in review_detail.get("paths", []):
            lines.append(f"- Artifact: `{path}`")
        lines.append("")
    lines.append("## Steps\n")
    lines.append("| Step | Status | Duration |")
    lines.append("|---|---|---:|")
    for s in report.steps:
        lines.append(f"| `{s.name}` | {s.status} | {s.duration_s:.1f}s |")
    lines.append("\n## RL training results\n")
    lines.append("| Symbol | Policy | Sharpe | PF | Trades | Win % | Passed |")
    lines.append("|---|---|---:|---:|---:|---:|:---:|")
    for s in report.steps:
        d = s.detail
        if not isinstance(d, dict) or "policy" not in d:
            continue
        lines.append(
            f"| `{d.get('symbol')}` | {d.get('policy')} | "
            f"{d.get('sharpe', 0):.2f} | {d.get('pf', 0):.2f} | "
            f"{d.get('trades', 0)} | {d.get('win_rate', 0):.1f} | "
            f"{'PASS' if d.get('passed') else 'fail'} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def sleep_host() -> None:
    """Suspend the Windows host. Requires the script to run with permission to
    set the SetSuspendState flag — works without admin on most Windows boxes."""
    import subprocess

    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command",
         "Add-Type -AssemblyName System.Windows.Forms; "
         "[System.Windows.Forms.Application]::SetSuspendState('Suspend',$false,$false)"],
        creationflags=0x00000008,  # DETACHED_PROCESS
    )


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="Fortuna nightly training orchestrator")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_BASKET,
                        help="cash-equity symbols to train (default: 10-symbol NSE F&O basket)")
    parser.add_argument("--include-futures", dest="include_futures",
                        action="store_true", default=True,
                        help="also train on the matching .FUT (rolling front-month) symbols")
    parser.add_argument("--no-futures", dest="include_futures",
                        action="store_false",
                        help="train cash equities only — skip .FUT symbols")
    parser.add_argument("--futures-symbols", nargs="*", default=None,
                        help="explicit futures basket (defaults to mirroring --symbols on .FUT)")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--data-source", default="smartapi")
    parser.add_argument("--policies", nargs="*", default=["CnnPolicy"],
                        choices=["MlpPolicy", "CnnPolicy"],
                        help="which policy types to train per symbol")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--use-subproc", action="store_true", default=True)
    parser.add_argument("--no-subproc", dest="use_subproc", action="store_false")
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--report-dir", default="reports/nightly")
    parser.add_argument("--ml-learning-dir", default="logs/agentic")
    parser.add_argument("--ml-model-kind", choices=("logistic", "hgb"), default="logistic")
    parser.add_argument("--ml-horizon-bars", type=int, default=3)
    parser.add_argument("--ml-eval-fraction", type=float, default=0.2)
    parser.add_argument("--ml-seed", type=int, default=42)
    parser.add_argument(
        "--promotion-review-dir",
        default="",
        help=(
            "Optional directory for emitted promotion review artifacts "
            "(default: <report-dir>/promotions)"
        ),
    )
    parser.add_argument(
        "--promotion-review-format",
        choices=["md", "json"],
        default="md",
        help="Artifact format for nightly promotion reviews",
    )
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--skip-promote", action="store_true")
    parser.add_argument(
        "--promote-include-rejected",
        action="store_true",
        help="promote best checkpoint even when advisory_ready=False",
    )
    parser.add_argument("--skip-deterministic", action="store_true",
                        help="skip PaperLeague walk-forward sweep on deterministic strategies")
    parser.add_argument("--strategy-dirs", nargs="*",
                        default=["strategies/intraday", "strategies/builtin"],
                        help="dirs containing deterministic JSON strategies to tune")
    parser.add_argument("--param-grid", default="strategies/grids/ema_grid.json")
    parser.add_argument("--paper-league-output", default="logs/paper_league")
    parser.add_argument("--sleep-after", action="store_true",
                        help="suspend the Windows host after completion")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip pre-flight Streamlit cleanup and RAM auto-tuning")
    parser.add_argument("--no-auto-tune-envs", dest="auto_tune_envs",
                        action="store_false", default=True,
                        help="keep --n-envs / --use-subproc even on low-RAM laptops")
    parser.add_argument("--max-symbols", type=int, default=None,
                        help="cap basket size (useful for debugging)")
    parser.add_argument(
        "--candidate-manifest",
        default="",
        help="Optional shortlist-driven training candidate manifest JSON",
    )
    parser.add_argument(
        "--use-training-candidates",
        action="store_true",
        help="Build the nightly basket from shortlist-derived training candidates",
    )
    parser.add_argument(
        "--candidate-target",
        choices=("all", "ml", "rl"),
        default="all",
        help="Which candidate subset to use when manifest-driven selection is enabled",
    )
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=0,
        help="Optional top-N cap when candidate-driven selection is enabled",
    )
    parser.add_argument(
        "--candidate-source",
        choices=("auto", "screener", "registry"),
        default="auto",
        help="Universe source when deriving training candidates on the fly",
    )
    parser.add_argument(
        "--candidate-selection-policy",
        choices=("ranked", "diversified"),
        default="ranked",
        help="How shortlisted training candidates are converted into the nightly basket",
    )
    parser.add_argument("--candidate-universe-limit", type=int, default=15)
    parser.add_argument("--candidate-analysis-limit", type=int, default=8)
    parser.add_argument(
        "--training-candidate-out",
        default="",
        help=(
            "Optional training-candidate manifest JSON path; "
            "candidate-driven runs emit one by default"
        ),
    )
    parser.add_argument(
        "--training-research-out",
        default="",
        help=(
            "Optional training research plan JSON path; "
            "candidate-driven runs emit one by default"
        ),
    )
    parser.add_argument(
        "--workflow-snapshot-out",
        default="",
        help="Optional workflow snapshot JSON path; candidate-driven runs emit one by default",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings = get_settings()

    with workflow_run_context() as nightly_run_id:
        with workflow_boundary(
            settings,
            event_name="nightly_train",
            module="scripts.nightly_train",
            workflow_id="nightly_train",
            run_id=nightly_run_id,
        ) as nightly_span:
            exit_code = _run_nightly_main(
                args,
                repo_root=repo_root,
                settings=settings,
                nightly_run_id=nightly_run_id,
                nightly_span=nightly_span,
            )
            return exit_code


def _run_nightly_main(
    args,
    *,
    repo_root: Path,
    settings,
    nightly_run_id: str,
    nightly_span,
) -> int:
    candidate_response = resolve_training_candidate_response(args)
    symbols, basket_meta = resolve_symbol_basket(args, candidate_response=candidate_response)

    models_root = Path(args.checkpoint_dir)

    report = NightlyReport(started_at=_now_iso(), basket=symbols)
    start = time.perf_counter()
    pf = None

    if not args.skip_preflight:
        logger.info("[nightly] === phase 0/5: pre-flight ===")
        pf = run_nightly_preflight(
            repo_root,
            requested_n_envs=args.n_envs,
            requested_use_subproc=args.use_subproc,
            kill_streamlit=True,
            auto_tune_envs=args.auto_tune_envs,
        )
        report.add(StepResult(
            name="preflight",
            status="ok",
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_s=0.0,
            detail=pf.as_detail(),
        ))
        emit_workflow_step(
            settings,
            event_name="nightly_train.preflight",
            module="scripts.nightly_train",
            workflow_id="nightly_train",
            run_id=nightly_run_id,
            context={"n_envs": pf.n_envs, "should_abort": pf.should_abort},
        )
        report.add(StepResult(
            name="basket_resolution",
            status="ok",
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_s=0.0,
            detail=basket_meta,
        ))
        if args.auto_tune_envs:
            args.n_envs = pf.n_envs
            args.use_subproc = pf.use_subproc
    else:
        report.add(StepResult(
            name="basket_resolution",
            status="ok",
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_s=0.0,
            detail=basket_meta,
        ))

    training_candidate_meta = emit_training_candidate_manifest(
        args,
        report_dir=Path(args.report_dir),
        candidate_response=candidate_response,
    )
    if training_candidate_meta is not None:
        report.add(StepResult(
            name="training_candidates",
            status="ok",
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_s=0.0,
            detail=training_candidate_meta,
        ))

    training_research_meta = emit_training_research_plan(
        args,
        report_dir=Path(args.report_dir),
    )
    if training_research_meta is not None:
        report.add(StepResult(
            name="training_research",
            status="ok",
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_s=0.0,
            detail=training_research_meta,
        ))

    execution_plan = _build_training_execution_plan(
        symbols,
        basket_meta if isinstance(basket_meta, dict) else None,
        training_research_meta if isinstance(training_research_meta, dict) else None,
    )
    report.add(StepResult(
        name="training_execution_target",
        status="ok",
        started_at=_now_iso(),
        ended_at=_now_iso(),
        duration_s=0.0,
        detail=execution_plan,
    ))

    workflow_meta = emit_workflow_snapshot(args, report_dir=Path(args.report_dir))
    workflow_snapshot_path = (
        str(workflow_meta.get("path", "")).strip()
        if isinstance(workflow_meta, dict)
        else ""
    )
    if workflow_meta is not None:
        report.add(StepResult(
            name="workflow_snapshot",
            status="ok",
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_s=0.0,
            detail=workflow_meta,
        ))

    lane_gating = bool(getattr(settings, "nightly_lane_gating_enabled", True))
    fail_fast = bool(getattr(settings, "nightly_fail_fast_global_blockers", True))
    if pf is not None:
        attach_lane_readiness(
            pf,
            settings,
            execution_plan=execution_plan,
            data_source=args.data_source,
            symbols=symbols,
        )
        lane_detail = pf.lane_readiness_detail()
        preflight_step = next((step for step in report.steps if step.name == "preflight"), None)
        if preflight_step is not None:
            preflight_step.detail = pf.as_detail()
    else:
        lane_detail = lane_readiness_for_step(
            settings,
            execution_plan=execution_plan,
            data_source=args.data_source,
            symbols=symbols,
        )
    report.add(StepResult(
        name="lane_readiness",
        status="ok",
        started_at=_now_iso(),
        ended_at=_now_iso(),
        duration_s=0.0,
        detail=lane_detail,
    ))
    scaling_detail = None
    if isinstance(lane_detail, dict):
        scaling_detail = lane_detail.get("scaling_posture")
    if scaling_detail is None and pf is not None:
        scaling_detail = pf.scaling_posture
    if scaling_detail is None:
        from fortuna.app.scaling_posture import build_scaling_posture_summary

        scaling_detail = build_scaling_posture_summary(
            settings,
            basket_count=len(symbols),
            physical_ram_gb=getattr(pf, "physical_ram_gb", None) if pf else None,
            n_envs=args.n_envs,
            use_subproc=args.use_subproc,
        ).to_dict()
    report.add(StepResult(
        name="scaling_posture",
        status="ok",
        started_at=_now_iso(),
        ended_at=_now_iso(),
        duration_s=0.0,
        detail=scaling_detail,
    ))
    emit_workflow_step(
        settings,
        event_name="nightly_train.lane_readiness",
        module="scripts.nightly_train",
        workflow_id="nightly_train",
        run_id=nightly_run_id,
        context={
            "global_blockers": (
                lane_detail.get("global_blockers") if isinstance(lane_detail, dict) else []
            ),
            "should_abort": (
                lane_detail.get("should_abort") if isinstance(lane_detail, dict) else False
            ),
        },
    )
    emit_workflow_step(
        settings,
        event_name="nightly_train.scaling_posture",
        module="scripts.nightly_train",
        workflow_id="nightly_train",
        run_id=nightly_run_id,
        context={
            "basket_posture": (
                scaling_detail.get("basket_posture") if isinstance(scaling_detail, dict) else None
            ),
            "basket_count": len(symbols),
        },
    )

    n_cash = sum(1 for s in symbols if not _is_rolling_future(s) and ".FUT" not in s.upper())
    n_fut = len(symbols) - n_cash
    logger.info(
        "[nightly] starting | basket=%d (%d cash + %d futures) | policies=%s | "
        "device=%s | n_envs=%d | use_subproc=%s | keep_awake=%s | mode=%s | target=%s",
        len(symbols), n_cash, n_fut, args.policies, args.device, args.n_envs,
        args.use_subproc, _KEEP_AWAKE_OK, basket_meta.get("mode"), execution_plan.get("target"),
    )

    backfill_disp = _lane_disposition(lane_detail if lane_gating else None, "backfill")
    ml_disp = _lane_disposition(lane_detail if lane_gating else None, "ml")
    rl_disp = _lane_disposition(lane_detail if lane_gating else None, "rl")
    promo_disp = _lane_disposition(lane_detail if lane_gating else None, "promotion")
    should_abort = lane_gating and _should_abort_after_lane_readiness(lane_detail)
    if should_abort:
        logger.warning(
            "[nightly] aborting expensive phases after lane readiness: %s",
            ", ".join(str(row) for row in lane_detail.get("global_blockers", []))
            if isinstance(lane_detail, dict)
            else "lane blockers",
        )

    # ---- Step 1: backfill data
    logger.info("[nightly] === phase 1/5: data backfill (%d symbols) ===", len(symbols))
    _run_backfill_phase(
        report,
        symbols,
        timeframe=args.timeframe,
        days=args.days,
        data_source=args.data_source,
        lane_gating=lane_gating,
        lane_detail=lane_detail if lane_gating else None,
        fail_fast=fail_fast,
    )

    # ---- Step 2: regime detector (multi-symbol)
    if not args.skip_regime:
        if lane_gating and backfill_disp != "run":
            row = _lane_row(lane_detail, "backfill")
            report.add(_skip_step(
                "regime_detector",
                {
                    "skip_class": str(row.get("skip_class", "missing_prerequisites")),
                    "reason": "backfill lane blocked",
                },
            ))
        else:
            logger.info("[nightly] === phase 2/5: regime detector ===")
            report.add(_timed(
                "regime_detector",
                train_regime_multi,
                symbols, args.timeframe, args.days, args.data_source,
                Path("models/regime/classifier.joblib"),
            ))

    # ---- Step 3: target-aware model refresh
    if bool(execution_plan.get("run_ml")):
        ml_symbols = _cash_equity_symbols(symbols)
        if lane_gating and ml_disp != "run":
            row = _lane_row(lane_detail, "ml")
            report.add(_skip_step(
                "ml_train",
                {
                    "skip_class": str(row.get("skip_class", "policy")),
                    "reason": str(row.get("detail", "ML lane not allowed")),
                    "target": execution_plan.get("target"),
                },
            ))
        elif ml_symbols:
            logger.info(
                "[nightly] === phase 3/5: ML scorer training (%d cash symbols) ===",
                len(ml_symbols),
            )
            ml_output_dir = settings.resolve_path(settings.ml_scorer_artifact_dir)
            for sym in ml_symbols:
                logger.info("[nightly] ml_train start: %s", sym)
                report.add(_timed(
                    f"ml_train:{sym}",
                    train_ml_scorer_symbol,
                    sym,
                    timeframe=args.timeframe,
                    days=args.days,
                    repo_root=repo_root,
                    learning_dir=args.ml_learning_dir,
                    output_dir=ml_output_dir,
                    model_kind=args.ml_model_kind,
                    horizon_bars=args.ml_horizon_bars,
                    eval_fraction=args.ml_eval_fraction,
                    seed=args.ml_seed,
                ))
        else:
            report.add(_skip_step(
                "ml_train",
                {
                    "skip_class": "policy",
                    "reason": "no_cash_symbols",
                    "target": execution_plan.get("target"),
                },
            ))
        if not bool(execution_plan.get("run_rl")):
            report.add(_skip_step(
                "rl_train",
                {
                    "skip_class": "policy",
                    "reason": "target_prefers_ml",
                    "target": execution_plan.get("target"),
                },
            ))
    if bool(execution_plan.get("run_rl")):
        if lane_gating and rl_disp != "run":
            row = _lane_row(lane_detail, "rl")
            report.add(_skip_step(
                "rl_train",
                {
                    "skip_class": str(row.get("skip_class", "policy")),
                    "reason": str(row.get("detail", "RL lane not allowed")),
                    "target": execution_plan.get("target"),
                },
            ))
        else:
            logger.info(
                "[nightly] === phase 3/5: RL training (%d symbols × %d policies) ===",
                len(symbols), len(args.policies),
            )
            for sym in symbols:
                for pol in args.policies:
                    logger.info("[nightly] rl_train start: %s %s", sym, pol)
                    report.add(_timed(
                        f"rl_train:{sym}:{pol}",
                        train_one_symbol,
                        sym, args.timeframe, pol, args.timesteps, args.days,
                        args.data_source, args.n_envs, args.device,
                        args.use_subproc, models_root,
                    ))
        if not bool(execution_plan.get("run_ml")):
            report.add(_skip_step(
                "ml_train",
                {
                    "skip_class": "policy",
                    "reason": "target_prefers_rl",
                    "target": execution_plan.get("target"),
                },
            ))
    elif not bool(execution_plan.get("run_ml")):
        report.add(_skip_step(
            "ml_train",
            {
                "skip_class": "policy",
                "reason": "target_prefers_rl",
                "target": execution_plan.get("target"),
            },
        ))

    # ---- Step 4: deterministic strategy walk-forward sweep
    if not args.skip_deterministic:
        if should_abort:
            report.add(_skip_step(
                "deterministic_sweep",
                {
                    "skip_class": "missing_prerequisites",
                    "reason": "all_requested_training_lanes_blocked",
                },
            ))
        else:
            logger.info("[nightly] === phase 4/5: deterministic strategy sweep ===")
            param_grid = Path(args.param_grid) if args.param_grid else None
            if param_grid and not param_grid.exists():
                logger.warning(
                    "[nightly] param-grid not found: %s (continuing without)",
                    param_grid,
                )
                param_grid = None
            for sym in symbols:
                for sdir in args.strategy_dirs:
                    sdir_path = Path(sdir)
                    if not sdir_path.exists():
                        continue
                    report.add(_timed(
                        f"deterministic:{sym}:{sdir_path.name}",
                        optimize_deterministic_strategies,
                        sym, args.timeframe, min(args.days, 60), args.data_source,
                        sdir_path, Path(args.paper_league_output), param_grid,
                    ))

    # ---- Step 5: promotion
    promotion_blocked = lane_gating and promo_disp == "block"
    if promotion_blocked:
        row = _lane_row(lane_detail, "promotion")
        report.add(_skip_step(
            "promote_best",
            {
                "skip_class": str(row.get("skip_class", "missing_prerequisites")),
                "reason": str(row.get("detail", "promotion lane blocked")),
                "target": execution_plan.get("target"),
            },
        ))
    elif not args.skip_promote and lane_gating and promo_disp == "skip":
        row = _lane_row(lane_detail, "promotion")
        report.add(_skip_step(
            "promote_best",
            {
                "skip_class": str(row.get("skip_class", "policy")),
                "reason": str(row.get("detail", "promotion skipped by policy")),
                "target": execution_plan.get("target"),
            },
        ))
    elif not args.skip_promote and bool(execution_plan.get("allow_ml_promotion")):
        logger.info("[nightly] === phase 5/5: promote best ML scorer ===")
        settings = get_settings()
        ml_validated_dir = settings.resolve_path(settings.ml_scorer_artifact_dir)
        ml_base = (
            ml_validated_dir.parent
            if ml_validated_dir.name == "validated"
            else ml_validated_dir
        )
        promote_step = _timed(
            "promote_best",
            promote_best_ml_scorer,
            models_root,
            ml_base,
            include_rejected=args.promote_include_rejected,
            eligible_symbols=_cash_equity_symbols(symbols),
            timeframe=args.timeframe,
            workflow_snapshot_path=workflow_snapshot_path or None,
        )
        report.add(promote_step)
        promote_detail = promote_step.detail if isinstance(promote_step.detail, dict) else {}
        if promote_step.status == "ok" and int(promote_detail.get("count", 0) or 0) > 0:
            review_dir = (
                Path(args.promotion_review_dir)
                if str(args.promotion_review_dir or "").strip()
                else Path(args.report_dir) / "promotions"
            )
            report.add(_timed(
                "promotion_reviews",
                emit_promotion_review_artifacts,
                promote_detail,
                models_root=models_root,
                report_dir=Path(args.report_dir),
                fmt=args.promotion_review_format,
                out_dir=review_dir,
            ))
    elif not args.skip_promote and bool(execution_plan.get("allow_rl_promotion")):
        logger.info("[nightly] === phase 5/5: promote best per symbol ===")
        promote_step = _timed(
            "promote_best",
            promote_best_per_symbol,
            models_root,
            include_rejected=args.promote_include_rejected,
            workflow_snapshot_path=workflow_snapshot_path or None,
        )
        report.add(promote_step)
        promote_detail = promote_step.detail if isinstance(promote_step.detail, dict) else {}
        if promote_step.status == "ok" and int(promote_detail.get("count", 0) or 0) > 0:
            review_dir = (
                Path(args.promotion_review_dir)
                if str(args.promotion_review_dir or "").strip()
                else Path(args.report_dir) / "promotions"
            )
            report.add(_timed(
                "promotion_reviews",
                emit_promotion_review_artifacts,
                promote_detail,
                models_root=models_root,
                report_dir=Path(args.report_dir),
                fmt=args.promotion_review_format,
                out_dir=review_dir,
            ))
    elif not args.skip_promote and str(execution_plan.get("target") or "") == "all":
        logger.info("[nightly] === phase 5/5: mixed ML/RL promotion bundle ===")
        settings = get_settings()
        ml_validated_dir = settings.resolve_path(settings.ml_scorer_artifact_dir)
        ml_base = (
            ml_validated_dir.parent
            if ml_validated_dir.name == "validated"
            else ml_validated_dir
        )
        ml_result = promote_best_ml_scorer(
            models_root,
            ml_base,
            include_rejected=args.promote_include_rejected,
            eligible_symbols=_cash_equity_symbols(symbols),
            timeframe=args.timeframe,
            workflow_snapshot_path=workflow_snapshot_path or None,
        )
        rl_result = promote_best_per_symbol(
            models_root,
            include_rejected=args.promote_include_rejected,
            workflow_snapshot_path=workflow_snapshot_path or None,
        )
        promote_detail = _merge_promotion_results(ml_result, rl_result)
        report.add(StepResult(
            name="promote_best",
            status="ok",
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_s=0.0,
            detail=promote_detail,
        ))
        if int(promote_detail.get("count", 0) or 0) > 0:
            review_dir = (
                Path(args.promotion_review_dir)
                if str(args.promotion_review_dir or "").strip()
                else Path(args.report_dir) / "promotions"
            )
            report.add(_timed(
                "promotion_reviews",
                emit_promotion_review_artifacts,
                promote_detail,
                models_root=models_root,
                report_dir=Path(args.report_dir),
                fmt=args.promotion_review_format,
                out_dir=review_dir,
            ))
    elif not args.skip_promote:
        report.add(_skip_step(
            "promote_best",
            {
                "skip_class": "policy",
                "reason": (
                    "mixed_target_requires_manual_review"
                    if str(execution_plan.get("target") or "") == "all"
                    else "target_prefers_ml"
                ),
                "target": execution_plan.get("target"),
            },
        ))

    # ---- finalize
    report.ended_at = _now_iso()
    report.total_duration_s = time.perf_counter() - start
    report.overall_status = finalize_overall_status(report)
    nightly_span.set_context(
        overall_status=report.overall_status,
        basket_count=len(symbols),
        duration_s=int(report.total_duration_s),
    )
    emit_workflow_step(
        settings,
        event_name="nightly_train.completed",
        module="scripts.nightly_train",
        workflow_id="nightly_train",
        run_id=nightly_run_id,
        status="ok" if report.overall_status == "ok" else "warn",
        context={"overall_status": report.overall_status},
    )

    from fortuna.app.scaling_posture import apply_nightly_report_retention

    retention = apply_nightly_report_retention(
        Path(args.report_dir),
        settings,
    )
    report.add(StepResult(
        name="artifact_retention",
        status="ok",
        started_at=_now_iso(),
        ended_at=_now_iso(),
        duration_s=0.0,
        detail=retention.to_dict(),
    ))
    md_path = write_report(report, Path(args.report_dir))
    json_path = md_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({**asdict(report)}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("[nightly] report -> %s", md_path)
    print(f"\n=== Nightly complete: {report.overall_status} ===")
    print(f"Total: {report.total_duration_s:.0f}s ({report.total_duration_s / 60:.1f} min)")
    print(f"Report: {md_path}")

    if args.sleep_after:
        logger.info("[nightly] suspending host...")
        time.sleep(5)
        # Release the power-state lock first, otherwise SetSuspendState
        # is a no-op while ES_SYSTEM_REQUIRED is held.
        if _KEEP_AWAKE_OK:
            _keep_awake_release()
        sleep_host()
    elif _KEEP_AWAKE_OK:
        # Always tidy up the lock on a clean exit even when not sleeping.
        _keep_awake_release()

    return 0 if report.overall_status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
