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

from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401,E402
from fortuna.utils.keep_awake import activate as _keep_awake_activate  # noqa: E402
from fortuna.utils.keep_awake import release as _keep_awake_release  # noqa: E402
from fortuna.utils.nightly_preflight import run_nightly_preflight  # noqa: E402
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
    return {"promoted": promoted, "count": len([p for p in promoted if "pointer" in p])}


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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    cash_symbols = args.symbols[: args.max_symbols] if args.max_symbols else args.symbols
    if args.include_futures:
        if args.futures_symbols is not None:
            futures_symbols = args.futures_symbols
        else:
            # Mirror the cash basket on .FUT (rolling front-month). Only the
            # ``.NS`` suffix is rewritten; pre-suffixed entries pass through.
            futures_symbols = [
                s.replace(".NS", ".FUT") if s.endswith(".NS") else s
                for s in cash_symbols
            ]
        symbols = list(cash_symbols) + list(futures_symbols)
    else:
        symbols = list(cash_symbols)

    models_root = Path(args.checkpoint_dir)

    report = NightlyReport(started_at=_now_iso(), basket=symbols)
    start = time.perf_counter()

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
        if args.auto_tune_envs:
            args.n_envs = pf.n_envs
            args.use_subproc = pf.use_subproc

    n_cash = sum(1 for s in symbols if not _is_rolling_future(s) and ".FUT" not in s.upper())
    n_fut = len(symbols) - n_cash
    logger.info(
        "[nightly] starting | basket=%d (%d cash + %d futures) | policies=%s | "
        "device=%s | n_envs=%d | use_subproc=%s | keep_awake=%s",
        len(symbols), n_cash, n_fut, args.policies, args.device, args.n_envs,
        args.use_subproc, _KEEP_AWAKE_OK,
    )

    # ---- Step 1: backfill data
    logger.info("[nightly] === phase 1/5: data backfill (%d symbols) ===", len(symbols))
    for sym in symbols:
        report.add(_timed(
            f"backfill:{sym}",
            backfill_symbol, sym, args.timeframe, args.days, args.data_source,
        ))

    # ---- Step 2: regime detector (multi-symbol)
    if not args.skip_regime:
        logger.info("[nightly] === phase 2/5: regime detector ===")
        report.add(_timed(
            "regime_detector",
            train_regime_multi,
            symbols, args.timeframe, args.days, args.data_source,
            Path("models/regime/classifier.joblib"),
        ))

    # ---- Step 3: RL training per (symbol, policy)
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

    # ---- Step 4: deterministic strategy walk-forward sweep
    if not args.skip_deterministic:
        logger.info("[nightly] === phase 4/5: deterministic strategy sweep ===")
        param_grid = Path(args.param_grid) if args.param_grid else None
        if param_grid and not param_grid.exists():
            logger.warning("[nightly] param-grid not found: %s (continuing without)", param_grid)
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
    if not args.skip_promote:
        logger.info("[nightly] === phase 5/5: promote best per symbol ===")
        report.add(_timed(
            "promote_best",
            promote_best_per_symbol,
            models_root,
            include_rejected=args.promote_include_rejected,
        ))

    # ---- finalize
    report.ended_at = _now_iso()
    report.total_duration_s = time.perf_counter() - start
    n_fail = sum(1 for s in report.steps if s.status == "fail")
    report.overall_status = "ok" if n_fail == 0 else f"partial ({n_fail} failures)"

    md_path = write_report(report, Path(args.report_dir))
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

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
