#!/usr/bin/env python
"""Run arena and print champion + latest-bar signal (uses venv python, no uv)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("FORTUNA_CONFIG", "configs/intraday.yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

from fortuna.arena import ArenaConfig, StrategyArena
from fortuna.arena.param_search import ParamSearchResult
from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.search.indicator_cache import IndicatorCache
from fortuna.search.signals import emit_signal, signal_at_bar
from fortuna.strategy.warmup import slice_for_backtest
from fortuna.utils.timing import timed_step


def _champion_strategy(board) -> tuple[ParamSearchResult | None, object]:
    winner = board.overall_winner()
    if not winner:
        return None, None
    for r in board.results:
        if r.best_report.candidate_id == winner.candidate_id:
            return r, r.best_strategy
    for r in board.results:
        if winner.strategy_name.startswith(r.base_name) or r.best_report.strategy_name == winner.strategy_name:
            return r, r.best_strategy
    return None, None


def _step(msg: str) -> None:
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    print("=" * 60, flush=True)
    print("Arena quick run", flush=True)
    print(f"  PID={os.getpid()}  CWD={Path.cwd()}", flush=True)
    print(f"  FORTUNA_CONFIG={os.environ.get('FORTUNA_CONFIG', '(default)')}", flush=True)
    print("=" * 60, flush=True)

    config = ArenaConfig(
        symbol="RELIANCE.NS",
        timeframe="5m",
        days=7,
        param_grid_path=Path("__none__"),
        max_candidates_per_strategy=15,
        max_workers=2,
        chunk_size=4,
        time_budget_sec=20.0,
        window_bars=78,
        warmup_bars=78,
        use_streaming=False,
        rank_by="profit_pct",
        output_dir=Path("logs/arena"),
    )
    from fortuna.compute.scheduler import get_scheduler

    _step("Compute backend")
    print(f"  {get_scheduler().status_line()}", flush=True)
    _step("Arena config")
    print(f"  symbol={config.symbol} timeframe={config.timeframe} days={config.days}", flush=True)
    print(
        f"  window_bars={config.window_bars} max_candidates={config.max_candidates_per_strategy}",
        flush=True,
    )
    print(
        f"  max_workers={config.max_workers} chunk_size={config.chunk_size} "
        f"time_budget={config.time_budget_sec}s",
        flush=True,
    )

    _step("Build StrategyArena")
    with timed_step("arena.init"):
        arena = StrategyArena(config)

    _step("Run arena (param search per strategy — watch [TIMING] lines)")
    with timed_step("run_arena_quick.total"):
        board = arena.run()
    winner = board.overall_winner()
    if not winner:
        print("No champion (all strategies failed).")
        return 1

    print("\n--- Leaderboard (best per strategy) ---")
    board.print_summary(top_n=10)

    _, champ_strategy = _champion_strategy(board)
    if champ_strategy is None:
        print(f"\nCHAMPION: {winner.strategy_name} (signal lookup skipped — see logs/arena)")
        return 0

    with timed_step("run_arena_quick.champion_signal") as d:
        ohlcv = arena._mdm.get_ohlcv(config.symbol, config.timeframe, days=config.days)
        d["bars"] = str(len(ohlcv))
        window = slice_for_backtest(
            ohlcv,
            window_bars=config.window_bars,
            warmup_bars=config.warmup_bars,
            strategy=champ_strategy,
        )
        cache = IndicatorCache()
        compiler = StrategyCompiler()
        enriched = cache.enrich(window, champ_strategy.indicators)
        bar_idx = len(enriched) - 1
        entry, exit_sig = signal_at_bar(champ_strategy, enriched, compiler, bar_idx)
        signal = emit_signal(champ_strategy, entry, exit_sig)

    print(f"\nCHAMPION: {winner.strategy_name}")
    print(f"  Profit: {winner.profit_pct:.2f}% | Win rate: {winner.win_ratio_pct:.1f}% | Trades: {winner.total_trades}")
    print(f"  Latest bar ({window.index[-1]}): **{signal}**")
    if signal == "BUY":
        print("  Suggestion: enter / add long (per strategy rules on last bar).")
    elif signal == "SELL":
        print("  Suggestion: exit long / take profit (per strategy rules on last bar).")
    else:
        print("  Suggestion: no new entry or exit on last bar (HOLD).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
