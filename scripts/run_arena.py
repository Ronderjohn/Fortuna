#!/usr/bin/env python
"""Run multi-strategy arena: parallel param search + TradingView-style leaderboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

from fortuna.arena import ArenaConfig, StrategyArena


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compete strategies with parameter search on streamed OHLCV (DuckDB cache)"
    )
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--strategies", default="strategies/generated")
    parser.add_argument(
        "--param-grid",
        default="strategies/grids/ema_grid.json",
        help="Optional JSON grid; keys that do not match a strategy are skipped",
    )
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=4, help="Min candidates per parallel batch")
    parser.add_argument("--window-bars", type=int, default=78)
    parser.add_argument("--warmup-bars", type=int, default=78)
    parser.add_argument("--stream-step", type=int, default=5, help="Bar step for streaming mode")
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument(
        "--rank-by",
        choices=["profit_pct", "win_ratio_pct", "composite_score"],
        default="profit_pct",
    )
    parser.add_argument(
        "--mode",
        choices=["stream", "full"],
        default="full",
        help="stream=rolling windows; full=single window param search per strategy",
    )
    parser.add_argument("--output", default="logs/arena")
    args = parser.parse_args()

    grid_path = Path(args.param_grid) if args.param_grid else None
    config = ArenaConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        data_source=args.data_source,
        strategy_dir=Path(args.strategies),
        param_grid_path=grid_path,
        max_candidates_per_strategy=args.max_candidates,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        time_budget_sec=args.time_budget,
        window_bars=args.window_bars,
        warmup_bars=args.warmup_bars,
        stream_step=args.stream_step,
        use_streaming=args.mode == "stream",
        rank_by=args.rank_by,
        output_dir=Path(args.output),
        eval_full_series=True,
    )

    StrategyArena(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
