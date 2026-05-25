#!/usr/bin/env python
"""
Run institutional-standard backtests on all strategies.

- 6+ months 5m data
- Train 70% / Validation 15% / OOS Test 15%
- NSE session + costs + filters + Monte Carlo
- TradingView-style reports per split
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FORTUNA_CONFIG", "configs/intraday.yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

from fortuna.config.settings import load_settings
from fortuna.backtesting.standard import InstitutionalBacktestConfig, InstitutionalPipeline
from fortuna.backtesting.standard.config import (
    ExecutionTiming,
    FilterThresholds,
    MarketCostModel,
    SessionRules,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Institutional 5m backtesting standard")
    cfg = load_settings()
    parser.add_argument("--symbol", default=cfg.default_symbol)
    parser.add_argument("--timeframe", default=cfg.default_timeframe)
    parser.add_argument("--days", type=int, default=cfg.default_days, help="History length to load")
    parser.add_argument(
        "--min-days",
        type=int,
        default=max(20, cfg.default_days - 5),
        help="Minimum calendar days required",
    )
    parser.add_argument("--init-cash", type=float, default=100_000.0)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--brokerage", type=float, default=0.0003)
    parser.add_argument(
        "--execution",
        choices=["on_close", "next_open"],
        default="on_close",
    )
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--output", default="logs/institutional")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print("INSTITUTIONAL BACKTESTING STANDARD", flush=True)
    print(f"  Symbol: {args.symbol}  Timeframe: {args.timeframe}  Days: {args.days}", flush=True)
    print(f"  Split: 70% train | 15% validation | 15% OOS test", flush=True)
    print(f"  Capital: {args.init_cash:,.0f}  Execution: {args.execution}", flush=True)
    print("=" * 72, flush=True)

    config = InstitutionalBacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        min_history_days=args.min_days,
        preferred_history_days=args.days,
        initial_capital=args.init_cash,
        costs=MarketCostModel(
            brokerage_rate=args.brokerage,
            slippage_rate=args.slippage,
        ),
        session=SessionRules(),
        execution=ExecutionTiming(args.execution),
        filters=FilterThresholds(min_trades=args.min_trades),
        output_dir=Path(args.output),
    )

    run_dir = InstitutionalPipeline(config).run()
    print(f"\nComplete. Results: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
