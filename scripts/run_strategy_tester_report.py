#!/usr/bin/env python
"""Generate TradingView-style strategy tester reports for all strategies."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FORTUNA_CONFIG", "configs/intraday.yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.config.settings import load_settings
from fortuna.data.manager import MarketDataManager
from fortuna.reporting.strategy_tester import (
    CostConfig,
    compare_reports,
    export_comparison,
    export_strategy_chart_bundle,
)
from fortuna.strategy.loader import load_strategy
from fortuna.strategies.builtin.dispatch import is_builtin_strategy, run_builtin_backtest


def _strategy_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for rel in [Path("strategies/intraday"), Path("strategies/generated"), Path("strategies/builtin")]:
        if not rel.is_dir():
            continue
        for p in sorted(rel.glob("*.json")):
            if p.name not in seen:
                seen.add(p.name)
                paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="TradingView-style strategy tester reports")
    _cfg = load_settings()
    parser.add_argument("--symbol", default=_cfg.default_symbol)
    parser.add_argument("--timeframe", default=_cfg.default_timeframe)
    parser.add_argument("--days", type=int, default=_cfg.default_days)
    parser.add_argument("--init-cash", type=float, default=100_000.0)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--output", default="logs/strategy_tester")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    costs = CostConfig(commission_rate=args.commission, slippage_rate=args.slippage)
    mdm = MarketDataManager(settings, data_source=settings.data_source)

    print(f"Loading {args.symbol} {args.timeframe} ({args.days} days)...", flush=True)
    ohlcv = mdm.get_ohlcv(args.symbol, args.timeframe, days=args.days)
    print(f"  Bars: {len(ohlcv)}", flush=True)

    out_root = Path(args.output) / f"{args.symbol}_{args.timeframe}"
    reports = []
    runner = NumPyBacktestRunner(settings)

    for path in _strategy_paths():
        strategy = load_strategy(path)
        print(f"\n--- {path.stem} ---", flush=True)

        if is_builtin_strategy(strategy):
            bt = run_builtin_backtest(strategy, ohlcv, symbol=args.symbol, init_cash=args.init_cash)
        else:
            bt = runner.run(strategy, ohlcv, symbol=args.symbol)

        _, report = export_strategy_chart_bundle(
            strategy,
            ohlcv,
            bt,
            out_root / path.stem,
            strategy_name=path.stem,
            symbol=args.symbol,
            timeframe=args.timeframe,
            init_cash=args.init_cash,
            costs=costs,
            generate_charts=not args.no_charts,
        )
        report.print_summary()
        reports.append(report)

    if reports:
        csv_p, json_p = export_comparison(reports, out_root)
        print(f"\nComparison: {csv_p}", flush=True)
        print(compare_reports(reports).to_string(index=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
