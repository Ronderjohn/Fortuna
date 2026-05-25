#!/usr/bin/env python
"""Benchmark sequential vs parallel strategy batch on one OHLCV frame."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FORTUNA_CONFIG", "configs/intraday.yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

from fortuna.app import AppConfig, ParallelStrategyRunner, list_strategy_paths
from fortuna.config.settings import load_settings
from fortuna.data.manager import MarketDataManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark strategy batch runner")
    cfg = load_settings()
    parser.add_argument("--symbol", default=cfg.default_symbol)
    parser.add_argument("--timeframe", default=cfg.default_timeframe)
    parser.add_argument("--days", type=int, default=cfg.default_days)
    parser.add_argument("--min-speedup", type=float, default=1.2)
    args = parser.parse_args()

    settings = load_settings()
    settings = settings.model_copy(update={"data_source": "smartapi"})
    mdm = MarketDataManager(settings, data_source="smartapi")
    ohlcv = mdm.get_ohlcv(args.symbol, args.timeframe, days=args.days)
    paths = list_strategy_paths(settings)
    print(f"Symbol={args.symbol} bars={len(ohlcv)} strategies={len(paths)}", flush=True)

    runner = ParallelStrategyRunner(settings, AppConfig.from_settings(settings))
    seq = runner.run_sequential(
        ohlcv, paths, symbol=args.symbol, timeframe=args.timeframe
    )
    par = runner.run_parallel(
        ohlcv, paths, symbol=args.symbol, timeframe=args.timeframe
    )
    speedup = seq.elapsed_ms / par.elapsed_ms if par.elapsed_ms > 0 else 0.0
    print(f"Sequential: {seq.elapsed_ms:.0f} ms", flush=True)
    print(f"Parallel:   {par.elapsed_ms:.0f} ms", flush=True)
    print(f"Speedup:    {speedup:.2f}x", flush=True)
    if speedup < args.min_speedup and len(paths) >= 3:
        print(f"FAIL: speedup {speedup:.2f} < {args.min_speedup}", flush=True)
        return 1
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
