#!/usr/bin/env python
"""Backfill OHLCV cache from Angel One SmartAPI getCandleData."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FORTUNA_CONFIG", "configs/intraday.yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import time

from fortuna.config.settings import load_settings
from fortuna.data.manager import MarketDataManager
from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults
from fortuna.utils.timing import timed_step


def _step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartAPI historical backfill to Parquet cache")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Refresh cache even if fresh")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("SmartAPI backfill", flush=True)
    print(f"  PID={os.getpid()}  CWD={Path.cwd()}", flush=True)
    print("=" * 60, flush=True)

    _step("Apply GPU/low-memory defaults")
    apply_low_spec_gpu_defaults()
    print(f"  FORTUNA_USE_GPU={os.environ.get('FORTUNA_USE_GPU')}", flush=True)
    print(f"  FORTUNA_LOW_MEMORY={os.environ.get('FORTUNA_LOW_MEMORY')}", flush=True)

    _step("Load settings (FORTUNA_CONFIG / intraday.yaml)")
    settings = load_settings()
    print(f"  data_source -> smartapi (was {settings.data_source})", flush=True)
    settings = settings.model_copy(update={"data_source": "smartapi"})

    _step("Create MarketDataManager")
    with timed_step("backfill.init_manager"):
        mdm = MarketDataManager(settings, data_source="smartapi")

    _step(
        f"Fetch + cache {args.symbol} {args.timeframe} ({args.days} days, force={args.force})"
    )
    with timed_step("backfill.get_ohlcv") as d:
        df = mdm.get_ohlcv(
            args.symbol,
            args.timeframe,
            days=args.days,
            force_refresh=args.force,
        )
        d["rows"] = str(len(df))

    _step("Result")
    print(f"  Bars: {len(df)}", flush=True)
    print(f"  Range: {df.index.min()} -> {df.index.max()}", flush=True)
    path = mdm.cache.get_path(args.symbol.upper(), args.timeframe)
    print(f"  Parquet: {path}", flush=True)
    if path and path.exists():
        print(f"  File size: {path.stat().st_size / 1024:.1f} KB", flush=True)
    print("=" * 60, flush=True)
    print("BACKFILL OK", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
