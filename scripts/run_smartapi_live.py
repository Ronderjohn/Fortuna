#!/usr/bin/env python
"""Stream SmartAPI WebSocket 2.0 snap quotes into OHLCV cache (5m bars by default)."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("SMARTAPI_USE_LIVE_FEED", "true")
os.environ.setdefault("FORTUNA_CONFIG", "configs/intraday.yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.config.settings import load_settings
from fortuna.data.manager import MarketDataManager
from fortuna.data.sources.smartapi_live import SmartAPILiveBarFeed
from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartAPI WebSocket 2.0 live bar feed")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--timeframe", default="5m")
    args = parser.parse_args()

    apply_low_spec_gpu_defaults()
    settings = load_settings()
    settings = settings.model_copy(update={"data_source": "smartapi"})
    mdm = MarketDataManager(settings, data_source="smartapi")

    print("Warming cache with historical backfill (if needed)...", flush=True)
    mdm.get_ohlcv(args.symbol, args.timeframe, days=7)

    feed = SmartAPILiveBarFeed(mdm, args.symbol, args.timeframe)
    stop = False

    def _handle_sig(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sig)

    print(
        f"Starting WebSocket 2.0 snap quote feed for {args.symbol} ({args.timeframe})...",
        flush=True,
    )
    print("Press Ctrl+C to stop.", flush=True)
    feed.start(background=True)

    try:
        while not stop:
            time.sleep(1)
    finally:
        feed.stop()
        print("Feed stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
