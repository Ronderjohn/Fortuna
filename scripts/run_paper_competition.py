#!/usr/bin/env python
"""Paper-trade all strategies with equal starting capital; compare wins and losses."""

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
from fortuna.paper.competition import PaperCompetition, PaperCompetitionConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Each strategy paper-trades the same history with identical starting cash. "
            "Reports profit, wins, and losses independently."
        )
    )
    cfg = load_settings()
    parser.add_argument("--symbol", default=cfg.default_symbol)
    parser.add_argument("--timeframe", default=cfg.default_timeframe)
    parser.add_argument("--days", type=int, default=cfg.default_days)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--init-cash", type=float, default=100_000.0)
    parser.add_argument("--output", default="logs/paper_competition")
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip per-strategy TradingView-style chart export under charts/",
    )
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("Paper competition", flush=True)
    print(f"  Symbol={args.symbol}  Timeframe={args.timeframe}  Days={args.days}", flush=True)
    print(f"  Init cash per strategy: {args.init_cash:,.0f}", flush=True)
    print("=" * 60, flush=True)

    config = PaperCompetitionConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        data_source=args.data_source,
        init_cash=args.init_cash,
        output_dir=Path(args.output),
        export_charts=not args.no_charts,
    )
    result = PaperCompetition(config).run()
    print(f"\nResults: {result.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
