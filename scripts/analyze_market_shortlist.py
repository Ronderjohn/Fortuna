#!/usr/bin/env python
"""Analyze the top liquid market-universe candidates through Fortuna's advisory stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.shortlist_analysis import analyze_market_shortlist  # noqa: E402
from fortuna.config.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a ranked Fortuna market shortlist")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--universe-limit", type=int, default=10, help="Universe rank count")
    parser.add_argument("--analysis-limit", type=int, default=5, help="How many symbols to analyze")
    parser.add_argument("--timeframe", default="5m", help="Analysis timeframe")
    parser.add_argument("--days", type=int, default=30, help="Lookback days for analysis")
    parser.add_argument(
        "--source",
        default="auto",
        choices=("auto", "screener", "registry"),
        help="Universe seed source",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    response = analyze_market_shortlist(
        settings=settings,
        universe_limit=args.universe_limit,
        analysis_limit=args.analysis_limit,
        timeframe=args.timeframe,
        days=args.days,
        source=args.source,
    )
    if not response.ok:
        message = response.error.message if response.error is not None else "unknown error"
        print(f"[fail] {message}")
        return 1

    print(
        f"[ok] source={response.source} timeframe={response.timeframe} "
        f"days={response.lookback_days} rows={len(response.items)}"
    )
    for idx, item in enumerate(response.items, start=1):
        action = item.decision.action if item.decision is not None else "NA"
        confidence = (
            f"{item.decision.confidence * 100:.0f}%"
            if item.decision is not None
            else "NA"
        )
        critique = item.critique.summary if item.critique is not None else "NA"
        print(
            f"{idx}. {item.symbol} | action={action} | confidence={confidence} | "
            f"liquidity={item.liquidity_score} | critique={critique}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
