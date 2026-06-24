#!/usr/bin/env python
"""Build a constrained portfolio allocation from the current shortlist briefing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.portfolio_allocator import build_portfolio_allocation
from fortuna.config.settings import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Allocate shortlisted setups under portfolio limits"
    )
    parser.add_argument("--universe-limit", type=int, default=10)
    parser.add_argument("--analysis-limit", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--max-per-exposure", type=int, default=1)
    parser.add_argument("--max-same-side", type=int, default=2)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--source", default="auto", choices=("auto", "screener", "registry"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    response = build_portfolio_allocation(
        settings=get_settings(),
        universe_limit=args.universe_limit,
        analysis_limit=args.analysis_limit,
        max_positions=args.max_positions,
        max_per_exposure=args.max_per_exposure,
        max_same_side=args.max_same_side,
        timeframe=args.timeframe,
        days=args.days,
        source=args.source,
    )
    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print(response.headline)
        for note in response.notes:
            print(f"- {note}")
        for item in response.items:
            weight = (
                f" weight={item.allocation_weight:.3f}"
                if item.allocation_weight is not None
                else ""
            )
            state = "selected" if item.selected else "skipped"
            print(
                f"{state}: {item.symbol} {item.action} verdict={item.verdict} "
                f"rank={item.selection_rank} reason={item.reason}{weight}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
