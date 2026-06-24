#!/usr/bin/env python
"""Build shortlist-derived ML/RL training candidates and optionally backfill them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.training_candidates import (  # noqa: E402
    backfill_training_candidates,
    build_training_candidates,
    export_training_candidates,
)
from fortuna.config.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Fortuna ML/RL training candidates")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--universe-limit", type=int, default=15)
    parser.add_argument("--analysis-limit", type=int, default=8)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--source",
        default="auto",
        choices=("auto", "screener", "registry"),
    )
    parser.add_argument(
        "--out",
        default="reports/training_candidates.json",
        help="Output manifest path",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Refresh OHLCV for selected symbols",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Force data cache refresh")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    response = build_training_candidates(
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

    out_path = settings.resolve_path(Path(args.out))
    export_training_candidates(response, out_path)
    print(
        f"[ok] source={response.source} timeframe={response.timeframe} "
        f"days={response.lookback_days} candidates={len(response.candidates)}"
    )
    for row in response.candidates:
        print(
            f"- {row.symbol} | rank={row.shortlist_rank} | "
            f"ml={row.ml_candidate} | rl={row.rl_candidate} | verdict={row.critique_verdict}"
        )
    print(f"[write] {out_path}")

    if args.backfill:
        refreshed = backfill_training_candidates(
            settings,
            response,
            timeframe=args.timeframe,
            days=args.days,
            force_refresh=args.force_refresh,
        )
        print(f"[backfill] refreshed {len(refreshed)} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
