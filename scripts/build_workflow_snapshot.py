#!/usr/bin/env python
"""Build and optionally export a multi-stage Fortuna workflow snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.operator_workflow import (  # noqa: E402
    build_workflow_snapshot,
    export_workflow_snapshot,
)
from fortuna.config.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Fortuna workflow snapshot")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--universe-limit", type=int, default=10)
    parser.add_argument("--analysis-limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--source",
        default="auto",
        choices=("auto", "screener", "registry"),
    )
    parser.add_argument(
        "--out",
        default="reports/workflow_snapshot.json",
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    snapshot = build_workflow_snapshot(
        settings=settings,
        universe_limit=args.universe_limit,
        analysis_limit=args.analysis_limit,
        candidate_limit=args.candidate_limit,
        timeframe=args.timeframe,
        days=args.days,
        source=args.source,
    )
    print(
        f"[ok] source={snapshot.source} timeframe={snapshot.timeframe} "
        f"days={snapshot.lookback_days}"
    )
    print(
        f"[universe] count={snapshot.universe.metrics.get('count', 0)} "
        f"top_liquidity={snapshot.universe.metrics.get('top_liquidity_score', 0.0)}"
    )
    print(f"[shortlist] count={snapshot.shortlist.metrics.get('count', 0)}")
    print(f"[briefing] candidates={snapshot.briefing.metrics.get('candidates', 0)}")
    print(
        f"[candidates] ml={snapshot.candidates.metrics.get('ml_count', 0)} "
        f"rl={snapshot.candidates.metrics.get('rl_count', 0)}"
    )

    out_path = settings.resolve_path(Path(args.out))
    export_workflow_snapshot(snapshot, out_path)
    print(f"[write] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
