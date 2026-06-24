#!/usr/bin/env python
"""Build a typed training-research plan and optionally refresh selected data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.training_research import (  # noqa: E402
    build_training_research_plan,
    export_training_research_plan,
)
from fortuna.config.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Fortuna training research plan")
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
    parser.add_argument("--ml-top-n", type=int, default=5)
    parser.add_argument("--rl-top-n", type=int, default=3)
    parser.add_argument(
        "--selection-policy",
        default="diversified",
        choices=("ranked", "diversified"),
    )
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument(
        "--refresh-target",
        default="all",
        choices=("all", "ml", "rl"),
    )
    parser.add_argument("--refresh-timeframe", default="")
    parser.add_argument("--refresh-days", type=int, default=0)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--out",
        default="reports/training_research_plan.json",
        help="Output plan path",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    response = build_training_research_plan(
        settings=settings,
        universe_limit=args.universe_limit,
        analysis_limit=args.analysis_limit,
        timeframe=args.timeframe,
        days=args.days,
        source=args.source,
        ml_top_n=args.ml_top_n,
        rl_top_n=args.rl_top_n,
        selection_policy=args.selection_policy,
        refresh_data=args.refresh_data,
        refresh_target=args.refresh_target,
        refresh_timeframe=args.refresh_timeframe or None,
        refresh_days=(args.refresh_days or None),
        force_refresh=args.force_refresh,
    )
    if not response.ok:
        message = response.error.message if response.error is not None else "unknown error"
        print(f"[fail] {message}")
        return 1

    out_path = settings.resolve_path(Path(args.out))
    export_training_research_plan(response, out_path)
    print(
        f"[ok] source={response.source} timeframe={response.timeframe} "
        f"days={response.lookback_days} rows={len(response.rows)} "
        f"policy={response.selection_policy}"
    )
    if response.nightly_alignment_summary:
        print(f"[nightly] {response.nightly_alignment_summary}")
    if response.nightly_alignment_target:
        print(
            "[nightly-target] "
            f"{response.nightly_alignment_target} "
            f"force_refresh={1 if response.nightly_alignment_force_refresh else 0}"
        )
    if response.discovery_recommended_refresh_target:
        print(f"[discovery-target] {response.discovery_recommended_refresh_target}")
    print(f"[ml] {', '.join(response.ml_symbols) if response.ml_symbols else 'none'}")
    print(f"[rl] {', '.join(response.rl_symbols) if response.rl_symbols else 'none'}")
    if response.refresh_requested:
        refreshed = [row.symbol for row in response.rows if row.refreshed]
        print(
            f"[refresh] target={response.refresh_target} "
            f"count={len(refreshed)} symbols={', '.join(refreshed) if refreshed else 'none'}"
        )
    print(f"[write] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
