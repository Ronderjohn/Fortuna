#!/usr/bin/env python
"""Render a concise briefing for the top shortlisted market setups."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.shortlist_briefing import build_shortlist_briefing  # noqa: E402
from fortuna.config.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Brief top shortlisted Fortuna setups")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--universe-limit", type=int, default=10)
    parser.add_argument("--analysis-limit", type=int, default=5)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--source",
        default="auto",
        choices=("auto", "screener", "registry"),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    response = build_shortlist_briefing(
        settings=settings,
        universe_limit=args.universe_limit,
        analysis_limit=args.analysis_limit,
        timeframe=args.timeframe,
        days=args.days,
        source=args.source,
    )
    if not response.ok:
        message = response.error.message if response.error is not None else "briefing failed"
        print(f"[fail] {message}")
        return 1
    print(f"[ok] {response.headline}")
    for idx, item in enumerate(response.items, start=1):
        print(
            f"{idx}. {item.symbol} | {item.action} | {item.confidence * 100:.0f}% | "
            f"{item.verdict} | {item.summary}"
        )
    if response.portfolio is not None:
        print(f"[portfolio] {response.portfolio.summary}")
        for note in response.portfolio.notes:
            print(f"  - {note.level}: {note.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
