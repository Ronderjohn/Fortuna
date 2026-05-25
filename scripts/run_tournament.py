#!/usr/bin/env python
"""Run intraday strategy tournament."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

from fortuna.config.settings import load_settings
from fortuna.tournament.config import TournamentConfig
from fortuna.tournament.runner import TournamentRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Fortuna intraday strategy tournament")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--strategies", default="strategies/generated")
    parser.add_argument("--param-grid", default=None, help="JSON file with param_grid")
    parser.add_argument("--window-bars", type=int, default=78)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--time-budget", type=float, default=30.0)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--bar-step",
        type=int,
        default=1,
        help="Evaluate every Nth bar (e.g. 10 on 90d 5m data)",
    )
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--tier", type=int, default=1)
    args = parser.parse_args()

    settings = load_settings()
    paths: list[Path] = []
    seen: set[str] = set()
    for rel in [Path(args.strategies), Path("strategies/builtin")]:
        strat_dir = settings.resolve_path(rel)
        if strat_dir.is_dir():
            for p in sorted(strat_dir.glob("*.json")):
                if p.name not in seen:
                    seen.add(p.name)
                    paths.append(p)

    cfg = TournamentConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        window_bars=args.window_bars,
        warmup_bars=args.window_bars,
        time_budget_sec=args.time_budget,
        max_workers=args.max_workers,
        max_candidates=args.max_candidates,
        tier=args.tier,
        parallel=args.parallel,
        data_source=args.data_source,
        days=args.days,
        bar_step=args.bar_step,
        strategy_paths=paths,
        param_grid_path=Path(args.param_grid) if args.param_grid else None,
        output_dir=settings.resolve_path(Path("logs/tournament")),
    )

    results = TournamentRunner(cfg, settings).run()
    print(f"Tournament complete: {len(results)} bars")
    if results:
        last = results[-1]
        print(f"Last signal: {last.signal} winner={last.winner_strategy} score={last.score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
