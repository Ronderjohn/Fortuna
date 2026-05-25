#!/usr/bin/env python
"""Walk-forward paper trading league with adaptive parameter learning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

from fortuna.paper import PaperLeague, PaperLeagueConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-trade strategies in walk-forward folds (black-box OOS). "
            "Train = learn params; test = compete on unseen bars; repeat."
        )
    )
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--strategies", default="strategies/generated")
    parser.add_argument("--param-grid", default="strategies/grids/ema_grid.json")
    parser.add_argument("--train-bars", type=int, default=156)
    parser.add_argument("--test-bars", type=int, default=78)
    parser.add_argument("--fold-step", type=int, default=78)
    parser.add_argument("--min-folds", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=36)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--time-budget", type=float, default=45.0)
    parser.add_argument("--init-cash", type=float, default=100_000.0)
    parser.add_argument(
        "--rank-by",
        choices=["profit_pct", "win_ratio_pct", "composite_score"],
        default="profit_pct",
    )
    parser.add_argument("--no-learning", action="store_true", help="Disable adaptive grid refinement")
    parser.add_argument("--output", default="logs/paper_league")
    args = parser.parse_args()

    grid = Path(args.param_grid) if args.param_grid else None
    config = PaperLeagueConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        data_source=args.data_source,
        strategy_dir=Path(args.strategies),
        param_grid_path=grid,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        fold_step_bars=args.fold_step,
        min_folds=args.min_folds,
        max_candidates_per_strategy=args.max_candidates,
        max_workers=args.max_workers,
        time_budget_sec=args.time_budget,
        init_cash=args.init_cash,
        rank_by=args.rank_by,
        enable_learning=not args.no_learning,
        output_dir=Path(args.output),
    )

    result = PaperLeague(config).run()
    print(f"Results: {result.run_dir}")
    if result.champion:
        print(
            f"Champion: {result.champion.strategy_name} | "
            f"last-fold profit {result.champion.profit_pct:.2f}% | "
            f"win {result.champion.win_ratio_pct:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
