"""CLI for running backtests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Fortuna strategy backtest")
    parser.add_argument("--strategy", required=True, help="Path to strategy JSON")
    parser.add_argument("--symbol", help="Override symbol")
    parser.add_argument("--timeframe", help="Override timeframe")
    parser.add_argument("--refresh", action="store_true", help="Force data refresh")
    parser.add_argument("--no-persist", action="store_true", help="Skip validated/rejected copy")
    parser.add_argument(
        "--full-metrics",
        action="store_true",
        help="Use full vectorbt stats() (slower)",
    )
    args = parser.parse_args(argv)

    from fortuna.agents.stubs import RuleBasedCriticAgent
    from fortuna.backtesting.engine import BacktestEngine
    from fortuna.config.settings import load_settings
    from fortuna.data.manager import MarketDataManager
    from fortuna.evaluation.ranker import StrategyRanker
    from fortuna.strategy.loader import load_strategy
    from fortuna.utils.logging import get_logger, setup_logging

    logger = get_logger(__name__)

    settings = load_settings()
    setup_logging(settings.log_level, settings.resolve_path(settings.logs_dir))

    strategy_path = Path(args.strategy)
    if not strategy_path.is_absolute():
        strategy_path = settings.resolve_path(strategy_path)

    strategy = load_strategy(strategy_path)
    symbol = (args.symbol or strategy.symbol).upper()
    timeframe = args.timeframe or strategy.timeframe

    logger.info("Loading data for %s %s", symbol, timeframe)
    mdm = MarketDataManager(settings)
    df = mdm.get_ohlcv(symbol, timeframe, force_refresh=args.refresh)

    engine = BacktestEngine(settings)
    result = engine.run(
        strategy,
        df,
        symbol=symbol,
        fast_metrics=not args.full_metrics,
    )

    ranker = StrategyRanker(settings)
    ranked = ranker.rank_one(strategy_path, result)

    critic = RuleBasedCriticAgent()
    critique = critic.critique(strategy, result.metrics)

    output = {
        "strategy": strategy.name,
        "symbol": symbol,
        "timeframe": timeframe,
        "metrics": result.metrics.to_dict(),
        "score": {
            "composite": ranked.score.composite,
            "passed": ranked.score.passed,
            "notes": ranked.score.notes,
        },
        "critic": {
            "approved": critique.approved,
            "issues": critique.issues,
            "suggestions": critique.suggestions,
        },
    }
    print(json.dumps(output, indent=2))

    if not args.no_persist:
        dest = ranker.persist(ranked)
        logger.info("Persisted to %s", dest)

    return 0 if ranked.score.passed else 1


if __name__ == "__main__":
    sys.exit(main())
