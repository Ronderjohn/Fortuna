"""Strategy scorer tests."""

from pathlib import Path

import pandas as pd

from fortuna.backtesting.engine import BacktestResult
from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.evaluation.ranker import RankedStrategy, StrategyRanker
from fortuna.evaluation.scorer import StrategyScore, StrategyScorer


def test_better_sharpe_scores_higher() -> None:
    scorer = StrategyScorer()
    good = BacktestMetrics(
        total_return=0.25,
        sharpe_ratio=1.5,
        max_drawdown=0.10,
        win_rate=0.55,
        expectancy=100.0,
        profit_factor=1.8,
        total_trades=50,
        final_value=125_000.0,
    )
    bad = BacktestMetrics(
        total_return=-0.10,
        sharpe_ratio=-0.5,
        max_drawdown=0.40,
        win_rate=0.40,
        expectancy=-50.0,
        profit_factor=0.7,
        total_trades=50,
        final_value=90_000.0,
    )
    assert scorer.score(good).composite > scorer.score(bad).composite


def test_low_trades_fails() -> None:
    scorer = StrategyScorer()
    m = BacktestMetrics(
        total_return=0.5,
        sharpe_ratio=2.0,
        max_drawdown=0.05,
        win_rate=0.9,
        expectancy=200.0,
        profit_factor=3.0,
        total_trades=3,
        final_value=150_000.0,
    )
    result = scorer.score(m)
    assert result.passed is False


def _ranked(
    name: str,
    *,
    total_trades: int,
    composite: float,
    total_return: float = 0.0,
) -> RankedStrategy:
    metrics = BacktestMetrics(
        total_return=total_return,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        expectancy=0.0,
        profit_factor=0.0,
        total_trades=total_trades,
        final_value=100_000.0 * (1 + total_return),
    )
    bt = BacktestResult(
        strategy_name=name,
        symbol="TEST",
        timeframe="5m",
        metrics=metrics,
        portfolio=None,
        enriched_data=pd.DataFrame(),
    )
    score = StrategyScore(
        composite=composite,
        sharpe_component=0.0,
        return_component=0.0,
        drawdown_penalty=0.0,
        profit_factor_component=0.0,
        consistency_component=0.0,
        robustness_penalty=0.0,
        passed=False,
        notes=[],
    )
    return RankedStrategy(strategy_path=Path(f"{name}.json"), result=bt, score=score)


def test_ranker_demotes_zero_trade_strategies() -> None:
    """Regression: ``rank_many`` used to sort purely by composite score.
    A zero-trade strategy (which gets neutral scores on most components and
    only a 0.15 robustness penalty) could outrank a real losing strategy.
    The fix is to make ``total_trades > 0`` the primary sort key."""
    ranker = StrategyRanker()
    zero_trade = _ranked("idle", total_trades=0, composite=60.0)
    traded_loser = _ranked("loser", total_trades=20, composite=25.0, total_return=-0.05)
    traded_winner = _ranked("winner", total_trades=30, composite=70.0, total_return=0.10)

    ordered = ranker.rank_many([zero_trade, traded_loser, traded_winner])
    names = [r.strategy_path.stem for r in ordered]
    assert names == ["winner", "loser", "idle"], (
        f"Zero-trade strategy must rank last; got order {names}"
    )
