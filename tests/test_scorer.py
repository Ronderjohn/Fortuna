"""Strategy scorer tests."""

from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.evaluation.scorer import StrategyScorer


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
