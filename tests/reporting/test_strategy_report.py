"""TradingView-style report tests."""

import pytest

from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.evaluation.scorer import StrategyScorer
from fortuna.reporting.strategy_report import build_report, reports_to_dataframe


def test_build_report_profit_and_win_ratio() -> None:
    metrics = BacktestMetrics(
        total_return=0.12,
        sharpe_ratio=1.1,
        max_drawdown=0.08,
        win_rate=0.55,
        expectancy=0.01,
        profit_factor=1.8,
        total_trades=20,
        final_value=112_000.0,
        gross_profit=0.15,
        gross_loss=0.03,
        avg_trade_return=0.006,
    )
    score = StrategyScorer().score(metrics)
    report = build_report(
        strategy_name="ema_test",
        candidate_id="c0",
        symbol="RELIANCE.NS",
        timeframe="5m",
        metrics=metrics,
        score=score,
        params_summary="ema_fast=12,ema_slow=26",
    )
    assert report.profit_pct == pytest.approx(12.0)
    assert report.win_ratio_pct == pytest.approx(55.0)
    assert report.total_trades == 20
    assert report.winning_trades == 11


def test_reports_to_dataframe_sortable() -> None:
    a = build_report(
        strategy_name="a",
        candidate_id="1",
        symbol="X",
        timeframe="5m",
        metrics=BacktestMetrics(0.1, 0, 0.05, 0.5, 0, 1, 5, 110),
    )
    b = build_report(
        strategy_name="b",
        candidate_id="2",
        symbol="X",
        timeframe="5m",
        metrics=BacktestMetrics(0.2, 0, 0.05, 0.6, 0, 1, 8, 120),
    )
    df = reports_to_dataframe([a, b])
    assert len(df) == 2
    assert "profit_pct" in df.columns
