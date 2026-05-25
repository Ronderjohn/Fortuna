"""Institutional backtesting standard tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fortuna.backtesting.standard.calendar import filter_session_bars
from fortuna.backtesting.standard.config import FilterThresholds, SessionRules, SplitRatios
from fortuna.backtesting.standard.filters import evaluate_strategy
from fortuna.reporting.strategy_tester.metrics import PerformanceMetrics, RiskMetrics
from fortuna.backtesting.standard.robustness import monte_carlo_trade_shuffle
from fortuna.backtesting.standard.splits import chronological_split
from fortuna.reporting.strategy_tester.trade import TradeRecord, TradeSide


def _ohlcv(n: int = 1000) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02 09:15", periods=n * 2, freq="5min")
    idx = idx[(idx.hour >= 9) & (idx.hour <= 15)][:n]
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
        },
        index=idx,
    )


def test_chronological_split_ratios() -> None:
    ohlcv = _ohlcv(1000)
    split = chronological_split(ohlcv, SplitRatios())
    n = len(ohlcv)
    assert len(split.train) == int(n * 0.70)
    assert len(split.validation) == int(n * 0.15)
    assert len(split.test) == n - len(split.train) - len(split.validation)


def test_session_filter() -> None:
    idx = pd.date_range("2025-01-01 08:00", periods=20, freq="5min")
    df = pd.DataFrame(
        {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        index=idx,
    )
    out = filter_session_bars(df, SessionRules())
    assert out.index.min().hour >= 9


def test_filter_rejects_low_pf() -> None:
    from fortuna.backtesting.standard.config import BenchmarkTargets

    perf = PerformanceMetrics(
        total_net_profit=-1000,
        gross_profit=500,
        gross_loss=1500,
        profit_factor=0.33,
        total_trades=120,
        winning_trades=40,
        losing_trades=80,
        win_rate_pct=33.3,
    )
    risk = RiskMetrics(max_drawdown_pct=0.2, sharpe_ratio=0.3, expectancy=-8)
    v = evaluate_strategy(
        perf, risk, thresholds=FilterThresholds(), benchmarks=BenchmarkTargets(), period_days=200
    )
    assert not v.passed
    assert any("Profit factor" in r for r in v.reasons)


def test_monte_carlo() -> None:
    trades = [
        TradeRecord(
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-01-01 10:00"),
            100,
            101,
            TradeSide.LONG,
            10,
            100,
            1,
            pd.Timedelta(hours=1),
        )
    ]
    mc = monte_carlo_trade_shuffle(trades, 100_000, iterations=100)
    assert mc.iterations == 100
    assert mc.median_final_equity > 0
