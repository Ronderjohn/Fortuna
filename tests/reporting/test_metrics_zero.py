"""Phase 1 completion 2.2: ``StrategyMetrics.zero()`` + empty-trade safety."""

from __future__ import annotations

import pandas as pd

from fortuna.backtesting.standard.config import BenchmarkTargets, FilterThresholds
from fortuna.backtesting.standard.filters import FilterVerdict, evaluate_strategy
from fortuna.reporting.strategy_tester.metrics import (
    StrategyMetrics,
    compute_all_metrics,
)


def test_zero_returns_valid_strategy_metrics():
    z = StrategyMetrics.zero()
    assert isinstance(z, StrategyMetrics)
    assert z.performance.total_trades == 0
    assert z.performance.profit_factor == 0.0
    assert z.risk.sharpe_ratio == -1e9
    assert z.risk.max_drawdown_pct == 0.0


def test_zero_metrics_fail_default_filter_thresholds():
    """``StrategyMetrics.zero()`` is designed to be unambiguously rejected."""
    z = StrategyMetrics.zero()
    verdict = evaluate_strategy(
        z.performance,
        z.risk,
        thresholds=FilterThresholds(),
        benchmarks=BenchmarkTargets(),
        period_days=180,
    )
    assert verdict.passed is False
    assert verdict.reasons, "FilterVerdict must list at least one rejection reason"


def test_compute_all_metrics_empty_trades_does_not_raise():
    """Phase 2 RL early training often produces fold episodes with zero trades."""
    metrics, equity = compute_all_metrics([], initial_capital=100_000.0)
    assert metrics.performance.total_trades == 0
    assert metrics.risk.sharpe_ratio == 0.0
    assert isinstance(equity, pd.DataFrame)
    assert equity.empty or "equity" in equity.columns


def test_filter_verdict_evaluate_classmethod_routes_through_evaluate_strategy():
    metrics = StrategyMetrics.zero()
    verdict = FilterVerdict.evaluate(metrics, period_days=180)
    assert isinstance(verdict, FilterVerdict)
    assert verdict.passed is False


def test_filter_verdict_pass_fail_sentinels_compare_to_real_verdicts():
    failing = FilterVerdict(passed=False, reasons=["x"])
    passing = FilterVerdict(passed=True)
    assert failing == FilterVerdict.FAIL
    assert passing == FilterVerdict.PASS
    assert failing != FilterVerdict.PASS
