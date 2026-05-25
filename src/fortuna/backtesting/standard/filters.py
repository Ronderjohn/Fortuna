"""Strategy acceptance / rejection rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fortuna.backtesting.standard.config import BenchmarkTargets, FilterThresholds
from fortuna.reporting.strategy_tester.metrics import PerformanceMetrics, RiskMetrics


@dataclass
class FilterVerdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0
    benchmarks_met: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "reasons": self.reasons,
            "score": round(self.score, 2),
            "benchmarks_met": {k: bool(v) for k, v in self.benchmarks_met.items()},
        }


def evaluate_strategy(
    perf: PerformanceMetrics,
    risk: RiskMetrics,
    *,
    thresholds: FilterThresholds,
    benchmarks: BenchmarkTargets,
    period_days: int,
) -> FilterVerdict:
    reasons: list[str] = []

    if perf.total_trades < thresholds.min_trades:
        if period_days >= 120:
            reasons.append(
                f"Too few trades: {perf.total_trades} < {thresholds.min_trades} "
                f"over {period_days}d"
            )

    if perf.profit_factor < thresholds.min_profit_factor:
        reasons.append(f"Profit factor {perf.profit_factor:.2f} < {thresholds.min_profit_factor}")

    if risk.max_drawdown_pct * 100 > thresholds.max_drawdown_pct:
        reasons.append(
            f"Max drawdown {risk.max_drawdown_pct * 100:.1f}% > {thresholds.max_drawdown_pct}%"
        )

    if risk.sharpe_ratio < thresholds.min_sharpe:
        reasons.append(f"Sharpe {risk.sharpe_ratio:.2f} < {thresholds.min_sharpe}")

    if risk.expectancy < thresholds.min_expectancy:
        reasons.append(f"Expectancy {risk.expectancy:.2f} not positive")

    bm = {
        "profit_factor_1.5": perf.profit_factor >= benchmarks.profit_factor,
        "sharpe_1.2": risk.sharpe_ratio >= benchmarks.sharpe,
        "max_dd_15pct": risk.max_drawdown_pct * 100 <= benchmarks.max_drawdown_pct,
        "risk_reward_1.5": risk.risk_reward_ratio >= benchmarks.risk_reward,
        "positive_expectancy": risk.expectancy > 0,
    }

    composite = _composite_rank_score(perf, risk)
    passed = len(reasons) == 0
    return FilterVerdict(passed=passed, reasons=reasons, score=composite, benchmarks_met=bm)


def _composite_rank_score(perf: PerformanceMetrics, risk: RiskMetrics) -> float:
    """
    Rank by risk-adjusted quality, NOT raw profit alone.

    Weights: Sharpe, PF, drawdown penalty, expectancy, win rate stability proxy.
    """
    pf = min(perf.profit_factor, 5.0)
    sharpe = max(min(risk.sharpe_ratio, 5.0), -5.0)
    dd_penalty = max(0.0, 1.0 - risk.max_drawdown_pct * 4)
    exp_norm = max(min(risk.expectancy / 500.0, 1.0), -1.0)
    trade_score = min(perf.total_trades / 200.0, 1.0)
    return (
        sharpe * 30.0
        + pf * 20.0
        + dd_penalty * 25.0
        + exp_norm * 15.0
        + trade_score * 10.0
    )
