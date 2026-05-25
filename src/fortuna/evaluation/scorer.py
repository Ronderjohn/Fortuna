"""Composite strategy scoring with anti-overfit penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.config.settings import Settings, get_settings
from fortuna.strategy.schema import StrategyMetadata


@dataclass
class StrategyScore:
    """Detailed scoring breakdown."""

    composite: float
    sharpe_component: float
    return_component: float
    drawdown_penalty: float
    profit_factor_component: float
    consistency_component: float
    robustness_penalty: float
    passed: bool
    notes: list[str]


class StrategyScorer:
    """Score strategies on risk-adjusted metrics, not win rate alone."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.weights = self.settings.scoring_weights

    def score(
        self,
        metrics: BacktestMetrics,
        metadata: Optional[StrategyMetadata] = None,
        monthly_returns_std: Optional[float] = None,
    ) -> StrategyScore:
        """Compute composite score (0-100) from backtest metrics."""
        notes: list[str] = []
        w = self.weights

        sharpe_c = self._normalize_sharpe(metrics.sharpe_ratio)
        return_c = self._normalize_return(metrics.total_return)
        dd_penalty = self._drawdown_penalty(metrics.max_drawdown)
        pf_c = self._normalize_profit_factor(metrics.profit_factor)
        consistency_c = self._consistency_score(monthly_returns_std)
        robustness_penalty = self._robustness_penalty(metrics, metadata, notes)

        raw = (
            w.sharpe * sharpe_c
            + w.total_return * return_c
            + w.drawdown_penalty * (1.0 - dd_penalty)
            + w.profit_factor * pf_c
            + w.consistency * consistency_c
        )
        composite = max(0.0, min(100.0, (raw - robustness_penalty) * 100.0))

        passed = composite >= self.settings.validation_score_threshold
        if metrics.total_trades < self.settings.min_trades:
            notes.append(f"Low trade count: {metrics.total_trades} < {self.settings.min_trades}")
            passed = False
        if metrics.profit_factor < self.settings.min_profit_factor:
            notes.append(f"Profit factor below minimum: {metrics.profit_factor:.2f}")
            passed = False
        if metrics.max_drawdown > self.settings.max_drawdown_threshold:
            notes.append(f"Max drawdown exceeds threshold: {metrics.max_drawdown:.1%}")

        return StrategyScore(
            composite=composite,
            sharpe_component=sharpe_c,
            return_component=return_c,
            drawdown_penalty=dd_penalty,
            profit_factor_component=pf_c,
            consistency_component=consistency_c,
            robustness_penalty=robustness_penalty,
            passed=passed,
            notes=notes,
        )

    def _normalize_sharpe(self, sharpe: float) -> float:
        return max(0.0, min(1.0, (sharpe + 1.0) / 3.0))

    def _normalize_return(self, total_return: float) -> float:
        return max(0.0, min(1.0, (total_return + 0.5) / 1.5))

    def _drawdown_penalty(self, max_drawdown: float) -> float:
        threshold = self.settings.max_drawdown_threshold
        if max_drawdown <= threshold * 0.5:
            return 0.0
        if max_drawdown >= threshold * 2:
            return 1.0
        return (max_drawdown - threshold * 0.5) / (threshold * 1.5)

    def _normalize_profit_factor(self, pf: float) -> float:
        if pf <= 0:
            return 0.0
        return max(0.0, min(1.0, pf / 3.0))

    def _consistency_score(self, monthly_std: Optional[float]) -> float:
        if monthly_std is None:
            return 0.5
        return max(0.0, min(1.0, 1.0 - monthly_std * 5.0))

    def _robustness_penalty(
        self,
        metrics: BacktestMetrics,
        metadata: Optional[StrategyMetadata],
        notes: list[str],
    ) -> float:
        penalty = 0.0
        if metrics.total_trades < 5:
            penalty += 0.15
            notes.append("Very few trades (overfitting risk)")
        if metrics.win_rate > 0.85 and metrics.total_trades < 30:
            penalty += 0.10
            notes.append("Suspiciously high win rate with few trades")
        if metadata and metadata.parameter_count > 8:
            penalty += 0.08
            notes.append("High parameter count")
        return penalty
