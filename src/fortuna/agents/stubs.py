"""Deterministic agent stubs for testing and architecture validation."""

from __future__ import annotations

from typing import Any

from fortuna.agents.base import (
    CritiqueResult,
    CriticAgent,
    OptimizerAgent,
    ResearchAgent,
    ResearchContext,
)
from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import StrategyDefinition


class NullResearchAgent(ResearchAgent):
    """Returns a bundled example strategy without LLM calls."""

    def __init__(self, default_strategy_path: str) -> None:
        self.default_strategy_path = default_strategy_path

    def propose_strategy(self, context: ResearchContext) -> StrategyDefinition:
        strategy = load_strategy(self.default_strategy_path)
        strategy.symbol = context.symbol
        strategy.timeframe = context.timeframe
        return strategy


class RuleBasedCriticAgent(CriticAgent):
    """Simple threshold-based critic (deterministic)."""

    def __init__(
        self,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.30,
        min_trades: int = 5,
    ) -> None:
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_trades = min_trades

    def critique(
        self,
        strategy: StrategyDefinition,
        metrics: BacktestMetrics,
    ) -> CritiqueResult:
        issues: list[str] = []
        suggestions: list[str] = []
        adjustment = 0.0

        if metrics.sharpe_ratio < self.min_sharpe:
            issues.append(f"Sharpe {metrics.sharpe_ratio:.2f} below {self.min_sharpe}")
            adjustment -= 5.0
        if metrics.max_drawdown > self.max_drawdown:
            issues.append(f"Max drawdown {metrics.max_drawdown:.1%} exceeds limit")
            adjustment -= 10.0
        if metrics.total_trades < self.min_trades:
            issues.append(f"Only {metrics.total_trades} trades — insufficient sample")
            adjustment -= 8.0
        if metrics.profit_factor < 1.0:
            issues.append("Profit factor below 1.0")
            suggestions.append("Tighten stops or refine entry filters")

        approved = len(issues) == 0
        return CritiqueResult(
            approved=approved,
            score_adjustment=adjustment,
            issues=issues,
            suggestions=suggestions,
        )


class PassThroughOptimizerAgent(OptimizerAgent):
    """Returns strategy unchanged — placeholder for grid/LLM optimizers."""

    def suggest_params(
        self,
        strategy: StrategyDefinition,
        history: list[dict[str, Any]],
    ) -> StrategyDefinition:
        return strategy
