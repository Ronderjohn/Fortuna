"""Abstract agent interfaces for Ollama / LangGraph integration later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.strategy.schema import StrategyDefinition


@dataclass
class ResearchContext:
    """Context passed to research agents when proposing strategies."""

    symbol: str
    timeframe: str
    market_regime: Optional[str] = None
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueResult:
    """Output from critic agent."""

    approved: bool
    score_adjustment: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class ResearchAgent(ABC):
    """Propose new strategy definitions (LLM-backed in future phases)."""

    @abstractmethod
    def propose_strategy(self, context: ResearchContext) -> StrategyDefinition:
        ...


class CriticAgent(ABC):
    """Critique strategies and backtest results."""

    @abstractmethod
    def critique(
        self,
        strategy: StrategyDefinition,
        metrics: BacktestMetrics,
    ) -> CritiqueResult:
        ...


class OptimizerAgent(ABC):
    """Suggest parameter improvements for strategies."""

    @abstractmethod
    def suggest_params(
        self,
        strategy: StrategyDefinition,
        history: list[dict[str, Any]],
    ) -> StrategyDefinition:
        ...
