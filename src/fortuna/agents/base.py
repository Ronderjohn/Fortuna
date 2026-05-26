"""Abstract agent interfaces for Ollama / LangGraph integration later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.strategy.schema import StrategyDefinition

if TYPE_CHECKING:
    import pandas as pd

    from fortuna.app.live_signals import SignalType
    from fortuna.features.position import PositionState


@dataclass
class ResearchContext:
    """Context passed to research agents when proposing strategies."""

    symbol: str
    timeframe: str
    market_regime: Optional[str] = None
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketContext:
    """Live snapshot used by Phase 2 RL agents.

    Distinct from ``ResearchContext`` (which is strategy-design level) — this
    is the per-bar runtime view the policy actually consumes.
    """

    symbol: str
    timeframe: str
    timestamp: "pd.Timestamp"
    latest_indicator_row: "pd.Series"
    position_state: "PositionState"
    ohlcv: Optional["pd.DataFrame"] = None
    market_regime: Optional[str] = None


@dataclass
class AgentProposal:
    """RL/agent-level proposal for the next action on the live bar."""

    signal: "SignalType"
    source: str = "rl_policy"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueResult:
    """Output from critic agent."""

    approved: bool
    score_adjustment: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class ResearchAgent(ABC):
    """Propose new strategies and/or live actions.

    Phase 1 `propose_strategy` keeps its original contract. Phase 2 adds a
    `propose` method for per-bar action proposals — concrete agents are free
    to implement only one of them; the unused method may simply raise
    ``NotImplementedError``.
    """

    @abstractmethod
    def propose_strategy(self, context: ResearchContext) -> StrategyDefinition:
        ...

    def propose(self, market_context: MarketContext) -> AgentProposal:
        """Default: agents that do not support live proposals raise."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement per-bar propose()"
        )


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
