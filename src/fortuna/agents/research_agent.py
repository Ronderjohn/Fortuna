"""``RLResearchAgent`` — wraps a trained policy as a Phase 2 ResearchAgent."""

from __future__ import annotations

from typing import Any

from fortuna.agents.base import (
    AgentProposal,
    MarketContext,
    ResearchAgent,
    ResearchContext,
)
from fortuna.app.live_signals import SignalType
from fortuna.rl.inference.signal_generator import RLSignalGenerator
from fortuna.strategy.schema import StrategyDefinition


class RLResearchAgent(ResearchAgent):
    """Per-bar action proposer backed by a trained RL policy.

    ``propose_strategy()`` is intentionally a no-op for now — the agent only
    contributes live signals; new strategy templates are still seeded by the
    Phase 1 ``NullResearchAgent`` until LLM-backed proposal lands.
    """

    def __init__(self, signal_generator: RLSignalGenerator) -> None:
        self._gen = signal_generator

    @property
    def is_available(self) -> bool:
        return self._gen.is_available

    def propose_strategy(self, context: ResearchContext) -> StrategyDefinition:
        raise NotImplementedError(
            "RLResearchAgent.propose_strategy is reserved for Phase 2.3+ "
            "(LLM-backed proposals). Use propose() for per-bar action signals."
        )

    def propose(self, market_context: MarketContext) -> AgentProposal:
        if not self._gen.is_available:
            return AgentProposal(
                signal=SignalType.HOLD,
                source="rl_unavailable",
                confidence=0.0,
                metadata={"reason": "no_policy_loaded"},
            )

        rl_signal = self._gen.predict(
            indicator_row=market_context.latest_indicator_row,
            position_state=market_context.position_state,
            timestamp=market_context.timestamp,
        )
        if rl_signal is None:
            return AgentProposal(
                signal=SignalType.HOLD,
                source="rl_unavailable",
                confidence=0.0,
                metadata={"reason": "predict_returned_none"},
            )

        meta: dict[str, Any] = dict(rl_signal.metadata or {})
        meta.update(
            {
                "action_id": rl_signal.action_id,
                "bar_close": rl_signal.bar_close,
            }
        )
        return AgentProposal(
            signal=rl_signal.signal,
            source=rl_signal.source,
            confidence=1.0,
            metadata=meta,
        )
