"""Internal deterministic multi-agent workflow for advisory decisions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from fortuna.agentic.agents import (
    DecisionSynthesizer,
    DeterministicSignalAgent,
    MarketContextAgent,
    NotificationComposer,
    PortfolioContextAgent,
    RiskReviewAgent,
    RLPolicyAgent,
    adjust_confidence,
)
from fortuna.agentic.models import (
    AgentDecision,
    AgentRunContext,
    DecisionRationale,
)
from fortuna.ml.agent import MLSignalScorerAgent
from fortuna.utils.logging import get_logger

if TYPE_CHECKING:
    from fortuna.ml.signal_scorer import SignalScorer

logger = get_logger(__name__)


class AgenticOrchestrator:
    """Run Fortuna's advisory agents and return an audited decision."""

    def __init__(
        self,
        *,
        paper_tracking: bool = False,
        ml_scorer: Optional["SignalScorer"] = None,
    ) -> None:
        self.paper_tracking = bool(paper_tracking)
        self._market = MarketContextAgent()
        self._deterministic = DeterministicSignalAgent()
        self._ml = MLSignalScorerAgent(ml_scorer)
        self._rl = RLPolicyAgent()
        self._portfolio = PortfolioContextAgent()
        self._risk = RiskReviewAgent()
        self._synth = DecisionSynthesizer()
        self._notify = NotificationComposer()

    def decide(self, context: AgentRunContext) -> AgentDecision:
        votes = self._collect_votes(context)

        action, confidence, rationale = self._synth.synthesize(context, votes)
        action, risk_notes, dissent = self._risk.review(context, action, votes)
        confidence = adjust_confidence(confidence, action, votes)
        if risk_notes or dissent:
            rationale = DecisionRationale(
                summary=rationale.summary,
                reasons=rationale.reasons,
                risk_notes=[*rationale.risk_notes, *risk_notes],
                dissent=[*rationale.dissent, *dissent],
            )

        regime = None
        rl_action = None
        for vote in votes:
            if vote.agent == "market_context":
                regime = regime or vote.metadata.get("regime")
        for sig in context.signals.values():
            regime = regime or getattr(sig, "regime", None)
            rl_action = rl_action or getattr(sig, "rl_action", None)

        decision = AgentDecision(
            symbol=context.symbol,
            action=action,
            confidence=round(float(confidence), 4),
            bar_time=context.bar_time,
            bar_close=context.bar_close,
            rationale=rationale,
            votes=votes,
            current_side=context.current_side,
            regime=regime,
            rl_action=rl_action,
            paper_tracking=self.paper_tracking,
            metadata=self._decision_metadata(context),
        )
        notification = self._notify.compose(decision)
        if notification is not None:
            decision = replace(decision, notification=notification)
        return decision

    def _collect_votes(self, context: AgentRunContext) -> list:
        votes = []
        agents = (
            self._market,
            self._deterministic,
            self._ml,
            self._rl,
            self._portfolio,
        )
        for agent in agents:
            try:
                votes.extend(agent.votes(context))
            except Exception as exc:  # noqa: BLE001
                logger.debug("[agentic] %s votes failed: %s", agent.name, exc)
        return votes

    def decide_many(self, contexts: list[AgentRunContext]) -> dict[str, AgentDecision]:
        return {ctx.symbol: self.decide(ctx) for ctx in contexts}

    def _decision_metadata(self, context: AgentRunContext) -> dict:
        """Keep transient inference objects out of durable decision artifacts."""
        metadata = dict(context.metadata)
        metadata.pop("enriched", None)
        return metadata
