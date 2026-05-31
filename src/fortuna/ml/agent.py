"""Adapter converting ML scorer output into an AgentVote for one strategy."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from fortuna.agentic.models import ActionRecommendation, AgentRunContext, AgentVote
from fortuna.ml.signal_scorer import SignalScorer, score_or_neutral


def _normalize_action(raw: object) -> ActionRecommendation:
    val = getattr(raw, "value", raw)
    text = str(val or "HOLD").upper()
    if text == "SELL_SHORT":
        text = "SELL"
    if text == "EXIT":
        return ActionRecommendation.EXIT_LONG
    if text in ActionRecommendation.__members__:
        return ActionRecommendation[text]
    if text in {a.value for a in ActionRecommendation}:
        return ActionRecommendation(text)
    return ActionRecommendation.HOLD


class MLSignalScorerAgent:
    """Scores deterministic strategy signals via the optional ML scorer."""

    name = "ml_signal_scorer"

    def __init__(self, scorer: Optional[SignalScorer] = None) -> None:
        self._scorer = scorer

    def votes(self, context: AgentRunContext) -> list[AgentVote]:
        if self._scorer is None:
            return []
        enriched = context.metadata.get("enriched")
        bar_idx = context.metadata.get("bar_idx")
        if enriched is None or bar_idx is None:
            return []
        if not isinstance(enriched, pd.DataFrame):
            return []
        try:
            idx = int(bar_idx)
        except (TypeError, ValueError):
            return []
        out: list[AgentVote] = []
        for strategy_name in sorted(context.signals):
            if str(strategy_name).startswith("RL:"):
                continue
            sig = context.signals[strategy_name]
            action = _normalize_action(getattr(sig, "action", "HOLD"))
            if not action.is_actionable:
                continue
            vote = self.vote_for_strategy(
                context,
                strategy_name=str(strategy_name),
                enriched=enriched,
                bar_idx=idx,
            )
            if vote is not None:
                out.append(vote)
        return out

    def vote_for_strategy(
        self,
        context: AgentRunContext,
        *,
        strategy_name: str,
        enriched: pd.DataFrame,
        bar_idx: int,
    ) -> Optional[AgentVote]:
        sig = context.signals.get(strategy_name)
        if sig is None:
            return None

        action = _normalize_action(getattr(sig, "action", "HOLD"))
        if not action.is_actionable:
            return None

        result = score_or_neutral(
            self._scorer,
            enriched,
            bar_idx,
            action=action.value,
            position_side=str(context.current_side or "FLAT"),
            bar_time=context.bar_time,
        )
        if not result.available:
            return None

        return AgentVote(
            agent=self.name,
            action=action,
            confidence=round(float(result.probability), 4),
            reason=result.reason,
            source=strategy_name,
            metadata={
                "model_id": result.model_id,
                "predicted_class": result.predicted_class,
                "strategy_name": strategy_name,
            },
        )
