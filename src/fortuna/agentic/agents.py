"""Small deterministic agents used by the internal orchestrator."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import pandas as pd

from fortuna.agentic.models import (
    ActionRecommendation,
    AgentRunContext,
    AgentVote,
    DecisionRationale,
    NotificationIntent,
)

_OPEN_MIN = 9 * 60 + 15
_CLOSE_MIN = 15 * 60 + 30
_SESSION_LEN = _CLOSE_MIN - _OPEN_MIN

_INDEPENDENT_AGENTS = frozenset(
    {"deterministic_signals", "ml_signal_scorer", "rl_policy", "portfolio_context"}
)
_SUPPORTING_AGENTS = frozenset({"deterministic_signals", "ml_signal_scorer", "rl_policy"})


def normalize_action(raw: Any) -> ActionRecommendation:
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


def _session_phase_and_progress(bar_time: Any) -> tuple[str, float]:
    if bar_time is None:
        return "unknown", 0.0
    ts = pd.Timestamp(bar_time)
    minutes = ts.hour * 60 + ts.minute
    progress = max(0.0, min(1.0, (minutes - _OPEN_MIN) / _SESSION_LEN))
    if minutes < _OPEN_MIN + 15:
        phase = "opening"
    elif minutes >= _CLOSE_MIN - 30:
        phase = "square-off"
    else:
        phase = "mid"
    return phase, progress


def adjust_confidence(
    base: float,
    action: ActionRecommendation,
    votes: list[AgentVote],
) -> float:
    """Boost agreement across independent agents; penalize ML/RL opposition."""
    conf = float(base)
    agreeing = [
        v
        for v in votes
        if v.agent in _INDEPENDENT_AGENTS
        and v.confidence >= 0.5
        and v.action == action
    ]
    extra = max(0, len(agreeing) - 1)
    conf += min(0.12, extra * 0.04)
    if action in {ActionRecommendation.BUY, ActionRecommendation.SELL}:
        opposite = (
            ActionRecommendation.SELL
            if action == ActionRecommendation.BUY
            else ActionRecommendation.BUY
        )
        for agent_name in ("ml_signal_scorer", "rl_policy"):
            for vote in votes:
                if (
                    vote.agent == agent_name
                    and vote.action == opposite
                    and vote.confidence >= 0.55
                ):
                    conf -= 0.10
    return max(0.0, min(0.99, conf))


class MarketContextAgent:
    """Audit-only vote summarizing regime and session phase."""

    name = "market_context"

    def votes(self, context: AgentRunContext) -> list[AgentVote]:
        regime: Optional[str] = None
        for sig in context.signals.values():
            raw = getattr(sig, "regime", None)
            if raw:
                regime = str(raw)
                break
        phase, progress = _session_phase_and_progress(context.bar_time)
        pct = int(progress * 100)
        return [
            AgentVote(
                agent=self.name,
                action=ActionRecommendation.HOLD,
                confidence=0.0,
                reason=f"Regime {regime or 'unknown'}; session {pct}% elapsed ({phase})",
                weight=0.0,
                metadata={
                    "regime": regime,
                    "session_progress": progress,
                    "session_phase": phase,
                },
            )
        ]


class DeterministicSignalAgent:
    """Turns per-strategy LiveSignal rows into weighted votes."""

    name = "deterministic_signals"

    def votes(self, context: AgentRunContext) -> list[AgentVote]:
        votes: list[AgentVote] = []
        for strategy_name, sig in sorted(context.signals.items()):
            if str(strategy_name).startswith("RL:"):
                continue
            action = normalize_action(getattr(sig, "action", "HOLD"))
            if action == ActionRecommendation.HOLD:
                continue
            confidence = 0.55
            rl_conf = getattr(sig, "rl_confidence", None)
            if rl_conf == "agree":
                confidence = 0.72
            elif rl_conf == "disagree":
                confidence = 0.35
            reason = f"{strategy_name} fired {action.value}"
            votes.append(
                AgentVote(
                    agent=self.name,
                    action=action,
                    confidence=confidence,
                    reason=reason,
                    source=str(strategy_name),
                    metadata={
                        "rl_confidence": rl_conf,
                        "regime": getattr(sig, "regime", None),
                    },
                )
            )
        return votes


class RLPolicyAgent:
    """Adds a vote from the standalone RL signal row when available."""

    name = "rl_policy"

    def votes(self, context: AgentRunContext) -> list[AgentVote]:
        out: list[AgentVote] = []
        for strategy_name, sig in context.signals.items():
            if not str(strategy_name).startswith("RL:"):
                continue
            action = normalize_action(getattr(sig, "action", "HOLD"))
            if action == ActionRecommendation.HOLD:
                continue
            out.append(
                AgentVote(
                    agent=self.name,
                    action=action,
                    confidence=0.68,
                    reason=f"{strategy_name} policy proposed {action.value}",
                    weight=1.2,
                    source=str(strategy_name),
                    metadata={"run_id": getattr(sig, "rl_run_id", None)},
                )
            )
        return out


class PortfolioContextAgent:
    """Prevents entry votes from acting like exits for existing positions."""

    name = "portfolio_context"

    def votes(self, context: AgentRunContext) -> list[AgentVote]:
        side = (context.current_side or "").upper()
        if side == "LONG":
            exits = _strategies_for(context.signals, {"EXIT_LONG", "EXIT"})
            if exits:
                return [
                    AgentVote(
                        agent=self.name,
                        action=ActionRecommendation.EXIT_LONG,
                        confidence=0.8,
                        reason="Open LONG has exit confirmation",
                        weight=1.5,
                        source=", ".join(exits),
                    )
                ]
            return [
                AgentVote(
                    agent=self.name,
                    action=ActionRecommendation.HOLD,
                    confidence=0.6,
                    reason="Open LONG, no exit confirmation",
                    weight=0.8,
                )
            ]
        if side == "SHORT":
            exits = _strategies_for(context.signals, {"EXIT_SHORT", "EXIT"})
            if exits:
                return [
                    AgentVote(
                        agent=self.name,
                        action=ActionRecommendation.EXIT_SHORT,
                        confidence=0.8,
                        reason="Open SHORT has exit confirmation",
                        weight=1.5,
                        source=", ".join(exits),
                    )
                ]
            return [
                AgentVote(
                    agent=self.name,
                    action=ActionRecommendation.HOLD,
                    confidence=0.6,
                    reason="Open SHORT, no exit confirmation",
                    weight=0.8,
                )
            ]
        return []


class RiskReviewAgent:
    """Adds conservative veto notes for weak or contradictory entry evidence."""

    name = "risk_review"

    def review(
        self,
        context: AgentRunContext,
        action: ActionRecommendation,
        votes: list[AgentVote],
    ) -> tuple[ActionRecommendation, list[str], list[str]]:
        risk_notes: list[str] = []
        dissent: list[str] = []
        if action in {ActionRecommendation.BUY, ActionRecommendation.SELL}:
            opposite = (
                ActionRecommendation.SELL
                if action == ActionRecommendation.BUY
                else ActionRecommendation.BUY
            )
            opposing = [
                v
                for v in votes
                if v.action == opposite
                and v.confidence >= 0.5
                and v.agent in _SUPPORTING_AGENTS
            ]
            supporting = [
                v
                for v in votes
                if v.action == action
                and v.confidence >= 0.5
                and v.agent in _SUPPORTING_AGENTS
            ]
            if opposing:
                dissent.append(
                    "Opposing votes: "
                    + ", ".join(f"{v.source or v.agent}:{v.action.value}" for v in opposing)
                )
            if not supporting:
                return ActionRecommendation.DO_NOT_ENTER, risk_notes, dissent
            if len(supporting) == 1 and supporting[0].agent == "deterministic_signals":
                ml_unfavorable = any(
                    v.agent == "ml_signal_scorer"
                    and v.metadata.get("predicted_class") == 0
                    and v.confidence >= 0.6
                    for v in votes
                )
                has_ml_rl_support = any(
                    v.agent in ("ml_signal_scorer", "rl_policy") and v.action == action
                    for v in votes
                )
                if ml_unfavorable:
                    risk_notes.append("ML scores entry unfavorably")
                    return ActionRecommendation.DO_NOT_ENTER, risk_notes, dissent
                if not has_ml_rl_support:
                    risk_notes.append("Single deterministic entry without ML/RL support")
                    return ActionRecommendation.DO_NOT_ENTER, risk_notes, dissent
        if context.is_holding and action == ActionRecommendation.SELL:
            risk_notes.append(
                "Long-term holding detected; SELL is advisory only, not an auto-short"
            )
        return action, risk_notes, dissent


class DecisionSynthesizer:
    """Weighted vote aggregation."""

    def synthesize(
        self,
        context: AgentRunContext,
        votes: list[AgentVote],
    ) -> tuple[ActionRecommendation, float, DecisionRationale]:
        if not votes:
            return (
                ActionRecommendation.HOLD,
                0.0,
                DecisionRationale(summary="No actionable agent votes", reasons=[]),
            )

        side = (context.current_side or "").upper()
        if side == "LONG":
            action = _score_actions(votes, allowed={ActionRecommendation.EXIT_LONG})
            if action is None:
                return (
                    ActionRecommendation.HOLD,
                    0.45,
                    DecisionRationale(
                        summary="Hold LONG; no exit won the vote",
                        reasons=_top_reasons(votes),
                    ),
                )
        elif side == "SHORT":
            action = _score_actions(votes, allowed={ActionRecommendation.EXIT_SHORT})
            if action is None:
                return (
                    ActionRecommendation.HOLD,
                    0.45,
                    DecisionRationale(
                        summary="Hold SHORT; no exit won the vote",
                        reasons=_top_reasons(votes),
                    ),
                )
        else:
            action = _score_actions(
                votes,
                allowed={ActionRecommendation.BUY, ActionRecommendation.SELL},
            )
            if action is None:
                return (
                    ActionRecommendation.HOLD,
                    0.35,
                    DecisionRationale(
                        summary="Stay flat; no entry won the vote",
                        reasons=_top_reasons(votes),
                    ),
                )

        scores = _scores(votes)
        total = sum(v for k, v in scores.items() if k != ActionRecommendation.HOLD)
        confidence = min(0.99, max(0.0, scores[action] / total if total else 0.0))
        return (
            action,
            confidence,
            DecisionRationale(
                summary=f"{action.value} won agent vote",
                reasons=(
                    _top_reasons([v for v in votes if v.action == action])
                    or _top_reasons(votes)
                ),
            ),
        )


class NotificationComposer:
    """Build concise Telegram-ready messages."""

    def compose(self, decision) -> Optional[NotificationIntent]:
        action = decision.action
        if not action.is_actionable and action != ActionRecommendation.DO_NOT_ENTER:
            return None
        if action == ActionRecommendation.DO_NOT_ENTER and not decision.current_side:
            return None
        price = f" @ {decision.bar_close:.2f}" if decision.bar_close is not None else ""
        conf = f"{decision.confidence * 100:.0f}%"
        reasons = "; ".join(decision.rationale.reasons[:2]) or decision.rationale.summary
        risk = "; ".join(decision.rationale.risk_notes[:2])
        lines = [
            f"Fortuna: {action.value} {decision.symbol}{price}",
            f"Confidence: {conf}",
            f"Why: {reasons}",
        ]
        for agent_name, label in (("ml_signal_scorer", "ML"), ("rl_policy", "RL")):
            for vote in decision.votes:
                if vote.agent == agent_name and vote.reason:
                    lines.append(f"{label}: {vote.reason}")
                    break
        if decision.regime:
            lines.append(f"Regime: {decision.regime}")
        if risk:
            lines.append(f"Risk: {risk}")
        mode = (
            "Paper learning: enabled (advisory + simulated tracking)"
            if decision.paper_tracking
            else "Paper learning: off (advisory only)"
        )
        lines.append(mode)
        lines.append("Advisory only — not an execution instruction.")
        return NotificationIntent(
            symbol=decision.symbol,
            action=action,
            message="\n".join(lines),
            dedupe_key=decision.notification_key,
            metadata={"decision_hash": decision.decision_hash},
        )


def _strategies_for(signals: dict[str, Any], actions: set[str]) -> list[str]:
    out: list[str] = []
    for name, sig in signals.items():
        if str(getattr(sig, "action", "")).upper() in actions:
            out.append(str(name))
    return sorted(set(out))


def _scores(votes: list[AgentVote]) -> Counter:
    scores: Counter = Counter()
    for vote in votes:
        scores[vote.action] += max(0.0, vote.confidence) * max(0.0, vote.weight)
    return scores


def _score_actions(
    votes: list[AgentVote],
    *,
    allowed: set[ActionRecommendation],
) -> Optional[ActionRecommendation]:
    scores = _scores([v for v in votes if v.action in allowed])
    if not scores:
        return None
    return scores.most_common(1)[0][0]


def _top_reasons(votes: list[AgentVote], n: int = 3) -> list[str]:
    ordered = sorted(votes, key=lambda v: v.confidence * v.weight, reverse=True)
    return [v.reason for v in ordered[:n]]
