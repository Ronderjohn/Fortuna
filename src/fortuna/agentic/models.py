"""Typed contracts for the agentic advisory workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ActionRecommendation(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"
    DO_NOT_ENTER = "DO_NOT_ENTER"

    @property
    def is_actionable(self) -> bool:
        return self in {
            ActionRecommendation.BUY,
            ActionRecommendation.SELL,
            ActionRecommendation.EXIT_LONG,
            ActionRecommendation.EXIT_SHORT,
        }

    @property
    def is_exit(self) -> bool:
        return self in {ActionRecommendation.EXIT_LONG, ActionRecommendation.EXIT_SHORT}


@dataclass(frozen=True)
class AgentVote:
    agent: str
    action: ActionRecommendation
    confidence: float
    reason: str
    weight: float = 1.0
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRationale:
    summary: str
    reasons: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    dissent: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NotificationIntent:
    symbol: str
    action: ActionRecommendation
    message: str
    dedupe_key: str
    destination: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotificationResult:
    sent: bool
    provider: str = ""
    message_id: Optional[str] = None
    error: Optional[str] = None
    deduped: bool = False
    throttled: bool = False
    policy_reason: str = ""


@dataclass(frozen=True)
class AgentDecision:
    symbol: str
    action: ActionRecommendation
    confidence: float
    bar_time: Optional[datetime]
    bar_close: Optional[float]
    rationale: DecisionRationale
    votes: list[AgentVote] = field(default_factory=list)
    current_side: Optional[str] = None
    regime: Optional[str] = None
    rl_action: Optional[str] = None
    notification: Optional[NotificationIntent] = None
    notification_result: Optional[NotificationResult] = None
    paper_tracking: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decision_hash(self) -> str:
        payload = {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": round(float(self.confidence), 4),
            "bar_time": self.bar_time.isoformat() if self.bar_time else None,
            "bar_close": self.bar_close,
            "votes": [
                {
                    "agent": v.agent,
                    "action": v.action.value,
                    "confidence": round(float(v.confidence), 4),
                    "reason": v.reason,
                    "source": v.source,
                }
                for v in self.votes
            ],
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def notification_key(self) -> str:
        t = self.bar_time.isoformat() if self.bar_time else "no-bar"
        return f"{self.symbol}|{self.action.value}|{t}|{self.decision_hash}"

    def with_notification_result(self, result: NotificationResult) -> "AgentDecision":
        return AgentDecision(
            symbol=self.symbol,
            action=self.action,
            confidence=self.confidence,
            bar_time=self.bar_time,
            bar_close=self.bar_close,
            rationale=self.rationale,
            votes=self.votes,
            current_side=self.current_side,
            regime=self.regime,
            rl_action=self.rl_action,
            notification=self.notification,
            notification_result=result,
            paper_tracking=self.paper_tracking,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class AgentRunContext:
    symbol: str
    timeframe: str
    signals: dict[str, Any]
    bar_time: Optional[datetime] = None
    bar_close: Optional[float] = None
    current_side: Optional[str] = None
    position_qty: int = 0
    is_holding: bool = False
    account_summary: dict[str, Any] = field(default_factory=dict)
    # Optional keys: bar_idx (int), enriched (indicator OHLCV DataFrame for ML).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperLearningEvent:
    ts: datetime
    symbol: str
    action: ActionRecommendation
    decision_hash: str
    event_type: str
    confidence: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
