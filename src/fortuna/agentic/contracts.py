"""Stable typed contracts for interactive advisory and model-health surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from fortuna.agentic.models import AgentDecision


class AdvisoryErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    RESOLVE_FAILED = "resolve_failed"
    LOAD_FAILED = "load_failed"
    DECISION_UNAVAILABLE = "decision_unavailable"
    INTERNAL = "internal"


@dataclass(frozen=True)
class AdvisoryError:
    code: AdvisoryErrorCode
    message: str
    details: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class InstrumentAnalysisRequest:
    symbol: str
    timeframe: str = "5m"
    days: int = 30
    force_refresh: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class InstrumentSnapshot:
    requested_symbol: str
    resolved_symbol: str
    segment_label: str
    timeframe: str
    lookback_days: int
    last_bar_time: Optional[datetime] = None
    last_close: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class DecisionSummary:
    action: str
    confidence: float
    summary: str
    reasons: tuple[str, ...] = ()
    risk_notes: tuple[str, ...] = ()
    action_plan: str = ""

    @classmethod
    def from_agent_decision(cls, decision: AgentDecision) -> DecisionSummary:
        action = decision.action.value
        return cls(
            action=action,
            confidence=float(decision.confidence),
            summary=decision.rationale.summary,
            reasons=tuple(decision.rationale.reasons),
            risk_notes=tuple(decision.rationale.risk_notes),
            action_plan=action_plan_for(action),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class SignalSummary:
    name: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class InstrumentAnalysisResponse:
    ok: bool
    request: InstrumentAnalysisRequest
    instrument: Optional[InstrumentSnapshot] = None
    decision: Optional[DecisionSummary] = None
    signals: tuple[SignalSummary, ...] = ()
    winning_strategy: Optional[str] = None
    disclaimer: str = "Advisory only — not an execution instruction."
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class InstrumentSearchHitSummary:
    symbol: str
    display: str
    segment: str
    tradingsymbol: str

    @classmethod
    def from_hit(cls, hit: Any) -> InstrumentSearchHitSummary:
        return cls(
            symbol=str(hit.symbol),
            display=str(hit.display),
            segment=str(getattr(hit, "segment", "EQUITY")),
            tradingsymbol=str(hit.tradingsymbol),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class InstrumentSearchResponse:
    ok: bool
    query: str
    hits: tuple[InstrumentSearchHitSummary, ...] = ()
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class PromotionRecord:
    run_id: Optional[str] = None
    symbol: Optional[str] = None
    promoted_by: Optional[str] = None
    artifact_dir: Optional[str] = None
    model_kind: Optional[str] = None

    @classmethod
    def from_audit_row(cls, row: Optional[dict[str, Any]]) -> Optional[PromotionRecord]:
        if not row:
            return None
        return cls(
            run_id=row.get("run_id"),
            symbol=row.get("symbol"),
            promoted_by=row.get("promoted_by"),
            artifact_dir=row.get("artifact_dir"),
            model_kind=row.get("model_kind"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class RlModelStatus:
    symbol: str = ""
    available: bool = False
    advisory_ready: bool = False
    load_error: Optional[str] = "no_generator"
    live_pointer: Optional[str] = None
    checkpoint_dir: str = ""
    last_promotion: Optional[PromotionRecord] = None
    run_id: Optional[str] = None
    verdict_passed: Optional[bool] = None
    policy_type: Optional[str] = None
    baseline_sharpe: Optional[float] = None
    beats_baseline: Optional[bool] = None
    oos_sharpe: Optional[float] = None
    oos_pf: Optional[float] = None
    oos_trades: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        if self.last_promotion is not None:
            out["last_promotion"] = self.last_promotion.to_dict()
        else:
            out["last_promotion"] = None
        return out


@dataclass(frozen=True)
class MlModelStatus:
    available: bool = False
    enabled: bool = False
    live_pointer: Optional[str] = None
    artifact_dir: str = ""
    load_error: Optional[str] = "not_loaded"
    last_promotion: Optional[PromotionRecord] = None
    run_id: Optional[str] = None
    verdict_passed: Optional[bool] = None
    advisory_ready: Optional[bool] = None
    feature_schema_hash: Optional[str] = None
    oos_precision: Optional[float] = None
    oos_roc_auc: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        if self.last_promotion is not None:
            out["last_promotion"] = self.last_promotion.to_dict()
        else:
            out["last_promotion"] = None
        return out


@dataclass(frozen=True)
class RegimeModelStatus:
    available: bool = False
    path: str = ""
    live_pointer: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class LearningRowSummary:
    symbol: str
    action: str
    bar_time: str
    resolved: bool
    paper_closed: bool
    realized_pnl_pct: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class LearningSummaryStatus:
    total_rows: int = 0
    resolved_rows: int = 0
    paper_closed_rows: int = 0
    avg_realized_pnl_pct: Optional[float] = None
    recent_rows: tuple[LearningRowSummary, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["recent_rows"] = [row.to_dict() for row in self.recent_rows]
        return out


@dataclass(frozen=True)
class AgenticDecisionRow:
    symbol: Optional[str]
    action: str
    confidence: float
    bar_time: str

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class AgenticStatusSummary:
    recent_decisions: tuple[AgenticDecisionRow, ...] = ()
    learning_summary: LearningSummaryStatus = field(default_factory=LearningSummaryStatus)

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["recent_decisions"] = [row.to_dict() for row in self.recent_decisions]
        out["learning_summary"] = self.learning_summary.to_dict()
        return out


@dataclass(frozen=True)
class ModelHealthResponse:
    registry_enabled: bool
    promotion_required: bool
    rl: RlModelStatus
    ml: MlModelStatus
    regime: RegimeModelStatus
    agentic: AgenticStatusSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_enabled": self.registry_enabled,
            "promotion_required": self.promotion_required,
            "rl": self.rl.to_dict(),
            "ml": self.ml.to_dict(),
            "regime": self.regime.to_dict(),
            "agentic": self.agentic.to_dict(),
        }


def action_plan_for(action: str) -> str:
    if action == "BUY":
        return (
            "Consider entry on the long side near current market "
            "structure; re-check on next closed bar."
        )
    if action == "SELL":
        return (
            "Consider short entry if your market/instrument permits it; "
            "re-check on next closed bar."
        )
    if action.startswith("EXIT"):
        return (
            "Protect profits or cut risk; current advisory favors "
            "exiting the open side."
        )
    if action == "DO_NOT_ENTER":
        return "Stay flat; current setup is not attractive enough to open a position."
    return "Wait for a clearer setup."


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
