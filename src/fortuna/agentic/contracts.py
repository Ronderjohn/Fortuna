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
    structured_signal: Optional["StructuredSignalDetails"] = None
    replay_risk: Optional["ReplayRiskSummary"] = None
    futures_lot_risk: Optional["FuturesLotRiskAssessment"] = None
    oi_enrichment: Optional["OpenInterestEnrichment"] = None
    disclaimer: str = "Advisory only — not an execution instruction."
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class SignalLayerUsage:
    deterministic: str = "used"
    ml: str = "not_ready"
    rl: str = "not_ready"

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ConfidenceComponent:
    name: str
    score: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class OpenInterestEnrichment:
    available: bool = False
    latest_open_interest: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    posture: str = ""
    confirms_direction: Optional[bool] = None
    summary: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class StructuredSignalDetails:
    source: str = "structured_signal_engine"
    verdict: str = "NO_TRADE"
    setup_type: str = ""
    trend_context: str = ""
    confirmation: str = ""
    invalidation: str = ""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward: Optional[float] = None
    market_regime: str = ""
    volatility_bucket: str = ""
    reason_to_avoid: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    confidence_components: tuple[ConfidenceComponent, ...] = ()
    feature_summary: dict[str, float] = field(default_factory=dict)
    oi_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ReplayRiskSummary:
    ok: bool = False
    sample_size: int = 0
    matched_setups: int = 0
    stop_hit_rate: Optional[float] = None
    target_hit_rate: Optional[float] = None
    avg_hold_bars: Optional[float] = None
    avg_mae_pct: Optional[float] = None
    avg_mfe_pct: Optional[float] = None
    worst_adverse_excursion_pct: Optional[float] = None
    avg_realized_risk_reward: Optional[float] = None
    volatility_posture: str = ""
    summary: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class FuturesLotRiskAssessment:
    available: bool = False
    lot_size: int = 0
    contract_price: Optional[float] = None
    contract_notional: Optional[float] = None
    stop_distance_per_unit: Optional[float] = None
    stop_loss_amount_per_lot: Optional[float] = None
    cost_adjusted_loss_per_lot: Optional[float] = None
    atr_noise_per_lot: Optional[float] = None
    volatility_bucket: str = ""
    entry_risk_bucket: str = ""
    summary: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class SignalResponse:
    resolved_symbol: str
    segment: str
    timeframe: str
    lookback_days: int
    signal_verdict: str
    confidence: float
    summary: str
    reasons: tuple[str, ...] = ()
    risk_notes: tuple[str, ...] = ()
    action_plan: str = ""
    data_freshness: str = ""
    market_context: Optional[str] = None
    used_layers: SignalLayerUsage = field(default_factory=SignalLayerUsage)
    request_modality: str = "text"
    forecast_used: bool = False
    forecast_summaries: tuple[str, ...] = ()
    agent_statuses: tuple[str, ...] = ()
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    invalidation: str = ""
    setup_type: str = ""
    risk_reward: Optional[float] = None
    replay_summary: str = ""
    lot_risk_summary: str = ""
    lot_stop_loss_amount: Optional[float] = None
    volatility_bucket: str = ""
    oi_summary: str = ""
    expert_notes: tuple[str, ...] = ()
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class MediaAttachment:
    kind: str
    file_name: str = ""
    mime_type: str = ""
    local_path: str = ""
    content_hash: str = ""
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class MultimodalRequestEnvelope:
    request_id: str
    channel: str
    chat_id: str
    modality: str
    raw_text: str = ""
    caption: str = ""
    attachment: Optional[MediaAttachment] = None
    conversation_symbol: str = ""
    received_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ChartImageAnalysisResult:
    ok: bool
    intent: str = "clarify"
    symbol: str = ""
    instrument_family: str = "unknown"
    timeframe: str = "5m"
    confidence: float = 0.0
    trend_bias: str = ""
    level_notes: tuple[str, ...] = ()
    summary: str = ""
    warnings: tuple[str, ...] = ()
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ForecastTaskRequest:
    task_name: str
    symbol: str
    timeframe: str
    lookback_days: int
    latest_bar_iso: str = ""
    closes: tuple[float, ...] = ()
    highs: tuple[float, ...] = ()
    lows: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ForecastTaskResult:
    task_name: str
    ok: bool
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    reliability: float = 0.0
    mode: str = "builtin"

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class SandboxExecutionResult:
    ok: bool
    mode: str = "builtin"
    generated_code: str = ""
    stdout: str = ""
    stderr: str = ""
    runtime_ms: int = 0
    validation_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ForecastRetentionSummary:
    temp_images_deleted: int = 0
    temp_code_deleted: int = 0
    audit_rows_pruned: int = 0
    session_rows_pruned: int = 0

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
class MarketUniverseCandidate:
    symbol: str
    display_name: str
    source: str
    avg_turnover: Optional[float] = None
    avg_volume: Optional[float] = None
    trend_pct: Optional[float] = None
    last_close: Optional[float] = None
    liquidity_score: Optional[float] = None
    regime: str = ""
    volume_ratio: Optional[float] = None
    activity_score: Optional[float] = None
    adaptive_score_adjustment: Optional[float] = None
    adaptive_row_count: int = 0
    sector: str = ""
    market_cap_bucket: str = ""
    operator_quality_score: Optional[float] = None
    fundamentals_overlay_adjustment: Optional[float] = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class MarketUniverseResponse:
    ok: bool
    source: str
    timeframe: str
    lookback_days: int
    nightly_alignment_summary: Optional[str] = None
    nightly_alignment_target: Optional[str] = None
    nightly_alignment_force_refresh: bool = False
    nightly_recent_window: int = 0
    nightly_recent_enabled: int = 0
    nightly_recent_aligned: int = 0
    nightly_recent_latest_status: Optional[str] = None
    nightly_recent_latest_basket_size: int = 0
    scout_liquidity_symbols: tuple[str, ...] = ()
    scout_activity_symbols: tuple[str, ...] = ()
    scout_volume_dense_symbols: tuple[str, ...] = ()
    scout_overlap_symbols: tuple[str, ...] = ()
    scout_summary: Optional[str] = None
    seed_source: Optional[str] = None
    scoring_source: Optional[str] = None
    provider_summary: Optional[str] = None
    fallback_from: Optional[str] = None
    fundamentals_overlay_enabled: bool = False
    fundamentals_overlay_source: Optional[str] = None
    fundamentals_overlay_summary: Optional[str] = None
    fundamentals_overlay_diagnostics: tuple[str, ...] = ()
    candidates: tuple[MarketUniverseCandidate, ...] = ()
    disclaimer: str = "Universe shortlist for research/advisory preparation only."
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["candidates"] = [row.to_dict() for row in self.candidates]
        return out


@dataclass(frozen=True)
class RiskCritique:
    severity: str
    summary: str
    concerns: tuple[str, ...] = ()
    supports: tuple[str, ...] = ()
    verdict: str = "watch"

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ShortlistAnalysisItem:
    symbol: str
    instrument: Optional[InstrumentSnapshot] = None
    decision: Optional[DecisionSummary] = None
    winning_strategy: Optional[str] = None
    liquidity_score: Optional[float] = None
    trend_pct: Optional[float] = None
    regime: str = ""
    volume_ratio: Optional[float] = None
    activity_score: Optional[float] = None
    sector: str = ""
    market_cap_bucket: str = ""
    operator_quality_score: Optional[float] = None
    fundamentals_overlay_adjustment: Optional[float] = None
    selection_rank: int = 0
    priority_score: Optional[float] = None
    exposure_penalty: float = 0.0
    critique: Optional[RiskCritique] = None
    source: str = "universe"
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ShortlistAnalysisResponse:
    ok: bool
    source: str
    timeframe: str
    lookback_days: int
    scout_liquidity_symbols: tuple[str, ...] = ()
    scout_activity_symbols: tuple[str, ...] = ()
    scout_volume_dense_symbols: tuple[str, ...] = ()
    scout_overlap_symbols: tuple[str, ...] = ()
    fundamentals_overlay_summary: Optional[str] = None
    fundamentals_overlay_diagnostics: tuple[str, ...] = ()
    items: tuple[ShortlistAnalysisItem, ...] = ()
    disclaimer: str = "Shortlist is advisory preparation only."
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["items"] = [row.to_dict() for row in self.items]
        return out


@dataclass(frozen=True)
class TrainingCandidate:
    symbol: str
    shortlist_rank: int
    selection_rank: int = 0
    liquidity_score: Optional[float] = None
    winning_strategy: Optional[str] = None
    decision_action: Optional[str] = None
    decision_confidence: Optional[float] = None
    critique_verdict: str = "watch"
    ml_candidate: bool = False
    rl_candidate: bool = False
    market_regime: str = ""
    volume_ratio: Optional[float] = None
    priority_score: Optional[float] = None
    adaptive_score_adjustment: Optional[float] = None
    remediation_target: Optional[str] = None
    remediation_pressure: Optional[float] = None
    setup_family_reinforcement: Optional[float] = None
    setup_family_verdict: Optional[str] = None
    exposure_penalty: float = 0.0
    rationale: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class TrainingCandidateResponse:
    ok: bool
    source: str
    timeframe: str
    lookback_days: int
    candidates: tuple[TrainingCandidate, ...] = ()
    disclaimer: str = "Training candidates are suggestions for data refresh and model prep."
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["candidates"] = [row.to_dict() for row in self.candidates]
        return out


@dataclass(frozen=True)
class TrainingResearchRow:
    symbol: str
    target: str
    universe_rank: int = 0
    shortlist_rank: int = 0
    selection_rank: int = 0
    liquidity_score: Optional[float] = None
    decision_action: Optional[str] = None
    critique_verdict: str = "watch"
    winning_strategy: Optional[str] = None
    market_regime: str = ""
    volume_ratio: Optional[float] = None
    priority_score: Optional[float] = None
    adaptive_score_adjustment: Optional[float] = None
    remediation_target: Optional[str] = None
    remediation_pressure: Optional[float] = None
    setup_family_reinforcement: Optional[float] = None
    setup_family_verdict: Optional[str] = None
    refreshed: bool = False
    rationale: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class TrainingResearchPlanResponse:
    ok: bool
    source: str
    timeframe: str
    lookback_days: int
    selection_policy: str = "ranked"
    refresh_requested: bool = False
    refresh_target: str = "all"
    refresh_timeframe: Optional[str] = None
    refresh_lookback_days: Optional[int] = None
    nightly_alignment_summary: Optional[str] = None
    nightly_alignment_target: Optional[str] = None
    nightly_alignment_force_refresh: bool = False
    nightly_recent_window: int = 0
    nightly_recent_enabled: int = 0
    nightly_recent_aligned: int = 0
    nightly_recent_latest_status: Optional[str] = None
    nightly_recent_latest_basket_size: int = 0
    universe_symbols: tuple[str, ...] = ()
    discovery_preferred_symbols: tuple[str, ...] = ()
    discovery_liquidity_symbols: tuple[str, ...] = ()
    discovery_activity_symbols: tuple[str, ...] = ()
    discovery_volume_dense_symbols: tuple[str, ...] = ()
    discovery_regime_mix: Optional[str] = None
    discovery_summary: Optional[str] = None
    discovery_posture: Optional[str] = None
    strategy_posture: Optional[str] = None
    candidate_posture: Optional[str] = None
    nightly_posture: Optional[str] = None
    effective_posture: Optional[str] = None
    discovery_recommended_refresh_target: Optional[str] = None
    effective_refresh_target: Optional[str] = None
    refresh_urgency: Optional[str] = None
    refresh_urgency_target: Optional[str] = None
    refresh_urgency_score: Optional[float] = None
    refresh_urgency_rows: int = 0
    refresh_urgency_summary: Optional[str] = None
    follow_up_action: Optional[str] = None
    discovery_follow_up_target: Optional[str] = None
    discovery_follow_up_summary: Optional[str] = None
    discovery_follow_up_action: Optional[str] = None
    research_follow_up_target: Optional[str] = None
    research_follow_up_summary: Optional[str] = None
    research_follow_up_action: Optional[str] = None
    execution_follow_up_target: Optional[str] = None
    execution_follow_up_summary: Optional[str] = None
    execution_follow_up_action: Optional[str] = None
    selected_target_mix: Optional[str] = None
    selected_regime_mix: Optional[str] = None
    scout_support_target: Optional[str] = None
    ml_symbols: tuple[str, ...] = ()
    rl_symbols: tuple[str, ...] = ()
    rows: tuple[TrainingResearchRow, ...] = ()
    disclaimer: str = (
        "Training research plan is for data refresh and model preparation only."
    )
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["rows"] = [row.to_dict() for row in self.rows]
        return out


@dataclass(frozen=True)
class PortfolioExposureNote:
    level: str
    message: str
    symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class PortfolioCritique:
    summary: str
    notes: tuple[PortfolioExposureNote, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["notes"] = [row.to_dict() for row in self.notes]
        return out


@dataclass(frozen=True)
class BriefingItem:
    symbol: str
    action: str
    confidence: float
    verdict: str
    summary: str
    selection_rank: int = 0
    priority_score: Optional[float] = None
    exposure_penalty: float = 0.0
    rationale: tuple[str, ...] = ()
    winning_strategy: Optional[str] = None
    liquidity_score: Optional[float] = None
    regime: str = ""
    volume_ratio: Optional[float] = None
    activity_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ShortlistBriefingResponse:
    ok: bool
    source: str
    timeframe: str
    lookback_days: int
    headline: str
    items: tuple[BriefingItem, ...] = ()
    portfolio: Optional[PortfolioCritique] = None
    disclaimer: str = "Briefing is advisory preparation only."
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["items"] = [row.to_dict() for row in self.items]
        out["portfolio"] = self.portfolio.to_dict() if self.portfolio is not None else None
        return out


@dataclass(frozen=True)
class AllocationDecision:
    symbol: str
    action: str
    verdict: str
    selected: bool
    reason: str
    selection_rank: int = 0
    allocation_rank: int = 0
    allocation_weight: Optional[float] = None
    priority_score: Optional[float] = None
    allocation_score: Optional[float] = None
    exposure_penalty: float = 0.0
    critic_penalty: float = 0.0
    critic_note: str = ""
    exposure_key: str = ""
    winning_strategy: Optional[str] = None
    regime: str = ""
    risk_bucket: str = ""
    sizing_hint: str = ""
    research_target: str = ""
    research_priority_score: Optional[float] = None
    research_refreshed: bool = False
    research_nightly_reports: int = 0
    research_nightly_refreshed_reports: int = 0
    research_nightly_promoted_reports: int = 0
    research_nightly_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class PortfolioAllocationResponse:
    ok: bool
    source: str
    timeframe: str
    lookback_days: int
    headline: str
    max_positions: int
    items: tuple[AllocationDecision, ...] = ()
    selected_symbols: tuple[str, ...] = ()
    skipped_symbols: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["items"] = [row.to_dict() for row in self.items]
        return out


@dataclass(frozen=True)
class AgentRoleStatus:
    name: str
    ok: bool
    headline: str
    focus_symbols: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class MultiAgentWorkflowResponse:
    ok: bool
    source: str
    timeframe: str
    lookback_days: int
    headline: str
    roles: tuple[AgentRoleStatus, ...] = ()
    universe: Optional[MarketUniverseResponse] = None
    shortlist: Optional[ShortlistAnalysisResponse] = None
    briefing: Optional[ShortlistBriefingResponse] = None
    candidates: Optional[TrainingCandidateResponse] = None
    allocation: Optional[PortfolioAllocationResponse] = None
    research: Optional[TrainingResearchPlanResponse] = None
    nightly_alignment: Optional[NightlyAlignmentStatus] = None
    disclaimer: str = "Advisory-only multi-agent workflow summary."
    error: Optional[AdvisoryError] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["roles"] = [row.to_dict() for row in self.roles]
        out["universe"] = self.universe.to_dict() if self.universe is not None else None
        out["shortlist"] = self.shortlist.to_dict() if self.shortlist is not None else None
        out["briefing"] = self.briefing.to_dict() if self.briefing is not None else None
        out["candidates"] = self.candidates.to_dict() if self.candidates is not None else None
        out["allocation"] = self.allocation.to_dict() if self.allocation is not None else None
        out["research"] = self.research.to_dict() if self.research is not None else None
        out["nightly_alignment"] = (
            self.nightly_alignment.to_dict() if self.nightly_alignment is not None else None
        )
        return out


@dataclass(frozen=True)
class WorkflowSnapshotSummary:
    path: str
    source: Optional[str] = None
    timeframe: Optional[str] = None
    lookback_days: Optional[int] = None
    team_role_count: int = 0
    team_ok_role_count: int = 0
    team_headline: Optional[str] = None
    team_nightly_report_count: int = 0
    team_nightly_aligned_reports: int = 0
    team_nightly_enabled_reports: int = 0
    team_nightly_latest_target_mix: Optional[str] = None
    team_nightly_latest_execution_target: Optional[str] = None
    team_nightly_latest_execution_model_family: Optional[str] = None
    team_nightly_latest_execution_selection_source: Optional[str] = None
    team_nightly_latest_promotion_review_model_kind: Optional[str] = None
    team_nightly_latest_promotion_review_model_kinds: Optional[str] = None
    team_nightly_latest_refreshed_target_mix: Optional[str] = None
    team_nightly_recommended_target: Optional[str] = None
    team_nightly_posture: Optional[str] = None
    team_nightly_recommended_force_refresh: bool = False
    team_nightly_recent_window: int = 0
    team_nightly_recent_enabled: int = 0
    team_nightly_recent_aligned: int = 0
    team_nightly_recent_latest_status: Optional[str] = None
    team_nightly_recent_latest_basket_size: int = 0
    team_research_headline: Optional[str] = None
    team_research_selection_policy: Optional[str] = None
    team_research_refresh_target: Optional[str] = None
    team_research_discovery_summary: Optional[str] = None
    team_research_discovery_recommended_target: Optional[str] = None
    team_research_discovery_posture: Optional[str] = None
    team_research_strategy_posture: Optional[str] = None
    team_research_candidate_posture: Optional[str] = None
    team_research_nightly_posture: Optional[str] = None
    team_research_effective_posture: Optional[str] = None
    team_research_effective_target: Optional[str] = None
    team_research_refresh_urgency: Optional[str] = None
    team_research_refresh_urgency_target: Optional[str] = None
    team_research_refresh_urgency_score: Optional[float] = None
    team_research_refresh_urgency_rows: int = 0
    team_research_refresh_urgency_summary: Optional[str] = None
    team_research_follow_up_action: Optional[str] = None
    team_research_discovery_follow_up_target: Optional[str] = None
    team_research_discovery_follow_up_summary: Optional[str] = None
    team_research_discovery_follow_up_action: Optional[str] = None
    team_research_research_follow_up_target: Optional[str] = None
    team_research_research_follow_up_summary: Optional[str] = None
    team_research_research_follow_up_action: Optional[str] = None
    team_research_execution_follow_up_target: Optional[str] = None
    team_research_execution_follow_up_summary: Optional[str] = None
    team_research_execution_follow_up_action: Optional[str] = None
    team_research_target_mismatch: bool = False
    team_research_refresh_requested: bool = False
    team_research_refreshed_count: int = 0
    team_research_ml_count: int = 0
    team_research_rl_count: int = 0
    team_research_selected_target_mix: Optional[str] = None
    team_research_selected_regime_mix: Optional[str] = None
    team_research_scout_support_target: Optional[str] = None
    team_research_scout_support_summary: Optional[str] = None
    team_research_scout_supported_symbols: Optional[str] = None
    team_research_scout_overlap_selected_count: int = 0
    team_research_scout_volume_dense_selected_count: int = 0
    team_research_alignment_summary: Optional[str] = None
    team_research_alignment_target_mix: Optional[str] = None
    team_research_alignment_overlap_count: int = 0
    team_research_alignment_selected_count: int = 0
    team_discovery_refresh_source: Optional[str] = None
    team_discovery_refresh_timeframe: Optional[str] = None
    team_discovery_refresh_days: int = 0
    team_discovery_refresh_requested: bool = False
    team_discovery_refreshed_count: int = 0
    team_discovery_alignment_summary: Optional[str] = None
    team_discovery_overlap_symbols: Optional[str] = None
    team_discovery_alignment_overlap_count: int = 0
    team_discovery_alignment_compare_count: int = 0
    team_artifact_follow_up_action: Optional[str] = None
    team_artifact_follow_up_target: Optional[str] = None
    team_artifact_recovery_posture: Optional[str] = None
    universe_count: int = 0
    universe_adaptive_count: int = 0
    universe_top_adaptive_note: Optional[str] = None
    universe_scout_summary: Optional[str] = None
    universe_liquidity_symbols: Optional[str] = None
    universe_activity_symbols: Optional[str] = None
    universe_volume_dense_symbols: Optional[str] = None
    universe_overlap_symbols: Optional[str] = None
    universe_seed_source: Optional[str] = None
    universe_provider_summary: Optional[str] = None
    shortlist_count: int = 0
    briefing_candidates: int = 0
    allocation_selected: int = 0
    allocation_skipped: int = 0
    allocation_max_positions: int = 0
    allocation_research_alignment_enabled: bool = False
    allocation_research_target_mix: Optional[str] = None
    allocation_refreshed_research_target_mix: Optional[str] = None
    allocation_critic_enabled: bool = False
    allocation_regime_mix: Optional[str] = None
    allocation_strategy_mix: Optional[str] = None
    allocation_risk_mix: Optional[str] = None
    allocation_freshness_mix: Optional[str] = None
    allocation_target_balance_summary: Optional[str] = None
    allocation_critic_summary: Optional[str] = None
    ml_count: int = 0
    rl_count: int = 0
    research_count: int = 0
    research_ml_count: int = 0
    research_rl_count: int = 0
    research_selection_policy: Optional[str] = None
    research_refresh_target: Optional[str] = None
    research_discovery_summary: Optional[str] = None
    research_discovery_recommended_target: Optional[str] = None
    research_discovery_posture: Optional[str] = None
    research_volume_dense_symbols: Optional[str] = None
    research_strategy_posture: Optional[str] = None
    research_candidate_posture: Optional[str] = None
    research_nightly_posture: Optional[str] = None
    research_effective_posture: Optional[str] = None
    research_effective_target: Optional[str] = None
    research_refresh_urgency: Optional[str] = None
    research_refresh_urgency_target: Optional[str] = None
    research_refresh_urgency_score: Optional[float] = None
    research_refresh_urgency_rows: int = 0
    research_refresh_urgency_summary: Optional[str] = None
    research_follow_up_action: Optional[str] = None
    research_discovery_follow_up_target: Optional[str] = None
    research_discovery_follow_up_summary: Optional[str] = None
    research_discovery_follow_up_action: Optional[str] = None
    research_research_follow_up_target: Optional[str] = None
    research_research_follow_up_summary: Optional[str] = None
    research_research_follow_up_action: Optional[str] = None
    research_execution_follow_up_target: Optional[str] = None
    research_execution_follow_up_summary: Optional[str] = None
    research_execution_follow_up_action: Optional[str] = None
    research_target_mismatch: bool = False
    research_refresh_requested: bool = False
    research_refreshed_count: int = 0
    research_selected_target_mix: Optional[str] = None
    research_selected_regime_mix: Optional[str] = None
    research_scout_support_target: Optional[str] = None
    research_scout_support_summary: Optional[str] = None
    research_scout_supported_symbols: Optional[str] = None
    research_scout_overlap_selected_count: int = 0
    research_scout_volume_dense_selected_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class PromotionRecord:
    run_id: Optional[str] = None
    symbol: Optional[str] = None
    promoted_by: Optional[str] = None
    promoted_at: Optional[str] = None
    artifact_dir: Optional[str] = None
    model_kind: Optional[str] = None
    workflow_snapshot_path: Optional[str] = None
    workflow_snapshot: Optional[WorkflowSnapshotSummary] = None

    @classmethod
    def from_audit_row(cls, row: Optional[dict[str, Any]]) -> Optional[PromotionRecord]:
        if not row:
            return None
        return cls(
            run_id=row.get("run_id"),
            symbol=row.get("symbol"),
            promoted_by=row.get("promoted_by"),
            promoted_at=row.get("promoted_at") or row.get("ts"),
            artifact_dir=row.get("artifact_dir"),
            model_kind=row.get("model_kind"),
            workflow_snapshot_path=row.get("workflow_snapshot_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["workflow_snapshot"] = (
            self.workflow_snapshot.to_dict() if self.workflow_snapshot is not None else None
        )
        return out


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
class NightlyAlignmentRow:
    path: str
    overall_status: str = ""
    basket_size: int = 0
    alignment_enabled: bool = False
    execution_target: Optional[str] = None
    execution_model_family: Optional[str] = None
    execution_selection_source: Optional[str] = None
    target_mix: Optional[str] = None
    refreshed_target_mix: Optional[str] = None
    workflow_research_alignment_summary: Optional[str] = None
    workflow_research_alignment_target_mix: Optional[str] = None
    workflow_research_alignment_recommended_target: Optional[str] = None
    workflow_research_refresh_urgency_summary: Optional[str] = None
    workflow_research_refresh_urgency_target: Optional[str] = None
    workflow_research_follow_up_action: Optional[str] = None
    workflow_research_discovery_follow_up_target: Optional[str] = None
    workflow_research_discovery_follow_up_summary: Optional[str] = None
    workflow_research_discovery_follow_up_action: Optional[str] = None
    workflow_research_research_follow_up_target: Optional[str] = None
    workflow_research_research_follow_up_summary: Optional[str] = None
    workflow_research_research_follow_up_action: Optional[str] = None
    workflow_research_execution_follow_up_target: Optional[str] = None
    workflow_research_execution_follow_up_summary: Optional[str] = None
    workflow_research_execution_follow_up_action: Optional[str] = None
    workflow_research_scout_support_target: Optional[str] = None
    workflow_discovery_alignment_summary: Optional[str] = None
    workflow_discovery_overlap_symbols: Optional[str] = None
    workflow_discovery_alignment_overlap_count: int = 0
    workflow_discovery_alignment_compare_count: int = 0
    training_research_discovery_preferred_count: int = 0
    training_research_discovery_regime_mix: Optional[str] = None
    training_research_discovery_summary: Optional[str] = None
    training_research_scout_support_target: Optional[str] = None
    promotion_review_model_kind: Optional[str] = None
    promotion_review_model_kinds: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class NightlyAlignmentStatus:
    report_count: int = 0
    aligned_reports: int = 0
    enabled_reports: int = 0
    latest_execution_target: Optional[str] = None
    latest_execution_model_family: Optional[str] = None
    latest_execution_selection_source: Optional[str] = None
    latest_target_mix: Optional[str] = None
    refreshed_aligned_reports: int = 0
    latest_refreshed_target_mix: Optional[str] = None
    latest_workflow_research_alignment_summary: Optional[str] = None
    latest_workflow_research_alignment_target_mix: Optional[str] = None
    latest_workflow_research_recommended_target: Optional[str] = None
    latest_workflow_research_refresh_urgency_summary: Optional[str] = None
    latest_workflow_research_refresh_urgency_target: Optional[str] = None
    latest_workflow_research_follow_up_action: Optional[str] = None
    latest_workflow_research_discovery_follow_up_target: Optional[str] = None
    latest_workflow_research_discovery_follow_up_summary: Optional[str] = None
    latest_workflow_research_discovery_follow_up_action: Optional[str] = None
    latest_workflow_research_research_follow_up_target: Optional[str] = None
    latest_workflow_research_research_follow_up_summary: Optional[str] = None
    latest_workflow_research_research_follow_up_action: Optional[str] = None
    latest_workflow_research_execution_follow_up_target: Optional[str] = None
    latest_workflow_research_execution_follow_up_summary: Optional[str] = None
    latest_workflow_research_execution_follow_up_action: Optional[str] = None
    latest_workflow_research_scout_support_target: Optional[str] = None
    latest_workflow_discovery_alignment_summary: Optional[str] = None
    latest_workflow_discovery_overlap_symbols: Optional[str] = None
    latest_workflow_discovery_warning: bool = False
    latest_training_research_discovery_preferred_count: int = 0
    latest_training_research_discovery_regime_mix: Optional[str] = None
    latest_training_research_discovery_summary: Optional[str] = None
    latest_training_research_discovery_recommended_target: Optional[str] = None
    latest_training_research_scout_support_target: Optional[str] = None
    latest_promotion_review_model_kind: Optional[str] = None
    latest_promotion_review_model_kinds: tuple[str, ...] = ()
    nightly_posture: Optional[str] = None
    recommended_action: Optional[str] = None
    recommended_refresh_target: Optional[str] = None
    effective_refresh_target: Optional[str] = None
    recommended_force_refresh: bool = False
    workflow_target_mismatch: bool = False
    recommended_cli_command: Optional[str] = None
    recommended_discovery_action: Optional[str] = None
    recommended_discovery_cli_command: Optional[str] = None
    recent_trend_window: int = 0
    recent_trend_enabled: int = 0
    recent_trend_aligned: int = 0
    recent_trend_latest_status: Optional[str] = None
    recent_trend_latest_basket_size: int = 0
    recent_rows: tuple[NightlyAlignmentRow, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["recent_rows"] = [row.to_dict() for row in self.recent_rows]
        return out


@dataclass(frozen=True)
class CrossArtifactAlignmentSummary:
    discovery_research_aligned: Optional[bool] = None
    research_execution_aligned: Optional[bool] = None
    promotion_research_aligned: Optional[bool] = None
    refreshed_basket_aligned: Optional[bool] = None
    overall_status: str = "unknown"
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class DualPromotionReviewSummary:
    both_present: bool = False
    ml_present: bool = False
    rl_present: bool = False
    nightly_review_kinds: tuple[str, ...] = ()
    execution_target: Optional[str] = None
    effective_refresh_target: Optional[str] = None
    posture: str = "unknown"
    summary: str = ""
    warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class NightlyArtifactLinkageSummary:
    overall_status: str = "unknown"
    replay_status: Optional[str] = None
    run_identifier: Optional[str] = None
    nightly_report_path: Optional[str] = None
    training_candidates_path: Optional[str] = None
    training_research_path: Optional[str] = None
    workflow_snapshot_path: Optional[str] = None
    promotion_review_paths: tuple[str, ...] = ()
    artifact_presence: dict[str, bool] = field(default_factory=dict)
    missing_artifacts: tuple[str, ...] = ()
    broken_links: tuple[str, ...] = ()
    linkage_warnings: tuple[str, ...] = ()
    summary: str = ""
    warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


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
    nightly_alignment: NightlyAlignmentStatus = field(default_factory=NightlyAlignmentStatus)

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["recent_decisions"] = [row.to_dict() for row in self.recent_decisions]
        out["learning_summary"] = self.learning_summary.to_dict()
        out["nightly_alignment"] = self.nightly_alignment.to_dict()
        return out


@dataclass(frozen=True)
class RuntimeReadinessRow:
    name: str
    severity: str
    category: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class RuntimeReadinessSummary:
    overall_posture: str
    nightly_posture: str
    summary: str
    blockers: tuple[RuntimeReadinessRow, ...] = ()
    warnings: tuple[RuntimeReadinessRow, ...] = ()
    deterministic_fallback_ok: bool = True
    agentic_enabled: bool = False
    ml_advisory_ready: bool = False
    rl_advisory_ready: bool = False
    scaling: Optional["ScalingPostureSummary"] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["blockers"] = [row.to_dict() for row in self.blockers]
        out["warnings"] = [row.to_dict() for row in self.warnings]
        out["scaling"] = self.scaling.to_dict() if self.scaling is not None else None
        return out


@dataclass(frozen=True)
class ComputeBudgetSummary:
    physical_ram_gb: Optional[float]
    cpu_count: Optional[int]
    machine_class: str
    rl_n_envs: int
    rl_use_subproc: bool
    recommended_max_symbols: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ScalingPostureSummary:
    basket_posture: str
    basket_count: int
    recommended_max_symbols: int
    within_recommended: bool
    compute: ComputeBudgetSummary
    summary: str
    blockers: tuple[RuntimeReadinessRow, ...] = ()
    warnings: tuple[RuntimeReadinessRow, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["compute"] = self.compute.to_dict()
        out["blockers"] = [row.to_dict() for row in self.blockers]
        out["warnings"] = [row.to_dict() for row in self.warnings]
        return out


@dataclass(frozen=True)
class ArtifactRetentionSummary:
    enabled: bool
    retained_count: int
    pruned_count: int
    policy: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ActivationBlockerRow:
    code: str
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class LaneActivationSummary:
    lane: str
    stage: str
    active: bool
    summary: str
    blockers: tuple[ActivationBlockerRow, ...] = ()
    recommended_command: Optional[str] = None
    candidate_run_id: Optional[str] = None
    live_pointer: Optional[str] = None
    artifact_dir: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out = _to_jsonable(self)
        out["blockers"] = [row.to_dict() for row in self.blockers]
        return out


@dataclass(frozen=True)
class ModelActivationSummary:
    ml: LaneActivationSummary
    rl: LaneActivationSummary
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ml": self.ml.to_dict(),
            "rl": self.rl.to_dict(),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ModelHealthResponse:
    registry_enabled: bool
    promotion_required: bool
    rl: RlModelStatus
    ml: MlModelStatus
    regime: RegimeModelStatus
    agentic: AgenticStatusSummary
    runtime_readiness: Optional[RuntimeReadinessSummary] = None
    activation: Optional[ModelActivationSummary] = None
    scaling: Optional[ScalingPostureSummary] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_enabled": self.registry_enabled,
            "promotion_required": self.promotion_required,
            "rl": self.rl.to_dict(),
            "ml": self.ml.to_dict(),
            "regime": self.regime.to_dict(),
            "agentic": self.agentic.to_dict(),
            "runtime_readiness": (
                self.runtime_readiness.to_dict() if self.runtime_readiness is not None else None
            ),
            "activation": self.activation.to_dict() if self.activation is not None else None,
            "scaling": self.scaling.to_dict() if self.scaling is not None else None,
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
