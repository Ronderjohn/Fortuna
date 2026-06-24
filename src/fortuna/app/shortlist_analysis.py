"""Analyze a ranked market universe and attach a conservative critique."""

from __future__ import annotations

from typing import Any, Callable, Optional

from fortuna.agentic.contracts import (
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    RiskCritique,
    ShortlistAnalysisItem,
    ShortlistAnalysisResponse,
)
from fortuna.app.advisory_service import analyze_instrument
from fortuna.app.exposure_ranking import advisory_priority_score, rank_with_exposure_penalties
from fortuna.app.market_universe import build_market_universe
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRegistry


def analyze_market_shortlist(
    *,
    settings: Settings,
    engine_factory: Optional[Callable[[], Any]] = None,
    registry: Optional[InstrumentRegistry] = None,
    universe_limit: int = 10,
    analysis_limit: int = 5,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
) -> ShortlistAnalysisResponse:
    reg = registry or InstrumentRegistry()
    reg.ensure_loaded()
    universe = build_market_universe(
        settings=settings,
        registry=reg,
        limit=max(analysis_limit, universe_limit),
        timeframe="1d",
        days=max(days, 20),
        source=source,
    )
    if not universe.ok:
        return ShortlistAnalysisResponse(
            ok=False,
            source=source,
            timeframe=timeframe,
            lookback_days=int(days),
            error=universe.error,
        )

    items: list[ShortlistAnalysisItem] = []
    discovery_overlap, activity_ranked = _discovery_alignment_sets(universe)
    for candidate in universe.candidates[: max(1, int(analysis_limit))]:
        request = InstrumentAnalysisRequest(symbol=candidate.symbol, timeframe=timeframe, days=days)
        result = analyze_instrument(
            request=request,
            settings=settings,
            engine_factory=engine_factory,
            registry=reg,
        )
        items.append(
            _to_shortlist_item(
                candidate,
                result,
                settings=settings,
                discovery_overlap=discovery_overlap,
                activity_ranked=activity_ranked,
            )
        )

    ranked = _rank_shortlist_items(items)
    return ShortlistAnalysisResponse(
        ok=True,
        source=universe.source,
        timeframe=timeframe,
        lookback_days=int(days),
        scout_liquidity_symbols=tuple(getattr(universe, "scout_liquidity_symbols", ()) or ()),
        scout_activity_symbols=tuple(getattr(universe, "scout_activity_symbols", ()) or ()),
        scout_volume_dense_symbols=tuple(
            getattr(universe, "scout_volume_dense_symbols", ()) or ()
        ),
        scout_overlap_symbols=tuple(getattr(universe, "scout_overlap_symbols", ()) or ()),
        fundamentals_overlay_summary=getattr(universe, "fundamentals_overlay_summary", None),
        fundamentals_overlay_diagnostics=tuple(
            getattr(universe, "fundamentals_overlay_diagnostics", ()) or ()
        ),
        items=tuple(ranked),
    )


def critique_analysis(response: InstrumentAnalysisResponse) -> RiskCritique:
    if not response.ok or response.decision is None:
        message = response.error.message if response.error is not None else "analysis unavailable"
        return RiskCritique(
            severity="high",
            summary=message,
            concerns=("No tradable advisory result available",),
            verdict="avoid",
        )

    decision = response.decision
    concerns: list[str] = []
    supports: list[str] = list(decision.reasons[:2])
    if decision.confidence < 0.55:
        concerns.append("Low confidence advisory")
    if not decision.reasons:
        concerns.append("Sparse rationale")
    if decision.action == "DO_NOT_ENTER":
        concerns.append("Current setup explicitly not attractive enough")
    if decision.action == "HOLD":
        concerns.append("No new entry edge detected")
    if decision.risk_notes:
        concerns.extend(decision.risk_notes[:2])
    if response.winning_strategy is None:
        concerns.append("No winning deterministic strategy surfaced")
    if not response.signals:
        concerns.append("No actionable signal rows")

    if (
        decision.action in {"BUY", "SELL"}
        and decision.confidence >= 0.65
        and not decision.risk_notes
    ):
        severity = "low"
        verdict = "candidate"
        summary = f"{decision.action} candidate with relatively clean support"
    elif decision.action in {"BUY", "SELL"}:
        severity = "medium"
        verdict = "watch"
        summary = f"{decision.action} setup exists but needs caution"
    else:
        severity = "high"
        verdict = "avoid"
        summary = f"{decision.action} is not an active entry candidate"

    return RiskCritique(
        severity=severity,
        summary=summary,
        concerns=tuple(dict.fromkeys(concerns)),
        supports=tuple(dict.fromkeys(supports)),
        verdict=verdict,
    )


def _to_shortlist_item(
    candidate,
    response: InstrumentAnalysisResponse,
    *,
    settings: Settings,
    discovery_overlap: set[str],
    activity_ranked: set[str],
) -> ShortlistAnalysisItem:
    if not response.ok:
        return ShortlistAnalysisItem(
            symbol=candidate.symbol,
            liquidity_score=candidate.liquidity_score,
            trend_pct=candidate.trend_pct,
            regime=candidate.regime,
            volume_ratio=candidate.volume_ratio,
            activity_score=candidate.activity_score,
            sector=getattr(candidate, "sector", "") or "",
            market_cap_bucket=getattr(candidate, "market_cap_bucket", "") or "",
            operator_quality_score=getattr(candidate, "operator_quality_score", None),
            fundamentals_overlay_adjustment=getattr(
                candidate, "fundamentals_overlay_adjustment", None
            ),
            priority_score=0.0,
            source=candidate.source,
            critique=critique_analysis(response),
            error=response.error,
        )
    critique = _apply_discovery_posture(
        critique_analysis(response),
        candidate=candidate,
        discovery_overlap=discovery_overlap,
        activity_ranked=activity_ranked,
    )
    priority_score = advisory_priority_score(
        confidence=float(response.decision.confidence),
        verdict=critique.verdict,
        liquidity_score=candidate.liquidity_score,
        action=response.decision.action,
    )
    priority_score = _apply_discovery_priority_adjustment(
        priority_score,
        candidate=candidate,
        settings=settings,
        discovery_overlap=discovery_overlap,
        activity_ranked=activity_ranked,
    )
    return ShortlistAnalysisItem(
        symbol=candidate.symbol,
        instrument=response.instrument,
        decision=response.decision,
        winning_strategy=response.winning_strategy,
        liquidity_score=candidate.liquidity_score,
        trend_pct=candidate.trend_pct,
        regime=candidate.regime,
        volume_ratio=candidate.volume_ratio,
        activity_score=candidate.activity_score,
        sector=getattr(candidate, "sector", "") or "",
        market_cap_bucket=getattr(candidate, "market_cap_bucket", "") or "",
        operator_quality_score=getattr(candidate, "operator_quality_score", None),
        fundamentals_overlay_adjustment=getattr(candidate, "fundamentals_overlay_adjustment", None),
        priority_score=priority_score,
        critique=critique,
        source=candidate.source,
        error=response.error,
    )


def _rank_shortlist_items(items: list[ShortlistAnalysisItem]) -> tuple[ShortlistAnalysisItem, ...]:
    ranked = rank_with_exposure_penalties(
        items,
        symbol_of=lambda row: row.symbol,
        action_of=lambda row: row.decision.action if row.decision is not None else "",
        base_score_of=lambda row: float(row.priority_score or 0.0),
        tie_breaker_of=lambda row: (
            0 if row.critique and row.critique.verdict == "candidate" else 1,
            row.symbol,
        ),
    )
    return tuple(
        ShortlistAnalysisItem(
            symbol=result.item.symbol,
            instrument=result.item.instrument,
            decision=result.item.decision,
            winning_strategy=result.item.winning_strategy,
            liquidity_score=result.item.liquidity_score,
            trend_pct=result.item.trend_pct,
            regime=result.item.regime,
            volume_ratio=result.item.volume_ratio,
            activity_score=result.item.activity_score,
            sector=result.item.sector,
            market_cap_bucket=result.item.market_cap_bucket,
            operator_quality_score=result.item.operator_quality_score,
            fundamentals_overlay_adjustment=result.item.fundamentals_overlay_adjustment,
            selection_rank=idx,
            priority_score=result.adjusted_score,
            exposure_penalty=result.exposure_penalty,
            critique=result.item.critique,
            source=result.item.source,
            error=result.item.error,
        )
        for idx, result in enumerate(ranked, start=1)
    )


def _discovery_alignment_sets(universe) -> tuple[set[str], set[str]]:
    candidates = tuple(getattr(universe, "candidates", ()) or ())
    if not candidates:
        return set(), set()
    liquidity_ranked = sorted(
        candidates,
        key=lambda row: (
            -(float(getattr(row, "liquidity_score", 0.0) or 0.0)),
            -(float(getattr(row, "avg_turnover", 0.0) or 0.0)),
            row.symbol,
        ),
    )
    activity_ranked_rows = [
        row
        for row in candidates
        if (
            getattr(row, "activity_score", None) is not None
            or getattr(row, "trend_pct", None) is not None
            or getattr(row, "volume_ratio", None) is not None
            or str(getattr(row, "regime", "") or "").strip()
        )
    ]
    activity_ranked_rows = sorted(
        activity_ranked_rows,
        key=lambda row: (
            -(float(getattr(row, "activity_score", 0.0) or 0.0)),
            -(abs(float(getattr(row, "trend_pct", 0.0) or 0.0))),
            row.symbol,
        ),
    )
    compare_count = min(3, len(liquidity_ranked), len(activity_ranked_rows))
    if compare_count <= 0:
        return set(), set()
    activity_top = {row.symbol for row in activity_ranked_rows[:compare_count]}
    overlap = {
        row.symbol for row in liquidity_ranked[:compare_count] if row.symbol in activity_top
    }
    return overlap, activity_top


def _apply_discovery_posture(
    critique: RiskCritique,
    *,
    candidate,
    discovery_overlap: set[str],
    activity_ranked: set[str],
) -> RiskCritique:
    supports = list(critique.supports)
    concerns = list(critique.concerns)
    if candidate.symbol in discovery_overlap:
        supports.append("Discovery overlap backed by liquidity and activity scouts")
    elif candidate.symbol in activity_ranked:
        concerns.append("Discovery support is not reinforced by both scouts")
    else:
        concerns.append("Discovery support is liquidity-led without activity confirmation")
    return RiskCritique(
        severity=critique.severity,
        summary=critique.summary,
        concerns=tuple(dict.fromkeys(concerns)),
        supports=tuple(dict.fromkeys(supports)),
        verdict=critique.verdict,
    )


def _apply_discovery_priority_adjustment(
    priority_score: float,
    *,
    candidate,
    settings: Settings,
    discovery_overlap: set[str],
    activity_ranked: set[str],
) -> float:
    if not bool(getattr(settings, "shortlist_discovery_alignment_enabled", True)):
        return priority_score
    if candidate.symbol in discovery_overlap:
        return round(
            priority_score
            + float(getattr(settings, "shortlist_discovery_alignment_boost", 0.05)),
            4,
        )
    if candidate.symbol in activity_ranked:
        return round(
            priority_score
            - float(getattr(settings, "shortlist_discovery_alignment_penalty", 0.03)),
            4,
        )
    return round(
        priority_score
        - float(getattr(settings, "shortlist_discovery_alignment_penalty", 0.03)),
        4,
    )
