"""Typed portfolio-allocation layer over shortlist briefing output."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fortuna.agentic.contracts import (
    AllocationDecision,
    PortfolioAllocationResponse,
)
from fortuna.agentic.learning import LearningExample
from fortuna.app.shortlist_briefing import build_shortlist_briefing


def build_portfolio_allocation(
    *,
    settings,
    engine_factory=None,
    registry=None,
    universe_limit: int = 10,
    analysis_limit: int = 5,
    max_positions: int = 3,
    max_per_exposure: int = 1,
    max_same_side: int = 2,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
) -> PortfolioAllocationResponse:
    briefing = build_shortlist_briefing(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    if not briefing.ok:
        return PortfolioAllocationResponse(
            ok=False,
            source=source,
            timeframe=timeframe,
            lookback_days=days,
            headline="Portfolio allocation unavailable",
            max_positions=max_positions,
            error=briefing.error,
        )

    account_context = _account_context(engine_factory, briefing.items)
    learning_biases = _learning_biases(engine_factory)
    discovery_alignment = _discovery_alignment(briefing.items)
    research_alignment = _training_research_alignment(
        settings=settings,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    ranked_items = _rank_allocation_items(
        briefing.items,
        settings=settings,
        learning_biases=learning_biases,
        research_alignment=research_alignment,
    )

    selected: list[AllocationDecision] = []
    skipped: list[AllocationDecision] = []
    exposure_counts = dict(account_context.exposure_counts)
    side_counts = dict(account_context.side_counts)
    regime_counts: dict[str, int] = defaultdict(int)
    risk_counts: dict[str, int] = defaultdict(int)
    max_high_risk = max(
        0,
        int(getattr(settings, "portfolio_allocator_max_high_risk_positions", 1)),
    )
    max_per_regime = max(0, int(getattr(settings, "portfolio_allocator_max_per_regime", 2)))

    for item in ranked_items:
        reason = _eligibility_reason(item.verdict, item.action)
        exposure_key = _exposure_key(item.symbol)
        learning_note = _learning_reason_suffix(item.action, exposure_key, learning_biases)
        regime_key = str(getattr(item, "regime", "") or "").upper()
        risk_bucket = _risk_bucket(item)
        research_support = _resolve_research_support(
            item.symbol,
            research_alignment=research_alignment,
        )
        critic_penalty, critic_note, critic_block = _portfolio_critic_assessment(
            item,
            selected=selected,
            account_context=account_context,
            risk_bucket=risk_bucket,
            discovery_alignment=discovery_alignment,
            research_support=research_support,
            settings=settings,
        )
        if reason is not None:
            skipped.append(
                AllocationDecision(
                    symbol=item.symbol,
                    action=item.action,
                    verdict=item.verdict,
                    selected=False,
                    reason=_join_reason(reason, learning_note),
                    selection_rank=item.selection_rank,
                    priority_score=item.priority_score,
                    allocation_score=_scored_allocation(item.priority_score, critic_penalty),
                    exposure_penalty=item.exposure_penalty,
                    critic_penalty=critic_penalty,
                    critic_note=critic_note,
                    exposure_key=exposure_key,
                    winning_strategy=str(getattr(item, "winning_strategy", "") or "") or None,
                    regime=str(getattr(item, "regime", "") or ""),
                    risk_bucket=risk_bucket,
                    sizing_hint="blocked",
                    research_target=research_support.target if research_support is not None else "",
                    research_priority_score=(
                        research_support.priority_score if research_support is not None else None
                    ),
                    research_refreshed=(
                        research_support.refreshed if research_support is not None else False
                    ),
                    research_nightly_reports=(
                        research_support.nightly_reports if research_support is not None else 0
                    ),
                    research_nightly_refreshed_reports=(
                        research_support.nightly_refreshed_reports
                        if research_support is not None
                        else 0
                    ),
                    research_nightly_promoted_reports=(
                        research_support.nightly_promoted_reports
                        if research_support is not None
                        else 0
                    ),
                    research_nightly_score=(
                        research_support.nightly_score if research_support is not None else 0.0
                    ),
                )
            )
            continue

        side = str(item.action or "").upper()
        if len(selected) >= max_positions:
            reason = f"max_positions={max_positions} reached"
        elif exposure_counts.get(exposure_key, 0) >= max_per_exposure:
            reason = f"exposure limit reached for {exposure_key}"
        elif side_counts.get(side, 0) >= max_same_side:
            reason = f"same-side limit reached for {side}"
        elif risk_bucket == "high" and risk_counts.get("high", 0) >= max_high_risk:
            reason = f"high-risk limit reached ({max_high_risk})"
        elif regime_key and regime_counts.get(regime_key, 0) >= max_per_regime:
            reason = f"regime limit reached for {regime_key}"
        elif critic_block:
            reason = critic_note or "portfolio critic blocked setup"
        else:
            reason = ""

        if reason:
            skipped.append(
                AllocationDecision(
                    symbol=item.symbol,
                    action=item.action,
                    verdict=item.verdict,
                    selected=False,
                    reason=_join_reason(reason, learning_note),
                    selection_rank=item.selection_rank,
                    priority_score=item.priority_score,
                    allocation_score=_scored_allocation(item.priority_score, critic_penalty),
                    exposure_penalty=item.exposure_penalty,
                    critic_penalty=critic_penalty,
                    critic_note=critic_note,
                    exposure_key=exposure_key,
                    winning_strategy=str(getattr(item, "winning_strategy", "") or "") or None,
                    regime=str(getattr(item, "regime", "") or ""),
                    risk_bucket=risk_bucket,
                    sizing_hint="blocked",
                    research_target=research_support.target if research_support is not None else "",
                    research_priority_score=(
                        research_support.priority_score if research_support is not None else None
                    ),
                    research_refreshed=(
                        research_support.refreshed if research_support is not None else False
                    ),
                    research_nightly_reports=(
                        research_support.nightly_reports if research_support is not None else 0
                    ),
                    research_nightly_refreshed_reports=(
                        research_support.nightly_refreshed_reports
                        if research_support is not None
                        else 0
                    ),
                    research_nightly_promoted_reports=(
                        research_support.nightly_promoted_reports
                        if research_support is not None
                        else 0
                    ),
                    research_nightly_score=(
                        research_support.nightly_score if research_support is not None else 0.0
                    ),
                )
            )
            continue

        exposure_counts[exposure_key] = exposure_counts.get(exposure_key, 0) + 1
        side_counts[side] = side_counts.get(side, 0) + 1
        if regime_key:
            regime_counts[regime_key] = regime_counts.get(regime_key, 0) + 1
        risk_counts[risk_bucket] = risk_counts.get(risk_bucket, 0) + 1
        selected.append(
            AllocationDecision(
                symbol=item.symbol,
                action=item.action,
                verdict=item.verdict,
                selected=True,
                reason=_join_reason(
                    _selected_reason(critic_note),
                    learning_note,
                ),
                selection_rank=item.selection_rank,
                allocation_rank=len(selected) + 1,
                priority_score=item.priority_score,
                allocation_score=_scored_allocation(item.priority_score, critic_penalty),
                exposure_penalty=item.exposure_penalty,
                critic_penalty=critic_penalty,
                critic_note=critic_note,
                exposure_key=exposure_key,
                winning_strategy=str(getattr(item, "winning_strategy", "") or "") or None,
                regime=str(getattr(item, "regime", "") or ""),
                risk_bucket=risk_bucket,
                research_target=research_support.target if research_support is not None else "",
                research_priority_score=(
                    research_support.priority_score if research_support is not None else None
                ),
                research_refreshed=(
                    research_support.refreshed if research_support is not None else False
                ),
                research_nightly_reports=(
                    research_support.nightly_reports if research_support is not None else 0
                ),
                research_nightly_refreshed_reports=(
                    research_support.nightly_refreshed_reports
                    if research_support is not None
                    else 0
                ),
                research_nightly_promoted_reports=(
                    research_support.nightly_promoted_reports
                    if research_support is not None
                    else 0
                ),
                research_nightly_score=(
                    research_support.nightly_score if research_support is not None else 0.0
                ),
            )
        )

    weighted_selected = _weighted_selected(selected, settings=settings)
    items = tuple(weighted_selected + skipped)
    selected_symbols = tuple(item.symbol for item in weighted_selected)
    skipped_symbols = tuple(item.symbol for item in skipped)
    notes = [
        f"max_positions={max_positions}",
        f"max_per_exposure={max_per_exposure}",
        f"max_same_side={max_same_side}",
        f"max_high_risk_positions={max_high_risk}",
        f"max_per_regime={max_per_regime}",
    ]
    if bool(getattr(settings, "portfolio_allocator_context_weighting_enabled", True)):
        notes.append("allocator_context_weighting=enabled")
    if bool(getattr(settings, "portfolio_allocator_critic_enabled", True)):
        notes.append("portfolio_critic=enabled")
    if bool(getattr(settings, "portfolio_allocator_discovery_alignment_enabled", True)):
        notes.append("portfolio_discovery_alignment=enabled")
    if bool(getattr(settings, "portfolio_allocator_research_alignment_enabled", True)):
        notes.append("portfolio_research_alignment=enabled")
    if bool(getattr(settings, "portfolio_allocator_freshness_preference_enabled", True)):
        notes.append("portfolio_freshness_preference=enabled")
    if bool(getattr(settings, "portfolio_allocator_same_target_balance_enabled", True)):
        notes.append("portfolio_same_target_balance=enabled")
    notes.extend(account_context.notes)
    notes.extend(_learning_notes(learning_biases))
    notes.extend(discovery_alignment.notes)
    notes.extend(research_alignment.notes)
    notes.extend(_allocation_mix_notes(weighted_selected))
    if briefing.portfolio is not None:
        notes.extend(note.message for note in briefing.portfolio.notes[:3])
    headline = (
        f"{len(weighted_selected)} selected, {len(skipped)} skipped under portfolio limits"
        if items
        else "No allocatable setups found"
    )
    return PortfolioAllocationResponse(
        ok=True,
        source=briefing.source,
        timeframe=timeframe,
        lookback_days=days,
        headline=headline,
        max_positions=max_positions,
        items=items,
        selected_symbols=selected_symbols,
        skipped_symbols=skipped_symbols,
        notes=tuple(notes),
    )


def _eligibility_reason(verdict: str, action: str) -> Optional[str]:
    verdict_text = str(verdict or "").strip().lower()
    action_text = str(action or "").strip().upper()
    if action_text not in {"BUY", "SELL"}:
        return f"action {action_text or '—'} is not allocatable"
    if verdict_text not in {"candidate", "watch"}:
        return f"verdict {verdict_text or '—'} is not allocatable"
    return None


def _exposure_key(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    return text.split(".", 1)[0]


@dataclass(frozen=True)
class _AccountContext:
    exposure_counts: dict[str, int]
    side_counts: dict[str, int]
    exposure_side_counts: dict[tuple[str, str], int]
    open_regime_side_counts: dict[tuple[str, str], int]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _LearningBias:
    avg_realized_pnl_pct: float
    paper_closed_rows: int
    score_adjustment: float


@dataclass(frozen=True)
class _ResearchSupport:
    target: str
    selection_rank: int
    priority_score: float
    refreshed: bool = False
    nightly_reports: int = 0
    nightly_refreshed_reports: int = 0
    nightly_promoted_reports: int = 0
    nightly_score: float = 0.0


@dataclass(frozen=True)
class _NightlyResearchFeedback:
    research_reports: int
    refreshed_reports: int
    promoted_reports: int
    executed_reports: int
    executed_refreshed_reports: int


@dataclass(frozen=True)
class _ResearchAlignment:
    by_symbol: dict[str, _ResearchSupport]
    by_exposure: dict[str, _ResearchSupport]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _DiscoveryAlignment:
    overlap_symbols: frozenset[str]
    notes: tuple[str, ...] = ()


def _empty_account_context() -> _AccountContext:
    return _AccountContext(
        exposure_counts={},
        side_counts={},
        exposure_side_counts={},
        open_regime_side_counts={},
    )


def _briefing_regime_by_symbol(briefing_items) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in briefing_items or ():
        symbol = str(getattr(item, "symbol", "") or "").upper()
        regime = str(getattr(item, "regime", "") or "").upper()
        if symbol and regime:
            out[symbol] = regime
            exposure_key = _exposure_key(symbol)
            if exposure_key:
                out.setdefault(exposure_key, regime)
    return out


def _account_context(engine_factory, briefing_items=()) -> _AccountContext:
    if engine_factory is None:
        return _empty_account_context()
    try:
        engine = engine_factory()
    except Exception:
        return _empty_account_context()
    account = getattr(engine, "live_account", None)
    if account is None:
        return _empty_account_context()
    try:
        positions = getattr(account, "positions", {}) or {}
    except Exception:
        positions = {}
    regime_by_symbol = _briefing_regime_by_symbol(briefing_items)
    exposure_counts: dict[str, int] = defaultdict(int)
    side_counts: dict[str, int] = defaultdict(int)
    exposure_side_counts: dict[tuple[str, str], int] = defaultdict(int)
    open_regime_side_counts: dict[tuple[str, str], int] = defaultdict(int)
    open_symbols: list[str] = []
    for symbol, pos in positions.items():
        sym = str(symbol)
        open_symbols.append(sym)
        exposure_key = _exposure_key(sym)
        exposure_counts[exposure_key] += 1
        side = str(getattr(getattr(pos, "side", None), "value", getattr(pos, "side", ""))).upper()
        if side:
            side_counts[side] += 1
            if exposure_key:
                exposure_side_counts[(exposure_key, side)] += 1
            regime = regime_by_symbol.get(str(sym).upper(), "") or regime_by_symbol.get(
                exposure_key,
                "",
            )
            if regime:
                open_regime_side_counts[(regime, side)] += 1
    notes: list[str] = []
    if open_symbols:
        notes.append(f"existing_open_positions={len(open_symbols)}")
    if exposure_side_counts:
        notes.append(
            "open_side_exposures: "
            + ", ".join(
                f"{side}:{exposure}={count}"
                for (exposure, side), count in sorted(exposure_side_counts.items())
            )
        )
    if open_regime_side_counts:
        notes.append(
            "open_regime_side_exposures: "
            + ", ".join(
                f"{side}:{regime}={count}"
                for (regime, side), count in sorted(open_regime_side_counts.items())
            )
        )
    return _AccountContext(
        exposure_counts=dict(exposure_counts),
        side_counts=dict(side_counts),
        exposure_side_counts=dict(exposure_side_counts),
        open_regime_side_counts=dict(open_regime_side_counts),
        notes=tuple(notes),
    )


def _learning_biases(engine_factory) -> dict[tuple[str, str], _LearningBias]:
    if engine_factory is None:
        return {}
    try:
        engine = engine_factory()
    except Exception:
        return {}
    store = getattr(engine, "_agentic_learning_store", None)
    if store is None:
        return {}
    try:
        rows = list(store.read_all(limit=100))
    except Exception:
        return {}
    grouped: dict[tuple[str, str], list[LearningExample]] = defaultdict(list)
    for row in rows:
        if not row.outcome.paper_closed:
            continue
        if row.outcome.realized_pnl_pct is None:
            continue
        grouped[(str(row.action or "").upper(), _exposure_key(row.symbol))].append(row)
    out: dict[tuple[str, str], _LearningBias] = {}
    for key, bucket in grouped.items():
        realized = [float(row.outcome.realized_pnl_pct or 0.0) for row in bucket]
        if not realized:
            continue
        avg_realized = sum(realized) / len(realized)
        score_adjustment = _learning_score_adjustment(avg_realized, len(realized))
        out[key] = _LearningBias(
            avg_realized_pnl_pct=round(avg_realized, 4),
            paper_closed_rows=len(realized),
            score_adjustment=score_adjustment,
        )
    return out


def _discovery_alignment(items) -> _DiscoveryAlignment:
    base_items = tuple(items or ())
    if not base_items:
        return _DiscoveryAlignment(overlap_symbols=frozenset())
    liquidity_ranked = sorted(
        base_items,
        key=lambda row: (
            -(float(getattr(row, "liquidity_score", 0.0) or 0.0)),
            row.symbol,
        ),
    )
    activity_ranked = [
        row
        for row in base_items
        if (
            getattr(row, "activity_score", None) is not None
            or getattr(row, "volume_ratio", None) is not None
            or str(getattr(row, "regime", "") or "").strip()
        )
    ]
    activity_ranked = sorted(
        activity_ranked,
        key=lambda row: (
            -(float(getattr(row, "activity_score", 0.0) or 0.0)),
            -(float(getattr(row, "volume_ratio", 0.0) or 0.0)),
            row.symbol,
        ),
    )
    compare_count = min(3, len(liquidity_ranked), len(activity_ranked))
    if compare_count <= 0:
        return _DiscoveryAlignment(overlap_symbols=frozenset())
    activity_symbols = frozenset(row.symbol for row in activity_ranked[:compare_count])
    overlap_symbols = frozenset(
        row.symbol for row in liquidity_ranked[:compare_count] if row.symbol in activity_symbols
    )
    notes: list[str] = []
    if overlap_symbols:
        notes.append("allocation_discovery_overlap_symbols: " + ", ".join(sorted(overlap_symbols)))
    return _DiscoveryAlignment(
        overlap_symbols=overlap_symbols,
        notes=tuple(notes),
    )


def _rank_allocation_items(
    items,
    *,
    settings,
    learning_biases: dict[tuple[str, str], _LearningBias],
    research_alignment=None,
):
    def _score(row) -> float:
        base = float(row.priority_score or 0.0)
        bias = learning_biases.get((str(row.action or "").upper(), _exposure_key(row.symbol)))
        if bias is not None:
            base += bias.score_adjustment
        if bool(getattr(settings, "portfolio_allocator_context_weighting_enabled", True)):
            base += _context_adjustment(row, settings=settings)
        if research_alignment is not None:
            base += _freshness_preference_adjustment(
                row,
                research_alignment=research_alignment,
                settings=settings,
            )
        return round(base, 4)

    ranked = sorted(
        items,
        key=lambda row: (
            -_score(row),
            int(row.selection_rank or 0),
            row.symbol,
        ),
    )
    out = []
    for row in ranked:
        out.append(
            row.__class__(
                symbol=row.symbol,
                action=row.action,
                confidence=row.confidence,
                verdict=row.verdict,
                summary=row.summary,
                selection_rank=row.selection_rank,
                priority_score=_score(row),
                exposure_penalty=row.exposure_penalty,
                rationale=row.rationale,
                winning_strategy=getattr(row, "winning_strategy", None),
                liquidity_score=row.liquidity_score,
                regime=getattr(row, "regime", ""),
                volume_ratio=getattr(row, "volume_ratio", None),
                activity_score=getattr(row, "activity_score", None),
            )
        )
    return out


def _context_adjustment(row, *, settings) -> float:
    regime = str(getattr(row, "regime", "") or "").upper()
    volume_ratio = float(getattr(row, "volume_ratio", 0.0) or 0.0)
    activity_score = float(getattr(row, "activity_score", 0.0) or 0.0)
    regime_weight = float(getattr(settings, "portfolio_allocator_regime_weight", 0.06))
    activity_weight = float(getattr(settings, "portfolio_allocator_activity_weight", 0.04))
    regime_adjustment = 0.0
    if regime == "TRENDING":
        regime_adjustment += regime_weight
    elif regime == "VOLATILE":
        regime_adjustment -= regime_weight * 0.5
    volume_adjustment = max(0.0, min(activity_weight, (volume_ratio - 1.0) * activity_weight))
    activity_adjustment = max(0.0, min(activity_weight, activity_score * (activity_weight / 2.0)))
    return round(regime_adjustment + volume_adjustment + activity_adjustment, 4)


def _freshness_preference_adjustment(row, *, research_alignment, settings) -> float:
    if not bool(getattr(settings, "portfolio_allocator_freshness_preference_enabled", True)):
        return 0.0
    weight = max(
        0.0,
        float(getattr(settings, "portfolio_allocator_freshness_preference_weight", 0.03)),
    )
    if weight <= 0.0:
        return 0.0
    support = _resolve_research_support(row.symbol, research_alignment=research_alignment)
    if support is None or not support.refreshed:
        return 0.0
    bonus = weight
    if support.nightly_refreshed_reports > 0:
        bonus += weight * 0.5
    elif support.nightly_reports > 0:
        bonus += weight * 0.25
    return round(min(weight * 2.0, bonus), 4)


def _risk_bucket(item) -> str:
    regime = str(getattr(item, "regime", "") or "").upper()
    confidence = float(getattr(item, "confidence", 0.0) or 0.0)
    verdict = str(getattr(item, "verdict", "") or "").lower()
    if verdict != "candidate" or confidence < 0.72 or regime == "VOLATILE":
        return "high"
    if confidence < 0.82 or regime == "RANGING":
        return "medium"
    return "low"


def _portfolio_critic_assessment(
    item,
    *,
    selected: list[AllocationDecision],
    account_context: _AccountContext,
    risk_bucket: str,
    discovery_alignment: _DiscoveryAlignment,
    research_support: _ResearchSupport | None,
    settings,
) -> tuple[float, str, bool]:
    if not bool(getattr(settings, "portfolio_allocator_critic_enabled", True)):
        return 0.0, "", False
    side = str(getattr(item, "action", "") or "").upper()
    regime = str(getattr(item, "regime", "") or "").upper()
    verdict = str(getattr(item, "verdict", "") or "").lower()
    item_symbol = str(getattr(item, "symbol", "") or "")
    item_exposure_key = _exposure_key(item_symbol)
    winning_strategy = str(getattr(item, "winning_strategy", "") or "").strip()
    winning_strategy_key = winning_strategy.upper()
    same_side = sum(1 for row in selected if str(row.action or "").upper() == side)
    same_regime = sum(1 for row in selected if str(row.regime or "").upper() == regime and regime)
    same_strategy_selected = [
        row
        for row in selected
        if (
            str(row.action or "").upper() == side
            and str(getattr(row, "winning_strategy", "") or "").strip().upper()
            == winning_strategy_key
            and winning_strategy_key
        )
    ]
    open_same_side = int(account_context.side_counts.get(side, 0))
    open_same_exposure_side = int(
        account_context.exposure_side_counts.get((item_exposure_key, side), 0)
    )
    penalty = 0.0
    reasons: list[str] = []
    if same_side + open_same_side > 0:
        penalty += float(getattr(settings, "portfolio_allocator_critic_same_side_penalty", 0.05))
        reasons.append(f"same-side crowding ({same_side + open_same_side})")
    if open_same_exposure_side > 0:
        penalty += float(
            getattr(settings, "portfolio_allocator_critic_open_same_exposure_penalty", 0.03)
        )
        reasons.append(f"same-side live exposure in {item_exposure_key}")
    open_same_regime_side = int(account_context.open_regime_side_counts.get((regime, side), 0))
    if same_regime > 0:
        penalty += float(
            getattr(settings, "portfolio_allocator_critic_same_regime_penalty", 0.04)
        )
        reasons.append(f"same-regime concentration ({regime})")
    elif open_same_regime_side > 0 and regime:
        penalty += float(
            getattr(settings, "portfolio_allocator_critic_same_regime_penalty", 0.04)
        )
        reasons.append(f"same-regime live exposure ({regime})")
    if same_strategy_selected:
        penalty += float(
            getattr(settings, "portfolio_allocator_critic_same_strategy_penalty", 0.035)
        )
        strongest_strategy_score = max(
            float(getattr(row, "priority_score", 0.0) or 0.0) for row in same_strategy_selected
        )
        current_priority = float(getattr(item, "priority_score", 0.0) or 0.0)
        if strongest_strategy_score > current_priority:
            reasons.append(
                f"duplicate setup family `{winning_strategy}` versus stronger current {side} basket"
            )
        else:
            reasons.append(f"duplicate setup family `{winning_strategy}` in current {side} basket")
    if same_side > 0 and same_regime > 0:
        strongest_same_side_score = max(
            (
                float(getattr(row, "priority_score", 0.0) or 0.0)
                for row in selected
                if str(row.action or "").upper() == side
            ),
            default=0.0,
        )
        current_priority = float(getattr(item, "priority_score", 0.0) or 0.0)
        if strongest_same_side_score > current_priority + 0.03:
            penalty += float(
                getattr(settings, "portfolio_allocator_critic_weaker_same_side_penalty", 0.025)
            )
            reasons.append(f"weaker conviction than current {side} basket")
    if verdict == "watch":
        penalty += float(getattr(settings, "portfolio_allocator_critic_watch_penalty", 0.04))
        reasons.append("watch-grade conviction")
    if risk_bucket == "high":
        penalty += float(
            getattr(settings, "portfolio_allocator_critic_high_risk_penalty", 0.06)
        )
        reasons.append("high-risk setup")
    if bool(getattr(settings, "portfolio_allocator_discovery_alignment_enabled", True)):
        item_has_overlap = item_symbol in discovery_alignment.overlap_symbols
        selected_side_has_overlap = any(
            str(getattr(row, "action", "") or "").upper() == side
            and str(getattr(row, "symbol", "") or "") in discovery_alignment.overlap_symbols
            for row in selected
        )
        if same_side > 0 and selected_side_has_overlap and not item_has_overlap:
            penalty += float(
                getattr(settings, "portfolio_allocator_critic_weaker_discovery_penalty", 0.025)
            )
            reasons.append(f"weaker discovery alignment than current {side} basket")
    if bool(getattr(settings, "portfolio_allocator_research_alignment_enabled", True)):
        support_weight = _research_target_weight(
            research_support.target if research_support is not None else ""
        )
        support_refreshed = bool(
            research_support.refreshed if research_support is not None else False
        )
        selected_side_support = max(
            (
                _research_target_weight(getattr(row, "research_target", ""))
                for row in selected
                if str(row.action or "").upper() == side
            ),
            default=0,
        )
        selected_side_has_refreshed_support = any(
            bool(getattr(row, "research_refreshed", False))
            for row in selected
            if str(row.action or "").upper() == side
        )
        selected_side_nightly_score = max(
            (
                float(getattr(row, "research_nightly_score", 0.0) or 0.0)
                for row in selected
                if str(row.action or "").upper() == side
            ),
            default=0.0,
        )
        support_nightly_score = float(
            research_support.nightly_score if research_support is not None else 0.0
        )
        if same_side + open_same_side > 0 and support_weight <= 0:
            penalty += float(
                getattr(settings, "portfolio_allocator_critic_missing_research_penalty", 0.04)
            )
            reasons.append("no current ML/RL research support")
        elif same_side > 0 and selected_side_support > support_weight:
            penalty += float(
                getattr(settings, "portfolio_allocator_critic_weaker_research_penalty", 0.03)
            )
            reasons.append(f"weaker ML/RL research support than current {side} basket")
        elif same_side > 0 and selected_side_has_refreshed_support and not support_refreshed:
            penalty += float(
                getattr(
                    settings,
                    "portfolio_allocator_critic_unrefreshed_research_penalty",
                    0.02,
                )
            )
            reasons.append(f"plan-only research support versus refreshed current {side} basket")
        if same_side > 0 and selected_side_nightly_score > support_nightly_score:
            penalty += float(
                getattr(
                    settings,
                    "portfolio_allocator_critic_weaker_nightly_research_penalty",
                    0.025,
                )
            )
            reasons.append(f"weaker recurring nightly research evidence than current {side} basket")
    if bool(getattr(settings, "portfolio_allocator_same_target_balance_enabled", True)):
        item_target = str(
            research_support.target if research_support is not None else ""
        ).strip().lower()
        if item_target and same_side > 0:
            same_target_selected = sum(
                1
                for row in selected
                if str(row.action or "").upper() == side
                and str(getattr(row, "research_target", "") or "").strip().lower()
                == item_target
            )
            if same_target_selected > 0:
                refreshed = bool(
                    research_support.refreshed if research_support is not None else False
                )
                high_conviction = float(getattr(item, "confidence", 0.0) or 0.0) >= 0.85
                if not (refreshed and high_conviction):
                    penalty += float(
                        getattr(
                            settings,
                            "portfolio_allocator_same_target_balance_penalty",
                            0.035,
                        )
                    )
                    reasons.append(
                        f"same {item_target.upper()} research target crowding"
                    )
    note = ""
    if reasons:
        note = "portfolio critic: " + ", ".join(reasons)
    threshold = float(getattr(settings, "portfolio_allocator_critic_block_threshold", 0.14))
    block = penalty >= threshold
    return round(penalty, 4), note, block


def _weighted_selected(selected: list[AllocationDecision], *, settings) -> list[AllocationDecision]:
    if not selected:
        return []
    max_single_weight = min(
        0.9,
        max(0.2, float(getattr(settings, "portfolio_allocator_max_single_weight", 0.6))),
    )
    scores = [
        max(float(item.allocation_score or item.priority_score or 0.0), 0.0)
        for item in selected
    ]
    total_score = sum(scores)
    if total_score <= 0:
        base_weights = [1.0 / len(selected)] * len(selected)
    else:
        base_weights = [score / total_score for score in scores]
    adjusted_weights = [
        _risk_adjusted_weight(item, weight)
        for item, weight in zip(selected, base_weights)
    ]
    adjusted_total = sum(adjusted_weights)
    if adjusted_total <= 0:
        adjusted_weights = [1.0 / len(selected)] * len(selected)
    else:
        adjusted_weights = [weight / adjusted_total for weight in adjusted_weights]
    capped_weights = _cap_weights(adjusted_weights, cap=max_single_weight)
    out: list[AllocationDecision] = []
    for item, weight in zip(selected, capped_weights):
        out.append(
            AllocationDecision(
                symbol=item.symbol,
                action=item.action,
                verdict=item.verdict,
                selected=item.selected,
                reason=item.reason,
                selection_rank=item.selection_rank,
                allocation_rank=item.allocation_rank,
                allocation_weight=round(weight, 3),
                priority_score=item.priority_score,
                allocation_score=item.allocation_score,
                exposure_penalty=item.exposure_penalty,
                critic_penalty=item.critic_penalty,
                critic_note=item.critic_note,
                exposure_key=item.exposure_key,
                winning_strategy=item.winning_strategy,
                regime=item.regime,
                risk_bucket=item.risk_bucket,
                sizing_hint=_sizing_hint(weight),
                research_target=item.research_target,
                research_priority_score=item.research_priority_score,
                research_refreshed=item.research_refreshed,
                research_nightly_reports=item.research_nightly_reports,
                research_nightly_refreshed_reports=item.research_nightly_refreshed_reports,
                research_nightly_promoted_reports=item.research_nightly_promoted_reports,
                research_nightly_score=item.research_nightly_score,
            )
        )
    return out


def _risk_adjusted_weight(item: AllocationDecision, weight: float) -> float:
    if item.risk_bucket == "high":
        return weight * 0.82
    if item.risk_bucket == "medium":
        return weight * 0.94
    return weight * 1.06


def _cap_weights(weights: list[float], *, cap: float) -> list[float]:
    if not weights:
        return []
    capped = list(weights)
    overflow = 0.0
    uncapped: list[int] = []
    for idx, weight in enumerate(capped):
        if weight > cap:
            overflow += weight - cap
            capped[idx] = cap
        else:
            uncapped.append(idx)
    if overflow > 0 and uncapped:
        uncapped_total = sum(capped[idx] for idx in uncapped)
        if uncapped_total > 0:
            for idx in uncapped:
                capped[idx] += overflow * (capped[idx] / uncapped_total)
    total = sum(capped)
    if total <= 0:
        return [1.0 / len(capped)] * len(capped)
    return [weight / total for weight in capped]


def _sizing_hint(weight: float) -> str:
    if weight >= 0.45:
        return "overweight"
    if weight >= 0.25:
        return "core"
    return "starter"


def _allocation_mix_notes(selected: list[AllocationDecision]) -> list[str]:
    if not selected:
        return []
    regimes: dict[str, int] = defaultdict(int)
    risk_buckets: dict[str, int] = defaultdict(int)
    strategies: dict[str, int] = defaultdict(int)
    research_targets: dict[str, int] = defaultdict(int)
    refreshed_targets: dict[str, int] = defaultdict(int)
    nightly_targets: dict[str, int] = defaultdict(int)
    promoted_nightly_targets: dict[str, int] = defaultdict(int)
    for row in selected:
        if row.regime:
            regimes[row.regime.upper()] += 1
        if row.risk_bucket:
            risk_buckets[row.risk_bucket] += 1
        if getattr(row, "winning_strategy", ""):
            strategy_key = str(row.winning_strategy).upper()
            strategies[strategy_key] += 1
        if getattr(row, "research_target", ""):
            target_key = str(row.research_target).lower()
            research_targets[target_key] += 1
            if bool(getattr(row, "research_refreshed", False)):
                refreshed_targets[target_key] += 1
            if int(getattr(row, "research_nightly_reports", 0) or 0) > 0:
                nightly_targets[target_key] += 1
            if int(getattr(row, "research_nightly_promoted_reports", 0) or 0) > 0:
                promoted_nightly_targets[target_key] += 1
    notes: list[str] = []
    if regimes:
        regime_text = ", ".join(f"{key}={value}" for key, value in sorted(regimes.items()))
        notes.append(f"selected_regimes: {regime_text}")
    if risk_buckets:
        risk_text = ", ".join(f"{key}={value}" for key, value in sorted(risk_buckets.items()))
        notes.append(f"allocation_risk_mix: {risk_text}")
    if strategies:
        strategy_text = ", ".join(f"{key}={value}" for key, value in sorted(strategies.items()))
        notes.append(f"allocation_strategy_mix: {strategy_text}")
    if research_targets:
        target_text = ", ".join(
            f"{key}={value}" for key, value in sorted(research_targets.items())
        )
        notes.append(f"allocation_research_targets: {target_text}")
    if refreshed_targets:
        refreshed_text = ", ".join(
            f"{key}={value}" for key, value in sorted(refreshed_targets.items())
        )
        notes.append(f"allocation_refreshed_research_targets: {refreshed_text}")
    if nightly_targets:
        nightly_text = ", ".join(
            f"{key}={value}" for key, value in sorted(nightly_targets.items())
        )
        notes.append(f"allocation_nightly_research_targets: {nightly_text}")
    if promoted_nightly_targets:
        promoted_nightly_text = ", ".join(
            f"{key}={value}" for key, value in sorted(promoted_nightly_targets.items())
        )
        notes.append(f"allocation_nightly_promoted_research_targets: {promoted_nightly_text}")
    refreshed_count = sum(1 for row in selected if bool(getattr(row, "research_refreshed", False)))
    if selected:
        notes.append(
            f"allocation_freshness_mix: refreshed={refreshed_count}/{len(selected)}"
        )
    if research_targets and len(selected) >= 2:
        dominant_target = max(research_targets, key=research_targets.get)
        dominant_count = research_targets[dominant_target]
        if dominant_count == len(selected):
            notes.append(
                f"allocation_target_balance_summary: concentrated={dominant_target}"
            )
        elif dominant_count >= max(2, int(len(selected) * 0.67)):
            notes.append(
                "allocation_target_balance_summary: "
                f"leaning={dominant_target} ({dominant_count}/{len(selected)})"
            )
        else:
            notes.append("allocation_target_balance_summary: balanced")
    critic_notes = [
        str(getattr(row, "critic_note", "") or "").strip()
        for row in selected
        if str(getattr(row, "critic_note", "") or "").strip()
    ]
    if critic_notes:
        notes.append(f"allocation_critic_summary: {critic_notes[0]}")
    return notes


def _training_research_alignment(
    *,
    settings,
    universe_limit: int,
    analysis_limit: int,
    timeframe: str,
    days: int,
    source: str,
) -> _ResearchAlignment:
    if not bool(getattr(settings, "portfolio_allocator_research_alignment_enabled", True)):
        return _ResearchAlignment(by_symbol={}, by_exposure={})
    try:
        from fortuna.app.training_research import build_training_research_plan

        response = build_training_research_plan(
            settings=settings,
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
            refresh_data=False,
        )
    except Exception:
        return _ResearchAlignment(by_symbol={}, by_exposure={})
    if not response.ok:
        return _ResearchAlignment(by_symbol={}, by_exposure={})
    nightly_feedback, nightly_report_count = _load_nightly_research_feedback(settings)
    effective_target = _effective_research_refresh_target(
        requested_target=str(getattr(response, "refresh_target", "") or "").strip() or None,
        discovery_target=(
            str(getattr(response, "discovery_recommended_refresh_target", "") or "").strip() or None
        ),
    )
    by_symbol: dict[str, _ResearchSupport] = {}
    by_exposure: dict[str, _ResearchSupport] = {}
    target_counts: dict[str, int] = defaultdict(int)
    executed_target_counts: dict[str, int] = defaultdict(int)
    executed_supported = 0
    for row in response.rows:
        target = _normalize_research_target_for_effective_focus(
            str(getattr(row, "target", "") or "").lower(),
            effective_target=effective_target,
        )
        symbol_key = str(getattr(row, "symbol", "") or "").upper()
        exposure_key = _exposure_key(symbol_key)
        feedback = nightly_feedback.get(exposure_key)
        support = _ResearchSupport(
            target=target,
            selection_rank=int(getattr(row, "selection_rank", 0) or 0),
            priority_score=float(getattr(row, "priority_score", 0.0) or 0.0),
            refreshed=bool(getattr(row, "refreshed", False)),
            nightly_reports=int(feedback.research_reports if feedback is not None else 0),
            nightly_refreshed_reports=int(
                feedback.refreshed_reports if feedback is not None else 0
            ),
            nightly_promoted_reports=int(feedback.promoted_reports if feedback is not None else 0),
            nightly_score=_nightly_research_score(
                feedback,
                report_count=nightly_report_count,
                settings=settings,
            ),
        )
        if feedback is not None and int(feedback.executed_reports or 0) > 0:
            executed_supported += 1
            if target:
                executed_target_counts[target] += 1
        by_symbol[symbol_key] = support
        current = by_exposure.get(exposure_key)
        if current is None or _research_support_sort_key(support) > _research_support_sort_key(
            current
        ):
            by_exposure[exposure_key] = support
        if target:
            target_counts[target] += 1
    notes: list[str] = []
    if response.rows:
        notes.append(f"research_plan_rows={len(response.rows)}")
    if target_counts:
        target_text = ", ".join(f"{key}={value}" for key, value in sorted(target_counts.items()))
        notes.append(f"research_plan_targets: {target_text}")
    discovery_target = (
        str(getattr(response, "discovery_recommended_refresh_target", "") or "").strip() or None
    )
    if discovery_target:
        notes.append(f"research_plan_discovery_target={discovery_target}")
    if effective_target:
        notes.append(f"research_plan_effective_target={effective_target}")
    refreshed_count = sum(1 for row in response.rows if bool(getattr(row, "refreshed", False)))
    if refreshed_count > 0:
        notes.append(f"research_plan_refreshed_rows={refreshed_count}")
    if nightly_report_count > 0:
        notes.append(f"research_nightly_reports={nightly_report_count}")
    nightly_supported = sum(1 for support in by_symbol.values() if support.nightly_reports > 0)
    if nightly_supported > 0:
        notes.append(f"research_nightly_supported_symbols={nightly_supported}")
    if executed_supported > 0:
        notes.append(f"research_nightly_executed_symbols={executed_supported}")
    if executed_target_counts:
        executed_text = ", ".join(
            f"{key}={value}" for key, value in sorted(executed_target_counts.items())
        )
        notes.append(f"research_nightly_executed_targets: {executed_text}")
    return _ResearchAlignment(
        by_symbol=by_symbol,
        by_exposure=by_exposure,
        notes=tuple(notes),
    )


def _resolve_research_support(
    symbol: str,
    *,
    research_alignment: _ResearchAlignment,
) -> _ResearchSupport | None:
    symbol_key = str(symbol or "").upper()
    if symbol_key in research_alignment.by_symbol:
        return research_alignment.by_symbol[symbol_key]
    return research_alignment.by_exposure.get(_exposure_key(symbol_key))


def _research_support_sort_key(support: _ResearchSupport) -> tuple[int, float, int]:
    return (
        1 if support.refreshed else 0,
        float(support.nightly_score or 0.0),
        _research_target_weight(support.target),
        float(support.priority_score or 0.0),
        -int(support.selection_rank or 0),
    )


def _research_target_weight(target: str) -> int:
    key = str(target or "").strip().lower()
    if key == "both":
        return 3
    if key == "rl":
        return 2
    if key == "ml":
        return 1
    return 0


def _effective_research_refresh_target(
    *,
    requested_target: str | None,
    discovery_target: str | None,
) -> str | None:
    normalized = [
        str(target or "").strip().lower()
        for target in (requested_target, discovery_target)
        if str(target or "").strip()
    ]
    if not normalized:
        return None
    concrete = {target for target in normalized if target in {"ml", "rl"}}
    if len(concrete) >= 2:
        return "all"
    if len(concrete) == 1:
        return next(iter(concrete))
    if "all" in normalized:
        return "all"
    return normalized[0]


def _normalize_research_target_for_effective_focus(
    target: str,
    *,
    effective_target: str | None,
) -> str:
    key = str(target or "").strip().lower()
    focus = str(effective_target or "").strip().lower()
    if focus not in {"ml", "rl"}:
        return key
    if key == "both":
        return focus
    if key == focus:
        return key
    if key in {"ml", "rl"}:
        return ""
    return key


def _load_nightly_research_feedback(
    settings,
) -> tuple[dict[str, _NightlyResearchFeedback], int]:
    if not bool(getattr(settings, "portfolio_allocator_nightly_feedback_enabled", True)):
        return {}, 0
    report_dir = settings.resolve_path(
        Path(getattr(settings, "portfolio_allocator_nightly_report_dir", "reports/nightly"))
    )
    if not report_dir.is_dir():
        return {}, 0
    lookback = max(
        1,
        int(getattr(settings, "portfolio_allocator_nightly_feedback_lookback_reports", 8)),
    )
    reports = sorted(report_dir.glob("*.json"), key=lambda path: path.name, reverse=True)[:lookback]
    if not reports:
        return {}, 0

    counts: dict[str, dict[str, int]] = {}
    used_reports = 0
    for report_path in reports:
        payload = _load_json_dict(report_path)
        if not payload or str(payload.get("overall_status", "")).strip().lower() != "ok":
            continue
        steps = payload.get("steps")
        step_rows = steps if isinstance(steps, list) else []
        research_detail = _step_detail(step_rows, "training_research")
        execution_detail = _step_detail(step_rows, "training_execution_target")
        promote_detail = _step_detail(step_rows, "promote_best")
        execution_target = _normalize_execution_target((execution_detail or {}).get("target"))
        execution_source = _normalize_execution_source(
            (execution_detail or {}).get("selection_source")
        )
        promote_count = _coerce_int((promote_detail or {}).get("count")) or 0
        report_used = False
        research_path_text = str((research_detail or {}).get("path", "") or "").strip()
        if research_path_text:
            research = _load_json_dict(
                _resolve_report_relative_path(report_path, research_path_text)
            )
            rows = research.get("rows") if research else None
            if isinstance(rows, list) and rows:
                report_used = True
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    key = _exposure_key(str(row.get("symbol", "") or ""))
                    if not key:
                        continue
                    bucket = counts.setdefault(
                        key,
                        {
                            "research_reports": 0,
                            "refreshed_reports": 0,
                            "promoted_reports": 0,
                            "executed_reports": 0,
                            "executed_refreshed_reports": 0,
                        },
                    )
                    bucket["research_reports"] += 1
                    row_refreshed = bool(row.get("refreshed", False))
                    if bool(row.get("refreshed", False)):
                        bucket["refreshed_reports"] += 1
                    if execution_source == "training_research" and _research_row_matches_execution(
                        row,
                        execution_target=execution_target,
                    ):
                        bucket["executed_reports"] += 1
                        if row_refreshed:
                            bucket["executed_refreshed_reports"] += 1
                    if promote_count > 0:
                        bucket["promoted_reports"] += 1
        if report_used:
            used_reports += 1
    return (
        {
            key: _NightlyResearchFeedback(
                research_reports=value["research_reports"],
                refreshed_reports=value["refreshed_reports"],
                promoted_reports=value["promoted_reports"],
                executed_reports=value["executed_reports"],
                executed_refreshed_reports=value["executed_refreshed_reports"],
            )
            for key, value in counts.items()
        },
        used_reports,
    )


def _nightly_research_score(
    feedback: _NightlyResearchFeedback | None,
    *,
    report_count: int,
    settings,
) -> float:
    if feedback is None or report_count <= 0:
        return 0.0
    refresh_bonus_weight = max(
        0.0,
        float(getattr(settings, "portfolio_allocator_nightly_refresh_bonus_weight", 0.65)),
    )
    promote_bonus_weight = max(
        0.0,
        float(getattr(settings, "portfolio_allocator_nightly_promote_bonus_weight", 0.3)),
    )
    research_weight = max(0.0, 1.0 - min(1.0, refresh_bonus_weight))
    execution_weight = min(1.0, refresh_bonus_weight + 0.2)
    evidence_reports = (
        (feedback.research_reports * research_weight)
        + (feedback.refreshed_reports * refresh_bonus_weight)
        + (feedback.executed_reports * execution_weight)
        + (feedback.executed_refreshed_reports * min(1.0, execution_weight + 0.1))
    )
    evidence_ratio = min(1.0, evidence_reports / float(report_count))
    promoted_ratio = min(1.0, float(feedback.promoted_reports) / float(report_count))
    base_weight = 1.0 - min(1.0, promote_bonus_weight)
    return round(
        min(
            1.0,
            (evidence_ratio * base_weight) + (promoted_ratio * promote_bonus_weight),
        ),
        4,
    )


def _normalize_execution_target(value: object) -> str:
    target = str(value or "").strip().lower()
    return target if target in {"ml", "rl", "all"} else ""


def _normalize_execution_source(value: object) -> str:
    source = str(value or "").strip().lower()
    return source if source in {"training_candidates", "training_research"} else ""


def _research_row_matches_execution(row: dict[str, object], *, execution_target: str) -> bool:
    target = str(row.get("target", "") or "").strip().lower()
    if execution_target == "ml":
        return target in {"ml", "both"}
    if execution_target == "rl":
        return target in {"rl", "both"}
    if execution_target == "all":
        return target in {"ml", "rl", "both"}
    return False


def _load_json_dict(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_report_relative_path(report_path: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip())
    if candidate.is_absolute():
        return candidate
    return report_path.parent / candidate


def _step_detail(step_rows: list, name: str) -> Optional[dict]:
    key = str(name or "").strip()
    for row in step_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("name", "")).strip() != key:
            continue
        detail = row.get("detail")
        return detail if isinstance(detail, dict) else None
    return None


def _coerce_int(value) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _learning_score_adjustment(avg_realized_pnl_pct: float, sample_size: int) -> float:
    if sample_size <= 0:
        return 0.0
    strength = min(0.18, abs(avg_realized_pnl_pct) / 10.0)
    if sample_size < 2:
        strength = min(strength, 0.05)
    if avg_realized_pnl_pct > 0.25:
        return round(strength, 4)
    if avg_realized_pnl_pct < -0.25:
        return round(-strength, 4)
    return 0.0


def _learning_reason_suffix(
    action: str,
    exposure_key: str,
    learning_biases: dict[tuple[str, str], _LearningBias],
) -> str:
    bias = learning_biases.get((str(action or "").upper(), exposure_key))
    if bias is None or bias.score_adjustment == 0.0:
        return ""
    direction = "boost" if bias.score_adjustment > 0 else "penalty"
    return (
        f"recent paper outcomes imply a {direction} "
        f"({bias.avg_realized_pnl_pct:+.2f}% avg over {bias.paper_closed_rows} closed rows)"
    )


def _learning_notes(learning_biases: dict[tuple[str, str], _LearningBias]) -> list[str]:
    notes: list[str] = []
    ranked = sorted(
        learning_biases.items(),
        key=lambda item: (item[1].score_adjustment, item[1].paper_closed_rows),
    )
    for (action, exposure_key), bias in ranked[:2]:
        if bias.score_adjustment >= 0:
            continue
        notes.append(
            f"recent {action} paper outcomes for {exposure_key} avg "
            f"{bias.avg_realized_pnl_pct:+.2f}% across {bias.paper_closed_rows} closed rows"
        )
    return notes


def _join_reason(primary: str, suffix: str) -> str:
    if not suffix:
        return primary
    return f"{primary}; {suffix}"


def _scored_allocation(priority_score: Optional[float], critic_penalty: float) -> Optional[float]:
    if priority_score is None:
        return None
    return round(float(priority_score) - float(critic_penalty or 0.0), 4)


def _selected_reason(critic_note: str) -> str:
    if critic_note:
        return f"selected under current portfolio constraints despite {critic_note}"
    return "selected under current portfolio constraints"
