"""Build a typed training-research plan from the universe and shortlist pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from fortuna.agentic.contracts import (
    TrainingCandidate,
    TrainingResearchPlanResponse,
    TrainingResearchRow,
)
from fortuna.app.market_universe import build_market_universe
from fortuna.app.model_status import build_nightly_alignment_status
from fortuna.app.training_candidates import (
    backfill_training_candidates,
    build_training_candidates,
    select_training_symbols,
)
from fortuna.config.settings import Settings
from fortuna.observability.recorder import workflow_boundary


def build_training_research_plan(
    *,
    settings: Settings,
    universe_limit: int = 15,
    analysis_limit: int = 8,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
    ml_top_n: int = 5,
    rl_top_n: int = 3,
    selection_policy: str = "diversified",
    refresh_data: bool = False,
    refresh_target: str = "all",
    refresh_timeframe: Optional[str] = None,
    refresh_days: Optional[int] = None,
    force_refresh: bool = False,
) -> TrainingResearchPlanResponse:
    with workflow_boundary(
        settings,
        event_name="training_research",
        module="fortuna.app.training_research",
        workflow_id="training_research",
        context={"source": source, "refresh_target": refresh_target},
    ) as span:
        response = _build_training_research_plan_impl(
            settings=settings,
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
            ml_top_n=ml_top_n,
            rl_top_n=rl_top_n,
            selection_policy=selection_policy,
            refresh_data=refresh_data,
            refresh_target=refresh_target,
            refresh_timeframe=refresh_timeframe,
            refresh_days=refresh_days,
            force_refresh=force_refresh,
        )
        span.set_context(
            ok=response.ok,
            row_count=len(response.rows),
            effective_refresh_target=response.effective_refresh_target,
        )
        if not response.ok:
            span.set_status("warn")
        return response


def _build_training_research_plan_impl(
    *,
    settings: Settings,
    universe_limit: int = 15,
    analysis_limit: int = 8,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
    ml_top_n: int = 5,
    rl_top_n: int = 3,
    selection_policy: str = "diversified",
    refresh_data: bool = False,
    refresh_target: str = "all",
    refresh_timeframe: Optional[str] = None,
    refresh_days: Optional[int] = None,
    force_refresh: bool = False,
) -> TrainingResearchPlanResponse:
    nightly_alignment = build_nightly_alignment_status(settings)
    nightly_alignment_summary = _nightly_alignment_summary(nightly_alignment)
    nightly_posture = getattr(nightly_alignment, "nightly_posture", None)
    universe = build_market_universe(
        settings=settings,
        limit=universe_limit,
        timeframe="1d",
        days=max(days, 20),
        source=source,
    )
    if not universe.ok:
        return TrainingResearchPlanResponse(
            ok=False,
            source=source,
            timeframe=timeframe,
            lookback_days=int(days),
            selection_policy=selection_policy,
            refresh_requested=refresh_data,
            refresh_target=refresh_target,
            refresh_timeframe=refresh_timeframe or timeframe,
            refresh_lookback_days=int(refresh_days or days),
            nightly_alignment_summary=nightly_alignment_summary,
            nightly_alignment_target=nightly_alignment.recommended_refresh_target,
            nightly_posture=nightly_posture,
            nightly_alignment_force_refresh=nightly_alignment.recommended_force_refresh,
            nightly_recent_window=nightly_alignment.recent_trend_window,
            nightly_recent_enabled=nightly_alignment.recent_trend_enabled,
            nightly_recent_aligned=nightly_alignment.recent_trend_aligned,
            nightly_recent_latest_status=nightly_alignment.recent_trend_latest_status,
            nightly_recent_latest_basket_size=nightly_alignment.recent_trend_latest_basket_size,
            error=universe.error,
        )

    discovery_preferred = _discovery_preferred_symbols(universe)
    discovery_liquidity = _discovery_liquidity_symbols(universe)
    discovery_activity = _discovery_activity_symbols(universe)
    discovery_volume_dense = _discovery_volume_dense_symbols(universe)
    discovery_regime_mix = _discovery_regime_mix(universe)
    discovery_posture = _discovery_posture(
        preferred_symbols=discovery_preferred,
        liquidity_candidates=_liquidity_ranked_candidates(universe),
        activity_candidates=_activity_ranked_candidates(universe),
        regime_mix=discovery_regime_mix,
    )
    strategy_posture = None
    discovery_summary = _discovery_summary(
        preferred_symbols=discovery_preferred,
        liquidity_symbols=discovery_liquidity,
        activity_symbols=discovery_activity,
        regime_mix=discovery_regime_mix,
        strategy_posture=strategy_posture,
        candidate_posture=None,
        nightly_posture=None,
        posture=discovery_posture,
    )
    discovery_recommended_refresh_target = _discovery_refresh_target_hint(
        posture=discovery_posture,
    )
    candidates = build_training_candidates(
        settings=settings,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    if not candidates.ok:
        return TrainingResearchPlanResponse(
            ok=False,
            source=universe.source,
            timeframe=timeframe,
            lookback_days=int(days),
            selection_policy=selection_policy,
            refresh_requested=refresh_data,
            refresh_target=refresh_target,
            refresh_timeframe=refresh_timeframe or timeframe,
            refresh_lookback_days=int(refresh_days or days),
            nightly_alignment_summary=nightly_alignment_summary,
            nightly_alignment_target=nightly_alignment.recommended_refresh_target,
            nightly_posture=nightly_posture,
            nightly_alignment_force_refresh=nightly_alignment.recommended_force_refresh,
            nightly_recent_window=nightly_alignment.recent_trend_window,
            nightly_recent_enabled=nightly_alignment.recent_trend_enabled,
            nightly_recent_aligned=nightly_alignment.recent_trend_aligned,
            nightly_recent_latest_status=nightly_alignment.recent_trend_latest_status,
            nightly_recent_latest_basket_size=nightly_alignment.recent_trend_latest_basket_size,
            universe_symbols=tuple(row.symbol for row in universe.candidates),
            discovery_preferred_symbols=discovery_preferred,
            discovery_liquidity_symbols=discovery_liquidity,
            discovery_activity_symbols=discovery_activity,
            discovery_regime_mix=discovery_regime_mix,
            discovery_summary=discovery_summary,
            discovery_recommended_refresh_target=discovery_recommended_refresh_target,
            error=candidates.error,
        )
    strategy_posture = _strategy_posture(candidates.candidates)
    candidate_posture = _candidate_posture(candidates.candidates)
    base_effective_posture = _merge_postures(discovery_posture, strategy_posture)
    candidate_effective_posture = (
        _merge_postures(base_effective_posture, candidate_posture)
        if str(base_effective_posture or "").strip().lower() in {"", "all"}
        else base_effective_posture
    )
    effective_posture = (
        _merge_postures(candidate_effective_posture, nightly_posture)
        if str(candidate_effective_posture or "").strip().lower() in {"", "all"}
        else candidate_effective_posture
    )
    candidate_posture_applied = (
        candidate_posture
        if candidate_posture
        and candidate_effective_posture
        and candidate_effective_posture != base_effective_posture
        else None
    )
    nightly_posture_applied = (
        nightly_posture
        if nightly_posture
        and effective_posture
        and effective_posture != candidate_effective_posture
        else None
    )
    discovery_summary = _discovery_summary(
        preferred_symbols=discovery_preferred,
        liquidity_symbols=discovery_liquidity,
        activity_symbols=discovery_activity,
        regime_mix=discovery_regime_mix,
        strategy_posture=strategy_posture,
        candidate_posture=candidate_posture_applied,
        nightly_posture=nightly_posture_applied,
        posture=effective_posture,
    )
    discovery_recommended_refresh_target = _discovery_refresh_target_hint(
        posture=effective_posture,
    )
    ml_symbols = tuple(
        select_training_symbols(
            candidates,
            target="ml",
            top_n=max(1, int(ml_top_n)),
            selection_policy=selection_policy,
            preferred_symbols=discovery_preferred,
        )
    )
    rl_symbols = tuple(
        select_training_symbols(
            candidates,
            target="rl",
            top_n=max(1, int(rl_top_n)),
            selection_policy=selection_policy,
            preferred_symbols=discovery_preferred,
        )
    )
    selected_symbols = _selected_symbols(
        refresh_target=refresh_target,
        ml_symbols=ml_symbols,
        rl_symbols=rl_symbols,
    )
    refreshed_symbols = (
        set(
            backfill_training_candidates(
                settings,
                candidates,
                timeframe=refresh_timeframe or timeframe,
                days=int(refresh_days or days),
                force_refresh=force_refresh,
                symbols=selected_symbols,
            )
        )
        if refresh_data and selected_symbols
        else set()
    )
    universe_ranks = {row.symbol: idx for idx, row in enumerate(universe.candidates, start=1)}
    discovery_preferred_symbols = set(discovery_preferred)
    discovery_volume_dense_symbols = set(discovery_volume_dense)
    rows = tuple(
        TrainingResearchRow(
            symbol=row.symbol,
            target=_row_target(row.symbol, ml_symbols=ml_symbols, rl_symbols=rl_symbols),
            universe_rank=universe_ranks.get(row.symbol, 0),
            shortlist_rank=row.shortlist_rank,
            selection_rank=row.selection_rank,
            liquidity_score=row.liquidity_score,
            decision_action=row.decision_action,
            critique_verdict=row.critique_verdict,
            winning_strategy=row.winning_strategy,
            market_regime=row.market_regime,
            volume_ratio=row.volume_ratio,
            priority_score=row.priority_score,
            adaptive_score_adjustment=row.adaptive_score_adjustment,
            remediation_target=row.remediation_target,
            remediation_pressure=row.remediation_pressure,
            setup_family_reinforcement=row.setup_family_reinforcement,
            setup_family_verdict=row.setup_family_verdict,
            refreshed=row.symbol in refreshed_symbols,
            rationale=_research_rationale(
                row.rationale,
                symbol=row.symbol,
                preferred_symbols=discovery_preferred_symbols,
                winning_strategy=row.winning_strategy,
                strategy_posture=strategy_posture,
                ml_candidate=bool(getattr(row, "ml_candidate", False)),
                rl_candidate=bool(getattr(row, "rl_candidate", False)),
                candidate_posture=candidate_posture_applied,
                volume_dense_symbols=discovery_volume_dense_symbols,
                setup_family_verdict=row.setup_family_verdict,
            ),
        )
        for row in candidates.candidates
        if row.symbol in selected_symbols
    )
    scout_support_target = _scout_support_target(
        rows=rows,
        preferred_symbols=discovery_preferred_symbols,
        volume_dense_symbols=discovery_volume_dense_symbols,
    )
    effective_refresh_target = _effective_refresh_target(
        requested_target=refresh_target,
        discovery_target=discovery_recommended_refresh_target,
        nightly_target=nightly_alignment.recommended_refresh_target,
        scout_target=scout_support_target,
    )
    selected_target_mix = _selected_target_mix(rows)
    selected_regime_mix = _selected_regime_mix(rows)
    (
        refresh_urgency,
        refresh_urgency_target,
        refresh_urgency_score,
        refresh_urgency_rows,
        refresh_urgency_summary,
        research_follow_up_action,
    ) = _refresh_urgency_summary(
        rows=rows,
        effective_refresh_target=effective_refresh_target,
    )
    (
        discovery_follow_up_target,
        discovery_follow_up_summary,
        discovery_follow_up_action,
        research_follow_up_target,
        research_follow_up_summary,
        execution_follow_up_target,
        execution_follow_up_summary,
        execution_follow_up_action,
        follow_up_action,
    ) = _follow_up_guidance(
        discovery_target=discovery_recommended_refresh_target,
        discovery_posture=discovery_posture,
        refresh_urgency_target=refresh_urgency_target,
        refresh_urgency_summary=refresh_urgency_summary,
        research_follow_up_action=research_follow_up_action,
        nightly_target=nightly_alignment.recommended_refresh_target,
        nightly_posture=nightly_posture,
        nightly_force_refresh=nightly_alignment.recommended_force_refresh,
        effective_refresh_target=effective_refresh_target,
        requested_refresh_target=refresh_target,
    )
    return TrainingResearchPlanResponse(
        ok=True,
        source=candidates.source,
        timeframe=timeframe,
        lookback_days=int(days),
        selection_policy=selection_policy,
        refresh_requested=refresh_data,
        refresh_target=refresh_target,
        refresh_timeframe=refresh_timeframe or timeframe,
        refresh_lookback_days=int(refresh_days or days),
        nightly_alignment_summary=nightly_alignment_summary,
        nightly_alignment_target=nightly_alignment.recommended_refresh_target,
        nightly_posture=nightly_posture,
        nightly_alignment_force_refresh=nightly_alignment.recommended_force_refresh,
        nightly_recent_window=nightly_alignment.recent_trend_window,
        nightly_recent_enabled=nightly_alignment.recent_trend_enabled,
        nightly_recent_aligned=nightly_alignment.recent_trend_aligned,
        nightly_recent_latest_status=nightly_alignment.recent_trend_latest_status,
        nightly_recent_latest_basket_size=nightly_alignment.recent_trend_latest_basket_size,
        universe_symbols=tuple(row.symbol for row in universe.candidates),
        discovery_preferred_symbols=discovery_preferred,
        discovery_liquidity_symbols=discovery_liquidity,
        discovery_activity_symbols=discovery_activity,
        discovery_volume_dense_symbols=discovery_volume_dense,
        discovery_regime_mix=discovery_regime_mix,
        discovery_summary=discovery_summary,
        discovery_posture=discovery_posture,
        strategy_posture=strategy_posture,
        candidate_posture=candidate_posture,
        effective_posture=effective_posture,
        discovery_recommended_refresh_target=discovery_recommended_refresh_target,
        effective_refresh_target=effective_refresh_target,
        scout_support_target=scout_support_target,
        refresh_urgency=refresh_urgency,
        refresh_urgency_target=refresh_urgency_target,
        refresh_urgency_score=refresh_urgency_score,
        refresh_urgency_rows=refresh_urgency_rows,
        refresh_urgency_summary=refresh_urgency_summary,
        follow_up_action=follow_up_action,
        discovery_follow_up_target=discovery_follow_up_target,
        discovery_follow_up_summary=discovery_follow_up_summary,
        discovery_follow_up_action=discovery_follow_up_action,
        research_follow_up_target=research_follow_up_target,
        research_follow_up_summary=research_follow_up_summary,
        research_follow_up_action=research_follow_up_action,
        execution_follow_up_target=execution_follow_up_target,
        execution_follow_up_summary=execution_follow_up_summary,
        execution_follow_up_action=execution_follow_up_action,
        selected_target_mix=selected_target_mix,
        selected_regime_mix=selected_regime_mix,
        ml_symbols=ml_symbols,
        rl_symbols=rl_symbols,
        rows=rows,
    )


def export_training_research_plan(
    response: TrainingResearchPlanResponse,
    out_path: Path | str,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response.to_dict(), indent=2), encoding="utf-8")
    return path


def _selected_symbols(
    *,
    refresh_target: str,
    ml_symbols: tuple[str, ...],
    rl_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    target = str(refresh_target or "all").strip().lower()
    if target == "ml":
        return ml_symbols
    if target == "rl":
        return rl_symbols
    ordered = list(rl_symbols) + [symbol for symbol in ml_symbols if symbol not in rl_symbols]
    return tuple(ordered)


def _row_target(
    symbol: str,
    *,
    ml_symbols: tuple[str, ...],
    rl_symbols: tuple[str, ...],
) -> str:
    in_ml = symbol in ml_symbols
    in_rl = symbol in rl_symbols
    if in_ml and in_rl:
        return "both"
    if in_rl:
        return "rl"
    if in_ml:
        return "ml"
    return "watch"


def _effective_refresh_target(
    *,
    requested_target: Optional[str],
    discovery_target: Optional[str],
    nightly_target: Optional[str],
    scout_target: Optional[str] = None,
) -> Optional[str]:
    requested = str(requested_target or "").strip().lower()
    if requested in {"ml", "rl"}:
        return requested
    scout = str(scout_target or "").strip().lower()
    normalized = [
        str(target or "").strip().lower()
        for target in (discovery_target, nightly_target)
        if str(target or "").strip()
    ]
    if not normalized:
        return scout if scout in {"ml", "rl"} else None
    concrete = {target for target in normalized if target in {"ml", "rl"}}
    if len(concrete) >= 2:
        if scout in concrete:
            return scout
        return "all"
    if len(concrete) == 1:
        return next(iter(concrete))
    if "all" in normalized:
        return scout if scout in {"ml", "rl"} else "all"
    return normalized[0]


def _nightly_alignment_summary(nightly_alignment) -> Optional[str]:
    window = int(getattr(nightly_alignment, "recent_trend_window", 0) or 0)
    if window <= 0:
        return None
    target = (
        str(getattr(nightly_alignment, "recommended_refresh_target", "") or "—").strip()
        or "—"
    )
    return (
        f"recent={int(getattr(nightly_alignment, 'recent_trend_aligned', 0) or 0)}/"
        f"{int(getattr(nightly_alignment, 'recent_trend_enabled', 0) or 0)} "
        f"over {window} run(s); "
        f"target={target} "
        "force_refresh="
        f"{1 if bool(getattr(nightly_alignment, 'recommended_force_refresh', False)) else 0}"
    )


def _discovery_preferred_symbols(universe) -> tuple[str, ...]:
    explicit = tuple(getattr(universe, "scout_overlap_symbols", ()) or ())
    if explicit:
        return explicit
    liquidity_ranked = _liquidity_ranked_candidates(universe)
    activity_ranked = _activity_ranked_candidates(universe)
    compare_count = min(3, len(liquidity_ranked), len(activity_ranked))
    if compare_count <= 0:
        return ()
    activity_top = {row.symbol for row in activity_ranked[:compare_count]}
    return tuple(
        row.symbol
        for row in liquidity_ranked[:compare_count]
        if row.symbol in activity_top
    )


def _discovery_liquidity_symbols(universe, limit: int = 3) -> tuple[str, ...]:
    explicit = tuple(getattr(universe, "scout_liquidity_symbols", ()) or ())
    if explicit:
        return explicit[: max(1, int(limit))]
    return tuple(
        row.symbol for row in _liquidity_ranked_candidates(universe)[: max(1, int(limit))]
    )


def _discovery_activity_symbols(universe, limit: int = 3) -> tuple[str, ...]:
    explicit = tuple(getattr(universe, "scout_activity_symbols", ()) or ())
    if explicit:
        return explicit[: max(1, int(limit))]
    return tuple(
        row.symbol for row in _activity_ranked_candidates(universe)[: max(1, int(limit))]
    )


def _discovery_volume_dense_symbols(universe, limit: int = 3) -> tuple[str, ...]:
    explicit = tuple(getattr(universe, "scout_volume_dense_symbols", ()) or ())
    if explicit:
        return explicit[: max(1, int(limit))]
    candidates = [
        row
        for row in getattr(universe, "candidates", ()) or ()
        if row.volume_ratio is not None
    ]
    return tuple(
        row.symbol
        for row in sorted(
            candidates,
            key=lambda row: (
                -(float(getattr(row, "volume_ratio", 0.0) or 0.0)),
                -(float(getattr(row, "activity_score", 0.0) or 0.0)),
                row.symbol,
            ),
        )[: max(1, int(limit))]
    )


def _liquidity_ranked_candidates(universe) -> list:
    candidates = tuple(getattr(universe, "candidates", ()) or ())
    return sorted(
        candidates,
        key=lambda row: (
            -(float(getattr(row, "liquidity_score", 0.0) or 0.0)),
            -(float(getattr(row, "avg_turnover", 0.0) or 0.0)),
            row.symbol,
        ),
    )


def _activity_ranked_candidates(universe) -> list:
    candidates = tuple(getattr(universe, "candidates", ()) or ())
    return sorted(
        [
            row
            for row in candidates
            if (
                getattr(row, "activity_score", None) is not None
                or getattr(row, "trend_pct", None) is not None
                or getattr(row, "volume_ratio", None) is not None
                or str(getattr(row, "regime", "") or "").strip()
            )
        ],
        key=lambda row: (
            -(float(getattr(row, "activity_score", 0.0) or 0.0)),
            -(abs(float(getattr(row, "trend_pct", 0.0) or 0.0))),
            row.symbol,
        ),
    )


def _discovery_regime_mix(universe) -> Optional[str]:
    counts: dict[str, int] = {}
    for row in getattr(universe, "candidates", ()) or ():
        regime = str(getattr(row, "regime", "") or "").strip().upper()
        if not regime:
            continue
        counts[regime] = counts.get(regime, 0) + 1
    if not counts:
        return None
    return ", ".join(f"{key.lower()}={value}" for key, value in sorted(counts.items()))


def _discovery_summary(
    *,
    preferred_symbols: tuple[str, ...],
    liquidity_symbols: tuple[str, ...],
    activity_symbols: tuple[str, ...],
    regime_mix: Optional[str],
    strategy_posture: Optional[str],
    candidate_posture: Optional[str],
    nightly_posture: Optional[str],
    posture: Optional[str],
) -> Optional[str]:
    parts: list[str] = []
    if preferred_symbols:
        parts.append(f"preferred={len(preferred_symbols)}")
    if liquidity_symbols:
        parts.append(f"liquidity={','.join(liquidity_symbols[:2])}")
    if activity_symbols:
        parts.append(f"activity={','.join(activity_symbols[:2])}")
    if regime_mix:
        parts.append(f"regimes={regime_mix}")
    if strategy_posture:
        parts.append(f"setup_posture={strategy_posture}")
    if candidate_posture:
        parts.append(f"candidate_posture={candidate_posture}")
    if nightly_posture:
        parts.append(f"nightly_posture={nightly_posture}")
    if posture:
        parts.append(f"posture={posture}")
    if not parts:
        return None
    return " | ".join(parts)


def _discovery_posture(
    *,
    preferred_symbols: tuple[str, ...],
    liquidity_candidates: list,
    activity_candidates: list,
    regime_mix: Optional[str],
) -> Optional[str]:
    focus_liquidity = liquidity_candidates[:3]
    focus_activity = activity_candidates[:3]
    if not preferred_symbols and not focus_liquidity and not focus_activity:
        return None
    counts = _parse_regime_mix(regime_mix)
    trending_like = counts.get("trending", 0) + counts.get("volatile", 0)
    ranging = counts.get("ranging", 0)
    activity_volume_ratio = _average_metric(focus_activity, "volume_ratio", default=1.0)
    activity_abs_trend = _average_abs_metric(focus_activity, "trend_pct")
    liquidity_volume_ratio = _average_metric(focus_liquidity, "volume_ratio", default=1.0)
    liquidity_abs_trend = _average_abs_metric(focus_liquidity, "trend_pct")

    rl_votes = 0
    ml_votes = 0
    if trending_like > ranging:
        rl_votes += 2
    elif ranging > trending_like:
        ml_votes += 2
    if activity_volume_ratio is not None and activity_volume_ratio >= 1.15:
        rl_votes += 1
    elif (
        activity_volume_ratio is not None
        and liquidity_volume_ratio is not None
        and activity_volume_ratio <= 1.05
        and liquidity_volume_ratio <= 1.05
    ):
        ml_votes += 1
    if activity_abs_trend is not None and activity_abs_trend >= 1.0:
        rl_votes += 1
    elif liquidity_abs_trend is not None and liquidity_abs_trend <= 0.6:
        ml_votes += 1
    if len(preferred_symbols) >= 2:
        if activity_volume_ratio is not None and activity_volume_ratio >= 1.1:
            rl_votes += 1
        elif liquidity_abs_trend is not None and liquidity_abs_trend <= 0.5:
            ml_votes += 1
    elif not preferred_symbols and abs(rl_votes - ml_votes) <= 1:
        return None

    if rl_votes <= 0 and ml_votes <= 0:
        return None
    if abs(rl_votes - ml_votes) <= 1:
        return "all"
    return "rl" if rl_votes > ml_votes else "ml"


def _discovery_refresh_target_hint(
    *,
    posture: Optional[str],
) -> Optional[str]:
    posture_key = str(posture or "").strip().lower()
    if posture_key == "rl":
        return "rl"
    if posture_key == "ml":
        return "ml"
    if posture_key == "all":
        return "all"
    return None


def _selected_target_mix(rows: Iterable[TrainingResearchRow]) -> Optional[str]:
    counts: dict[str, int] = {}
    for row in rows:
        target = str(getattr(row, "target", "") or "").strip().lower()
        if not target:
            continue
        counts[target] = counts.get(target, 0) + 1
    if not counts:
        return None
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _selected_regime_mix(rows: Iterable[TrainingResearchRow]) -> Optional[str]:
    counts: dict[str, int] = {}
    for row in rows:
        regime = str(getattr(row, "market_regime", "") or "").strip().lower()
        if not regime:
            continue
        counts[regime] = counts.get(regime, 0) + 1
    if not counts:
        return None
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _scout_support_target(
    *,
    rows: Iterable[TrainingResearchRow],
    preferred_symbols: set[str],
    volume_dense_symbols: set[str],
) -> Optional[str]:
    ml_votes = 0
    rl_votes = 0
    for row in rows:
        symbol = str(getattr(row, "symbol", "") or "").strip().upper()
        if symbol not in preferred_symbols and symbol not in volume_dense_symbols:
            continue
        target = str(getattr(row, "target", "") or "").strip().lower()
        if target in {"both", "all"}:
            ml_votes += 1
            rl_votes += 1
        elif target == "ml":
            ml_votes += 1
        elif target == "rl":
            rl_votes += 1
    if ml_votes <= 0 and rl_votes <= 0:
        return None
    if ml_votes == rl_votes:
        return "all"
    return "ml" if ml_votes > rl_votes else "rl"


def _refresh_urgency_summary(
    *,
    rows: Iterable[TrainingResearchRow],
    effective_refresh_target: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[float], int, Optional[str], Optional[str]]:
    scored_rows: list[tuple[str, float]] = []
    target_scores: dict[str, float] = {}
    for row in rows:
        pressure = getattr(row, "remediation_pressure", None)
        if pressure is None or float(pressure) <= 0.0:
            continue
        target = str(getattr(row, "remediation_target", "") or "").strip().lower()
        if target not in {"ml", "rl"}:
            continue
        pressure_value = round(float(pressure), 4)
        scored_rows.append((target, pressure_value))
        target_scores[target] = round(target_scores.get(target, 0.0) + pressure_value, 4)
    row_count = len(scored_rows)
    if row_count == 0:
        return None, None, None, 0, None, None

    total_pressure = round(sum(score for _, score in scored_rows), 4)
    max_pressure = round(max(score for _, score in scored_rows), 4)
    target = _refresh_urgency_target(
        target_scores=target_scores,
        effective_refresh_target=effective_refresh_target,
    )
    urgency = _refresh_urgency_level(
        row_count=row_count,
        total_pressure=total_pressure,
        max_pressure=max_pressure,
    )
    target_text = "ML/RL" if target == "all" else str(target or "mixed").upper()
    summary = (
        f"{urgency} {target_text} remediation pressure on {row_count} row(s); "
        f"max=+{max_pressure:.2f} total=+{total_pressure:.2f}"
    )
    action = (
        f"Prioritize {target_text} refresh follow-up from remediation pressure "
        f"across {row_count} selected row(s)."
    )
    return urgency, target, max_pressure, row_count, summary, action


def _follow_up_guidance(
    *,
    discovery_target: Optional[str],
    discovery_posture: Optional[str],
    refresh_urgency_target: Optional[str],
    refresh_urgency_summary: Optional[str],
    research_follow_up_action: Optional[str],
    nightly_target: Optional[str],
    nightly_posture: Optional[str],
    nightly_force_refresh: bool,
    effective_refresh_target: Optional[str],
    requested_refresh_target: Optional[str],
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    normalized_discovery_target = _normalize_follow_up_target(discovery_target)
    discovery_summary = None
    discovery_action = None
    if normalized_discovery_target is not None:
        discovery_target_text = _follow_up_target_text(normalized_discovery_target)
        posture_text = str(discovery_posture or "").strip().lower() or "mixed"
        discovery_summary = (
            "Discovery scouts currently lean "
            f"{discovery_target_text} from {posture_text} universe posture."
        )
        discovery_action = (
            "Refresh market universe and shortlist review with "
            f"{discovery_target_text} discovery focus before the next training cycle."
        )

    research_target = _normalize_follow_up_target(refresh_urgency_target)
    research_summary = refresh_urgency_summary or None

    normalized_nightly_target = _normalize_follow_up_target(nightly_target)
    normalized_effective_target = _normalize_follow_up_target(effective_refresh_target)
    normalized_requested_target = _normalize_follow_up_target(requested_refresh_target)
    execution_target = normalized_nightly_target or normalized_effective_target
    execution_summary = None
    execution_action = None
    if execution_target is not None and (
        nightly_force_refresh
        or normalized_nightly_target is not None
        or (
            normalized_requested_target is not None
            and normalized_effective_target is not None
            and normalized_requested_target != normalized_effective_target
        )
    ):
        execution_target_text = _follow_up_target_text(execution_target)
        posture_text = str(nightly_posture or "").strip().lower() or "mixed"
        execution_summary = (
            "Nightly execution drift currently leans "
            f"{execution_target_text} from {posture_text} execution posture."
        )
        execution_action = (
            "Review nightly execution path and retarget execution candidate selection "
            f"toward {execution_target_text} before the next promotion or nightly cycle."
        )

    follow_up_action = (
        research_follow_up_action
        or discovery_action
        or execution_action
    )
    return (
        normalized_discovery_target,
        discovery_summary,
        discovery_action,
        research_target,
        research_summary,
        execution_target,
        execution_summary,
        execution_action,
        follow_up_action,
    )


def _normalize_follow_up_target(target: Optional[str]) -> Optional[str]:
    value = str(target or "").strip().lower()
    if value in {"ml", "rl", "all"}:
        return value
    return None


def _follow_up_target_text(target: str) -> str:
    normalized = _normalize_follow_up_target(target) or "all"
    if normalized == "ml":
        return "ML-focused"
    if normalized == "rl":
        return "RL-focused"
    return "full ML/RL"


def _refresh_urgency_target(
    *,
    target_scores: dict[str, float],
    effective_refresh_target: Optional[str],
) -> Optional[str]:
    if not target_scores:
        return None
    if len(target_scores) == 1:
        return next(iter(target_scores))
    ordered = sorted(target_scores.items(), key=lambda item: (-item[1], item[0]))
    top_target, top_score = ordered[0]
    second_score = ordered[1][1]
    effective_target = str(effective_refresh_target or "").strip().lower()
    if abs(top_score - second_score) <= 0.015:
        if effective_target in {"ml", "rl"} and effective_target in target_scores:
            return effective_target
        return "all"
    return top_target


def _refresh_urgency_level(
    *,
    row_count: int,
    total_pressure: float,
    max_pressure: float,
) -> str:
    if row_count >= 3 or total_pressure >= 0.11 or max_pressure >= 0.06:
        return "high"
    if row_count >= 2 or total_pressure >= 0.06 or max_pressure >= 0.035:
        return "medium"
    return "low"


def _parse_regime_mix(mix: Optional[str]) -> dict[str, int]:
    text = str(mix or "").strip()
    if not text:
        return {}
    out: dict[str, int] = {}
    for part in text.split(","):
        key_text, _, value_text = part.strip().partition("=")
        key = key_text.strip().lower()
        if not key:
            continue
        try:
            value = int(value_text.strip() or "0")
        except ValueError:
            value = 0
        out[key] = value
    return out


def _average_metric(
    rows: Iterable,
    attr: str,
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    values: list[float] = []
    for row in rows:
        value = getattr(row, attr, None)
        if value is None:
            continue
        values.append(float(value))
    if values:
        return sum(values) / len(values)
    return default


def _average_abs_metric(rows: Iterable, attr: str) -> Optional[float]:
    values: list[float] = []
    for row in rows:
        value = getattr(row, attr, None)
        if value is None:
            continue
        values.append(abs(float(value)))
    if not values:
        return None
    return sum(values) / len(values)


def _research_rationale(
    rationale: tuple[str, ...],
    *,
    symbol: str,
    preferred_symbols: Iterable[str],
    winning_strategy: Optional[str],
    strategy_posture: Optional[str],
    ml_candidate: bool,
    rl_candidate: bool,
    candidate_posture: Optional[str],
    volume_dense_symbols: Iterable[str],
    setup_family_verdict: Optional[str] = None,
) -> tuple[str, ...]:
    out = list(rationale)
    if symbol in preferred_symbols:
        out.append("Discovery overlap favored by both liquidity and activity scouts")
    if symbol in volume_dense_symbols:
        out.append("Discovery volume-dense cohort reinforced this symbol for retraining priority")
    if _strategy_matches_posture(winning_strategy, strategy_posture):
        out.append(
            "Setup-family posture reinforced this symbol for current "
            f"{str(strategy_posture or '').upper()} training focus"
        )
    if setup_family_verdict in {"winner", "loser"} and winning_strategy:
        out.append(
            "Durable setup-family evidence reinforced "
            f"{winning_strategy} as recurring {setup_family_verdict} for research priority"
        )
    if _candidate_matches_posture(
        ml_candidate=ml_candidate,
        rl_candidate=rl_candidate,
        candidate_posture=candidate_posture,
    ):
        out.append(
            "Candidate-cohort posture reinforced this symbol for current "
            f"{str(candidate_posture or '').upper()} training focus"
        )
    return tuple(dict.fromkeys(out))


_RL_STYLE_STRATEGIES = frozenset({"ORB", "VWAP", "IVORB", "LSVWAP", "TREND_FOLLOW"})
_ML_STYLE_STRATEGIES = frozenset({"MMTS", "MEAN_REVERSION"})


def _strategy_posture(rows: Iterable[TrainingCandidate]) -> Optional[str]:
    rl_votes, ml_votes = _strategy_posture_votes(rows, use_family_tiebreak=False)
    if rl_votes <= 0 and ml_votes <= 0:
        return None
    if abs(rl_votes - ml_votes) > 0.35:
        return "rl" if rl_votes > ml_votes else "ml"
    tie_rl, tie_ml = _strategy_posture_votes(rows, use_family_tiebreak=True)
    if tie_rl <= 0 and tie_ml <= 0:
        return None
    if abs(tie_rl - tie_ml) <= 0.35:
        return "all"
    return "rl" if tie_rl > tie_ml else "ml"


def _strategy_posture_votes(
    rows: Iterable[TrainingCandidate],
    *,
    use_family_tiebreak: bool,
) -> tuple[float, float]:
    rl_votes = 0.0
    ml_votes = 0.0
    for row in rows:
        strategy = str(getattr(row, "winning_strategy", "") or "").strip().upper()
        if not strategy:
            continue
        weight = max(0.5, float(getattr(row, "priority_score", 0.0) or 0.0))
        family_multiplier = _setup_family_posture_multiplier(row) if use_family_tiebreak else 1.0
        if strategy in _RL_STYLE_STRATEGIES:
            rl_votes += (
                weight
                * family_multiplier
                * (1.15 if bool(getattr(row, "rl_candidate", False)) else 0.85)
            )
        elif strategy in _ML_STYLE_STRATEGIES:
            ml_votes += (
                weight
                * family_multiplier
                * (1.1 if bool(getattr(row, "ml_candidate", False)) else 0.8)
            )
    return rl_votes, ml_votes


def _setup_family_posture_multiplier(row: TrainingCandidate) -> float:
    verdict = str(getattr(row, "setup_family_verdict", "") or "").strip().lower()
    reinforcement = float(getattr(row, "setup_family_reinforcement", 0.0) or 0.0)
    strategy = str(getattr(row, "winning_strategy", "") or "").strip().upper()
    if not verdict or reinforcement == 0.0 or not strategy:
        return 1.0
    aligned = (
        (verdict == "winner" and reinforcement > 0.0)
        or (verdict == "loser" and reinforcement < 0.0)
    )
    if not aligned:
        return 1.0
    if verdict == "winner":
        if strategy in _RL_STYLE_STRATEGIES or strategy in _ML_STYLE_STRATEGIES:
            return 1.0 + min(0.15, abs(reinforcement))
        return 1.0
    return max(0.85, 1.0 - min(0.15, abs(reinforcement)))


def _candidate_posture(rows: Iterable[TrainingCandidate]) -> Optional[str]:
    rl_votes = 0.0
    ml_votes = 0.0
    for row in rows:
        priority = float(getattr(row, "priority_score", 0.0) or 0.0)
        adaptive = float(getattr(row, "adaptive_score_adjustment", 0.0) or 0.0)
        weight = max(0.35, priority + (adaptive * 0.75))
        regime = str(getattr(row, "market_regime", "") or "").strip().upper()
        if bool(getattr(row, "rl_candidate", False)):
            rl_votes += weight + (0.08 if regime in {"TRENDING", "VOLATILE"} else 0.0)
        if bool(getattr(row, "ml_candidate", False)):
            ml_votes += weight + (0.08 if regime == "RANGING" else 0.0)
    if rl_votes <= 0 and ml_votes <= 0:
        return None
    if abs(rl_votes - ml_votes) <= 0.45:
        return "all"
    return "rl" if rl_votes > ml_votes else "ml"


def _merge_postures(
    discovery_posture: Optional[str],
    strategy_posture: Optional[str],
) -> Optional[str]:
    discovery_key = str(discovery_posture or "").strip().lower()
    strategy_key = str(strategy_posture or "").strip().lower()
    if not discovery_key:
        return strategy_key or None
    if not strategy_key:
        return discovery_key or None
    if discovery_key == strategy_key:
        return discovery_key
    if discovery_key == "all":
        return strategy_key
    if strategy_key == "all":
        return discovery_key
    return "all"


def _strategy_matches_posture(
    winning_strategy: Optional[str],
    strategy_posture: Optional[str],
) -> bool:
    strategy = str(winning_strategy or "").strip().upper()
    posture = str(strategy_posture or "").strip().lower()
    if posture == "rl":
        return strategy in _RL_STYLE_STRATEGIES
    if posture == "ml":
        return strategy in _ML_STYLE_STRATEGIES
    return False


def _candidate_matches_posture(
    *,
    ml_candidate: bool,
    rl_candidate: bool,
    candidate_posture: Optional[str],
) -> bool:
    posture = str(candidate_posture or "").strip().lower()
    if posture == "rl":
        return rl_candidate
    if posture == "ml":
        return ml_candidate
    return False


__all__ = [
    "TrainingResearchPlanResponse",
    "build_training_research_plan",
    "export_training_research_plan",
]
