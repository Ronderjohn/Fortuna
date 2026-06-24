"""Build ML/RL training candidates from the ranked advisory shortlist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from fortuna.agentic.contracts import (
    NightlyAlignmentStatus,
    ShortlistAnalysisItem,
    TrainingCandidate,
    TrainingCandidateResponse,
)
from fortuna.app.exposure_ranking import (
    advisory_priority_score,
    normalize_exposure_key,
    rank_with_exposure_penalties,
)
from fortuna.app.model_status import build_nightly_alignment_status
from fortuna.app.outcome_bias import (
    load_adaptive_outcome_policy,
    setup_family_reinforcement_rationale,
)
from fortuna.app.shortlist_analysis import analyze_market_shortlist
from fortuna.config.settings import Settings
from fortuna.data.manager import MarketDataManager
from fortuna.observability.recorder import workflow_boundary


@dataclass(frozen=True)
class _NightlyCandidateFeedback:
    symbol_reports: int
    ml_reports: int
    rl_reports: int
    research_reports: int
    research_ml_reports: int
    research_rl_reports: int
    refreshed_reports: int
    refreshed_ml_reports: int
    refreshed_rl_reports: int
    promoted_reports: int
    executed_reports: int
    executed_ml_reports: int
    executed_rl_reports: int


def build_training_candidates(
    *,
    settings: Settings,
    universe_limit: int = 15,
    analysis_limit: int = 8,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
) -> TrainingCandidateResponse:
    with workflow_boundary(
        settings,
        event_name="training_candidates",
        module="fortuna.app.training_candidates",
        workflow_id="training_candidates",
        context={"source": source},
    ) as span:
        response = _build_training_candidates_impl(
            settings=settings,
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )
        ml_count = sum(1 for row in response.candidates if row.ml_candidate)
        rl_count = sum(1 for row in response.candidates if row.rl_candidate)
        span.set_context(
            ok=response.ok,
            source=response.source,
            count=len(response.candidates),
            ml_count=ml_count,
            rl_count=rl_count,
        )
        if not response.ok:
            span.set_status("warn")
        return response


def _build_training_candidates_impl(
    *,
    settings: Settings,
    universe_limit: int = 15,
    analysis_limit: int = 8,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
) -> TrainingCandidateResponse:
    shortlist = analyze_market_shortlist(
        settings=settings,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    if not shortlist.ok:
        return TrainingCandidateResponse(
            ok=False,
            source=source,
            timeframe=timeframe,
            lookback_days=days,
            error=shortlist.error,
        )

    candidates: list[TrainingCandidate] = []
    adaptive_policy = load_adaptive_outcome_policy(
        settings,
        enabled=bool(getattr(settings, "training_candidate_adaptive_weighting_enabled", True)),
        max_adjustment=float(getattr(settings, "training_candidate_adaptive_max_boost", 0.18)),
    )
    nightly_alignment = build_nightly_alignment_status(settings)
    nightly_feedback = _load_recent_nightly_feedback(settings)
    discovery_posture_hint = _discovery_posture_hint(shortlist.items)
    scout_overlap_symbols = {
        _symbol_key(symbol)
        for symbol in (getattr(shortlist, "scout_overlap_symbols", ()) or ())
        if _symbol_key(symbol)
    }
    scout_volume_dense_symbols = {
        _symbol_key(symbol)
        for symbol in (getattr(shortlist, "scout_volume_dense_symbols", ()) or ())
        if _symbol_key(symbol)
    }
    for idx, item in enumerate(shortlist.items, start=1):
        candidate = _candidate_from_item(
            item,
            rank=idx,
            adaptive_policy=adaptive_policy,
            nightly_alignment=nightly_alignment,
            nightly_feedback=nightly_feedback,
            discovery_posture_hint=discovery_posture_hint,
            scout_overlap_symbols=scout_overlap_symbols,
            scout_volume_dense_symbols=scout_volume_dense_symbols,
            settings=settings,
        )
        if candidate is not None:
            candidates.append(candidate)
    candidates = _rerank_candidates(candidates)
    return TrainingCandidateResponse(
        ok=True,
        source=shortlist.source,
        timeframe=timeframe,
        lookback_days=days,
        candidates=tuple(candidates),
    )


def export_training_candidates(
    response: TrainingCandidateResponse,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = response.to_dict()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_training_candidates_manifest(path: Path | str) -> TrainingCandidateResponse:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    error_raw = payload.get("error")
    error = None
    if isinstance(error_raw, dict):
        from fortuna.agentic.contracts import AdvisoryError, AdvisoryErrorCode

        code_raw = str(error_raw.get("code", "internal"))
        try:
            code = AdvisoryErrorCode(code_raw)
        except ValueError:
            code = AdvisoryErrorCode.INTERNAL
        error = AdvisoryError(
            code=code,
            message=str(error_raw.get("message", "")),
            details={str(k): str(v) for k, v in (error_raw.get("details") or {}).items()},
        )
    candidates = tuple(
        TrainingCandidate(
            symbol=str(row.get("symbol", "")),
            shortlist_rank=int(row.get("shortlist_rank", 0)),
            selection_rank=int(row.get("selection_rank", 0) or 0),
            liquidity_score=_float_or_none(row.get("liquidity_score")),
            winning_strategy=row.get("winning_strategy"),
            decision_action=row.get("decision_action"),
            decision_confidence=_float_or_none(row.get("decision_confidence")),
            critique_verdict=str(row.get("critique_verdict", "watch")),
            ml_candidate=bool(row.get("ml_candidate", False)),
            rl_candidate=bool(row.get("rl_candidate", False)),
            market_regime=str(row.get("market_regime", "")),
            volume_ratio=_float_or_none(row.get("volume_ratio")),
            priority_score=_float_or_none(row.get("priority_score")),
            adaptive_score_adjustment=_float_or_none(row.get("adaptive_score_adjustment")),
            remediation_target=str(row.get("remediation_target", "") or "").strip() or None,
            remediation_pressure=_float_or_none(row.get("remediation_pressure")),
            setup_family_reinforcement=_float_or_none(row.get("setup_family_reinforcement")),
            setup_family_verdict=str(row.get("setup_family_verdict", "") or "").strip() or None,
            exposure_penalty=float(row.get("exposure_penalty", 0.0) or 0.0),
            rationale=tuple(str(x) for x in row.get("rationale", []) if str(x).strip()),
        )
        for row in payload.get("candidates", [])
    )
    return TrainingCandidateResponse(
        ok=bool(payload.get("ok", False)),
        source=str(payload.get("source", "unknown")),
        timeframe=str(payload.get("timeframe", "5m")),
        lookback_days=int(payload.get("lookback_days", 30)),
        candidates=candidates,
        disclaimer=str(
            payload.get(
                "disclaimer",
                "Training candidates are suggestions for data refresh and model prep.",
            )
        ),
        error=error,
    )


def select_training_symbols(
    response: TrainingCandidateResponse,
    *,
    target: str,
    top_n: Optional[int] = None,
    selection_policy: str = "ranked",
    preferred_symbols: Optional[Iterable[str]] = None,
) -> list[str]:
    key = str(target).strip().lower()
    if key not in {"ml", "rl", "all"}:
        raise ValueError("target must be one of: ml, rl, all")
    policy = str(selection_policy or "ranked").strip().lower()
    if policy not in {"ranked", "diversified"}:
        raise ValueError("selection_policy must be one of: ranked, diversified")
    selected: Iterable[TrainingCandidate]
    if key == "ml":
        selected = (row for row in response.candidates if row.ml_candidate)
    elif key == "rl":
        selected = (row for row in response.candidates if row.rl_candidate)
    else:
        selected = response.candidates
    preferred = {
        _symbol_key(symbol)
        for symbol in (preferred_symbols or ())
        if _symbol_key(symbol)
    }
    out = [
        row.symbol
        for row in sorted(
            selected,
            key=lambda r: (
                0 if _symbol_key(r.symbol) in preferred else 1,
                r.selection_rank if int(r.selection_rank or 0) > 0 else r.shortlist_rank,
                r.shortlist_rank,
                r.symbol,
            ),
        )
    ]
    limit = None if top_n is None else max(1, int(top_n))
    if policy == "diversified":
        return _select_diversified_symbols(response, symbols=out, target=key, top_n=limit)
    if limit is not None:
        return out[:limit]
    return out


def backfill_training_candidates(
    settings: Settings,
    response: TrainingCandidateResponse,
    *,
    timeframe: str,
    days: int,
    force_refresh: bool = False,
    target: str = "all",
    top_n: Optional[int] = None,
    selection_policy: str = "ranked",
    symbols: Optional[Iterable[str]] = None,
) -> list[str]:
    mdm = MarketDataManager(settings=settings, data_source="smartapi")
    if symbols is not None:
        selected_symbols = {
            _symbol_key(symbol)
            for symbol in symbols
            if _symbol_key(symbol)
        }
    else:
        selected_symbols = {
            _symbol_key(symbol)
            for symbol in select_training_symbols(
                response,
                target=target,
                top_n=top_n,
                selection_policy=selection_policy,
            )
        }
    refreshed: list[str] = []
    for row in response.candidates:
        if not (row.ml_candidate or row.rl_candidate):
            continue
        if _symbol_key(row.symbol) not in selected_symbols:
            continue
        mdm.get_ohlcv(
            row.symbol,
            timeframe,
            days=days,
            force_refresh=force_refresh,
        )
        refreshed.append(row.symbol)
    return refreshed


def _candidate_from_item(
    item: ShortlistAnalysisItem,
    *,
    rank: int,
    adaptive_policy,
    nightly_alignment: NightlyAlignmentStatus,
    nightly_feedback: tuple[dict[str, _NightlyCandidateFeedback], int],
    discovery_posture_hint: Optional[str],
    scout_overlap_symbols: set[str],
    scout_volume_dense_symbols: set[str],
    settings: Settings,
) -> Optional[TrainingCandidate]:
    if item.decision is None or item.critique is None:
        return None
    action = item.decision.action
    confidence = float(item.decision.confidence)
    verdict = item.critique.verdict
    ml_candidate = verdict in {"candidate", "watch"} and action in {
        "BUY",
        "SELL",
        "DO_NOT_ENTER",
    }
    rl_candidate = verdict == "candidate" and action in {"BUY", "SELL"} and confidence >= 0.6
    rationale = list(item.decision.reasons[:2])
    rationale.append(item.critique.summary)
    if item.critique.concerns:
        rationale.append(item.critique.concerns[0])
    symbol_bias = adaptive_policy.symbol_biases.get(normalize_exposure_key(item.symbol))
    action_bias = adaptive_policy.action_biases.get(str(action or "").upper())
    regime_bias = adaptive_policy.regime_biases.get(str(item.regime or "").upper())
    family_key = str(item.winning_strategy or "").strip().upper()
    durable_family = adaptive_policy.durable_setup_family_biases.get(family_key)
    strategy_bias = (
        None
        if durable_family is not None
        else adaptive_policy.signal_biases.get(family_key)
    )
    global_bias = adaptive_policy.global_bias
    base_priority = advisory_priority_score(
        confidence=confidence,
        verdict=verdict,
        liquidity_score=item.liquidity_score,
        action=action,
    )
    activity_bonus = _training_activity_bonus(
        volume_ratio=item.volume_ratio,
        regime=item.regime,
        settings=settings,
    )
    if activity_bonus != 0.0:
        base_priority = round(base_priority + activity_bonus, 4)
        rationale.append(
            "Discovery activity context "
            f"volume_ratio={float(item.volume_ratio or 1.0):.2f} "
            f"regime={str(item.regime or 'unknown').upper()} "
            f"priority_adj={activity_bonus:+.2f}"
        )
    posture_bonus = _training_discovery_posture_bonus(
        volume_ratio=item.volume_ratio,
        regime=item.regime,
        posture_hint=discovery_posture_hint,
        settings=settings,
    )
    if posture_bonus != 0.0 and discovery_posture_hint:
        base_priority = round(base_priority + posture_bonus, 4)
        rationale.append(
            "Discovery posture "
            f"{discovery_posture_hint.upper()} favored "
            f"{str(item.regime or 'unknown').upper()} "
            f"volume_ratio={float(item.volume_ratio or 1.0):.2f} "
            f"priority_adj={posture_bonus:+.2f}"
        )
    scout_bonus = _training_scout_cohort_bonus(
        item=item,
        scout_overlap_symbols=scout_overlap_symbols,
        scout_volume_dense_symbols=scout_volume_dense_symbols,
        settings=settings,
    )
    if scout_bonus != 0.0:
        base_priority = round(base_priority + scout_bonus, 4)
        rationale.extend(
            _training_scout_cohort_rationale(
                item=item,
                scout_overlap_symbols=scout_overlap_symbols,
                scout_volume_dense_symbols=scout_volume_dense_symbols,
                adjustment=scout_bonus,
            )
        )
    bias_components = tuple(
        bias.score_adjustment
        for bias in (symbol_bias, action_bias, regime_bias, strategy_bias, global_bias)
        if bias is not None and bias.score_adjustment != 0.0
    )
    total_bias = sum(bias_components)
    if total_bias != 0.0:
        base_priority = round(base_priority + total_bias, 4)
        direction = "boost" if total_bias > 0 else "penalty"
        rationale.append(
            "Recent paper outcomes imply a "
            f"{direction} ({total_bias:+.2f} score adj)"
        )
    if action_bias is not None and action:
        rationale.append(
            "Action context "
            f"{action} weighted_avg={action_bias.weighted_avg_realized_pnl_pct:+.2f}%"
        )
    if regime_bias is not None and item.regime:
        rationale.append(
            "Regime context "
            f"{item.regime} weighted_avg={regime_bias.weighted_avg_realized_pnl_pct:+.2f}%"
        )
    if strategy_bias is not None and item.winning_strategy:
        rationale.append(
            "Strategy context "
            f"{item.winning_strategy} weighted_avg="
            f"{strategy_bias.weighted_avg_realized_pnl_pct:+.2f}%"
        )
    setup_family_boost = 0.0
    setup_family_verdict = None
    if durable_family is not None:
        setup_family_boost = float(durable_family.score_adjustment or 0.0)
        setup_family_verdict = durable_family.verdict
        base_priority = round(base_priority + setup_family_boost, 4)
        rationale.append(setup_family_reinforcement_rationale(durable_family))
    nightly_symbol_feedback, nightly_report_count = nightly_feedback
    nightly_boost = _nightly_feedback_adjustment(
        item,
        nightly_symbol_feedback.get(normalize_exposure_key(item.symbol)),
        report_count=nightly_report_count,
        max_adjustment=float(
            getattr(settings, "training_candidate_nightly_feedback_max_boost", 0.08)
        ),
        settings=settings,
    )
    if nightly_boost != 0.0:
        base_priority = round(base_priority + nightly_boost, 4)
        rationale.append(
            _nightly_feedback_rationale(
                item,
                nightly_symbol_feedback.get(normalize_exposure_key(item.symbol)),
                report_count=nightly_report_count,
                adjustment=nightly_boost,
            )
        )
    remediation_boost, remediation_target, remediation_note = _remediation_pressure_adjustment(
        item,
        symbol_bias=symbol_bias,
        action_bias=action_bias,
        regime_bias=regime_bias,
        strategy_bias=strategy_bias,
        global_bias=global_bias,
        ml_candidate=ml_candidate,
        rl_candidate=rl_candidate,
        nightly_alignment=nightly_alignment,
        settings=settings,
    )
    if remediation_boost != 0.0:
        base_priority = round(base_priority + remediation_boost, 4)
        if remediation_note:
            rationale.append(remediation_note)
    if item.regime:
        regime_line = f"Market regime {item.regime}"
        if item.volume_ratio is not None:
            regime_line += f" (volume_ratio={item.volume_ratio:.2f})"
        rationale.append(regime_line)
    return TrainingCandidate(
        symbol=item.symbol,
        shortlist_rank=int(item.selection_rank or rank),
        selection_rank=rank,
        liquidity_score=item.liquidity_score,
        winning_strategy=item.winning_strategy,
        decision_action=action,
        decision_confidence=confidence,
        critique_verdict=verdict,
        ml_candidate=ml_candidate,
        rl_candidate=rl_candidate,
        market_regime=item.regime,
        volume_ratio=item.volume_ratio,
        priority_score=base_priority,
        adaptive_score_adjustment=round(
            total_bias + nightly_boost + remediation_boost + setup_family_boost + scout_bonus,
            4,
        )
        if (
            total_bias + nightly_boost + remediation_boost + setup_family_boost + scout_bonus
        )
        != 0.0
        else None,
        remediation_target=remediation_target,
        remediation_pressure=round(remediation_boost, 4) if remediation_boost != 0.0 else None,
        setup_family_reinforcement=round(setup_family_boost, 4)
        if setup_family_boost != 0.0
        else None,
        setup_family_verdict=setup_family_verdict,
        rationale=tuple(dict.fromkeys(rationale)),
    )


def _rerank_candidates(candidates: list[TrainingCandidate]) -> list[TrainingCandidate]:
    ranked = rank_with_exposure_penalties(
        candidates,
        symbol_of=lambda row: row.symbol,
        action_of=lambda row: str(row.decision_action or ""),
        base_score_of=lambda row: float(row.priority_score or 0.0),
        tie_breaker_of=lambda row: (row.shortlist_rank, row.symbol),
    )
    return [
        TrainingCandidate(
            symbol=result.item.symbol,
            shortlist_rank=result.item.shortlist_rank,
            selection_rank=idx,
            liquidity_score=result.item.liquidity_score,
            winning_strategy=result.item.winning_strategy,
            decision_action=result.item.decision_action,
            decision_confidence=result.item.decision_confidence,
            critique_verdict=result.item.critique_verdict,
            ml_candidate=result.item.ml_candidate,
            rl_candidate=result.item.rl_candidate,
            market_regime=result.item.market_regime,
            volume_ratio=result.item.volume_ratio,
            priority_score=result.adjusted_score,
            adaptive_score_adjustment=result.item.adaptive_score_adjustment,
            remediation_target=result.item.remediation_target,
            remediation_pressure=result.item.remediation_pressure,
            setup_family_reinforcement=result.item.setup_family_reinforcement,
            setup_family_verdict=result.item.setup_family_verdict,
            exposure_penalty=result.exposure_penalty,
            rationale=_candidate_rationale_with_penalty(
                result.item.rationale,
                exposure_penalty=result.exposure_penalty,
                symbol=result.item.symbol,
            ),
        )
        for idx, result in enumerate(ranked, start=1)
    ]


def _training_activity_bonus(
    *,
    volume_ratio: Optional[float],
    regime: str,
    settings: Settings,
) -> float:
    volume_weight = max(
        0.0,
        float(getattr(settings, "training_candidate_volume_ratio_bonus_weight", 0.08)),
    )
    trending_weight = max(
        0.0,
        float(getattr(settings, "training_candidate_trending_bonus_weight", 0.04)),
    )
    volume_component = 0.0
    if volume_ratio is not None:
        volume_component = min(1.0, max(0.0, float(volume_ratio) - 1.0) / 0.5) * volume_weight
    regime_key = str(regime or "").strip().upper()
    regime_component = 0.0
    if regime_key == "TRENDING":
        regime_component = trending_weight
    elif regime_key == "VOLATILE":
        regime_component = trending_weight * 0.5
    return round(volume_component + regime_component, 4)


def _remediation_pressure_adjustment(
    item: ShortlistAnalysisItem,
    *,
    symbol_bias,
    action_bias,
    regime_bias,
    strategy_bias,
    global_bias,
    ml_candidate: bool,
    rl_candidate: bool,
    nightly_alignment: NightlyAlignmentStatus,
    settings: Settings,
) -> tuple[float, Optional[str], Optional[str]]:
    if not bool(getattr(settings, "training_candidate_remediation_pressure_enabled", True)):
        return 0.0, None, None
    max_boost = max(
        0.0,
        float(getattr(settings, "training_candidate_remediation_max_boost", 0.06)),
    )
    if max_boost <= 0.0:
        return 0.0, None, None

    negative_learning = sum(
        abs(float(getattr(bias, "score_adjustment", 0.0) or 0.0)) * weight
        for bias, weight in (
            (symbol_bias, 0.35),
            (action_bias, 0.2),
            (regime_bias, 0.15),
            (strategy_bias, 0.15),
            (global_bias, 0.15),
        )
        if bias is not None and float(getattr(bias, "score_adjustment", 0.0) or 0.0) < 0.0
    )
    learning_pressure = min(max_boost * 0.55, negative_learning)

    remediation_target = (
        str(getattr(nightly_alignment, "effective_refresh_target", "") or "").strip().lower()
        or str(getattr(nightly_alignment, "recommended_refresh_target", "") or "").strip().lower()
        or str(getattr(nightly_alignment, "nightly_posture", "") or "").strip().lower()
    )
    if remediation_target not in {"ml", "rl", "all"}:
        remediation_target = ""
    matches_target = _candidate_matches_remediation_target(
        remediation_target=remediation_target,
        ml_candidate=ml_candidate,
        rl_candidate=rl_candidate,
    )
    nightly_pressure = 0.0
    if matches_target:
        if bool(getattr(nightly_alignment, "recommended_force_refresh", False)):
            nightly_pressure += max_boost * 0.25
        if bool(getattr(nightly_alignment, "workflow_target_mismatch", False)):
            nightly_pressure += max_boost * 0.2
        if str(getattr(nightly_alignment, "recommended_action", "") or "").strip():
            nightly_pressure += max_boost * 0.15
        if remediation_target in {"ml", "rl"}:
            nightly_pressure += max_boost * 0.1
        elif remediation_target == "all":
            nightly_pressure += max_boost * 0.05
    total_pressure = round(min(max_boost, learning_pressure + nightly_pressure), 4)
    if total_pressure <= 0.0:
        return 0.0, remediation_target or None, None
    note = (
        "Remediation pressure "
        f"target={(remediation_target or 'mixed').upper()} "
        f"learning={learning_pressure:+.2f} nightly={nightly_pressure:+.2f} "
        f"priority_adj={total_pressure:+.2f}"
    )
    return total_pressure, remediation_target or None, note


def _candidate_matches_remediation_target(
    *,
    remediation_target: str,
    ml_candidate: bool,
    rl_candidate: bool,
) -> bool:
    if remediation_target == "ml":
        return ml_candidate
    if remediation_target == "rl":
        return rl_candidate
    if remediation_target == "all":
        return ml_candidate or rl_candidate
    return False


def _training_discovery_posture_bonus(
    *,
    volume_ratio: Optional[float],
    regime: str,
    posture_hint: Optional[str],
    settings: Settings,
) -> float:
    posture_key = str(posture_hint or "").strip().lower()
    if posture_key not in {"ml", "rl"}:
        return 0.0
    volume_weight = max(
        0.0,
        float(getattr(settings, "training_candidate_volume_ratio_bonus_weight", 0.08)),
    )
    trending_weight = max(
        0.0,
        float(getattr(settings, "training_candidate_trending_bonus_weight", 0.04)),
    )
    regime_key = str(regime or "").strip().upper()
    volume_value = float(volume_ratio) if volume_ratio is not None else 1.0
    if posture_key == "rl":
        bonus = 0.0
        if regime_key == "TRENDING":
            bonus += trending_weight * 0.5
        elif regime_key == "VOLATILE":
            bonus += trending_weight * 0.25
        if volume_value >= 1.15:
            bonus += volume_weight * 0.25
        return round(bonus, 4)

    bonus = 0.0
    if regime_key == "RANGING":
        bonus += trending_weight
    if 0.95 <= volume_value <= 1.08:
        bonus += volume_weight * 0.3
    return round(bonus, 4)


def _discovery_posture_hint(items: Iterable[ShortlistAnalysisItem]) -> Optional[str]:
    rows = tuple(items or ())
    if not rows:
        return None
    trending_like = 0
    ranging = 0
    volume_ratios: list[float] = []
    for item in rows[:5]:
        regime_key = str(item.regime or "").strip().upper()
        if regime_key in {"TRENDING", "VOLATILE"}:
            trending_like += 1
        elif regime_key == "RANGING":
            ranging += 1
        if item.volume_ratio is not None:
            volume_ratios.append(float(item.volume_ratio))
    avg_volume_ratio = (sum(volume_ratios) / len(volume_ratios)) if volume_ratios else None
    if trending_like > ranging and avg_volume_ratio is not None and avg_volume_ratio >= 1.1:
        return "rl"
    if ranging >= trending_like and avg_volume_ratio is not None and avg_volume_ratio <= 1.05:
        return "ml"
    if trending_like == ranging and avg_volume_ratio is not None:
        if avg_volume_ratio >= 1.1:
            return "rl"
        if avg_volume_ratio <= 1.05:
            return "ml"
    if trending_like > ranging:
        return "rl"
    if ranging > trending_like:
        return "ml"
    return "all"


def _training_scout_cohort_bonus(
    *,
    item: ShortlistAnalysisItem,
    scout_overlap_symbols: set[str],
    scout_volume_dense_symbols: set[str],
    settings: Settings,
) -> float:
    symbol_key = _symbol_key(item.symbol)
    if not symbol_key:
        return 0.0
    overlap_boost = 0.0
    volume_dense_boost = 0.0
    if symbol_key in scout_overlap_symbols:
        overlap_boost = max(
            0.0,
            float(getattr(settings, "training_candidate_scout_overlap_boost", 0.04)),
        )
    if symbol_key in scout_volume_dense_symbols:
        volume_dense_boost = max(
            0.0,
            float(getattr(settings, "training_candidate_scout_volume_dense_boost", 0.03)),
        )
    return round(overlap_boost + volume_dense_boost, 4)


def _training_scout_cohort_rationale(
    *,
    item: ShortlistAnalysisItem,
    scout_overlap_symbols: set[str],
    scout_volume_dense_symbols: set[str],
    adjustment: float,
) -> tuple[str, ...]:
    symbol_key = _symbol_key(item.symbol)
    lines: list[str] = []
    if symbol_key in scout_overlap_symbols:
        lines.append(
            "Discovery overlap cohort reinforced this symbol across liquidity and activity scouts"
        )
    if symbol_key in scout_volume_dense_symbols:
        volume_ratio = float(item.volume_ratio or 1.0)
        lines.append(
            "Volume-dense scout cohort kept this symbol in rotation "
            f"(volume_ratio={volume_ratio:.2f}, priority_adj={adjustment:+.2f})"
        )
    return tuple(lines)


def _candidate_rationale_with_penalty(
    rationale: tuple[str, ...],
    *,
    exposure_penalty: float,
    symbol: str,
) -> tuple[str, ...]:
    out = list(rationale)
    if exposure_penalty > 0:
        out.append(
            "Exposure penalty "
            f"{exposure_penalty:.2f} applied for crowding/overlap in "
            f"{normalize_exposure_key(symbol)}"
        )
    return tuple(dict.fromkeys(out))


def _float_or_none(value) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _load_recent_nightly_feedback(
    settings: Settings,
) -> tuple[dict[str, _NightlyCandidateFeedback], int]:
    if not bool(getattr(settings, "training_candidate_nightly_feedback_enabled", True)):
        return {}, 0
    report_dir = settings.resolve_path(
        Path(getattr(settings, "training_candidate_nightly_report_dir", "reports/nightly"))
    )
    if not report_dir.is_dir():
        return {}, 0
    lookback = max(
        1,
        int(getattr(settings, "training_candidate_nightly_feedback_lookback_reports", 8)),
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
        basket_detail = _step_detail(step_rows, "basket_resolution")
        candidate_detail = _step_detail(step_rows, "training_candidates")
        research_detail = _step_detail(step_rows, "training_research")
        execution_detail = _step_detail(step_rows, "training_execution_target")
        promote_detail = _step_detail(step_rows, "promote_best")
        mode = str((basket_detail or {}).get("mode", "") or "").strip()
        if mode not in {"candidate_manifest", "derived_training_candidates"}:
            continue
        execution_target = _normalize_execution_target((execution_detail or {}).get("target"))
        execution_source = _normalize_execution_source(
            (execution_detail or {}).get("selection_source")
        )
        promote_count = _coerce_int((promote_detail or {}).get("count")) or 0
        report_used = False
        manifest_path_text = str((candidate_detail or {}).get("path", "") or "").strip()
        if manifest_path_text:
            manifest_path = _resolve_report_relative_path(report_path, manifest_path_text)
            manifest = _load_json_dict(manifest_path)
            candidate_rows = manifest.get("candidates") if manifest else None
            if isinstance(candidate_rows, list) and candidate_rows:
                report_used = True
                for row in candidate_rows:
                    if not isinstance(row, dict):
                        continue
                    key = normalize_exposure_key(str(row.get("symbol", "") or ""))
                    if not key:
                        continue
                    bucket = counts.setdefault(
                        key,
                        {
                            "symbol_reports": 0,
                            "ml_reports": 0,
                            "rl_reports": 0,
                            "research_reports": 0,
                            "research_ml_reports": 0,
                            "research_rl_reports": 0,
                            "refreshed_reports": 0,
                            "refreshed_ml_reports": 0,
                            "refreshed_rl_reports": 0,
                            "promoted_reports": 0,
                            "executed_reports": 0,
                            "executed_ml_reports": 0,
                            "executed_rl_reports": 0,
                        },
                    )
                    bucket["symbol_reports"] += 1
                    if bool(row.get("ml_candidate", False)):
                        bucket["ml_reports"] += 1
                    if bool(row.get("rl_candidate", False)):
                        bucket["rl_reports"] += 1
                    if execution_source == "training_candidates":
                        _apply_candidate_execution_feedback(
                            bucket,
                            row,
                            execution_target=execution_target,
                        )
                    if promote_count > 0:
                        bucket["promoted_reports"] += 1
        research_path_text = str((research_detail or {}).get("path", "") or "").strip()
        if research_path_text:
            research_path = _resolve_report_relative_path(report_path, research_path_text)
            research = _load_json_dict(research_path)
            research_rows = research.get("rows") if research else None
            if isinstance(research_rows, list) and research_rows:
                report_used = True
                for row in research_rows:
                    if not isinstance(row, dict):
                        continue
                    key = normalize_exposure_key(str(row.get("symbol", "") or ""))
                    if not key:
                        continue
                    bucket = counts.setdefault(
                        key,
                        {
                            "symbol_reports": 0,
                            "ml_reports": 0,
                            "rl_reports": 0,
                            "research_reports": 0,
                            "research_ml_reports": 0,
                            "research_rl_reports": 0,
                            "refreshed_reports": 0,
                            "refreshed_ml_reports": 0,
                            "refreshed_rl_reports": 0,
                            "promoted_reports": 0,
                            "executed_reports": 0,
                            "executed_ml_reports": 0,
                            "executed_rl_reports": 0,
                        },
                    )
                    bucket["research_reports"] += 1
                    target = str(row.get("target", "") or "").strip().lower()
                    if target in {"ml", "both"}:
                        bucket["research_ml_reports"] += 1
                    if target in {"rl", "both"}:
                        bucket["research_rl_reports"] += 1
                    if bool(row.get("refreshed", False)):
                        bucket["refreshed_reports"] += 1
                        if target in {"ml", "both"}:
                            bucket["refreshed_ml_reports"] += 1
                        if target in {"rl", "both"}:
                            bucket["refreshed_rl_reports"] += 1
                    if execution_source == "training_research":
                        _apply_research_execution_feedback(
                            bucket,
                            row,
                            execution_target=execution_target,
                        )
                    if promote_count > 0:
                        bucket["promoted_reports"] += 1
        if report_used:
            used_reports += 1
    return (
        {
            key: _NightlyCandidateFeedback(
                symbol_reports=value["symbol_reports"],
                ml_reports=value["ml_reports"],
                rl_reports=value["rl_reports"],
                research_reports=value["research_reports"],
                research_ml_reports=value["research_ml_reports"],
                research_rl_reports=value["research_rl_reports"],
                refreshed_reports=value["refreshed_reports"],
                refreshed_ml_reports=value["refreshed_ml_reports"],
                refreshed_rl_reports=value["refreshed_rl_reports"],
                promoted_reports=value["promoted_reports"],
                executed_reports=value["executed_reports"],
                executed_ml_reports=value["executed_ml_reports"],
                executed_rl_reports=value["executed_rl_reports"],
            )
            for key, value in counts.items()
        },
        used_reports,
    )


def _nightly_feedback_adjustment(
    item: ShortlistAnalysisItem,
    feedback: Optional[_NightlyCandidateFeedback],
    *,
    report_count: int,
    max_adjustment: float,
    settings: Settings,
) -> float:
    if feedback is None or report_count <= 0 or max_adjustment <= 0:
        return 0.0
    refresh_bonus_weight = max(
        0.0,
        float(getattr(settings, "training_candidate_nightly_refresh_bonus_weight", 0.7)),
    )
    research_weight = max(0.0, 1.0 - min(1.0, refresh_bonus_weight))
    execution_weight = min(1.0, refresh_bonus_weight + 0.2)
    evidence_reports = feedback.symbol_reports
    verdict = str(item.critique.verdict if item.critique is not None else "" or "").lower()
    if verdict == "candidate":
        evidence_reports = max(
            evidence_reports,
            feedback.rl_reports,
            feedback.executed_rl_reports,
        )
        evidence_reports += feedback.research_rl_reports * research_weight
        evidence_reports += feedback.refreshed_rl_reports * refresh_bonus_weight
        evidence_reports += feedback.executed_rl_reports * execution_weight
    elif verdict == "watch":
        evidence_reports = max(
            evidence_reports,
            feedback.ml_reports,
            feedback.executed_ml_reports,
        )
        evidence_reports += feedback.research_ml_reports * research_weight
        evidence_reports += feedback.refreshed_ml_reports * refresh_bonus_weight
        evidence_reports += feedback.executed_ml_reports * execution_weight
    else:
        evidence_reports += feedback.research_reports * (research_weight * 0.5)
        evidence_reports += feedback.refreshed_reports * (refresh_bonus_weight * 0.5)
        evidence_reports += feedback.executed_reports * (execution_weight * 0.5)
    evidence_ratio = min(1.0, evidence_reports / report_count)
    promoted_ratio = min(1.0, feedback.promoted_reports / report_count)
    adjustment = (evidence_ratio * max_adjustment * 0.65) + (promoted_ratio * max_adjustment * 0.35)
    return round(min(max_adjustment, adjustment), 4)


def _nightly_feedback_rationale(
    item: ShortlistAnalysisItem,
    feedback: Optional[_NightlyCandidateFeedback],
    *,
    report_count: int,
    adjustment: float,
) -> str:
    if feedback is None or report_count <= 0:
        return ""
    evidence_reports = feedback.symbol_reports
    verdict = str(item.critique.verdict if item.critique is not None else "" or "").lower()
    if verdict == "candidate":
        evidence_reports = max(evidence_reports, feedback.rl_reports, feedback.executed_rl_reports)
        research_reports = feedback.research_rl_reports
        refreshed_reports = feedback.refreshed_rl_reports
        executed_reports = feedback.executed_rl_reports
    elif verdict == "watch":
        evidence_reports = max(evidence_reports, feedback.ml_reports, feedback.executed_ml_reports)
        research_reports = feedback.research_ml_reports
        refreshed_reports = feedback.refreshed_ml_reports
        executed_reports = feedback.executed_ml_reports
    else:
        research_reports = feedback.research_reports
        refreshed_reports = feedback.refreshed_reports
        executed_reports = feedback.executed_reports
    return (
        "Recent nightly evidence kept this symbol in "
        f"{evidence_reports}/{report_count} candidate manifests "
        f"and {research_reports}/{report_count} research plans "
        f"(refreshed={refreshed_reports}, executed={executed_reports}, "
        f"promotions={feedback.promoted_reports}, +{adjustment:.2f} score adj)"
    )


def _normalize_execution_target(value: object) -> str:
    target = str(value or "").strip().lower()
    return target if target in {"ml", "rl", "all"} else ""


def _normalize_execution_source(value: object) -> str:
    source = str(value or "").strip().lower()
    return source if source in {"training_candidates", "training_research"} else ""


def _apply_candidate_execution_feedback(
    bucket: dict[str, int],
    row: dict[str, object],
    *,
    execution_target: str,
) -> None:
    if execution_target == "ml" and bool(row.get("ml_candidate", False)):
        bucket["executed_reports"] += 1
        bucket["executed_ml_reports"] += 1
    elif execution_target == "rl" and bool(row.get("rl_candidate", False)):
        bucket["executed_reports"] += 1
        bucket["executed_rl_reports"] += 1
    elif execution_target == "all":
        matched = False
        if bool(row.get("ml_candidate", False)):
            bucket["executed_ml_reports"] += 1
            matched = True
        if bool(row.get("rl_candidate", False)):
            bucket["executed_rl_reports"] += 1
            matched = True
        if matched:
            bucket["executed_reports"] += 1


def _apply_research_execution_feedback(
    bucket: dict[str, int],
    row: dict[str, object],
    *,
    execution_target: str,
) -> None:
    target = str(row.get("target", "") or "").strip().lower()
    if execution_target == "ml" and target in {"ml", "both"}:
        bucket["executed_reports"] += 1
        bucket["executed_ml_reports"] += 1
    elif execution_target == "rl" and target in {"rl", "both"}:
        bucket["executed_reports"] += 1
        bucket["executed_rl_reports"] += 1
    elif execution_target == "all" and target in {"ml", "rl", "both"}:
        bucket["executed_reports"] += 1
        if target in {"ml", "both"}:
            bucket["executed_ml_reports"] += 1
        if target in {"rl", "both"}:
            bucket["executed_rl_reports"] += 1


def _load_json_dict(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _step_detail(steps, name: str) -> Optional[dict]:
    for row in steps:
        if not isinstance(row, dict):
            continue
        if str(row.get("name", "")).strip() != name:
            continue
        detail = row.get("detail")
        return detail if isinstance(detail, dict) else None
    return None


def _resolve_report_relative_path(report_path: Path, artifact_path: str) -> Path:
    candidate = Path(str(artifact_path))
    if candidate.is_absolute():
        return candidate
    base = report_path.parent
    if len(report_path.parents) >= 3:
        base = report_path.parents[2]
    return base / candidate


def _coerce_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _symbol_key(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def _select_diversified_symbols(
    response: TrainingCandidateResponse,
    *,
    symbols: list[str],
    target: str,
    top_n: Optional[int],
) -> list[str]:
    if not symbols:
        return []
    by_symbol = {row.symbol: row for row in response.candidates}
    ordered_rows = [by_symbol[symbol] for symbol in symbols if symbol in by_symbol]
    if top_n is None or top_n >= len(ordered_rows):
        limit = len(ordered_rows)
    else:
        limit = top_n
    buckets: dict[tuple[str, str], list[TrainingCandidate]] = {}
    bucket_order: list[tuple[str, str]] = []
    for row in ordered_rows:
        key = (
            str(row.decision_action or "NA").upper(),
            str(row.market_regime or "UNKNOWN").upper(),
        )
        if key not in buckets:
            buckets[key] = []
            bucket_order.append(key)
        buckets[key].append(row)
    picked: list[str] = []
    while len(picked) < limit:
        progressed = False
        for key in bucket_order:
            bucket = buckets[key]
            if not bucket:
                continue
            picked.append(bucket.pop(0).symbol)
            progressed = True
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked
