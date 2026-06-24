"""Build a liquid, trend-aware market universe for advisory and training flows."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from fortuna.agentic.contracts import (
    AdvisoryError,
    AdvisoryErrorCode,
    MarketUniverseCandidate,
    MarketUniverseResponse,
)
from fortuna.app.fundamentals_overlay import (
    apply_fundamentals_overlay_to_candidate,
    load_fundamentals_overlay,
)
from fortuna.app.model_status import build_nightly_alignment_status
from fortuna.app.outcome_bias import (
    AdaptiveOutcomePolicy,
    OutcomeBias,
    exposure_key,
    load_adaptive_outcome_policy,
)
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRegistry
from fortuna.observability.recorder import workflow_boundary

_SYMBOL_COLUMNS = ("symbol", "ticker", "stock", "nse code", "nsecode")
_NAME_COLUMNS = ("name", "company", "company name")
_VOLUME_COLUMNS = ("volume", "avg volume", "average volume")
_PRICE_COLUMNS = ("cmp", "current price", "price", "ltp", "close")


@dataclass(frozen=True)
class UniverseSeed:
    symbol: str
    display_name: str
    source: str
    screener_volume: Optional[float] = None
    screener_price: Optional[float] = None


@dataclass(frozen=True)
class _NightlyUniverseFeedback:
    symbol_reports: int
    research_reports: int
    refreshed_reports: int
    promoted_reports: int
    executed_reports: int
    executed_lane_reports: int = 0
    executed_lane: str = ""


@dataclass(frozen=True)
class _UniverseSeedResolution:
    seeds: tuple[UniverseSeed, ...]
    resolved_source: str
    seed_source: str
    fallback_from: Optional[str] = None


def build_market_universe(
    *,
    settings: Settings,
    registry: Optional[InstrumentRegistry] = None,
    data_provider: Optional[Callable[[str, str, int], pd.DataFrame]] = None,
    limit: Optional[int] = None,
    timeframe: str = "1d",
    days: int = 30,
    source: str = "auto",
) -> MarketUniverseResponse:
    with workflow_boundary(
        settings,
        event_name="market_universe",
        module="fortuna.app.market_universe",
        workflow_id="market_universe",
        context={"source": source, "limit": limit},
    ) as span:
        response = _build_market_universe_impl(
            settings=settings,
            registry=registry,
            data_provider=data_provider,
            limit=limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )
        span.set_context(
            ok=response.ok,
            source=response.source,
            candidate_count=len(response.candidates),
        )
        if not response.ok:
            span.set_status("warn")
        return response


def _build_market_universe_impl(
    *,
    settings: Settings,
    registry: Optional[InstrumentRegistry] = None,
    data_provider: Optional[Callable[[str, str, int], pd.DataFrame]] = None,
    limit: Optional[int] = None,
    timeframe: str = "1d",
    days: int = 30,
    source: str = "auto",
) -> MarketUniverseResponse:
    reg = registry or InstrumentRegistry()
    reg.ensure_loaded()
    nightly_alignment = build_nightly_alignment_status(settings)
    nightly_alignment_summary = _nightly_alignment_summary(nightly_alignment)
    out_limit = max(1, int(limit or getattr(settings, "market_universe_default_limit", 15)))
    normalized_source = str(source or "auto").strip().lower()
    if normalized_source not in {"auto", "screener", "registry", "smartapi"}:
        return MarketUniverseResponse(
            ok=False,
            source=normalized_source,
            timeframe=timeframe,
            lookback_days=int(days),
            nightly_alignment_summary=nightly_alignment_summary,
            nightly_alignment_target=nightly_alignment.recommended_refresh_target,
            nightly_alignment_force_refresh=nightly_alignment.recommended_force_refresh,
            nightly_recent_window=nightly_alignment.recent_trend_window,
            nightly_recent_enabled=nightly_alignment.recent_trend_enabled,
            nightly_recent_aligned=nightly_alignment.recent_trend_aligned,
            nightly_recent_latest_status=nightly_alignment.recent_trend_latest_status,
            nightly_recent_latest_basket_size=nightly_alignment.recent_trend_latest_basket_size,
            error=AdvisoryError(
                code=AdvisoryErrorCode.INVALID_REQUEST,
                message="Universe source must be one of: auto, screener, registry, smartapi",
            ),
        )

    seed_resolution = _resolve_universe_seeds(settings, reg, normalized_source)
    seeds = list(seed_resolution.seeds)
    resolved_source = seed_resolution.resolved_source
    seed_source = seed_resolution.seed_source
    fallback_from = seed_resolution.fallback_from
    if not seeds:
        return MarketUniverseResponse(
            ok=False,
            source=resolved_source,
            timeframe=timeframe,
            lookback_days=int(days),
            nightly_alignment_summary=nightly_alignment_summary,
            nightly_alignment_target=nightly_alignment.recommended_refresh_target,
            nightly_alignment_force_refresh=nightly_alignment.recommended_force_refresh,
            nightly_recent_window=nightly_alignment.recent_trend_window,
            nightly_recent_enabled=nightly_alignment.recent_trend_enabled,
            nightly_recent_aligned=nightly_alignment.recent_trend_aligned,
            nightly_recent_latest_status=nightly_alignment.recent_trend_latest_status,
            nightly_recent_latest_basket_size=nightly_alignment.recent_trend_latest_basket_size,
            seed_source=seed_source,
            fallback_from=fallback_from,
            provider_summary=_build_provider_summary(
                seed_source=seed_source,
                scoring_source="seed_snapshot",
                fallback_from=fallback_from,
                resolved_source=resolved_source,
            ),
            error=AdvisoryError(
                code=AdvisoryErrorCode.LOAD_FAILED,
                message=(
                    "No market-universe candidates available from SmartAPI, "
                    "Screener, or registry"
                ),
            ),
        )

    outcome_biases = _load_outcome_biases(settings)
    nightly_feedback = _load_nightly_feedback(settings)
    overlay = load_fundamentals_overlay(settings, reg)
    fetch = data_provider or _default_data_provider(settings)
    candidates: list[MarketUniverseCandidate] = []
    excluded_count = 0
    fetch_limit = max(out_limit * 3, out_limit)
    for seed in seeds[:fetch_limit]:
        frame = _safe_fetch(fetch, seed.symbol, timeframe, days)
        if frame is not None and not frame.empty:
            candidate = _candidate_from_ohlcv(seed, frame, settings=settings)
        else:
            candidate = _candidate_from_seed_only(seed, settings=settings)
        candidate = _apply_outcome_bias(candidate, outcome_biases)
        candidate = _apply_nightly_feedback(candidate, nightly_feedback, settings=settings)
        if candidate is not None:
            overlay_row = overlay.rows.get(exposure_key(candidate.symbol))
            if overlay_row is not None:
                updated = apply_fundamentals_overlay_to_candidate(
                    candidate,
                    overlay_row,
                    settings=settings,
                )
                if updated is None:
                    excluded_count += 1
                    continue
                candidate = updated
            candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda c: (
            -(c.liquidity_score or -1.0),
            -(c.activity_score or -1.0),
            -(abs(c.trend_pct) if c.trend_pct is not None else -1.0),
            c.symbol,
        ),
    )
    scout_liquidity_symbols = _scout_liquidity_symbols(ranked)
    scout_activity_symbols = _scout_activity_symbols(ranked)
    scout_volume_dense_symbols = _scout_volume_dense_symbols(ranked)
    scout_overlap_symbols = _scout_overlap_symbols(
        liquidity_symbols=scout_liquidity_symbols,
        activity_symbols=scout_activity_symbols,
    )
    ohlcv_ranked_count = sum(
        1 for row in ranked if any(str(note) == "ohlcv_ranked" for note in row.notes)
    )
    scoring_source = "smartapi_ohlcv" if ohlcv_ranked_count > 0 else "seed_snapshot"
    provider_summary = _build_provider_summary(
        seed_source=seed_source,
        scoring_source=scoring_source,
        fallback_from=fallback_from,
        resolved_source=resolved_source,
    )
    overlay_summary = overlay.summary
    if overlay.enabled and excluded_count:
        overlay_summary = f"{overlay_summary} excluded={excluded_count}"
    return MarketUniverseResponse(
        ok=True,
        source=resolved_source,
        timeframe=timeframe,
        lookback_days=int(days),
        nightly_alignment_summary=nightly_alignment_summary,
        nightly_alignment_target=nightly_alignment.recommended_refresh_target,
        nightly_alignment_force_refresh=nightly_alignment.recommended_force_refresh,
        nightly_recent_window=nightly_alignment.recent_trend_window,
        nightly_recent_enabled=nightly_alignment.recent_trend_enabled,
        nightly_recent_aligned=nightly_alignment.recent_trend_aligned,
        nightly_recent_latest_status=nightly_alignment.recent_trend_latest_status,
        nightly_recent_latest_basket_size=nightly_alignment.recent_trend_latest_basket_size,
        scout_liquidity_symbols=scout_liquidity_symbols,
        scout_activity_symbols=scout_activity_symbols,
        scout_volume_dense_symbols=scout_volume_dense_symbols,
        scout_overlap_symbols=scout_overlap_symbols,
        scout_summary=_scout_summary(
            liquidity_symbols=scout_liquidity_symbols,
            activity_symbols=scout_activity_symbols,
            volume_dense_symbols=scout_volume_dense_symbols,
            overlap_symbols=scout_overlap_symbols,
        ),
        seed_source=seed_source,
        scoring_source=scoring_source,
        provider_summary=provider_summary,
        fallback_from=fallback_from,
        fundamentals_overlay_enabled=overlay.enabled,
        fundamentals_overlay_source=overlay.source_path,
        fundamentals_overlay_summary=overlay_summary if overlay.enabled else None,
        fundamentals_overlay_diagnostics=overlay.diagnostics,
        candidates=tuple(ranked[:out_limit]),
    )


def _default_data_provider(settings: Settings) -> Callable[[str, str, int], pd.DataFrame]:
    from fortuna.data.manager import MarketDataManager

    manager = MarketDataManager(settings, data_source="smartapi")
    return lambda symbol, timeframe, days: manager.get_ohlcv(
        symbol,
        timeframe,
        days=days,
        force_refresh=False,
    )


def _safe_fetch(
    fetch: Callable[[str, str, int], pd.DataFrame],
    symbol: str,
    timeframe: str,
    days: int,
) -> Optional[pd.DataFrame]:
    try:
        frame = fetch(symbol, timeframe, days)
    except Exception:
        return None
    if frame is None or frame.empty:
        return None
    return frame


def _candidate_from_ohlcv(
    seed: UniverseSeed,
    frame: pd.DataFrame,
    *,
    settings: Settings,
) -> Optional[MarketUniverseCandidate]:
    needed = {"close", "volume"}
    if not needed.issubset(frame.columns):
        return _candidate_from_seed_only(seed, settings=settings)
    recent = frame.tail(min(len(frame), 20)).copy()
    if recent.empty:
        return _candidate_from_seed_only(seed, settings=settings)
    close = pd.to_numeric(recent["close"], errors="coerce")
    volume = pd.to_numeric(recent["volume"], errors="coerce")
    turnover = close * volume
    avg_turnover = _safe_mean(turnover)
    avg_volume = _safe_mean(volume)
    last_close = _last_numeric(close)
    trend_pct = _trend_pct(close)
    volume_ratio = _volume_ratio(volume)
    activity_score = _activity_score(volume_ratio=volume_ratio, trend_pct=trend_pct)
    regime = _market_regime(trend_pct=trend_pct, volume_ratio=volume_ratio)
    liquidity_score = _liquidity_score(
        avg_turnover,
        avg_volume,
        trend_pct,
        settings=settings,
        volume_ratio=volume_ratio,
        activity_score=activity_score,
    )
    notes: list[str] = []
    if trend_pct is not None:
        direction = "uptrend" if trend_pct >= 0 else "downtrend"
        notes.append(direction)
    if regime:
        notes.append(f"regime={regime.lower()}")
    if volume_ratio is not None:
        notes.append(f"volume_ratio={volume_ratio:.2f}")
    notes.append("ohlcv_ranked")
    return MarketUniverseCandidate(
        symbol=seed.symbol,
        display_name=seed.display_name,
        source=seed.source,
        avg_turnover=avg_turnover,
        avg_volume=avg_volume,
        trend_pct=trend_pct,
        last_close=last_close,
        liquidity_score=liquidity_score,
        regime=regime,
        volume_ratio=volume_ratio,
        activity_score=activity_score,
        adaptive_score_adjustment=None,
        adaptive_row_count=0,
        notes=tuple(notes),
    )


def _candidate_from_seed_only(
    seed: UniverseSeed,
    *,
    settings: Settings,
) -> Optional[MarketUniverseCandidate]:
    avg_turnover = None
    if seed.screener_price is not None and seed.screener_volume is not None:
        avg_turnover = float(seed.screener_price) * float(seed.screener_volume)
    avg_volume = float(seed.screener_volume) if seed.screener_volume is not None else None
    last_close = float(seed.screener_price) if seed.screener_price is not None else None
    liquidity_score = _liquidity_score(
        avg_turnover,
        avg_volume,
        None,
        settings=settings,
    )
    if liquidity_score is None:
        return None
    return MarketUniverseCandidate(
        symbol=seed.symbol,
        display_name=seed.display_name,
        source=seed.source,
        avg_turnover=avg_turnover,
        avg_volume=avg_volume,
        trend_pct=None,
        last_close=last_close,
        liquidity_score=liquidity_score,
        regime="",
        volume_ratio=None,
        activity_score=None,
        adaptive_score_adjustment=None,
        adaptive_row_count=0,
        notes=("screener_snapshot_only",),
    )


def _load_outcome_biases(settings: Settings) -> AdaptiveOutcomePolicy:
    return load_adaptive_outcome_policy(
        settings,
        enabled=bool(getattr(settings, "market_universe_adaptive_weighting_enabled", True)),
        max_adjustment=float(getattr(settings, "market_universe_adaptive_max_boost", 0.2)),
    )


def _apply_outcome_bias(
    candidate: Optional[MarketUniverseCandidate],
    policy: AdaptiveOutcomePolicy,
) -> Optional[MarketUniverseCandidate]:
    if candidate is None or candidate.liquidity_score is None:
        return candidate
    symbol_bias = policy.symbol_biases.get(exposure_key(candidate.symbol))
    action_bias = policy.action_biases.get(_preferred_action(candidate))
    regime_bias = policy.regime_biases.get(str(candidate.regime or "").upper())
    signal_bias, signal_keys = _signal_bias_for_candidate(candidate, policy=policy)
    global_bias = policy.global_bias
    total_adjustment = sum(
        bias.score_adjustment
        for bias in (symbol_bias, action_bias, regime_bias, signal_bias, global_bias)
        if bias is not None
    )
    if total_adjustment == 0.0:
        return candidate
    notes = list(candidate.notes)
    notes.append(
        _adaptive_note(
            total_adjustment,
            symbol_bias,
            action_bias,
            regime_bias,
            signal_bias,
            signal_keys,
            global_bias,
        )
    )
    return MarketUniverseCandidate(
        symbol=candidate.symbol,
        display_name=candidate.display_name,
        source=candidate.source,
        avg_turnover=candidate.avg_turnover,
        avg_volume=candidate.avg_volume,
        trend_pct=candidate.trend_pct,
        last_close=candidate.last_close,
        liquidity_score=round(candidate.liquidity_score + total_adjustment, 6),
        regime=candidate.regime,
        volume_ratio=candidate.volume_ratio,
        activity_score=candidate.activity_score,
        adaptive_score_adjustment=round(total_adjustment, 4),
        adaptive_row_count=max(
            getattr(symbol_bias, "paper_closed_rows", 0),
            getattr(action_bias, "paper_closed_rows", 0),
            getattr(signal_bias, "paper_closed_rows", 0),
            getattr(global_bias, "paper_closed_rows", 0),
        ),
        notes=tuple(notes),
    )


def _signal_bias_for_candidate(
    candidate: MarketUniverseCandidate,
    *,
    policy: AdaptiveOutcomePolicy,
) -> tuple[Optional[OutcomeBias], tuple[str, ...]]:
    signal_keys = _preferred_signal_keys(candidate)
    matched = [
        policy.signal_biases[key]
        for key in signal_keys
        if key in policy.signal_biases and policy.signal_biases[key].score_adjustment != 0.0
    ]
    if not matched:
        return None, signal_keys
    count = float(len(matched))
    return (
        OutcomeBias(
            avg_realized_pnl_pct=round(
                sum(bias.avg_realized_pnl_pct for bias in matched) / count,
                4,
            ),
            paper_closed_rows=sum(int(bias.paper_closed_rows) for bias in matched),
            score_adjustment=round(
                sum(bias.score_adjustment for bias in matched) / count,
                4,
            ),
            weighted_avg_realized_pnl_pct=round(
                sum(bias.weighted_avg_realized_pnl_pct for bias in matched) / count,
                4,
            ),
        ),
        signal_keys,
    )


def _load_nightly_feedback(
    settings: Settings,
) -> tuple[dict[str, _NightlyUniverseFeedback], int]:
    if not bool(getattr(settings, "market_universe_nightly_feedback_enabled", True)):
        return {}, 0
    report_dir = settings.resolve_path(
        Path(getattr(settings, "market_universe_nightly_report_dir", "reports/nightly"))
    )
    if not report_dir.is_dir():
        return {}, 0
    lookback = max(
        1,
        int(getattr(settings, "market_universe_nightly_feedback_lookback_reports", 8)),
    )
    if bool(getattr(settings, "market_universe_long_horizon_feedback_enabled", False)):
        lookback = max(
            lookback,
            int(getattr(settings, "market_universe_long_horizon_lookback_reports", 12)),
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
            manifest = _load_json_dict(
                _resolve_report_relative_path(report_path, manifest_path_text)
            )
            rows = manifest.get("candidates") if manifest else None
            if isinstance(rows, list) and rows:
                report_used = True
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    key = exposure_key(str(row.get("symbol", "") or ""))
                    if not key:
                        continue
                    bucket = counts.setdefault(
                        key,
                        {
                            "symbol_reports": 0,
                            "research_reports": 0,
                            "refreshed_reports": 0,
                            "promoted_reports": 0,
                            "executed_reports": 0,
                            "executed_lane_reports": 0,
                            "executed_lane": "",
                        },
                    )
                    bucket["symbol_reports"] += 1
                    candidate_lane = _candidate_research_lane(row)
                    if _execution_matches_research_lane(
                        execution_target,
                        candidate_lane,
                    ):
                        bucket["executed_lane_reports"] += 1
                        if candidate_lane and not bucket["executed_lane"]:
                            bucket["executed_lane"] = candidate_lane
                    if (
                        execution_source == "training_candidates"
                        and _candidate_row_matches_execution(
                            row,
                            execution_target=execution_target,
                        )
                    ):
                        bucket["executed_reports"] += 1
                    if promote_count > 0:
                        bucket["promoted_reports"] += 1
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
                    key = exposure_key(str(row.get("symbol", "") or ""))
                    if not key:
                        continue
                    bucket = counts.setdefault(
                        key,
                        {
                            "symbol_reports": 0,
                            "research_reports": 0,
                            "refreshed_reports": 0,
                            "promoted_reports": 0,
                            "executed_reports": 0,
                            "executed_lane_reports": 0,
                            "executed_lane": "",
                        },
                    )
                    bucket["research_reports"] += 1
                    research_lane = str(row.get("target", "") or "").strip().lower()
                    if _execution_matches_research_lane(
                        execution_target,
                        research_lane,
                    ):
                        bucket["executed_lane_reports"] += 1
                        if research_lane and not bucket["executed_lane"]:
                            bucket["executed_lane"] = research_lane
                    if execution_source == "training_research" and _research_row_matches_execution(
                        row,
                        execution_target=execution_target,
                    ):
                        bucket["executed_reports"] += 1
                    if bool(row.get("refreshed", False)):
                        bucket["refreshed_reports"] += 1
                    if promote_count > 0:
                        bucket["promoted_reports"] += 1
        if report_used:
            used_reports += 1
    return (
        {
            key: _NightlyUniverseFeedback(
                symbol_reports=value["symbol_reports"],
                research_reports=value["research_reports"],
                refreshed_reports=value["refreshed_reports"],
                promoted_reports=value["promoted_reports"],
                executed_reports=value["executed_reports"],
                executed_lane_reports=value.get("executed_lane_reports", 0),
                executed_lane=str(value.get("executed_lane", "") or ""),
            )
            for key, value in counts.items()
        },
        used_reports,
    )


def _apply_nightly_feedback(
    candidate: Optional[MarketUniverseCandidate],
    nightly_feedback: tuple[dict[str, _NightlyUniverseFeedback], int],
    *,
    settings: Settings,
) -> Optional[MarketUniverseCandidate]:
    if candidate is None or candidate.liquidity_score is None:
        return candidate
    symbol_feedback, report_count = nightly_feedback
    feedback = symbol_feedback.get(exposure_key(candidate.symbol))
    max_adjustment = max(
        0.0,
        float(getattr(settings, "market_universe_nightly_feedback_max_boost", 0.1)),
    )
    adjustment = _nightly_feedback_adjustment(
        feedback,
        report_count=report_count,
        max_adjustment=max_adjustment,
        settings=settings,
    )
    lane_adjustment = _executed_lane_adjustment(
        feedback,
        report_count=report_count,
        max_adjustment=max_adjustment,
        settings=settings,
    )
    total_adjustment = round(min(max_adjustment, adjustment + lane_adjustment), 4)
    if total_adjustment == 0.0:
        return candidate
    notes = list(candidate.notes)
    notes.append(
        "nightly_feedback="
        f"{feedback.symbol_reports}/{report_count}"
        f" research={feedback.research_reports}"
        f" refreshed={feedback.refreshed_reports}"
        f" executed={feedback.executed_reports}"
        f" promotions={feedback.promoted_reports} adj={total_adjustment:+.2f}"
    )
    if lane_adjustment != 0.0 and feedback is not None:
        notes.append(
            "executed_lane="
            f"{feedback.executed_lane or '—'}"
            f" reports={feedback.executed_lane_reports}"
            f" boost={lane_adjustment:+.2f}"
        )
    prior_adjustment = float(candidate.adaptive_score_adjustment or 0.0)
    return MarketUniverseCandidate(
        symbol=candidate.symbol,
        display_name=candidate.display_name,
        source=candidate.source,
        avg_turnover=candidate.avg_turnover,
        avg_volume=candidate.avg_volume,
        trend_pct=candidate.trend_pct,
        last_close=candidate.last_close,
        liquidity_score=round(candidate.liquidity_score + total_adjustment, 6),
        regime=candidate.regime,
        volume_ratio=candidate.volume_ratio,
        activity_score=candidate.activity_score,
        adaptive_score_adjustment=round(prior_adjustment + total_adjustment, 4),
        adaptive_row_count=max(candidate.adaptive_row_count, feedback.symbol_reports),
        notes=tuple(notes),
    )


def _nightly_feedback_adjustment(
    feedback: Optional[_NightlyUniverseFeedback],
    *,
    report_count: int,
    max_adjustment: float,
    settings: Settings,
) -> float:
    if feedback is None or report_count <= 0 or max_adjustment <= 0:
        return 0.0
    refresh_bonus_weight = max(
        0.0,
        float(getattr(settings, "market_universe_nightly_refresh_bonus_weight", 0.65)),
    )
    research_weight = max(0.0, 1.0 - min(1.0, refresh_bonus_weight))
    evidence_reports = (
        feedback.symbol_reports
        + (feedback.research_reports * research_weight)
        + (feedback.refreshed_reports * refresh_bonus_weight)
        + (feedback.executed_reports * min(1.0, refresh_bonus_weight + 0.2))
    )
    evidence_ratio = min(1.0, evidence_reports / report_count)
    promoted_ratio = min(1.0, feedback.promoted_reports / report_count)
    adjustment = (evidence_ratio * max_adjustment * 0.7) + (promoted_ratio * max_adjustment * 0.3)
    return round(min(max_adjustment, adjustment), 4)


def _executed_lane_adjustment(
    feedback: Optional[_NightlyUniverseFeedback],
    *,
    report_count: int,
    max_adjustment: float,
    settings: Settings,
) -> float:
    if not bool(getattr(settings, "market_universe_long_horizon_feedback_enabled", False)):
        return 0.0
    if feedback is None or report_count <= 0 or max_adjustment <= 0:
        return 0.0
    if feedback.executed_lane_reports <= 0:
        return 0.0
    lane_weight = max(
        0.0,
        float(getattr(settings, "market_universe_executed_lane_boost_weight", 0.04)),
    )
    if lane_weight <= 0.0:
        return 0.0
    lane_ratio = min(1.0, feedback.executed_lane_reports / report_count)
    return round(min(max_adjustment, lane_ratio * lane_weight), 4)


def _candidate_research_lane(row: dict[str, Any]) -> str:
    ml_candidate = bool(row.get("ml_candidate", False))
    rl_candidate = bool(row.get("rl_candidate", False))
    if ml_candidate and rl_candidate:
        return "both"
    if ml_candidate:
        return "ml"
    if rl_candidate:
        return "rl"
    return str(row.get("target", "") or "").strip().lower()


def _execution_matches_research_lane(
    execution_target: str,
    research_lane: str,
) -> bool:
    lane = str(research_lane or "").strip().lower()
    target = str(execution_target or "").strip().lower()
    if not lane or not target:
        return False
    if lane == "both":
        return target in {"all", "ml", "rl"}
    if target == "all":
        return True
    return target == lane


def _normalize_execution_target(value: object) -> str:
    target = str(value or "").strip().lower()
    return target if target in {"ml", "rl", "all"} else ""


def _normalize_execution_source(value: object) -> str:
    source = str(value or "").strip().lower()
    return source if source in {"training_candidates", "training_research"} else ""


def _candidate_row_matches_execution(row: dict[str, Any], *, execution_target: str) -> bool:
    if execution_target == "ml":
        return bool(row.get("ml_candidate", False))
    if execution_target == "rl":
        return bool(row.get("rl_candidate", False))
    if execution_target == "all":
        return bool(row.get("ml_candidate", False) or row.get("rl_candidate", False))
    return False


def _research_row_matches_execution(row: dict[str, Any], *, execution_target: str) -> bool:
    target = str(row.get("target", "") or "").strip().lower()
    if execution_target == "ml":
        return target in {"ml", "both"}
    if execution_target == "rl":
        return target in {"rl", "both"}
    if execution_target == "all":
        return target in {"ml", "rl", "both"}
    return False


def _safe_mean(series: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _last_numeric(series: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def _trend_pct(close: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])
    if start == 0:
        return None
    return ((end / start) - 1.0) * 100.0


def _liquidity_score(
    avg_turnover: Optional[float],
    avg_volume: Optional[float],
    trend_pct: Optional[float],
    *,
    settings: Settings,
    volume_ratio: Optional[float] = None,
    activity_score: Optional[float] = None,
) -> Optional[float]:
    if avg_turnover is None and avg_volume is None:
        return None
    weights = _normalized_universe_weights(settings)
    turnover_term = min(1.0, math.log1p(max(0.0, float(avg_turnover or 0.0))) / 16.0)
    volume_term = min(1.0, math.log1p(max(0.0, float(avg_volume or 0.0))) / 14.0)
    trend_term = min(1.0, abs(float(trend_pct or 0.0)) / 5.0)
    volume_ratio_term = min(1.0, max(0.0, float(volume_ratio or 1.0) - 1.0) / 0.8)
    activity_term = min(1.0, max(0.0, float(activity_score or 0.0)) / 1.5)
    return round(
        (
            turnover_term * weights["turnover"]
            + volume_term * weights["volume"]
            + trend_term * weights["trend"]
            + volume_ratio_term * weights["volume_ratio"]
            + activity_term * weights["activity"]
        )
        * 10.0,
        6,
    )


def _normalized_universe_weights(settings: Settings) -> dict[str, float]:
    raw = {
        "turnover": max(0.0, float(getattr(settings, "market_universe_turnover_weight", 0.46))),
        "volume": max(0.0, float(getattr(settings, "market_universe_volume_weight", 0.18))),
        "trend": max(0.0, float(getattr(settings, "market_universe_trend_weight", 0.08))),
        "volume_ratio": max(
            0.0,
            float(getattr(settings, "market_universe_volume_ratio_weight", 0.10)),
        ),
        "activity": max(0.0, float(getattr(settings, "market_universe_activity_weight", 0.18))),
    }
    total = sum(raw.values())
    if total <= 0.0:
        return {
            "turnover": 0.46,
            "volume": 0.18,
            "trend": 0.08,
            "volume_ratio": 0.10,
            "activity": 0.18,
        }
    return {key: value / total for key, value in raw.items()}


def _volume_ratio(volume: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(volume, errors="coerce").dropna()
    if len(clean) < 4:
        return None
    midpoint = max(1, len(clean) // 2)
    early = float(clean.iloc[:midpoint].mean())
    late = float(clean.iloc[midpoint:].mean())
    if early <= 0:
        return None
    return late / early


def _activity_score(
    *,
    volume_ratio: Optional[float],
    trend_pct: Optional[float],
) -> Optional[float]:
    if volume_ratio is None and trend_pct is None:
        return None
    trend_component = min(2.0, abs(float(trend_pct or 0.0)) / 3.0)
    volume_component = min(2.0, max(0.0, float(volume_ratio or 1.0) - 1.0) * 4.0)
    return round(trend_component * 0.55 + volume_component * 0.45, 6)


def _market_regime(
    *,
    trend_pct: Optional[float],
    volume_ratio: Optional[float],
) -> str:
    trend = float(trend_pct or 0.0)
    vol_ratio = float(volume_ratio or 1.0)
    if abs(trend) >= 4.0 or (abs(trend) >= 2.0 and vol_ratio >= 1.15):
        return "TRENDING"
    if abs(trend) <= 1.0 and vol_ratio >= 1.3:
        return "VOLATILE"
    return "RANGING"


def _preferred_action(candidate: MarketUniverseCandidate) -> str:
    trend = float(candidate.trend_pct or 0.0)
    if trend > 0:
        return "BUY"
    if trend < 0:
        return "SELL"
    return ""


def _preferred_signal_keys(candidate: MarketUniverseCandidate) -> tuple[str, ...]:
    regime = str(candidate.regime or "").upper()
    volume_ratio = float(candidate.volume_ratio or 1.0)
    trend = abs(float(candidate.trend_pct or 0.0))
    keys: list[str] = []
    if regime == "TRENDING":
        if volume_ratio >= 1.05:
            keys.extend(["ORB", "VWAP"])
        elif trend >= 4.0:
            keys.append("VWAP")
    elif regime == "VOLATILE":
        keys.extend(["IVORB", "LSVWAP", "ORB"])
    elif regime == "RANGING":
        keys.append("MMTS")
    return tuple(dict.fromkeys(keys))


def _scout_liquidity_symbols(
    candidates: list[MarketUniverseCandidate],
    limit: int = 3,
) -> tuple[str, ...]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            -(float(getattr(row, "liquidity_score", 0.0) or 0.0)),
            -(float(getattr(row, "avg_turnover", 0.0) or 0.0)),
            row.symbol,
        ),
    )
    return tuple(row.symbol for row in ranked[: max(1, int(limit))])


def _scout_activity_symbols(
    candidates: list[MarketUniverseCandidate],
    limit: int = 3,
) -> tuple[str, ...]:
    ranked = sorted(
        [
            row
            for row in candidates
            if (
                getattr(row, "activity_score", None) is not None
                or getattr(row, "trend_pct", None) is not None
                or getattr(row, "volume_ratio", None) is not None
            )
        ],
        key=lambda row: (
            -(float(getattr(row, "activity_score", 0.0) or 0.0)),
            -(abs(float(getattr(row, "trend_pct", 0.0) or 0.0))),
            row.symbol,
        ),
    )
    return tuple(row.symbol for row in ranked[: max(1, int(limit))])


def _scout_volume_dense_symbols(
    candidates: list[MarketUniverseCandidate],
    limit: int = 3,
) -> tuple[str, ...]:
    ranked = sorted(
        [row for row in candidates if getattr(row, "volume_ratio", None) is not None],
        key=lambda row: (
            -(float(getattr(row, "volume_ratio", 0.0) or 0.0)),
            -(float(getattr(row, "activity_score", 0.0) or 0.0)),
            -(float(getattr(row, "avg_turnover", 0.0) or 0.0)),
            row.symbol,
        ),
    )
    return tuple(row.symbol for row in ranked[: max(1, int(limit))])


def _scout_overlap_symbols(
    *,
    liquidity_symbols: tuple[str, ...],
    activity_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    activity_top = set(activity_symbols)
    return tuple(symbol for symbol in liquidity_symbols if symbol in activity_top)


def _scout_summary(
    *,
    liquidity_symbols: tuple[str, ...],
    activity_symbols: tuple[str, ...],
    volume_dense_symbols: tuple[str, ...],
    overlap_symbols: tuple[str, ...],
) -> Optional[str]:
    parts: list[str] = []
    if liquidity_symbols:
        parts.append(f"liquidity={','.join(liquidity_symbols[:2])}")
    if activity_symbols:
        parts.append(f"activity={','.join(activity_symbols[:2])}")
    if volume_dense_symbols:
        parts.append(f"volume_dense={','.join(volume_dense_symbols[:2])}")
    if overlap_symbols:
        parts.append(f"overlap={','.join(overlap_symbols[:2])}")
    if not parts:
        return None
    return " | ".join(parts)


def _adaptive_note(
    total_adjustment: float,
    symbol_bias,
    action_bias,
    regime_bias,
    signal_bias,
    signal_keys: tuple[str, ...],
    global_bias,
) -> str:
    direction = "boost" if total_adjustment > 0 else "penalty"
    parts = [f"adaptive_{direction}={total_adjustment:+.2f}"]
    if symbol_bias is not None:
        parts.append(
            "symbol="
            f"{symbol_bias.score_adjustment:+.2f}"
            f" weighted_avg={symbol_bias.weighted_avg_realized_pnl_pct:+.2f}%"
            f" rows={symbol_bias.paper_closed_rows}"
        )
    if action_bias is not None:
        parts.append(
            "action="
            f"{action_bias.score_adjustment:+.2f}"
            f" weighted_avg={action_bias.weighted_avg_realized_pnl_pct:+.2f}%"
            f" rows={action_bias.paper_closed_rows}"
        )
    if regime_bias is not None:
        parts.append(
            "regime="
            f"{regime_bias.score_adjustment:+.2f}"
            f" weighted_avg={regime_bias.weighted_avg_realized_pnl_pct:+.2f}%"
            f" rows={regime_bias.paper_closed_rows}"
        )
    if signal_bias is not None:
        label = ",".join(signal_keys) if signal_keys else "signal"
        parts.append(
            f"signal[{label}]="
            f"{signal_bias.score_adjustment:+.2f}"
            f" weighted_avg={signal_bias.weighted_avg_realized_pnl_pct:+.2f}%"
            f" rows={signal_bias.paper_closed_rows}"
        )
    if global_bias is not None:
        parts.append(
            "global="
            f"{global_bias.score_adjustment:+.2f}"
            f" weighted_avg={global_bias.weighted_avg_realized_pnl_pct:+.2f}%"
            f" rows={global_bias.paper_closed_rows}"
        )
    return " ".join(parts)


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


def resolve_market_universe_cli_source(settings: Settings) -> str:
    """Pick a concrete CLI `--source` for discovery refresh recommendations."""
    if bool(getattr(settings, "market_universe_auto_prefer_smartapi", False)):
        return "smartapi"
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    if screener_path.is_file():
        return "screener"
    return "smartapi"


def _resolve_universe_seeds(
    settings: Settings,
    registry: InstrumentRegistry,
    source: str,
) -> _UniverseSeedResolution:
    normalized = str(source or "auto").strip().lower()
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    tradable_path = settings.resolve_path(settings.market_universe_tradable_universe_csv)
    prefer_smartapi = bool(getattr(settings, "market_universe_auto_prefer_smartapi", False))
    seed_limit = max(1, int(getattr(settings, "market_universe_smartapi_seed_limit", 75)))

    def smartapi_seeds() -> list[UniverseSeed]:
        if tradable_path.is_file():
            seeds = _load_tradable_universe_seeds(tradable_path, registry)
            if seeds:
                return seeds
        return _load_smartapi_registry_seeds(registry, limit=seed_limit)

    if normalized == "registry":
        return _UniverseSeedResolution(
            seeds=tuple(_registry_fallback_seeds(registry)),
            resolved_source="registry",
            seed_source="registry",
        )

    if normalized == "screener":
        seeds: list[UniverseSeed] = []
        if screener_path.is_file():
            seeds = _load_screener_seeds(screener_path, registry)
        if seeds:
            return _UniverseSeedResolution(
                seeds=tuple(seeds),
                resolved_source="screener",
                seed_source="screener",
            )
        fallback = _registry_fallback_seeds(registry)
        return _UniverseSeedResolution(
            seeds=tuple(fallback),
            resolved_source="registry",
            seed_source="screener",
            fallback_from="screener",
        )

    if normalized == "smartapi":
        seeds = smartapi_seeds()
        if seeds:
            return _UniverseSeedResolution(
                seeds=tuple(seeds),
                resolved_source="smartapi",
                seed_source="smartapi",
            )
        fallback = _registry_fallback_seeds(registry)
        return _UniverseSeedResolution(
            seeds=tuple(fallback),
            resolved_source="registry",
            seed_source="smartapi",
            fallback_from="smartapi",
        )

    if prefer_smartapi:
        seeds = smartapi_seeds()
        if seeds:
            return _UniverseSeedResolution(
                seeds=tuple(seeds),
                resolved_source="smartapi",
                seed_source="smartapi",
            )
        if screener_path.is_file():
            seeds = _load_screener_seeds(screener_path, registry)
            if seeds:
                return _UniverseSeedResolution(
                    seeds=tuple(seeds),
                    resolved_source="screener",
                    seed_source="screener",
                    fallback_from="smartapi",
                )
    else:
        if screener_path.is_file():
            seeds = _load_screener_seeds(screener_path, registry)
            if seeds:
                return _UniverseSeedResolution(
                    seeds=tuple(seeds),
                    resolved_source="screener",
                    seed_source="screener",
                )
        seeds = smartapi_seeds()
        if seeds:
            return _UniverseSeedResolution(
                seeds=tuple(seeds),
                resolved_source="smartapi",
                seed_source="smartapi",
            )

    fallback = _registry_fallback_seeds(registry)
    return _UniverseSeedResolution(
        seeds=tuple(fallback),
        resolved_source="registry",
        seed_source="registry",
        fallback_from="auto",
    )


def _build_provider_summary(
    *,
    seed_source: str,
    scoring_source: str,
    resolved_source: str,
    fallback_from: Optional[str] = None,
) -> str:
    parts = [f"seeds={seed_source}", f"scoring={scoring_source}"]
    if fallback_from and fallback_from != resolved_source:
        parts.append(f"fallback={fallback_from}->{resolved_source}")
    return " ".join(parts)


def _load_tradable_universe_seeds(path: Path, registry: InstrumentRegistry) -> list[UniverseSeed]:
    rows: list[UniverseSeed] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            normalized = {str(k or "").strip().lower(): v for k, v in raw.items()}
            symbol_text = _first_value(normalized, _SYMBOL_COLUMNS)
            if not symbol_text:
                continue
            resolved = _resolve_seed_symbol(symbol_text, registry)
            if resolved is None:
                continue
            display_name = _first_value(normalized, _NAME_COLUMNS) or resolved.replace(".NS", "")
            rows.append(
                UniverseSeed(
                    symbol=resolved,
                    display_name=str(display_name).strip(),
                    source="smartapi",
                )
            )
    deduped: dict[str, UniverseSeed] = {}
    for row in rows:
        deduped.setdefault(row.symbol, row)
    return list(deduped.values())


def _load_smartapi_registry_seeds(
    registry: InstrumentRegistry,
    *,
    limit: int,
) -> list[UniverseSeed]:
    catalog_fn = getattr(registry, "catalog", None)
    if callable(catalog_fn):
        hits = catalog_fn()
    else:
        hits = registry.search("", limit=limit)
    out: list[UniverseSeed] = []
    for hit in hits:
        if getattr(hit, "segment", "") != "EQUITY":
            continue
        out.append(
            UniverseSeed(
                symbol=str(hit.symbol),
                display_name=str(hit.display).split(" — ")[0],
                source="smartapi",
            )
        )
        if len(out) >= limit:
            break
    return out


def _load_screener_seeds(path: Path, registry: InstrumentRegistry) -> list[UniverseSeed]:
    rows: list[UniverseSeed] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            normalized = {str(k or "").strip().lower(): v for k, v in raw.items()}
            symbol_text = _first_value(normalized, _SYMBOL_COLUMNS)
            if not symbol_text:
                continue
            resolved = _resolve_seed_symbol(symbol_text, registry)
            if resolved is None:
                continue
            display_name = _first_value(normalized, _NAME_COLUMNS) or resolved.replace(".NS", "")
            rows.append(
                UniverseSeed(
                    symbol=resolved,
                    display_name=str(display_name).strip(),
                    source="screener",
                    screener_volume=_parse_float(_first_value(normalized, _VOLUME_COLUMNS)),
                    screener_price=_parse_float(_first_value(normalized, _PRICE_COLUMNS)),
                )
            )
    deduped: dict[str, UniverseSeed] = {}
    for row in rows:
        deduped.setdefault(row.symbol, row)
    return list(deduped.values())


def _registry_fallback_seeds(registry: InstrumentRegistry) -> list[UniverseSeed]:
    hits = registry.search("", limit=25)
    out: list[UniverseSeed] = []
    for hit in hits:
        if getattr(hit, "segment", "") != "EQUITY":
            continue
        out.append(
            UniverseSeed(
                symbol=str(hit.symbol),
                display_name=str(hit.display).split(" — ")[0],
                source="registry",
            )
        )
    return out


def _resolve_seed_symbol(text: str, registry: InstrumentRegistry) -> Optional[str]:
    candidate = str(text or "").strip().upper()
    if not candidate:
        return None
    variants = (candidate, f"{candidate}.NS", f"{candidate}-EQ")
    for item in variants:
        try:
            ref = registry.resolve(item)
            if ref.is_future or ref.is_option:
                continue
            return f"{ref.symbol}.NS"
        except Exception:
            continue
    return None


def _parse_float(value: Any) -> Optional[float]:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_value(row: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row and str(row[key] or "").strip():
            return str(row[key]).strip()
    return None


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
