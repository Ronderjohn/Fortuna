"""Shared recent-outcome weighting helpers for advisory symbol ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from fortuna.agentic.learning import LearningExample
from fortuna.agentic.store import AgenticLearningStore
from fortuna.config.settings import Settings


@dataclass(frozen=True)
class OutcomeBias:
    avg_realized_pnl_pct: float
    paper_closed_rows: int
    score_adjustment: float
    weighted_avg_realized_pnl_pct: float


@dataclass(frozen=True)
class SetupFamilyReinforcement:
    family: str
    verdict: str
    paper_closed_rows: int
    weighted_avg_realized_pnl_pct: float
    score_adjustment: float


@dataclass(frozen=True)
class AdaptiveOutcomePolicy:
    symbol_biases: dict[str, OutcomeBias]
    action_biases: dict[str, OutcomeBias]
    regime_biases: dict[str, OutcomeBias]
    signal_biases: dict[str, OutcomeBias]
    durable_setup_family_biases: dict[str, SetupFamilyReinforcement]
    global_bias: Optional[OutcomeBias] = None


def load_symbol_outcome_biases(
    settings: Settings,
    *,
    enabled: bool,
    max_adjustment: float,
) -> dict[str, OutcomeBias]:
    return load_adaptive_outcome_policy(
        settings,
        enabled=enabled,
        max_adjustment=max_adjustment,
    ).symbol_biases


def setup_family_key_from_row(row: LearningExample) -> str:
    metadata = row.metadata or {}
    for key in ("primary_signal", "winning_strategy", "strategy_name"):
        value = str(metadata.get(key, "") or "").strip().upper()
        if value:
            return value
    return ""


def setup_family_reinforcement_rationale(reinforcement: SetupFamilyReinforcement) -> str:
    return (
        f"Setup-family reinforcement {reinforcement.family}: recurring "
        f"{reinforcement.verdict} ({reinforcement.score_adjustment:+.2f}, "
        f"{reinforcement.paper_closed_rows} durable rows, "
        f"weighted_avg={reinforcement.weighted_avg_realized_pnl_pct:+.2f}%)"
    )


def load_adaptive_outcome_policy(
    settings: Settings,
    *,
    enabled: bool,
    max_adjustment: float,
) -> AdaptiveOutcomePolicy:
    empty = AdaptiveOutcomePolicy(
        symbol_biases={},
        action_biases={},
        regime_biases={},
        signal_biases={},
        durable_setup_family_biases={},
    )
    if not enabled:
        return empty
    try:
        store = AgenticLearningStore(settings.resolve_path(settings.agentic_log_dir))
        rows = list(store.read_all(limit=250))
    except Exception:
        return empty

    closed_rows = [row for row in rows if _has_realized_outcome(row)]
    if not closed_rows:
        return empty

    max_boost = max(0.0, float(max_adjustment))
    recency_halflife = max(
        1.0,
        float(getattr(settings, "market_universe_adaptive_recency_halflife", 18.0)),
    )
    weighted_rows = _weighted_rows(closed_rows, halflife=recency_halflife)
    symbol_biases = _group_biases(
        weighted_rows,
        key_fn=lambda row: exposure_key(row.symbol),
        max_adjustment=max_boost,
    )
    if bool(getattr(settings, "market_universe_long_horizon_feedback_enabled", False)):
        long_halflife = max(
            recency_halflife + 1.0,
            float(getattr(settings, "market_universe_long_horizon_halflife", 48.0)),
        )
        long_weighted_rows = _weighted_rows(closed_rows, halflife=long_halflife)
        long_symbol_biases = _group_biases(
            long_weighted_rows,
            key_fn=lambda row: exposure_key(row.symbol),
            max_adjustment=max_boost,
        )
        symbol_biases = _blend_horizon_symbol_biases(
            symbol_biases,
            long_symbol_biases,
            max_adjustment=max_boost,
        )
    action_biases = _group_biases(
        weighted_rows,
        key_fn=lambda row: str(row.action or "").upper(),
        max_adjustment=max_boost * float(
            getattr(settings, "market_universe_action_bias_weight", 0.45)
        ),
    )
    regime_biases = _group_biases(
        weighted_rows,
        key_fn=lambda row: str((row.metadata or {}).get("regime", "") or "").upper(),
        max_adjustment=max_boost * float(
            getattr(settings, "market_universe_regime_bias_weight", 0.35)
        ),
    )
    signal_biases = _group_biases(
        weighted_rows,
        key_fn=setup_family_key_from_row,
        max_adjustment=max_boost * float(
            getattr(settings, "market_universe_signal_bias_weight", 0.25)
        ),
    )
    durable_setup_family_biases = _durable_setup_family_biases(
        weighted_rows,
        settings=settings,
    )
    if durable_setup_family_biases:
        signal_biases = {
            key: value
            for key, value in signal_biases.items()
            if key not in durable_setup_family_biases
        }
    global_bias = _summarize_bias(
        weighted_rows,
        max_adjustment=max_boost * float(
            getattr(settings, "market_universe_global_bias_weight", 0.2)
        ),
    )
    return AdaptiveOutcomePolicy(
        symbol_biases=symbol_biases,
        action_biases=action_biases,
        regime_biases=regime_biases,
        signal_biases=signal_biases,
        durable_setup_family_biases=durable_setup_family_biases,
        global_bias=global_bias,
    )


def adaptive_score_adjustment(
    avg_realized_pnl_pct: float,
    sample_size: int,
    *,
    max_adjustment: float,
) -> float:
    if sample_size <= 0:
        return 0.0
    strength = min(max_adjustment, abs(avg_realized_pnl_pct) / 10.0)
    if sample_size < 2:
        strength = min(strength, max_adjustment * 0.35)
    if avg_realized_pnl_pct > 0.25:
        return round(strength, 4)
    if avg_realized_pnl_pct < -0.25:
        return round(-strength, 4)
    return 0.0


def exposure_key(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    return text.split(".", 1)[0]


def _has_realized_outcome(row: LearningExample) -> bool:
    return bool(row.outcome.paper_closed) and row.outcome.realized_pnl_pct is not None


def _durable_setup_family_biases(
    weighted_rows: list[tuple[LearningExample, float]],
    *,
    settings: Settings,
) -> dict[str, SetupFamilyReinforcement]:
    if not bool(
        getattr(settings, "training_candidate_setup_family_reinforcement_enabled", True)
    ):
        return {}
    min_durable_rows = max(
        1,
        int(getattr(settings, "training_candidate_setup_family_min_durable_rows", 2)),
    )
    max_boost = max(
        0.0,
        float(getattr(settings, "training_candidate_setup_family_max_boost", 0.05)),
    )
    if max_boost <= 0.0:
        return {}

    grouped: dict[str, list[tuple[float, float]]] = {}
    for row, weight in weighted_rows:
        key = setup_family_key_from_row(row)
        if not key:
            continue
        grouped.setdefault(key, []).append(
            (float(row.outcome.realized_pnl_pct or 0.0), float(weight))
        )

    out: dict[str, SetupFamilyReinforcement] = {}
    for family, values in grouped.items():
        if len(values) < min_durable_rows:
            continue
        bias = _summarize_bias(values, max_adjustment=max_boost)
        if bias is None or bias.score_adjustment == 0.0:
            continue
        verdict = "winner" if bias.score_adjustment > 0.0 else "loser"
        out[family] = SetupFamilyReinforcement(
            family=family,
            verdict=verdict,
            paper_closed_rows=bias.paper_closed_rows,
            weighted_avg_realized_pnl_pct=bias.weighted_avg_realized_pnl_pct,
            score_adjustment=bias.score_adjustment,
        )
    return out


def _weighted_rows(
    rows: list[LearningExample],
    *,
    halflife: float,
) -> list[tuple[LearningExample, float]]:
    ordered = sorted(rows, key=lambda row: row.bar_time or "")
    total = len(ordered)
    out: list[tuple[LearningExample, float]] = []
    for idx, row in enumerate(ordered):
        distance = total - idx - 1
        weight = math.pow(0.5, distance / halflife)
        out.append((row, weight))
    return out


def _group_biases(
    weighted_rows: list[tuple[LearningExample, float]],
    *,
    key_fn,
    max_adjustment: float,
) -> dict[str, OutcomeBias]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row, weight in weighted_rows:
        key = str(key_fn(row) or "").strip().upper()
        if not key:
            continue
        grouped.setdefault(key, []).append((float(row.outcome.realized_pnl_pct or 0.0), weight))
    out: dict[str, OutcomeBias] = {}
    for key, values in grouped.items():
        bias = _summarize_bias(values, max_adjustment=max_adjustment)
        if bias is not None and bias.score_adjustment != 0.0:
            out[key] = bias
    return out


def _blend_horizon_symbol_biases(
    short_biases: dict[str, OutcomeBias],
    long_biases: dict[str, OutcomeBias],
    *,
    max_adjustment: float,
) -> dict[str, OutcomeBias]:
    out = dict(short_biases)
    for key, long_bias in long_biases.items():
        short_bias = short_biases.get(key)
        if short_bias is not None:
            blended = (short_bias.score_adjustment * 0.55) + (
                long_bias.score_adjustment * 0.45
            )
            row_count = max(short_bias.paper_closed_rows, long_bias.paper_closed_rows)
            weighted_avg = (
                (short_bias.weighted_avg_realized_pnl_pct * 0.55)
                + (long_bias.weighted_avg_realized_pnl_pct * 0.45)
            )
            avg_realized = (
                (short_bias.avg_realized_pnl_pct * 0.55)
                + (long_bias.avg_realized_pnl_pct * 0.45)
            )
        else:
            blended = long_bias.score_adjustment * 0.65
            row_count = long_bias.paper_closed_rows
            weighted_avg = long_bias.weighted_avg_realized_pnl_pct
            avg_realized = long_bias.avg_realized_pnl_pct
        capped = max(-max_adjustment, min(max_adjustment, blended))
        if capped == 0.0:
            out.pop(key, None)
            continue
        out[key] = OutcomeBias(
            avg_realized_pnl_pct=round(avg_realized, 4),
            paper_closed_rows=row_count,
            score_adjustment=round(capped, 4),
            weighted_avg_realized_pnl_pct=round(weighted_avg, 4),
        )
    return out


def _summarize_bias(
    weighted_rows: list[tuple[float, float]] | list[tuple[LearningExample, float]],
    *,
    max_adjustment: float,
) -> Optional[OutcomeBias]:
    values: list[tuple[float, float]] = []
    for left, weight in weighted_rows:
        if isinstance(left, LearningExample):
            if left.outcome.realized_pnl_pct is None:
                continue
            values.append((float(left.outcome.realized_pnl_pct), float(weight)))
        else:
            values.append((float(left), float(weight)))
    if not values:
        return None
    realized = [value for value, _ in values]
    avg_realized = sum(realized) / len(realized)
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    weighted_avg = sum(value * weight for value, weight in values) / total_weight
    adjustment = adaptive_score_adjustment(
        weighted_avg,
        len(values),
        max_adjustment=max_adjustment,
    )
    if adjustment == 0.0:
        return None
    return OutcomeBias(
        avg_realized_pnl_pct=round(avg_realized, 4),
        paper_closed_rows=len(values),
        score_adjustment=adjustment,
        weighted_avg_realized_pnl_pct=round(weighted_avg, 4),
    )
