"""Shared exposure-aware ranking helpers for shortlist-derived workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ExposureRankResult(Generic[T]):
    item: T
    selection_rank: int
    adjusted_score: float
    exposure_penalty: float


def rank_with_exposure_penalties(
    items: list[T],
    *,
    symbol_of: Callable[[T], str],
    action_of: Callable[[T], str],
    base_score_of: Callable[[T], float],
    tie_breaker_of: Optional[Callable[[T], tuple]] = None,
) -> list[ExposureRankResult[T]]:
    tie_breaker = tie_breaker_of or (lambda _: ())
    ranked = sorted(
        items,
        key=lambda row: (-float(base_score_of(row)),) + tuple(tie_breaker(row)),
    )
    action_counts: dict[str, int] = {}
    exposure_counts: dict[tuple[str, str], int] = {}
    penalized: list[tuple[T, float, float]] = []
    for row in ranked:
        action = str(action_of(row) or "").upper()
        exposure_key = normalize_exposure_key(symbol_of(row))
        penalty = 0.0
        if action in {"BUY", "SELL"}:
            if action_counts.get(action, 0) >= 2:
                penalty += 0.12
            if exposure_counts.get((action, exposure_key), 0) >= 1:
                penalty += 0.35
            action_counts[action] = action_counts.get(action, 0) + 1
            exposure_counts[(action, exposure_key)] = (
                exposure_counts.get((action, exposure_key), 0) + 1
            )
        penalized.append((row, float(base_score_of(row)) - penalty, penalty))
    penalized.sort(
        key=lambda row: (
            -row[1],
            *tuple(tie_breaker(row[0])),
        )
    )
    return [
        ExposureRankResult(
            item=row,
            selection_rank=idx,
            adjusted_score=round(score, 4),
            exposure_penalty=round(penalty, 4),
        )
        for idx, (row, score, penalty) in enumerate(penalized, start=1)
    ]


def advisory_priority_score(
    *,
    confidence: float,
    verdict: str,
    liquidity_score: Optional[float],
    action: str,
) -> float:
    verdict_bonus = {"candidate": 0.35, "watch": 0.12, "avoid": -0.15}.get(verdict, 0.0)
    action_bonus = 0.2 if action in {"BUY", "SELL"} else 0.05
    liquidity_bonus = min(0.25, float(liquidity_score or 0.0) / 100.0)
    return float(confidence) + verdict_bonus + action_bonus + liquidity_bonus


def normalize_exposure_key(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    return text.split(".", 1)[0]
