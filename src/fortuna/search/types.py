"""Search and evaluation result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TierResult:
    candidate_id: str
    score: float
    passed: bool
    signal: str
    entry: bool
    exit: bool
    total_trades: int
    elapsed_ms: float
    metrics_dict: dict


@dataclass
class EvaluationResult:
    """Ranked batch evaluation output."""

    results: list[TierResult]
    elapsed_ms: float
    timed_out: bool
