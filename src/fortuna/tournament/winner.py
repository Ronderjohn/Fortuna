"""Winner selection from ranked evaluation results."""

from __future__ import annotations

from fortuna.search.candidate import CandidateStrategy
from fortuna.search.types import TierResult


def select_winner(results: list[TierResult]) -> TierResult | None:
    """Pick highest score; tie-break prefers passed candidates."""
    if not results:
        return None
    return max(results, key=lambda r: (r.score, r.passed, r.total_trades))
