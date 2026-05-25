"""Winner selection from ranked evaluation results."""

from __future__ import annotations

from fortuna.search.candidate import CandidateStrategy
from fortuna.search.types import TierResult


def select_winner(results: list[TierResult]) -> TierResult | None:
    """Pick the strongest candidate that has actual evidence behind it.

    Sort key (descending precedence):
    1. ``total_trades > 0`` — a zero-trade candidate has no track record and
       should never beat a candidate that actually traded, even if its
       composite score happens to be higher due to scoring quirks.
    2. ``passed`` — passes the validation threshold.
    3. ``score`` — composite score.
    4. ``total_trades`` — tie-break by sample size.
    """
    if not results:
        return None
    return max(
        results,
        key=lambda r: (r.total_trades > 0, r.passed, r.score, r.total_trades),
    )
