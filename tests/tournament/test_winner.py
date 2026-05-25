"""Winner selection tests."""

from fortuna.search.types import TierResult
from fortuna.tournament.winner import select_winner


def _tier(
    cid: str,
    score: float,
    *,
    passed: bool = True,
    total_trades: int = 1,
) -> TierResult:
    return TierResult(
        candidate_id=cid,
        score=score,
        passed=passed,
        signal="BUY",
        entry=True,
        exit=False,
        total_trades=total_trades,
        elapsed_ms=1.0,
        metrics_dict={},
    )


def test_select_winner_highest_score() -> None:
    winner = select_winner([_tier("a", 10.0), _tier("b", 50.0), _tier("c", 30.0)])
    assert winner is not None
    assert winner.candidate_id == "b"


def test_select_winner_empty() -> None:
    assert select_winner([]) is None


def test_zero_trade_candidate_never_beats_traded() -> None:
    """Regression: a candidate with 0 trades and a high composite score
    must not win over a candidate with real trades — even a losing one.
    The composite scorer's penalties don't fully cancel out a no-trade
    strategy's neutral components, so the primary sort key has to be
    ``total_trades > 0``."""
    zero_trade = _tier("zero", score=80.0, passed=False, total_trades=0)
    traded_loser = _tier("loser", score=10.0, passed=False, total_trades=12)
    winner = select_winner([zero_trade, traded_loser])
    assert winner is not None
    assert winner.candidate_id == "loser"


def test_passed_beats_failed_among_traded() -> None:
    """Within traded candidates, ``passed`` ranks above ``score`` so a
    validated strategy doesn't lose to a higher-scoring failed one."""
    passed_low = _tier("passed", score=40.0, passed=True, total_trades=10)
    failed_high = _tier("failed", score=60.0, passed=False, total_trades=10)
    winner = select_winner([passed_low, failed_high])
    assert winner is not None
    assert winner.candidate_id == "passed"
