"""Winner selection tests."""

from fortuna.search.types import TierResult
from fortuna.tournament.winner import select_winner


def _tier(cid: str, score: float, passed: bool = True) -> TierResult:
    return TierResult(
        candidate_id=cid,
        score=score,
        passed=passed,
        signal="BUY",
        entry=True,
        exit=False,
        total_trades=1,
        elapsed_ms=1.0,
        metrics_dict={},
    )


def test_select_winner_highest_score() -> None:
    winner = select_winner([_tier("a", 10.0), _tier("b", 50.0), _tier("c", 30.0)])
    assert winner is not None
    assert winner.candidate_id == "b"


def test_select_winner_empty() -> None:
    assert select_winner([]) is None
