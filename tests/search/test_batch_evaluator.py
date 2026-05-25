"""Batch evaluator tests."""

from pathlib import Path

from fortuna.search.batch_evaluator import BatchEvaluator
from fortuna.search.candidate import CandidateStrategy
from fortuna.strategy.loader import load_strategy


def test_batch_evaluator_ranks_candidate(
    sample_ohlcv,
    ema_crossover_path: Path,
) -> None:
    base = load_strategy(ema_crossover_path)
    cand = CandidateStrategy(strategy=base, candidate_id="c0")
    window = sample_ohlcv.iloc[-40:].copy()
    out = BatchEvaluator(max_workers=1).evaluate_all(
        [cand], window, tier=1, parallel=False
    )
    assert len(out.results) == 1
    assert out.results[0].candidate_id == "c0"
    assert out.results[0].score > -1e8
