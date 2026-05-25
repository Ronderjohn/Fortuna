"""Process-pool worker for parallel candidate evaluation."""

from __future__ import annotations

import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.config.settings import get_settings
from fortuna.evaluation.scorer import StrategyScorer
from fortuna.search.candidate import CandidateStrategy
from fortuna.search.evaluator import evaluate_candidate
from fortuna.search.indicator_cache import IndicatorCache
from fortuna.strategy.schema import StrategyDefinition


def eval_chunk(
    candidates_data: list[tuple[str, dict]],
    window_records: dict,
    tier: int,
) -> list[dict]:
    window = pd.DataFrame.from_dict(window_records, orient="index")
    window.index = pd.to_datetime(window.index)
    settings = get_settings()
    cache = IndicatorCache()
    compiler = StrategyCompiler()
    runner = NumPyBacktestRunner(settings)
    scorer = StrategyScorer(settings)

    out: list[dict] = []
    for cid, strat_dict in candidates_data:
        cand = CandidateStrategy(
            strategy=StrategyDefinition.model_validate(strat_dict),
            candidate_id=cid,
        )
        tr = evaluate_candidate(
            cand,
            window,
            indicator_cache=cache,
            compiler=compiler,
            runner=runner,
            scorer=scorer,
            tier=tier,
        )
        out.append(
            {
                "candidate_id": tr.candidate_id,
                "score": tr.score,
                "passed": tr.passed,
                "signal": tr.signal,
                "entry": tr.entry,
                "exit": tr.exit,
                "total_trades": tr.total_trades,
                "elapsed_ms": tr.elapsed_ms,
                "metrics_dict": tr.metrics_dict,
            }
        )
    return out
