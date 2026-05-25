"""Single-candidate evaluation tiers."""

from __future__ import annotations

import time

import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.strategies.builtin.dispatch import is_builtin_strategy, run_builtin_backtest
from fortuna.evaluation.scorer import StrategyScorer
from fortuna.search.candidate import CandidateStrategy
from fortuna.search.indicator_cache import IndicatorCache
from fortuna.search.signals import emit_signal, signal_at_bar
from fortuna.search.types import TierResult

__all__ = ["TierResult", "evaluate_candidate", "emit_signal", "signal_at_bar"]


def evaluate_candidate(
    candidate: CandidateStrategy,
    window: pd.DataFrame,
    *,
    indicator_cache: IndicatorCache,
    compiler: StrategyCompiler,
    runner: NumPyBacktestRunner,
    scorer: StrategyScorer,
    tier: int = 1,
) -> TierResult:
    """Evaluate one candidate on a trailing window."""
    t0 = time.perf_counter()
    strategy = candidate.strategy
    enriched = indicator_cache.enrich(window, strategy.indicators)
    bar_idx = len(enriched) - 1
    entry, exit_sig = signal_at_bar(strategy, enriched, compiler, bar_idx)
    signal = emit_signal(strategy, entry, exit_sig)

    if tier == 0:
        score = 1.0 if signal in ("BUY", "SELL") else 0.0
        elapsed = (time.perf_counter() - t0) * 1000
        return TierResult(
            candidate_id=candidate.candidate_id,
            score=score,
            passed=signal != "HOLD",
            signal=signal,
            entry=entry,
            exit=exit_sig,
            total_trades=0,
            elapsed_ms=elapsed,
            metrics_dict={},
        )

    try:
        if is_builtin_strategy(strategy):
            result = run_builtin_backtest(
                strategy,
                window,
                symbol=strategy.symbol,
                init_cash=float(runner.settings.init_cash),
            )
        else:
            result = runner.run(strategy, window, symbol=strategy.symbol)
        score_obj = scorer.score(result.metrics, strategy.metadata)
        elapsed = (time.perf_counter() - t0) * 1000
        return TierResult(
            candidate_id=candidate.candidate_id,
            score=score_obj.composite,
            passed=score_obj.passed,
            signal=signal,
            entry=entry,
            exit=exit_sig,
            total_trades=result.metrics.total_trades,
            elapsed_ms=elapsed,
            metrics_dict=result.metrics.to_dict(),
        )
    except Exception:
        elapsed = (time.perf_counter() - t0) * 1000
        return TierResult(
            candidate_id=candidate.candidate_id,
            score=-1e9,
            passed=False,
            signal=signal,
            entry=entry,
            exit=exit_sig,
            total_trades=0,
            elapsed_ms=elapsed,
            metrics_dict={},
        )
