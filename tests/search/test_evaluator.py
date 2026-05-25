"""Tier-0 signal evaluation tests."""

from fortuna.search.evaluator import evaluate_candidate
from fortuna.search.indicator_cache import IndicatorCache
from fortuna.search.signals import emit_signal
from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.evaluation.scorer import StrategyScorer
from fortuna.search.candidate import CandidateStrategy
from fortuna.strategy.loader import load_strategy
from pathlib import Path


def test_emit_signal_hold() -> None:
    from fortuna.strategy.schema import StrategyDefinition, TradeSide

    s = StrategyDefinition.model_validate(
        {"name": "x", "side": TradeSide.LONG, "indicators": [], "rules": {}}
    )
    assert emit_signal(s, entry=False, exit_sig=False) == "HOLD"


def test_evaluate_candidate_tier0(
    sample_ohlcv,
    ema_crossover_path: Path,
) -> None:
    base = load_strategy(ema_crossover_path)
    cand = CandidateStrategy(strategy=base, candidate_id="t0")
    window = sample_ohlcv.iloc[-30:].copy()
    result = evaluate_candidate(
        cand,
        window,
        indicator_cache=IndicatorCache(),
        compiler=StrategyCompiler(),
        runner=NumPyBacktestRunner(),
        scorer=StrategyScorer(),
        tier=0,
    )
    assert result.signal in ("BUY", "SELL", "HOLD")
    assert result.candidate_id == "t0"
