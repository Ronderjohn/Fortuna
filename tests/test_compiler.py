"""Strategy compiler tests."""

import pandas as pd

from fortuna.strategy.loader import load_strategy


def test_ema_crossover_compiles(
    ema_crossover_path,
    indicator_engine,
    strategy_compiler,
    sample_ohlcv: pd.DataFrame,
) -> None:
    strategy = load_strategy(ema_crossover_path)
    enriched = indicator_engine.compute(sample_ohlcv, strategy.indicators)
    entries, exits = strategy_compiler.compile(strategy, enriched)
    assert isinstance(entries, pd.Series)
    assert isinstance(exits, pd.Series)
    assert entries.dtype == bool
