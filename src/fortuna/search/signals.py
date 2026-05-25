"""Signal helpers for search and tournament tiers."""

from __future__ import annotations

import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.strategy.schema import StrategyDefinition, TradeSide


def signal_at_bar(
    strategy: StrategyDefinition,
    enriched: pd.DataFrame,
    compiler: StrategyCompiler,
    bar_idx: int,
) -> tuple[bool, bool]:
    entries, exits = compiler.compile(strategy, enriched)
    return bool(entries.iloc[bar_idx]), bool(exits.iloc[bar_idx])


def emit_signal(
    strategy: StrategyDefinition,
    entry: bool,
    exit_sig: bool,
) -> str:
    if entry and not exit_sig:
        return "SELL" if strategy.side == TradeSide.SHORT else "BUY"
    if exit_sig and not entry:
        return "BUY" if strategy.side == TradeSide.SHORT else "SELL"
    return "HOLD"
