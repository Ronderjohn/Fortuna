"""Dispatch to builtin strategy engines (MMTS, ORB, etc.)."""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from fortuna.backtesting.engine import BacktestResult
from fortuna.strategy.schema import StrategyDefinition
from fortuna.strategies.builtin.mmts import is_mmts_strategy, run_mmts_backtest
from fortuna.strategies.builtin.orb import is_orb_strategy, run_orb_backtest

_BUILTIN: dict[str, Callable[..., BacktestResult]] = {
    "mmts": run_mmts_backtest,
    "orb": run_orb_backtest,
}


def builtin_engine_id(strategy: StrategyDefinition) -> Optional[str]:
    engine = getattr(strategy.metadata, "engine", None)
    if engine and engine in _BUILTIN:
        return engine
    if is_mmts_strategy(strategy):
        return "mmts"
    if is_orb_strategy(strategy):
        return "orb"
    return None


def is_builtin_strategy(strategy: StrategyDefinition) -> bool:
    return builtin_engine_id(strategy) is not None


def run_builtin_backtest(
    strategy: StrategyDefinition,
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    *,
    init_cash: float = 100_000.0,
) -> BacktestResult:
    engine = builtin_engine_id(strategy)
    if not engine:
        raise ValueError(f"Not a builtin strategy: {strategy.name}")
    return _BUILTIN[engine](strategy, df, symbol=symbol, init_cash=init_cash)
