"""Builtin strategies (MMTS, ORB, etc.)."""

from fortuna.strategies.builtin.dispatch import (
    is_builtin_strategy,
    run_builtin_backtest,
)
from fortuna.strategies.builtin.mmts import is_mmts_strategy, run_mmts_backtest
from fortuna.strategies.builtin.orb import is_orb_strategy, run_orb_backtest

__all__ = [
    "is_builtin_strategy",
    "is_mmts_strategy",
    "is_orb_strategy",
    "run_builtin_backtest",
    "run_mmts_backtest",
    "run_orb_backtest",
]
