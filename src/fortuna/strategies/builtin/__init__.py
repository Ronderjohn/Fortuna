"""Builtin strategies (Pine-parity engines: MMTS, ORB, LS-VWAP, IVWAP-ORB)."""

from fortuna.strategies.builtin.dispatch import (
    is_builtin_strategy,
    run_builtin_backtest,
)
from fortuna.strategies.builtin.ivorb import is_ivorb_strategy, run_ivorb_backtest
from fortuna.strategies.builtin.lsvwap import is_lsvwap_strategy, run_lsvwap_backtest
from fortuna.strategies.builtin.mmts import is_mmts_strategy, run_mmts_backtest
from fortuna.strategies.builtin.orb import is_orb_strategy, run_orb_backtest

__all__ = [
    "is_builtin_strategy",
    "is_mmts_strategy",
    "is_orb_strategy",
    "is_lsvwap_strategy",
    "is_ivorb_strategy",
    "run_builtin_backtest",
    "run_mmts_backtest",
    "run_orb_backtest",
    "run_lsvwap_backtest",
    "run_ivorb_backtest",
]
