"""Fortuna dashboard services (Streamlit backend)."""

from fortuna.app.config import AppConfig
from fortuna.app.parallel_runner import (
    BatchRunResult,
    ParallelStrategyRunner,
    StrategyRunResult,
)
from fortuna.app.strategy_paths import list_strategy_paths
from fortuna.app.symbol_catalog import SymbolCatalog

__all__ = [
    "AppConfig",
    "BatchRunResult",
    "FortunaSessionEngine",
    "ParallelStrategyRunner",
    "StrategyRunResult",
    "list_strategy_paths",
    "SymbolCatalog",
]


def __getattr__(name: str):
    if name == "FortunaSessionEngine":
        from fortuna.app.session_engine import FortunaSessionEngine

        return FortunaSessionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
