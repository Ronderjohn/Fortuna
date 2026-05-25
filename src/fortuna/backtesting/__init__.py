"""Backtesting engines and metrics."""

from fortuna.backtesting.engine import BacktestEngine, BacktestResult
from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner

try:
    from fortuna.backtesting.standard import InstitutionalBacktestConfig, InstitutionalPipeline
except ImportError:
    InstitutionalBacktestConfig = None  # type: ignore
    InstitutionalPipeline = None  # type: ignore

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestMetrics",
    "InstitutionalBacktestConfig",
    "InstitutionalPipeline",
    "NumPyBacktestRunner",
]
