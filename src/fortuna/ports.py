"""Protocol interfaces for modular swapping (agent-ready architecture)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol

import pandas as pd

from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.strategy.schema import IndicatorSpec, StrategyDefinition

if TYPE_CHECKING:
    from fortuna.backtesting.engine import BacktestResult


class IndicatorProvider(Protocol):
    def compute(self, df: pd.DataFrame, indicators: list[IndicatorSpec]) -> pd.DataFrame:
        ...


class SignalCompiler(Protocol):
    def compile(
        self, strategy: StrategyDefinition, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        ...


class BacktestRunner(Protocol):
    def run(
        self,
        strategy: StrategyDefinition,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> "BacktestResult":
        ...


class MetricsCalculator(Protocol):
    def extract(
        self,
        portfolio: Any,
        init_cash: float,
        *,
        fast: bool = False,
    ) -> BacktestMetrics:
        ...


class MarketDataSource(Protocol):
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        ...


class TournamentRunnerProtocol(Protocol):
    def run(self) -> list[Any]:
        ...
