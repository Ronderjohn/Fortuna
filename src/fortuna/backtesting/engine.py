"""Vectorbt backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.freq import timeframe_to_freq
from fortuna.backtesting.metrics import BacktestMetrics, MetricsExtractor
from fortuna.backtesting.risk import vectorbt_sl_tp
from fortuna.config.settings import Settings, get_settings
from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.schema import StrategyDefinition, TradeSide
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


def _import_vectorbt():
    try:
        import vectorbt as vbt
    except ImportError as e:
        raise ImportError(
            "vectorbt is not installed. Run: uv sync --group vbt"
        ) from e
    return vbt


@dataclass
class BacktestResult:
    """Container for backtest output."""

    strategy_name: str
    symbol: str
    timeframe: str
    metrics: BacktestMetrics
    portfolio: object
    enriched_data: pd.DataFrame
    trade_returns: list[float] | None = None
    equity_curve: object | None = None


class BacktestEngine:
    """Run deterministic backtests from strategy definitions (vectorbt)."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        indicator_engine: Optional[IndicatorEngine] = None,
        compiler: Optional[StrategyCompiler] = None,
        metrics_extractor: Optional[MetricsExtractor] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.indicator_engine = indicator_engine or IndicatorEngine()
        self.compiler = compiler or StrategyCompiler()
        self.metrics_extractor = metrics_extractor or MetricsExtractor()

    def run(
        self,
        strategy: StrategyDefinition,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        *,
        fast_metrics: bool = False,
    ) -> BacktestResult:
        """Execute backtest on OHLCV data."""
        vbt = _import_vectorbt()
        symbol = symbol or strategy.symbol
        enriched = self.indicator_engine.compute(df, strategy.indicators)
        entries, exits = self.compiler.compile(strategy, enriched)

        close = enriched["close"]
        sl_stop, tp_stop = vectorbt_sl_tp(strategy, enriched)

        short_entries = None
        short_exits = None
        if strategy.side == TradeSide.SHORT:
            entries, exits = pd.Series(False, index=close.index), entries
            short_entries, short_exits = exits, entries
            entries = pd.Series(False, index=close.index)
        elif strategy.side == TradeSide.BOTH:
            short_entries = exits
            short_exits = entries

        sizing = strategy.risk.position_sizing
        size = sizing.value
        is_fraction = sizing.type.value == "fixed_fraction"

        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=self.settings.init_cash,
            fees=self.settings.fees,
            slippage=self.settings.slippage,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            size=size,
            size_type="percent" if is_fraction else "amount",
            freq=timeframe_to_freq(strategy.timeframe),
        )

        metrics = self.metrics_extractor.extract(
            portfolio,
            self.settings.init_cash,
            fast=fast_metrics,
        )
        logger.info(
            "Backtest %s on %s: return=%.2f%% sharpe=%.2f mdd=%.2f%%",
            strategy.name,
            symbol,
            metrics.total_return * 100,
            metrics.sharpe_ratio,
            metrics.max_drawdown * 100,
        )

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=strategy.timeframe,
            metrics=metrics,
            portfolio=portfolio,
            enriched_data=enriched,
        )
