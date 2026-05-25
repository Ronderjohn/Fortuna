"""Lightweight long-only backtest runner for fast tests (no vectorbt)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.engine import BacktestResult
from fortuna.backtesting.metrics import BacktestMetrics, metrics_from_equity
from fortuna.backtesting.risk import percent_sl_tp
from fortuna.config.settings import Settings, get_settings
from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.schema import StrategyDefinition, TradeSide
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class NumPyBacktestRunner:
    """Deterministic long-only backtest using NumPy (percent SL/TP)."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        indicator_engine: Optional[IndicatorEngine] = None,
        compiler: Optional[StrategyCompiler] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.indicator_engine = indicator_engine or IndicatorEngine()
        self.compiler = compiler or StrategyCompiler()

    def run(
        self,
        strategy: StrategyDefinition,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> BacktestResult:
        if strategy.side != TradeSide.LONG:
            raise NotImplementedError("NumPyBacktestRunner supports long-only in Phase 1")

        symbol = symbol or strategy.symbol
        enriched = self.indicator_engine.compute(df, strategy.indicators)
        entries, exits = self.compiler.compile(strategy, enriched)

        close = enriched["close"].to_numpy(dtype=np.float64, copy=False)
        entry_sig = entries.to_numpy(dtype=bool, copy=False)
        exit_sig = exits.to_numpy(dtype=bool, copy=False)

        sl_pct, tp_pct = percent_sl_tp(strategy)
        size_frac = float(strategy.risk.position_sizing.value)
        fees = float(self.settings.fees)
        init_cash = float(self.settings.init_cash)

        equity, trade_count, trade_returns = self._simulate(
            close,
            entry_sig,
            exit_sig,
            init_cash=init_cash,
            size_frac=size_frac,
            fees=fees,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
        )

        metrics = metrics_from_equity(equity, init_cash, trade_count, trade_returns)
        logger.info(
            "NumPy backtest %s on %s: return=%.2f%% trades=%s",
            strategy.name,
            symbol,
            metrics.total_return * 100,
            metrics.total_trades,
        )

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=strategy.timeframe,
            metrics=metrics,
            portfolio=None,
            enriched_data=enriched,
            trade_returns=trade_returns,
            equity_curve=equity,
        )

    @staticmethod
    def _simulate(
        close: np.ndarray,
        entry_sig: np.ndarray,
        exit_sig: np.ndarray,
        *,
        init_cash: float,
        size_frac: float,
        fees: float,
        sl_pct: Optional[float],
        tp_pct: Optional[float],
    ) -> tuple[np.ndarray, int, list[float]]:
        n = close.shape[0]
        equity = np.empty(n, dtype=np.float64)
        cash = init_cash
        shares = 0.0
        entry_price = 0.0
        trades = 0
        trade_returns: list[float] = []

        for i in range(n):
            price = close[i]
            if shares > 0:
                ret = (price - entry_price) / entry_price if entry_price > 0 else 0.0
                hit_sl = sl_pct is not None and ret <= -sl_pct
                hit_tp = tp_pct is not None and ret >= tp_pct
                if hit_sl or hit_tp or exit_sig[i]:
                    trade_returns.append(ret)
                    proceeds = shares * price * (1.0 - fees)
                    cash += proceeds
                    shares = 0.0
                    entry_price = 0.0
            elif entry_sig[i] and shares == 0:
                invest = cash * size_frac
                if invest > 0 and price > 0:
                    shares = (invest * (1.0 - fees)) / price
                    cash -= invest
                    entry_price = price
                    trades += 1

            equity[i] = cash + shares * price

        if shares > 0:
            cash += shares * close[-1] * (1.0 - fees)
            equity[-1] = cash

        return equity, trades, trade_returns
