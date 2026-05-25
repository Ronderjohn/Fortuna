"""Lightweight long/short backtest runner for fast tests (no vectorbt).

Supports all three ``TradeSide`` modes via the dual-side simulator:

- ``side: long``  — uses the primary ``entry_conditions`` / ``exit_conditions``
  rules for long entries / exits; never opens shorts.
- ``side: short`` — uses the primary rules for short entries / exits; never
  opens longs.
- ``side: both``  — primary rules drive longs; ``short_entry_conditions`` /
  ``short_exit_conditions`` drive shorts. The simulator records every leg
  with a direction in the trade log so the chart and live-signal panel can
  render BUY / SELL / EXIT markers correctly.

The dual-side simulator treats opposite-direction entries as **flips** when
the strategy is already in a position: closing the existing leg at the
current bar's close and opening the new leg on the same bar. This matches
Pine's default ``strategy.entry`` behavior with ``pyramiding = 0`` for the
direction-switch case.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.engine import BacktestResult
from fortuna.backtesting.metrics import metrics_from_equity
from fortuna.backtesting.risk import percent_sl_tp
from fortuna.config.settings import Settings, get_settings
from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.schema import StrategyDefinition
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class NumPyBacktestRunner:
    """Deterministic long/short backtest using NumPy (percent SL/TP)."""

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
        symbol = symbol or strategy.symbol
        enriched = self.indicator_engine.compute(df, strategy.indicators)
        long_e, long_x, short_e, short_x = self.compiler.compile_dual(strategy, enriched)

        close = enriched["close"].to_numpy(dtype=np.float64, copy=False)
        sl_pct, tp_pct = percent_sl_tp(strategy)
        size_frac = float(strategy.risk.position_sizing.value)
        fees = float(self.settings.fees)
        init_cash = float(self.settings.init_cash)

        equity, trade_returns, trades_df = self._simulate_dual(
            close=close,
            index=enriched.index,
            long_entry=long_e.to_numpy(dtype=bool, copy=False),
            long_exit=long_x.to_numpy(dtype=bool, copy=False),
            short_entry=short_e.to_numpy(dtype=bool, copy=False),
            short_exit=short_x.to_numpy(dtype=bool, copy=False),
            init_cash=init_cash,
            size_frac=size_frac,
            fees=fees,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
        )

        metrics = metrics_from_equity(equity, init_cash, len(trade_returns), trade_returns)
        enriched.attrs["trades"] = trades_df
        enriched.attrs["equity_curve"] = equity
        # Surface the per-bar entry/exit signals so the live-signal layer can
        # see them without re-compiling the strategy.
        enriched["long_entry"] = long_e.to_numpy(dtype=bool, copy=False)
        enriched["short_entry"] = short_e.to_numpy(dtype=bool, copy=False)

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
    def _simulate_dual(
        *,
        close: np.ndarray,
        index: pd.Index,
        long_entry: np.ndarray,
        long_exit: np.ndarray,
        short_entry: np.ndarray,
        short_exit: np.ndarray,
        init_cash: float,
        size_frac: float,
        fees: float,
        sl_pct: Optional[float],
        tp_pct: Optional[float],
    ) -> tuple[np.ndarray, list[float], pd.DataFrame]:
        """Long+short fill loop with percent SL/TP and direction flips.

        Order of operations per bar (matches Pine's ``process_orders_on_close``):
        1. If in a position, check SL / TP / matching exit signal — close if hit.
        2. Check the opposite-direction entry — if it fires while in a
           position, close the current leg and flip on the same bar.
        3. If flat, check the same-direction entry and open a fresh leg.
        """
        n = close.shape[0]
        equity = np.empty(n, dtype=np.float64)
        cash = init_cash
        side = 0  # 0=flat, 1=long, -1=short
        shares = 0.0
        entry_price = 0.0
        entry_idx = -1
        trade_returns: list[float] = []
        trade_rows: list[dict[str, Any]] = []

        def close_at(exit_price: float, reason: str, i: int) -> None:
            nonlocal cash, side, shares, entry_price, entry_idx
            if side == 0 or shares <= 0:
                return
            if side == 1:
                ret = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                cash += shares * exit_price * (1.0 - fees)
                direction = "long"
            else:
                ret = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0
                cash += shares * (entry_price - exit_price) - shares * exit_price * fees
                direction = "short"
            trade_returns.append(ret)
            trade_rows.append(
                {
                    "entry_time": index[entry_idx] if entry_idx >= 0 else index[i],
                    "exit_time": index[i],
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret * 100.0,
                    "exit_reason": reason,
                }
            )
            side = 0
            shares = 0.0
            entry_price = 0.0
            entry_idx = -1

        def open_position(direction: int, i: int) -> None:
            nonlocal cash, side, shares, entry_price, entry_idx
            price = close[i]
            if price <= 0:
                return
            invest = cash * size_frac
            if invest <= 0:
                return
            shares = (invest * (1.0 - fees)) / price
            if direction == 1:
                cash -= invest
            # For shorts, cash is unchanged at entry (we owe ``shares`` units);
            # P&L settles via the ``entry_price - exit_price`` term at close.
            side = direction
            entry_price = price
            entry_idx = i

        for i in range(n):
            price = close[i]

            if side == 1:
                ret = (price - entry_price) / entry_price if entry_price > 0 else 0.0
                if sl_pct is not None and ret <= -sl_pct:
                    close_at(price, "stop", i)
                elif tp_pct is not None and ret >= tp_pct:
                    close_at(price, "target", i)
                elif long_exit[i]:
                    close_at(price, "signal", i)
            elif side == -1:
                ret = (entry_price - price) / entry_price if entry_price > 0 else 0.0
                if sl_pct is not None and ret <= -sl_pct:
                    close_at(price, "stop", i)
                elif tp_pct is not None and ret >= tp_pct:
                    close_at(price, "target", i)
                elif short_exit[i]:
                    close_at(price, "signal", i)

            # Direction-flip entry — close the current leg first so the new
            # entry below opens cleanly from the FLAT state.
            if side == 1 and short_entry[i]:
                close_at(price, "flip", i)
            elif side == -1 and long_entry[i]:
                close_at(price, "flip", i)

            if side == 0:
                if long_entry[i]:
                    open_position(1, i)
                elif short_entry[i]:
                    open_position(-1, i)

            if side == 1:
                equity[i] = cash + shares * price
            elif side == -1:
                equity[i] = cash + shares * (entry_price - price)
            else:
                equity[i] = cash

        # Mark-to-market close of any final position on the last bar so the
        # equity curve reflects realised + open P&L consistently.
        if side != 0:
            close_at(close[-1], "session_end", n - 1)
            equity[-1] = cash

        return equity, trade_returns, pd.DataFrame(trade_rows)
