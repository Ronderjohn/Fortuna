"""Bridge BacktestResult -> TradeRecord ledger."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.engine import BacktestResult
from fortuna.reporting.strategy_tester.costs import CostConfig
from fortuna.reporting.strategy_tester.trade import TradeRecord, TradeSide
from fortuna.strategy.schema import StrategyDefinition


def trades_from_backtest(
    strategy: StrategyDefinition,
    bt: BacktestResult,
    ohlcv: pd.DataFrame,
    *,
    initial_capital: float,
    costs: Optional[CostConfig] = None,
) -> list[TradeRecord]:
    """Reconstruct trade ledger from backtest signals and return series."""
    costs = costs or CostConfig()
    returns = list(bt.trade_returns or [])
    if not returns:
        return []

    compiler = StrategyCompiler()
    entries, exits = compiler.compile(strategy, bt.enriched_data)
    entry_mask = entries.fillna(False).to_numpy(dtype=bool)
    exit_mask = exits.fillna(False).to_numpy(dtype=bool)
    entry_idx = np.where(entry_mask)[0]
    close = ohlcv["close"].to_numpy(dtype=np.float64)
    index = ohlcv.index

    trades: list[TradeRecord] = []
    equity = float(initial_capital)
    ei_ptr = 0

    for ret in returns:
        if ei_ptr >= len(entry_idx):
            break
        ei = int(entry_idx[ei_ptr])
        ei_ptr += 1
        entry_px = float(close[ei])
        exit_idx = ei
        for j in range(ei + 1, len(close)):
            exit_idx = j
            if exit_mask[j]:
                break
        exit_px = float(close[exit_idx])
        qty = (equity * (1 - costs.commission_rate)) / entry_px if entry_px > 0 else 0.0
        entry_not = entry_px * qty
        exit_not = exit_px * qty
        comm, slip = costs.round_trip_cost(entry_not, exit_not)
        pnl = ret * equity - comm - slip if abs(ret) < 1 else (exit_px - entry_px) * qty - comm - slip

        trades.append(
            TradeRecord(
                entry_time=pd.Timestamp(index[ei]),
                exit_time=pd.Timestamp(index[exit_idx]),
                entry_price=entry_px,
                exit_price=exit_px,
                side=TradeSide.LONG,
                qty=qty,
                pnl=pnl,
                pnl_percent=ret * 100.0,
                holding_time=pd.Timestamp(index[exit_idx]) - pd.Timestamp(index[ei]),
                commission=comm,
                slippage=slip,
            )
        )
        equity += pnl
    return trades
