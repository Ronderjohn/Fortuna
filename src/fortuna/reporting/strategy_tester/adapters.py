"""Convert backtest outputs into TradeRecord ledgers."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from fortuna.reporting.strategy_tester.costs import CostConfig
from fortuna.reporting.strategy_tester.trade import TradeRecord, TradeSide


def _as_timestamp(ts) -> pd.Timestamp:
    return pd.Timestamp(ts)


def trades_from_dataframe(
    df: pd.DataFrame,
    *,
    costs: Optional[CostConfig] = None,
    default_qty: Optional[float] = None,
) -> list[TradeRecord]:
    """
  Accept a trade log with columns (flexible names):
    entry_time, exit_time, entry_price, exit_price, side/direction,
    qty (optional), pnl (optional), commission, slippage.
    """
    if df.empty:
        return []

    costs = costs or CostConfig()
    rows: list[TradeRecord] = []

    for _, r in df.iterrows():
        entry_t = _as_timestamp(r.get("entry_time", r.get("entry")))
        exit_t = _as_timestamp(r.get("exit_time", r.get("exit")))
        entry_px = float(r["entry_price"])
        exit_px = float(r["exit_price"])
        side_raw = str(r.get("side", r.get("direction", "LONG"))).upper()
        side = TradeSide.SHORT if side_raw in ("SHORT", "SELL", "-1") else TradeSide.LONG

        qty = float(r["qty"]) if "qty" in r and pd.notna(r["qty"]) else None
        if qty is None and default_qty is not None:
            qty = default_qty
        if qty is None or qty <= 0:
            notional = entry_px * 1.0
            qty = max(notional / entry_px, 1.0) if entry_px > 0 else 1.0

        if "pnl" in r and pd.notna(r["pnl"]):
            pnl = float(r["pnl"])
            comm = float(r.get("commission", 0.0) or 0.0)
            slip = float(r.get("slippage", 0.0) or 0.0)
        else:
            if side == TradeSide.LONG:
                gross = (exit_px - entry_px) * qty
            else:
                gross = (entry_px - exit_px) * qty
            entry_not = entry_px * qty
            exit_not = exit_px * qty
            comm, slip = costs.round_trip_cost(entry_not, exit_not)
            pnl = gross - comm - slip

        entry_not = entry_px * qty
        pnl_pct = (pnl / entry_not * 100.0) if entry_not > 0 else 0.0
        if "pnl_percent" in r and pd.notna(r["pnl_percent"]):
            pnl_pct = float(r["pnl_percent"])
        elif "return_pct" in r and pd.notna(r["return_pct"]):
            pnl_pct = float(r["return_pct"])

        meta = {}
        if "exit_reason" in r and pd.notna(r["exit_reason"]):
            meta["exit_reason"] = str(r["exit_reason"])

        rows.append(
            TradeRecord(
                entry_time=entry_t,
                exit_time=exit_t,
                entry_price=entry_px,
                exit_price=exit_px,
                side=side,
                qty=qty,
                pnl=pnl,
                pnl_percent=pnl_pct,
                holding_time=exit_t - entry_t,
                commission=comm,
                slippage=slip,
                metadata=meta,
            )
        )
    return rows


def trades_from_return_series(
    ohlcv: pd.DataFrame,
    trade_returns: Sequence[float],
    entry_mask: np.ndarray,
    *,
    initial_capital: float,
    costs: Optional[CostConfig] = None,
    side: TradeSide = TradeSide.LONG,
) -> list[TradeRecord]:
    """
    Build trades from per-trade return fractions and entry signal mask.

    Used for NumPy long-only backtests that only expose return list.
    """
    costs = costs or CostConfig()
    close = ohlcv["close"].to_numpy(dtype=np.float64)
    index = ohlcv.index
    entries_idx = np.where(entry_mask)[0]
    trades: list[TradeRecord] = []

    deploy = initial_capital
    for i, ret in enumerate(trade_returns):
        if i >= len(entries_idx):
            break
        ei = int(entries_idx[i])
        entry_px = float(close[ei])
        exit_idx = min(ei + 1, len(close) - 1)
        for j in range(ei + 1, len(close)):
            exit_idx = j
            break
        exit_px = float(close[exit_idx])
        qty = (deploy * (1 - costs.commission_rate)) / entry_px if entry_px > 0 else 0.0
        entry_not = entry_px * qty
        exit_not = exit_px * qty
        comm, slip = costs.round_trip_cost(entry_not, exit_not)
        pnl = ret * deploy - comm - slip
        trades.append(
            TradeRecord(
                entry_time=_as_timestamp(index[ei]),
                exit_time=_as_timestamp(index[exit_idx]),
                entry_price=entry_px,
                exit_price=exit_px,
                side=side,
                qty=qty,
                pnl=pnl,
                pnl_percent=ret * 100.0,
                holding_time=_as_timestamp(index[exit_idx]) - _as_timestamp(index[ei]),
                commission=comm,
                slippage=slip,
            )
        )
        deploy += pnl
    return trades


def trades_from_mmts_dataframe(
    trades_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    initial_capital: float,
    costs: Optional[CostConfig] = None,
) -> list[TradeRecord]:
    """Convert MMTS simulator output to full trade ledger."""
    if trades_df.empty:
        return []

    costs = costs or CostConfig()
    deploy = float(initial_capital)
    out: list[TradeRecord] = []

    for _, r in trades_df.iterrows():
        direction = str(r.get("direction", "long")).lower()
        side = TradeSide.SHORT if direction == "short" else TradeSide.LONG
        entry_px = float(r["entry_price"])
        exit_px = float(r["exit_price"])
        ret_frac = float(r.get("return_pct", 0.0)) / 100.0
        qty = (deploy * (1 - costs.commission_rate)) / entry_px if entry_px > 0 else 0.0
        entry_not = entry_px * qty
        exit_not = exit_px * qty
        comm, slip = costs.round_trip_cost(entry_not, exit_not)
        if side == TradeSide.LONG:
            gross = (exit_px - entry_px) * qty
        else:
            gross = (entry_px - exit_px) * qty
        pnl = gross - comm - slip
        if abs(ret_frac) > 1e-12 and abs(pnl) < 1e-9:
            pnl = ret_frac * deploy

        out.append(
            TradeRecord(
                entry_time=_as_timestamp(r["entry_time"]),
                exit_time=_as_timestamp(r["exit_time"]),
                entry_price=entry_px,
                exit_price=exit_px,
                side=side,
                qty=qty,
                pnl=pnl,
                pnl_percent=(pnl / entry_not * 100.0) if entry_not > 0 else 0.0,
                holding_time=_as_timestamp(r["exit_time"]) - _as_timestamp(r["entry_time"]),
                commission=comm,
                slippage=slip,
                metadata={"exit_reason": str(r.get("exit_reason", ""))},
            )
        )
        deploy += pnl
    return out


def bar_equity_series(
    candle_timestamps: pd.DatetimeIndex,
    equity_values: np.ndarray | Sequence[float],
) -> pd.Series:
    return pd.Series(np.asarray(equity_values, dtype=np.float64), index=candle_timestamps)
