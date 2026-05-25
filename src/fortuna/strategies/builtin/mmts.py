"""
MMTS — Measured Move Trend Strategy (Pine Script parity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.engine import BacktestResult
from fortuna.backtesting.metrics import metrics_from_equity
from fortuna.indicators.kernels import atr_array, ema_array, sma_array
from fortuna.strategy.schema import StrategyDefinition


@dataclass(frozen=True)
class MMTSParams:
    fast_ema: int = 20
    slow_ema: int = 50
    pullback_len: int = 5
    atr_len: int = 14
    atr_mult: float = 1.5
    bb_len: int = 20
    bb_mult: float = 2.0
    commission_rate: float = 0.0005
    # TradingView-style behavior tunables:
    # "fast"  → trend filter is `close vs fast_ema` (fires sooner; default,
    #            matches TV's MMTS published variants on consecutive bear / bull
    #            bars after a peak / trough).
    # "slow"  → trend filter is `fast_ema vs slow_ema` (the classical EMA-cross
    #            interpretation; produces fewer but slower-reacting signals).
    trend_mode: str = "fast"
    # When True, applies an extra `close < bb_basis` / `close > bb_basis`
    # confluence filter. Off by default to match TV signal density.
    use_bb_filter: bool = False
    # How many concurrent additions are allowed in the same direction
    # (pyramiding). TV's MMTS shows SELL markers on consecutive bear bars
    # because it pyramids — set to 1 to disable.
    max_pyramids: int = 3
    pyramid_size_frac: float = 0.5


def is_mmts_strategy(strategy: StrategyDefinition) -> bool:
    engine = getattr(strategy.metadata, "engine", None)
    return engine == "mmts" or strategy.name.lower().startswith("mmts")


def mmts_params_from_strategy(strategy: StrategyDefinition) -> MMTSParams:
    raw: dict[str, Any] = {}
    for tag in strategy.metadata.tags:
        if tag.startswith("mmts_") and "=" in tag:
            k, v = tag.split("=", 1)
            raw[k] = v
    return MMTSParams(
        fast_ema=int(raw.get("mmts_fast_ema", 20)),
        slow_ema=int(raw.get("mmts_slow_ema", 50)),
        pullback_len=int(raw.get("mmts_pullback_len", 5)),
        atr_len=int(raw.get("mmts_atr_len", 14)),
        atr_mult=float(raw.get("mmts_atr_mult", 1.5)),
        bb_len=int(raw.get("mmts_bb_len", 20)),
        bb_mult=float(raw.get("mmts_bb_mult", 2.0)),
        trend_mode=str(raw.get("mmts_trend_mode", "fast")).lower(),
        use_bb_filter=str(raw.get("mmts_use_bb_filter", "false")).lower() == "true",
        max_pyramids=int(raw.get("mmts_max_pyramids", 3)),
        pyramid_size_frac=float(raw.get("mmts_pyramid_size_frac", 0.5)),
    )


def compute_mmts_signals(df: pd.DataFrame, params: MMTSParams) -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)

    ema_fast = ema_array(close, params.fast_ema)
    ema_slow = ema_array(close, params.slow_ema)

    # Trend filter — TradingView's MMTS reacts off price vs fast EMA, which
    # fires SELLs as soon as the trend rolls over rather than waiting for a
    # full EMA(20)/EMA(50) crossover. We keep "slow" available for the
    # classical interpretation.
    if params.trend_mode == "slow":
        bull_trend = ema_fast > ema_slow
        bear_trend = ema_fast < ema_slow
    else:  # "fast"
        bull_trend = close > ema_fast
        bear_trend = close < ema_fast

    basis = sma_array(close, params.bb_len)
    dev = pd.Series(close).rolling(params.bb_len).std(ddof=0).to_numpy(dtype=np.float64) * params.bb_mult

    low_s = pd.Series(low, index=df.index)
    high_s = pd.Series(high, index=df.index)

    prior_low_min = low_s.shift(1).rolling(params.pullback_len).min().to_numpy()
    prior_high_max = high_s.shift(1).rolling(params.pullback_len).max().to_numpy()

    bull_breakout = close > prior_high_max
    bear_breakout = close < prior_low_min

    long_cond = bull_trend & bull_breakout
    short_cond = bear_trend & bear_breakout
    if params.use_bb_filter:
        long_cond = long_cond & (close > basis)
        short_cond = short_cond & (close < basis)

    swing_low = low_s.rolling(params.pullback_len).min().to_numpy()
    swing_high = high_s.rolling(params.pullback_len).max().to_numpy()
    measured_move = swing_high - swing_low

    atr = atr_array(high, low, close, params.atr_len)

    out = df.copy()
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["bb_basis"] = basis
    out["bb_upper"] = basis + dev
    out["bb_lower"] = basis - dev
    out["atr"] = atr
    out["long_entry"] = long_cond
    out["short_entry"] = short_cond
    out["long_stop"] = close - atr * params.atr_mult
    out["long_target"] = close + measured_move
    out["short_stop"] = close + atr * params.atr_mult
    out["short_target"] = close - measured_move
    return out


def _mark_equity(cash: float, side: int, shares: float, entry: float, price: float) -> float:
    if side == 0:
        return cash
    if side == 1:
        return cash + shares * price
    return cash + shares * (entry - price)


def simulate_mmts(
    df: pd.DataFrame,
    params: MMTSParams,
    *,
    init_cash: float = 100_000.0,
    size_frac: float = 1.0,
    signals: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, list[float], pd.DataFrame]:
    """Run fill simulation. Pass ``signals`` to reuse ORB/other precomputed entries."""
    sig = signals if signals is not None else compute_mmts_signals(df, params)
    close = sig["close"].to_numpy(dtype=np.float64)
    high = sig["high"].to_numpy(dtype=np.float64)
    low = sig["low"].to_numpy(dtype=np.float64)
    long_entry = sig["long_entry"].fillna(False).to_numpy(dtype=bool)
    short_entry = sig["short_entry"].fillna(False).to_numpy(dtype=bool)
    long_stop = sig["long_stop"].to_numpy(dtype=np.float64)
    long_target = sig["long_target"].to_numpy(dtype=np.float64)
    short_stop = sig["short_stop"].to_numpy(dtype=np.float64)
    short_target = sig["short_target"].to_numpy(dtype=np.float64)

    fee = params.commission_rate
    n = len(sig)
    equity = np.empty(n, dtype=np.float64)
    cash = init_cash
    side = 0
    shares = 0.0
    entry_price = 0.0
    entry_time = None
    stop_px = target_px = np.nan
    # Pyramid tracker: how many adds in the current direction (max=params.max_pyramids).
    pyramid_count = 0
    trade_returns: list[float] = []
    trade_rows: list[dict[str, Any]] = []

    def close_at(exit_price: float, reason: str, i: int) -> None:
        nonlocal cash, side, shares, entry_price, entry_time, stop_px, target_px
        nonlocal pyramid_count
        if side == 0:
            return
        if side == 1:
            ret = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
            cash = cash + shares * exit_price * (1.0 - fee)
            direction = "long"
        else:
            ret = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0
            cash = cash + shares * (entry_price - exit_price) - shares * exit_price * fee
            direction = "short"
        trade_returns.append(ret)
        trade_rows.append(
            {
                "entry_time": entry_time,
                "exit_time": sig.index[i],
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
        entry_time = None
        stop_px = target_px = np.nan
        pyramid_count = 0

    def open_long(i: int) -> None:
        nonlocal cash, side, shares, entry_price, entry_time, stop_px, target_px
        px = close[i]
        eq = _mark_equity(cash, side, shares, entry_price, px)
        deploy = eq * size_frac
        if deploy <= 0 or px <= 0:
            return
        shares = deploy / px * (1.0 - fee)
        cash = eq - deploy
        side = 1
        entry_price = px
        entry_time = sig.index[i]
        stop_px = long_stop[i]
        target_px = long_target[i]

    def open_short(i: int) -> None:
        nonlocal cash, side, shares, entry_price, entry_time, stop_px, target_px
        px = close[i]
        eq = _mark_equity(cash, side, shares, entry_price, px)
        deploy = eq * size_frac
        if deploy <= 0 or px <= 0:
            return
        shares = deploy / px * (1.0 - fee)
        cash = eq
        side = -1
        entry_price = px
        entry_time = sig.index[i]
        stop_px = short_stop[i]
        target_px = short_target[i]

    def add_pyramid(direction: int, i: int) -> None:
        """Add to an existing position in the same direction; records a fresh
        trade entry so the chart can show a SELL/BUY marker on each pyramid bar
        (matches TradingView's MMTS visual where every continuation bar gets a
        signal marker)."""
        nonlocal cash, shares, entry_price, stop_px, target_px, pyramid_count
        if pyramid_count >= max(0, params.max_pyramids - 1):
            return
        px = close[i]
        eq = _mark_equity(cash, side, shares, entry_price, px)
        # Add a fractional lot so the position isn't over-leveraged.
        add_deploy = eq * size_frac * params.pyramid_size_frac
        if add_deploy <= 0 or px <= 0:
            return
        add_shares = add_deploy / px * (1.0 - fee)
        if direction == 1:
            cash = cash - add_deploy
        # Volume-weighted average entry so the ledger stays consistent.
        total_shares = shares + add_shares
        if total_shares > 0:
            entry_price = (entry_price * shares + px * add_shares) / total_shares
        shares = total_shares
        # Refresh stop/target from current bar (trailing measured move).
        if direction == 1:
            stop_px = long_stop[i]
            target_px = long_target[i]
        else:
            stop_px = short_stop[i]
            target_px = short_target[i]
        pyramid_count += 1
        # Record a per-pyramid open-and-immediate-marker so the chart shows
        # the continuation. Exit time is the pyramid bar so the marker lands
        # on the right candle; closing the parent position later still records
        # the cumulative trade exit separately.
        trade_rows.append(
            {
                "entry_time": sig.index[i],
                "exit_time": sig.index[i],
                "direction": "long" if direction == 1 else "short",
                "entry_price": px,
                "exit_price": px,
                "return_pct": 0.0,
                "exit_reason": "pyramid",
            }
        )

    for i in range(n):
        px = close[i]

        if side == 1:
            if np.isfinite(stop_px) and low[i] <= stop_px:
                close_at(stop_px, "stop", i)
            elif np.isfinite(target_px) and high[i] >= target_px:
                close_at(target_px, "target", i)
        elif side == -1:
            if np.isfinite(stop_px) and high[i] >= stop_px:
                close_at(stop_px, "stop", i)
            elif np.isfinite(target_px) and low[i] <= target_px:
                close_at(target_px, "target", i)

        if long_entry[i]:
            if side <= 0:
                if side == -1:
                    close_at(px, "reverse", i)
                if side == 0:
                    open_long(i)
            elif side == 1 and params.max_pyramids > 1:
                add_pyramid(1, i)
        elif short_entry[i]:
            if side >= 0:
                if side == 1:
                    close_at(px, "reverse", i)
                if side == 0:
                    open_short(i)
            elif side == -1 and params.max_pyramids > 1:
                add_pyramid(-1, i)

        equity[i] = _mark_equity(cash, side, shares, entry_price, px)

    return equity, trade_returns, pd.DataFrame(trade_rows)


def run_mmts_backtest(
    strategy: StrategyDefinition,
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    *,
    init_cash: float = 100_000.0,
) -> BacktestResult:
    params = mmts_params_from_strategy(strategy)
    size_frac = float(strategy.risk.position_sizing.value)
    equity, trade_returns, trades_df = simulate_mmts(
        df, params, init_cash=init_cash, size_frac=size_frac
    )
    metrics = metrics_from_equity(equity, init_cash, len(trade_returns), trade_returns)
    enriched = compute_mmts_signals(df, params)
    enriched.attrs["trades"] = trades_df
    enriched.attrs["equity_curve"] = equity
    return BacktestResult(
        strategy_name=strategy.name,
        symbol=symbol or strategy.symbol,
        timeframe=strategy.timeframe,
        metrics=metrics,
        portfolio=None,
        enriched_data=enriched,
        trade_returns=trade_returns,
        equity_curve=equity,
    )
