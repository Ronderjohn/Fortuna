"""
IVWAP-ORB — Institutional VWAP Opening Range Strategy (Pine parity).

Source: ``Strategies.md`` (TradingView Pine v6 strategy by the same name).

Setup family: an NSE-intraday breakout-with-pullback setup. The first
``orb_minutes`` of the session (09:15 → 09:15+N IST) define an opening range
high/low. After the OR closes, signals fire on:

1. **Trend** — close > VWAP, close > EMA(20), EMA(20) > VWAP (mirror for shorts).
2. **Breakout** — first bar after the OR window with close > OR-high (long)
   or close < OR-low (short), confirmed by a volume spike (volume >
   ``volume_mult`` × SMA(20, volume)).
3. **Pullback** — bar's low touched the EMA(20) or VWAP (long) / high touched
   them for shorts. This is the institutional "re-test entry" filter.
4. **Confirming candle** — bullish candle (close > open and close > close[1])
   for longs, mirror for shorts.
5. **Supertrend** (optional) — direction aligned with the trade.

Exits:
- Hard **stop / target** at ATR×1.0 and ``risk_reward`` × that risk.
- **Trailing EMA(20) exit** — close the position the moment the bar closes on
  the wrong side of EMA(20). This is the differentiator from plain ORB.

Pyramiding is zero in the Pine source — only one position at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.engine import BacktestResult
from fortuna.backtesting.metrics import metrics_from_equity
from fortuna.indicators.kernels import atr_array, ema_array, sma_array
from fortuna.strategies.builtin.lsvwap import (
    _minutes_of_day,
    _mark_equity,
    session_anchored_vwap,
    supertrend_array,
)
from fortuna.strategy.schema import StrategyDefinition


@dataclass(frozen=True)
class IVORBParams:
    """Knobs for :func:`compute_ivorb_signals` / :func:`simulate_ivorb`.

    Defaults mirror the Pine source verbatim. Override per JSON strategy via
    metadata tags of the form ``ivorb_<key>=<value>``.
    """

    orb_minutes: int = 15
    ema_len: int = 20
    atr_len: int = 14
    atr_stop_mult: float = 1.0
    risk_reward: float = 2.0
    volume_mult: float = 1.2
    vol_sma_len: int = 20

    use_supertrend: bool = True
    st_atr_period: int = 10
    st_factor: float = 3.0

    # NSE session start (IST) — defines the OR window anchor.
    session_open_hour: int = 9
    session_open_minute: int = 15

    # 0.02% per side per Pine ``commission_value = 0.02`` (percent).
    commission_rate: float = 0.0002


# ───────────────────────── strategy dispatch helpers ─────────────────────────


def is_ivorb_strategy(strategy: StrategyDefinition) -> bool:
    engine = getattr(strategy.metadata, "engine", None)
    if engine == "ivorb":
        return True
    name = strategy.name.lower()
    return name.startswith("ivorb") or name.startswith("ivwap_orb")


def ivorb_params_from_strategy(strategy: StrategyDefinition) -> IVORBParams:
    raw: dict[str, Any] = {}
    for tag in strategy.metadata.tags:
        if tag.startswith("ivorb_") and "=" in tag:
            k, v = tag.split("=", 1)
            raw[k] = v

    def _f(key: str, default: float) -> float:
        return float(raw.get(f"ivorb_{key}", default))

    def _i(key: str, default: int) -> int:
        return int(raw.get(f"ivorb_{key}", default))

    def _b(key: str, default: bool) -> bool:
        return str(raw.get(f"ivorb_{key}", str(default))).lower() == "true"

    return IVORBParams(
        orb_minutes=_i("orb_minutes", 15),
        ema_len=_i("ema_len", 20),
        atr_len=_i("atr_len", 14),
        atr_stop_mult=_f("atr_stop_mult", 1.0),
        risk_reward=_f("risk_reward", 2.0),
        volume_mult=_f("volume_mult", 1.2),
        vol_sma_len=_i("vol_sma_len", 20),
        use_supertrend=_b("use_supertrend", True),
        st_atr_period=_i("st_atr_period", 10),
        st_factor=_f("st_factor", 3.0),
    )


# ────────────────────────────── signal builder ───────────────────────────────


def compute_ivorb_signals(df: pd.DataFrame, params: IVORBParams) -> pd.DataFrame:
    """Add IVWAP-ORB indicator columns + ``long_entry`` / ``short_entry`` flags."""
    out = df.copy()
    n = len(df)
    if n == 0:
        return out

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)

    ema = ema_array(close, params.ema_len)
    vwap = session_anchored_vwap(df)
    atr = atr_array(high, low, close, params.atr_len)
    vol_sma = sma_array(vol, params.vol_sma_len)
    vol_spike = vol > vol_sma * params.volume_mult

    st, st_dir = supertrend_array(high, low, close, params.st_atr_period, params.st_factor)
    bull_st = st_dir < 0
    bear_st = st_dir > 0

    # Build the opening-range envelope: for each calendar day, find the bars
    # whose IST wall-clock falls inside [open, open + orb_minutes] and take
    # the running max/min over that subset. The envelope is then broadcast to
    # every bar of the same day so the breakout check is just `close > or_high`.
    open_min = params.session_open_hour * 60 + params.session_open_minute
    end_min = open_min + params.orb_minutes
    mins = _minutes_of_day(df.index)
    inside_orb = (mins >= open_min) & (mins <= end_min)

    or_high = np.full(n, np.nan, dtype=np.float64)
    or_low = np.full(n, np.nan, dtype=np.float64)
    dates = df.index.date
    unique_dates = pd.unique(dates)
    for d in unique_dates:
        day_mask = dates == d
        idxs = np.where(day_mask)[0]
        if idxs.size == 0:
            continue
        in_orb_idxs = idxs[inside_orb[idxs]]
        if in_orb_idxs.size == 0:
            # Day's first bar starts after the OR window — fall back to its
            # first ``orb_minutes`` / bar-interval worth of bars.
            in_orb_idxs = idxs[: max(1, params.orb_minutes // 5)]
        oh = float(high[in_orb_idxs].max())
        ol = float(low[in_orb_idxs].min())
        for i in idxs:
            or_high[i] = oh
            or_low[i] = ol

    bull_trend = (close > vwap) & (close > ema) & (ema > vwap)
    bear_trend = (close < vwap) & (close < ema) & (ema < vwap)

    long_breakout = (~inside_orb) & np.isfinite(or_high) & (close > or_high) & vol_spike
    short_breakout = (~inside_orb) & np.isfinite(or_low) & (close < or_low) & vol_spike

    pullback_long = (low <= ema) | (low <= vwap)
    pullback_short = (high >= ema) | (high >= vwap)

    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    bullish_candle = (close > open_) & (close > close_prev)
    bearish_candle = (close < open_) & (close < close_prev)

    st_long_ok = bull_st if params.use_supertrend else np.ones(n, dtype=bool)
    st_short_ok = bear_st if params.use_supertrend else np.ones(n, dtype=bool)

    long_entry = (
        bull_trend & long_breakout & pullback_long & bullish_candle & st_long_ok
    )
    short_entry = (
        bear_trend & short_breakout & pullback_short & bearish_candle & st_short_ok
    )

    long_stop = close - atr * params.atr_stop_mult
    short_stop = close + atr * params.atr_stop_mult
    long_target = close + (close - long_stop) * params.risk_reward
    short_target = close - (short_stop - close) * params.risk_reward

    out["ema_fast"] = ema
    out["vwap"] = vwap
    out["supertrend"] = st
    out["atr"] = atr
    out["volume_sma"] = vol_sma
    out["or_high"] = or_high
    out["or_low"] = or_low
    out["long_entry"] = long_entry
    out["short_entry"] = short_entry
    out["long_stop"] = long_stop
    out["long_target"] = long_target
    out["short_stop"] = short_stop
    out["short_target"] = short_target
    return out


# ────────────────────────────── fill simulator ───────────────────────────────


def simulate_ivorb(
    df: pd.DataFrame,
    params: IVORBParams,
    *,
    init_cash: float = 100_000.0,
    size_frac: float = 1.0,
    signals: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, list[float], pd.DataFrame]:
    """Long/short fill loop with ATR stop, R-multiple target, and a **trailing
    EMA(20) exit** that fires the moment the bar closes on the wrong side of
    EMA(20). Pyramiding=0 — only one position at a time.
    """
    sig = signals if signals is not None else compute_ivorb_signals(df, params)
    close = sig["close"].to_numpy(dtype=np.float64)
    high = sig["high"].to_numpy(dtype=np.float64)
    low = sig["low"].to_numpy(dtype=np.float64)
    ema = sig["ema_fast"].to_numpy(dtype=np.float64)
    long_entry = sig["long_entry"].fillna(False).to_numpy(dtype=bool)
    short_entry = sig["short_entry"].fillna(False).to_numpy(dtype=bool)
    long_stop_a = sig["long_stop"].to_numpy(dtype=np.float64)
    long_target_a = sig["long_target"].to_numpy(dtype=np.float64)
    short_stop_a = sig["short_stop"].to_numpy(dtype=np.float64)
    short_target_a = sig["short_target"].to_numpy(dtype=np.float64)

    fee = params.commission_rate
    n = len(sig)
    equity = np.empty(n, dtype=np.float64)

    cash = init_cash
    side = 0
    shares = 0.0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None
    stop_px = target_px = np.nan
    trade_returns: list[float] = []
    trade_rows: list[dict[str, Any]] = []

    def close_at(exit_price: float, reason: str, i: int) -> None:
        nonlocal cash, side, shares, entry_price, entry_time, stop_px, target_px
        if side == 0 or shares <= 0:
            return
        qty = shares
        if side == 1:
            cash += qty * exit_price * (1.0 - fee)
            ret = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
            direction = "long"
        else:
            cash += qty * (entry_price - exit_price) - qty * exit_price * fee
            ret = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0
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

    def open_position(direction: int, i: int) -> None:
        nonlocal cash, side, shares, entry_price, entry_time, stop_px, target_px
        px = close[i]
        eq = _mark_equity(cash, side, shares, entry_price, px)
        deploy = eq * size_frac
        if deploy <= 0 or px <= 0:
            return
        qty = deploy / px * (1.0 - fee)
        if direction == 1:
            cash = eq - deploy
            stop_px = long_stop_a[i]
            target_px = long_target_a[i]
        else:
            cash = eq
            stop_px = short_stop_a[i]
            target_px = short_target_a[i]
        side = direction
        shares = qty
        entry_price = px
        entry_time = sig.index[i]

    for i in range(n):
        px = close[i]
        e = ema[i]

        # Intra-bar exits first (stop has priority over target).
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

        # Trailing EMA(20) bar-close exit — Pine's `strategy.close` after fills.
        if side == 1 and np.isfinite(e) and px < e:
            close_at(px, "ema_trail", i)
        elif side == -1 and np.isfinite(e) and px > e:
            close_at(px, "ema_trail", i)

        # Entries only while flat — Pine pyramiding=0.
        if side == 0:
            if long_entry[i]:
                open_position(1, i)
            elif short_entry[i]:
                open_position(-1, i)

        equity[i] = _mark_equity(cash, side, shares, entry_price, px)

    return equity, trade_returns, pd.DataFrame(trade_rows)


# ────────────────────────────── public entry point ───────────────────────────


def run_ivorb_backtest(
    strategy: StrategyDefinition,
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    *,
    init_cash: float = 100_000.0,
) -> BacktestResult:
    params = ivorb_params_from_strategy(strategy)
    size_frac = float(strategy.risk.position_sizing.value)
    equity, trade_returns, trades_df = simulate_ivorb(
        df, params, init_cash=init_cash, size_frac=size_frac
    )
    metrics = metrics_from_equity(equity, init_cash, len(trade_returns), trade_returns)
    enriched = compute_ivorb_signals(df, params)
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
