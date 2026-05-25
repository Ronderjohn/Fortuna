"""
LS-VWAP — Institutional Liquidity Sweep + VWAP Reclaim Strategy (Pine parity).

Source: ``Strategies.md`` (TradingView Pine v6 strategy by the same name).

Setup family: an intraday reversal-with-trend setup that fires only when ALL of
the following confluence conditions are met on a single bar:

1. **Trend** — close > EMA(20), EMA(20) > VWAP, higher highs + higher lows
   for the last 2 bars, and the higher-timeframe (15m) EMA(20) is bullish
   (mirror for shorts).
2. **Liquidity sweep** — the prior bar's low (or high) is taken out
   intra-bar but reclaimed by the close (`low < low[1] and close > low[1]`).
3. **VWAP reclaim** — current close on the right side of VWAP with a
   confirming candle direction.
4. **Volume spike** — bar volume > ``volume_mult`` × SMA(20, volume).
5. **Session** — bar is inside the morning (09:25–11:00 IST) or afternoon
   (13:45–14:45 IST) window.
6. **Supertrend** (optional) — direction aligned with the trade.

Exits use a **two-target partial exit** (50% off at 1R, the rest at 2R) with
an ATR-based stop anchored under (or above) the swept liquidity. ``pyramiding``
is zero in the Pine source — at most one open position at a time.
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
class LSVWAPParams:
    """Knobs for :func:`compute_lsvwap_signals` / :func:`simulate_lsvwap`.

    Defaults mirror the Pine source verbatim. Each knob is overridable per
    JSON strategy via metadata tags of the form ``lsvwap_<key>=<value>``.
    """

    ema_len: int = 20
    atr_len: int = 14
    atr_stop_mult: float = 1.0
    volume_mult: float = 1.5
    vol_sma_len: int = 20
    rr_target1: float = 1.0
    rr_target2: float = 2.0
    target1_qty_frac: float = 0.5

    enable_session: bool = True
    # Morning + afternoon session windows (minutes from midnight IST).
    session1_start_min: int = 9 * 60 + 25   # 09:25
    session1_end_min: int = 11 * 60         # 11:00
    session2_start_min: int = 13 * 60 + 45  # 13:45
    session2_end_min: int = 14 * 60 + 45    # 14:45

    htf_minutes: int = 15

    use_supertrend: bool = True
    st_atr_period: int = 10
    st_factor: float = 3.0

    # 0.03% per side per Pine ``commission_value = 0.03`` (percent).
    commission_rate: float = 0.0003


# ───────────────────────── strategy dispatch helpers ─────────────────────────


def is_lsvwap_strategy(strategy: StrategyDefinition) -> bool:
    engine = getattr(strategy.metadata, "engine", None)
    if engine == "lsvwap":
        return True
    name = strategy.name.lower()
    return name.startswith("lsvwap") or name.startswith("ls_vwap")


def lsvwap_params_from_strategy(strategy: StrategyDefinition) -> LSVWAPParams:
    raw: dict[str, Any] = {}
    for tag in strategy.metadata.tags:
        if tag.startswith("lsvwap_") and "=" in tag:
            k, v = tag.split("=", 1)
            raw[k] = v

    def _f(key: str, default: float) -> float:
        return float(raw.get(f"lsvwap_{key}", default))

    def _i(key: str, default: int) -> int:
        return int(raw.get(f"lsvwap_{key}", default))

    def _b(key: str, default: bool) -> bool:
        return str(raw.get(f"lsvwap_{key}", str(default))).lower() == "true"

    return LSVWAPParams(
        ema_len=_i("ema_len", 20),
        atr_len=_i("atr_len", 14),
        atr_stop_mult=_f("atr_stop_mult", 1.0),
        volume_mult=_f("volume_mult", 1.5),
        vol_sma_len=_i("vol_sma_len", 20),
        rr_target1=_f("rr_target1", 1.0),
        rr_target2=_f("rr_target2", 2.0),
        target1_qty_frac=_f("target1_qty_frac", 0.5),
        enable_session=_b("enable_session", True),
        htf_minutes=_i("htf_minutes", 15),
        use_supertrend=_b("use_supertrend", True),
        st_atr_period=_i("st_atr_period", 10),
        st_factor=_f("st_factor", 3.0),
    )


# ───────────────────────────── indicator helpers ─────────────────────────────


def session_anchored_vwap(df: pd.DataFrame) -> np.ndarray:
    """Pine ``ta.vwap`` parity — cumulative VWAP that resets at each session
    (calendar-day) boundary."""
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)
    typical = (high + low + close) / 3.0
    n = len(df)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    dates = df.index.date
    cum_pv = 0.0
    cum_v = 0.0
    prev_d = None
    for i in range(n):
        d = dates[i]
        if d != prev_d:
            cum_pv = 0.0
            cum_v = 0.0
        cum_pv += typical[i] * vol[i]
        cum_v += vol[i]
        out[i] = cum_pv / cum_v if cum_v > 0 else typical[i]
        prev_d = d
    return out


def supertrend_array(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """TradingView ``ta.supertrend`` parity.

    Returns ``(supertrend_line, direction)`` where ``direction < 0`` is
    bullish (long-only setup) and ``direction > 0`` is bearish, matching
    Pine semantics so the caller can use ``direction < 0`` as the bull flag.

    Handles the ATR warmup correctly: ATR is NaN for the first ``period``
    bars, so the trailing bands must be seeded from the current basic band
    on the first non-NaN ATR bar (otherwise the bands carry NaN forward and
    the direction is pinned to its initial value forever — a subtle bug
    that silently disables the indicator on any series with ATR warmup).
    """
    n = close.shape[0]
    atr = atr_array(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    st = np.full(n, np.nan, dtype=np.float64)
    direction = np.zeros(n, dtype=np.int64)
    if n == 0:
        return st, direction

    seeded = False
    direction[0] = -1  # default bullish until the first non-NaN ATR bar
    for i in range(1, n):
        if np.isnan(atr[i]):
            direction[i] = direction[i - 1]
            continue
        if not seeded:
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            direction[i] = -1 if close[i] >= lower[i] else 1
            st[i] = final_lower[i] if direction[i] == -1 else final_upper[i]
            seeded = True
            continue
        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]
        if upper[i] < prev_upper or close[i - 1] > prev_upper:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = prev_upper
        if lower[i] > prev_lower or close[i - 1] < prev_lower:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = prev_lower
        prev_dir = direction[i - 1]
        if prev_dir == 1 and close[i] > final_upper[i - 1]:
            direction[i] = -1
        elif prev_dir == -1 and close[i] < final_lower[i - 1]:
            direction[i] = 1
        else:
            direction[i] = prev_dir
        st[i] = final_lower[i] if direction[i] == -1 else final_upper[i]
    return st, direction


def _htf_ema_aligned(df: pd.DataFrame, htf_minutes: int, ema_len: int) -> np.ndarray:
    """Pine ``request.security(... lookahead_off)`` parity — compute EMA on
    the higher-timeframe close series, then carry the value forward onto the
    intraday index so each bar sees only data from already-closed HTF bars."""
    if htf_minutes <= 0:
        return ema_array(df["close"].to_numpy(dtype=np.float64), ema_len)
    rule = f"{htf_minutes}min"
    htf = df["close"].resample(rule).last().dropna()
    if len(htf) < 1:
        return np.full(len(df), np.nan, dtype=np.float64)
    htf_e = ema_array(htf.to_numpy(dtype=np.float64), ema_len)
    aligned = pd.Series(htf_e, index=htf.index).reindex(df.index, method="ffill")
    return aligned.to_numpy(dtype=np.float64)


def _minutes_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since midnight IST for each bar (tz-naive IST wall-clock)."""
    return (index.hour * 60 + index.minute).to_numpy(dtype=np.int64)


# ────────────────────────────── signal builder ───────────────────────────────


def compute_lsvwap_signals(df: pd.DataFrame, params: LSVWAPParams) -> pd.DataFrame:
    """Add LS-VWAP indicator columns + ``long_entry`` / ``short_entry`` flags."""
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
    vol_spike = vol > (vol_sma * params.volume_mult)

    st, st_dir = supertrend_array(high, low, close, params.st_atr_period, params.st_factor)
    bull_st = st_dir < 0
    bear_st = st_dir > 0

    htf_ema = _htf_ema_aligned(df, params.htf_minutes, params.ema_len)
    htf_close = pd.Series(close, index=df.index).resample(f"{params.htf_minutes}min").last()
    htf_close_aligned = htf_close.reindex(df.index, method="ffill").to_numpy(dtype=np.float64)
    htf_bull = htf_close_aligned > htf_ema
    htf_bear = htf_close_aligned < htf_ema

    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    higher_highs = high > high_prev
    higher_lows = low > low_prev
    lower_highs = high < high_prev
    lower_lows = low < low_prev

    # Liquidity sweep: prior bar's low/high taken out intra-bar but reclaimed.
    bull_sweep = (low < low_prev) & (close > low_prev)
    bear_sweep = (high > high_prev) & (close < high_prev)

    bull_reclaim = (close > vwap) & (open_ < close) & (close > ema)
    bear_reclaim = (close < vwap) & (open_ > close) & (close < ema)

    bull_trend = (
        (close > ema)
        & (ema > vwap)
        & higher_highs
        & higher_lows
        & htf_bull
    )
    bear_trend = (
        (close < ema)
        & (ema < vwap)
        & lower_highs
        & lower_lows
        & htf_bear
    )

    if params.enable_session:
        mins = _minutes_of_day(df.index)
        in_session = (
            (mins >= params.session1_start_min) & (mins <= params.session1_end_min)
        ) | (
            (mins >= params.session2_start_min) & (mins <= params.session2_end_min)
        )
    else:
        in_session = np.ones(n, dtype=bool)

    st_long_ok = bull_st if params.use_supertrend else np.ones(n, dtype=bool)
    st_short_ok = bear_st if params.use_supertrend else np.ones(n, dtype=bool)

    long_entry = (
        bull_trend & bull_sweep & bull_reclaim & vol_spike & in_session & st_long_ok
    )
    short_entry = (
        bear_trend & bear_sweep & bear_reclaim & vol_spike & in_session & st_short_ok
    )

    # Stop is anchored under the swept liquidity (min of current + prior low) for
    # longs / max for shorts, with an ATR cushion — same as Pine.
    swept_low = np.minimum(low, low_prev)
    swept_high = np.maximum(high, high_prev)
    long_stop = swept_low - atr * params.atr_stop_mult
    short_stop = swept_high + atr * params.atr_stop_mult

    long_risk = close - long_stop
    short_risk = short_stop - close
    long_tp1 = close + long_risk * params.rr_target1
    long_tp2 = close + long_risk * params.rr_target2
    short_tp1 = close - short_risk * params.rr_target1
    short_tp2 = close - short_risk * params.rr_target2

    out["ema_fast"] = ema
    out["vwap"] = vwap
    out["supertrend"] = st
    out["atr"] = atr
    out["volume_sma"] = vol_sma
    out["long_entry"] = long_entry
    out["short_entry"] = short_entry
    out["long_stop"] = long_stop
    out["long_tp1"] = long_tp1
    out["long_tp2"] = long_tp2
    out["short_stop"] = short_stop
    out["short_tp1"] = short_tp1
    out["short_tp2"] = short_tp2
    return out


# ────────────────────────────── fill simulator ───────────────────────────────


def _mark_equity(cash: float, side: int, shares: float, entry: float, price: float) -> float:
    if side == 0:
        return cash
    if side == 1:
        return cash + shares * price
    return cash + shares * (entry - price)


def simulate_lsvwap(
    df: pd.DataFrame,
    params: LSVWAPParams,
    *,
    init_cash: float = 100_000.0,
    size_frac: float = 1.0,
    signals: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, list[float], pd.DataFrame]:
    """Long/short fill loop with two-target partial exits (50% off at 1R,
    remainder at 2R), ATR stop. Matches Pine ``pyramiding=0`` — only one
    position at a time, no adds."""
    sig = signals if signals is not None else compute_lsvwap_signals(df, params)
    close = sig["close"].to_numpy(dtype=np.float64)
    high = sig["high"].to_numpy(dtype=np.float64)
    low = sig["low"].to_numpy(dtype=np.float64)
    long_entry = sig["long_entry"].fillna(False).to_numpy(dtype=bool)
    short_entry = sig["short_entry"].fillna(False).to_numpy(dtype=bool)
    long_stop_a = sig["long_stop"].to_numpy(dtype=np.float64)
    long_tp1_a = sig["long_tp1"].to_numpy(dtype=np.float64)
    long_tp2_a = sig["long_tp2"].to_numpy(dtype=np.float64)
    short_stop_a = sig["short_stop"].to_numpy(dtype=np.float64)
    short_tp1_a = sig["short_tp1"].to_numpy(dtype=np.float64)
    short_tp2_a = sig["short_tp2"].to_numpy(dtype=np.float64)

    fee = params.commission_rate
    n = len(sig)
    equity = np.empty(n, dtype=np.float64)

    cash = init_cash
    side = 0
    shares = 0.0
    initial_shares = 0.0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None
    stop_px = tp1_px = tp2_px = np.nan
    tp1_taken = False
    trade_returns: list[float] = []
    trade_rows: list[dict[str, Any]] = []

    def _record(direction: str, exit_price: float, exit_idx: int, reason: str, qty: float) -> None:
        if entry_price <= 0 or qty <= 0:
            return
        if direction == "long":
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price
        trade_returns.append(ret)
        trade_rows.append(
            {
                "entry_time": entry_time,
                "exit_time": sig.index[exit_idx],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": ret * 100.0,
                "exit_reason": reason,
            }
        )

    def close_partial(fraction: float, exit_price: float, reason: str, i: int) -> None:
        """Close ``fraction`` of the initial position at ``exit_price``."""
        nonlocal cash, shares
        if side == 0 or initial_shares <= 0:
            return
        qty = min(initial_shares * fraction, shares)
        if qty <= 0:
            return
        if side == 1:
            cash += qty * exit_price * (1.0 - fee)
        else:
            cash += qty * (entry_price - exit_price) - qty * exit_price * fee
        shares -= qty
        _record("long" if side == 1 else "short", exit_price, i, reason, qty)

    def close_all(exit_price: float, reason: str, i: int) -> None:
        nonlocal cash, side, shares, initial_shares, entry_price, entry_time
        nonlocal stop_px, tp1_px, tp2_px, tp1_taken
        if side == 0 or shares <= 0:
            return
        qty = shares
        if side == 1:
            cash += qty * exit_price * (1.0 - fee)
            direction = "long"
        else:
            cash += qty * (entry_price - exit_price) - qty * exit_price * fee
            direction = "short"
        _record(direction, exit_price, i, reason, qty)
        side = 0
        shares = 0.0
        initial_shares = 0.0
        entry_price = 0.0
        entry_time = None
        stop_px = tp1_px = tp2_px = np.nan
        tp1_taken = False

    def open_position(direction: int, i: int) -> None:
        nonlocal cash, side, shares, initial_shares, entry_price, entry_time
        nonlocal stop_px, tp1_px, tp2_px, tp1_taken
        px = close[i]
        eq = _mark_equity(cash, side, shares, entry_price, px)
        deploy = eq * size_frac
        if deploy <= 0 or px <= 0:
            return
        qty = deploy / px * (1.0 - fee)
        if direction == 1:
            cash = eq - deploy
            stop_px = long_stop_a[i]
            tp1_px = long_tp1_a[i]
            tp2_px = long_tp2_a[i]
        else:
            cash = eq
            stop_px = short_stop_a[i]
            tp1_px = short_tp1_a[i]
            tp2_px = short_tp2_a[i]
        side = direction
        shares = qty
        initial_shares = qty
        entry_price = px
        entry_time = sig.index[i]
        tp1_taken = False

    for i in range(n):
        px = close[i]

        # Intra-bar exits in priority order: stop → final target → partial target.
        if side == 1:
            if np.isfinite(stop_px) and low[i] <= stop_px:
                close_all(stop_px, "stop", i)
            elif np.isfinite(tp2_px) and high[i] >= tp2_px:
                close_all(tp2_px, "target2", i)
            elif (not tp1_taken) and np.isfinite(tp1_px) and high[i] >= tp1_px:
                close_partial(params.target1_qty_frac, tp1_px, "target1", i)
                tp1_taken = True
        elif side == -1:
            if np.isfinite(stop_px) and high[i] >= stop_px:
                close_all(stop_px, "stop", i)
            elif np.isfinite(tp2_px) and low[i] <= tp2_px:
                close_all(tp2_px, "target2", i)
            elif (not tp1_taken) and np.isfinite(tp1_px) and low[i] <= tp1_px:
                close_partial(params.target1_qty_frac, tp1_px, "target1", i)
                tp1_taken = True

        # Entries only while flat — Pine pyramiding=0.
        if side == 0:
            if long_entry[i]:
                open_position(1, i)
            elif short_entry[i]:
                open_position(-1, i)

        equity[i] = _mark_equity(cash, side, shares, entry_price, px)

    return equity, trade_returns, pd.DataFrame(trade_rows)


# ────────────────────────────── public entry point ───────────────────────────


def run_lsvwap_backtest(
    strategy: StrategyDefinition,
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    *,
    init_cash: float = 100_000.0,
) -> BacktestResult:
    params = lsvwap_params_from_strategy(strategy)
    size_frac = float(strategy.risk.position_sizing.value)
    equity, trade_returns, trades_df = simulate_lsvwap(
        df, params, init_cash=init_cash, size_frac=size_frac
    )
    metrics = metrics_from_equity(equity, init_cash, len(trade_returns), trade_returns)
    enriched = compute_lsvwap_signals(df, params)
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
