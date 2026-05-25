"""
Opening Range Breakout (ORB) — popular NSE intraday approach.

Reference practice (widely used, moderate win rate ~40–55% with discipline):
- Define opening range from first N minutes after 09:15 (default 15m = 3×5m bars)
- Long on close above OR high; short on close below OR low
- ATR-based stop; optional R-multiple target
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.engine import BacktestResult
from fortuna.backtesting.metrics import metrics_from_equity
from fortuna.indicators.kernels import atr_array
from fortuna.strategy.schema import StrategyDefinition


@dataclass(frozen=True)
class ORBParams:
    opening_bars: int = 3
    atr_len: int = 14
    atr_stop_mult: float = 1.5
    risk_reward: float = 2.0
    commission_rate: float = 0.0005
    square_off_hour: int = 15
    square_off_minute: int = 15


def is_orb_strategy(strategy: StrategyDefinition) -> bool:
    engine = getattr(strategy.metadata, "engine", None)
    return engine == "orb" or strategy.name.lower().startswith("orb")


def orb_params_from_strategy(strategy: StrategyDefinition) -> ORBParams:
    raw: dict[str, Any] = {}
    for tag in strategy.metadata.tags:
        if tag.startswith("orb_") and "=" in tag:
            k, v = tag.split("=", 1)
            raw[k] = v
    return ORBParams(
        opening_bars=int(raw.get("orb_opening_bars", 3)),
        atr_len=int(raw.get("orb_atr_len", 14)),
        atr_stop_mult=float(raw.get("orb_atr_mult", 1.5)),
        risk_reward=float(raw.get("orb_rr", 2.0)),
    )


def _session_date(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.date


def compute_orb_signals(df: pd.DataFrame, params: ORBParams) -> pd.DataFrame:
    out = df.copy()
    n = len(df)
    dates = _session_date(df.index)
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    unique_dates = pd.unique(dates)
    for d in unique_dates:
        mask = dates == d
        idxs = np.where(mask)[0]
        if len(idxs) < params.opening_bars + 2:
            continue
        ob = idxs[: params.opening_bars]
        oh = float(df["high"].iloc[ob].max())
        ol = float(df["low"].iloc[ob].min())
        for i in idxs:
            or_high[i] = oh
            or_low[i] = ol
        post = idxs[params.opening_bars :]
        for j, i in enumerate(post):
            c = float(df["close"].iloc[i])
            prev_c = float(df["close"].iloc[i - 1]) if i > 0 else c
            if c > oh and prev_c <= oh:
                long_entry[i] = True
            elif c < ol and prev_c >= ol:
                short_entry[i] = True

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    atr = atr_array(high, low, close, params.atr_len)

    out["or_high"] = or_high
    out["or_low"] = or_low
    out["atr"] = atr
    out["long_entry"] = long_entry
    out["short_entry"] = short_entry
    out["long_stop"] = close - atr * params.atr_stop_mult
    out["long_target"] = close + atr * params.atr_stop_mult * params.risk_reward
    out["short_stop"] = close + atr * params.atr_stop_mult
    out["short_target"] = close - atr * params.atr_stop_mult * params.risk_reward
    return out


def simulate_orb(
    df: pd.DataFrame,
    params: ORBParams,
    *,
    init_cash: float,
    size_frac: float = 1.0,
) -> tuple[np.ndarray, list[float], pd.DataFrame]:
    """Long/short simulator (same fill model as MMTS builtin)."""
    from fortuna.strategies.builtin.mmts import MMTSParams, simulate_mmts

    sig = compute_orb_signals(df, params)
    mp = MMTSParams(
        atr_len=params.atr_len,
        atr_mult=params.atr_stop_mult,
        commission_rate=params.commission_rate,
    )
    return simulate_mmts(df, mp, init_cash=init_cash, size_frac=size_frac, signals=sig)


def run_orb_backtest(
    strategy: StrategyDefinition,
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    *,
    init_cash: float = 100_000.0,
) -> BacktestResult:
    params = orb_params_from_strategy(strategy)
    size_frac = float(strategy.risk.position_sizing.value)
    equity, trade_returns, trades_df = simulate_orb(
        df, params, init_cash=init_cash, size_frac=size_frac
    )
    metrics = metrics_from_equity(equity, init_cash, len(trade_returns), trade_returns)
    enriched = compute_orb_signals(df, params)
    enriched.attrs["trades"] = trades_df
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
