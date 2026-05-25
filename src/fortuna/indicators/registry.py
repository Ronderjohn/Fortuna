"""Pure indicator computation functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fortuna.indicators.kernels import atr_array, ema_array, rsi_array, sma_array

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _series(df: pd.DataFrame, source: str) -> pd.Series:
    """Resolve price source to a pandas Series."""
    source = source.lower()
    if source in df.columns:
        return df[source]
    if source == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    if source == "ohlc4":
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    raise KeyError(f"Unknown price source: {source}")


def _to_series(arr: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(arr, index=index, dtype=float)


def compute_ema(df: pd.DataFrame, source: str, window: int = 14) -> pd.Series:
    close = _series(df, source).to_numpy(dtype=np.float64, copy=False)
    return _to_series(ema_array(close, window), df.index)


def compute_sma(df: pd.DataFrame, source: str, window: int = 14) -> pd.Series:
    close = _series(df, source).to_numpy(dtype=np.float64, copy=False)
    return _to_series(sma_array(close, window), df.index)


def compute_rsi(df: pd.DataFrame, source: str, window: int = 14) -> pd.Series:
    close = _series(df, source).to_numpy(dtype=np.float64, copy=False)
    return _to_series(rsi_array(close, window), df.index)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    return _to_series(atr_array(high, low, close, window), df.index)


def compute_vwap(df: pd.DataFrame, source: str = "hlc3") -> pd.Series:
    """Session-cumulative VWAP (resets are not applied across days in Phase 1)."""
    typical = _series(df, source)
    volume = df["volume"].replace(0, pd.NA)
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol


def compute_macd(
    df: pd.DataFrame,
    source: str,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
) -> pd.DataFrame:
    s = _series(df, source)
    fast = s.ewm(span=window_fast, adjust=False).mean()
    slow = s.ewm(span=window_slow, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=window_sign, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": macd - signal,
        }
    )


def compute_bollinger(
    df: pd.DataFrame,
    source: str,
    window: int = 20,
    window_dev: float = 2.0,
) -> pd.DataFrame:
    s = _series(df, source)
    mid = s.rolling(window).mean()
    std = s.rolling(window).std()
    return pd.DataFrame(
        {
            "bb_upper": mid + window_dev * std,
            "bb_middle": mid,
            "bb_lower": mid - window_dev * std,
        }
    )


def compute_volume_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    vol = df["volume"].to_numpy(dtype=np.float64, copy=False)
    return _to_series(sma_array(vol, window), df.index)


def compute_rolling_high(df: pd.DataFrame, source: str, window: int = 20, shift: int = 1) -> pd.Series:
    """Prior-bar rolling high (breakout level)."""
    s = _series(df, source)
    return s.rolling(window).max().shift(shift)


def compute_rolling_low(df: pd.DataFrame, source: str, window: int = 20, shift: int = 1) -> pd.Series:
    s = _series(df, source)
    return s.rolling(window).min().shift(shift)


INDICATOR_REGISTRY: dict[str, Any] = {
    "ema": compute_ema,
    "sma": compute_sma,
    "rsi": compute_rsi,
    "atr": compute_atr,
    "vwap": compute_vwap,
    "macd": compute_macd,
    "bollinger": compute_bollinger,
    "volume_sma": compute_volume_sma,
    "rolling_high": compute_rolling_high,
    "rolling_low": compute_rolling_low,
}
