"""Numba-accelerated indicator kernels (float64 arrays)."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def ema_array(close: np.ndarray, window: int) -> np.ndarray:
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    if n < window:
        return out
    alpha = 2.0 / (window + 1.0)
    out[window - 1] = np.mean(close[:window])
    for i in range(window, n):
        out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def sma_array(close: np.ndarray, window: int) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    running = 0.0
    for i in range(n):
        running += close[i]
        if i >= window:
            running -= close[i - window]
        if i >= window - 1:
            out[i] = running / window
    return out


@njit(cache=True)
def rsi_array(close: np.ndarray, window: int) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, window + 1):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / window
    avg_loss = losses / window
    if avg_loss == 0:
        out[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[window] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(window + 1, n):
        delta = close[i] - close[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


@njit(cache=True)
def atr_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n = close.shape[0]
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return sma_array(tr, window)
