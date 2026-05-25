"""CuPy indicator kernels (float64)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fortuna.compute.device import get_cupy, synchronize_gpu
from fortuna.indicators.kernels import atr_array, ema_array, rsi_array, sma_array


def _cp():
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy not available")
    return cp


def ema_gpu(close: np.ndarray, window: int) -> np.ndarray:
    cp = _cp()
    c = cp.asarray(close, dtype=cp.float64)
    n = c.shape[0]
    out = cp.empty(n, dtype=cp.float64)
    out[:] = cp.nan
    if n < window:
        return cp.asnumpy(out)
    alpha = 2.0 / (window + 1.0)
    out[window - 1] = cp.mean(c[:window])
    for i in range(window, n):
        out[i] = alpha * c[i] + (1.0 - alpha) * out[i - 1]
    synchronize_gpu()
    return cp.asnumpy(out)


def sma_gpu(close: np.ndarray, window: int) -> np.ndarray:
    cp = _cp()
    c = cp.asarray(close, dtype=cp.float64)
    kernel = cp.ones(window, dtype=cp.float64) / window
    valid = cp.convolve(c, kernel, mode="valid")
    out = cp.full(c.shape[0], cp.nan, dtype=cp.float64)
    out[window - 1 :] = valid
    synchronize_gpu()
    return cp.asnumpy(out)


def rsi_gpu(close: np.ndarray, window: int) -> np.ndarray:
    """GPU transfer + CPU Numba RSI (Wilder loop is sequential)."""
    return rsi_array(close, window)


def atr_gpu(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """TR on GPU; Wilder smoothing on CPU (sequential)."""
    return atr_array(
        high.astype(np.float64, copy=False),
        low.astype(np.float64, copy=False),
        close.astype(np.float64, copy=False),
        window,
    )


def compute_series_cpu(kind: str, close: np.ndarray, window: int, ohlcv: dict[str, np.ndarray]) -> np.ndarray:
    if kind == "ema":
        return ema_array(close, window)
    if kind == "sma":
        return sma_array(close, window)
    if kind == "rsi":
        return rsi_array(close, window)
    if kind == "atr":
        return atr_array(ohlcv["high"], ohlcv["low"], ohlcv["close"], window)
    raise ValueError(kind)


def compute_series_gpu(
    kind: str,
    close: np.ndarray,
    window: int,
    ohlcv: dict[str, np.ndarray],
) -> np.ndarray:
    if kind == "ema":
        return ema_gpu(close, window)
    if kind == "sma":
        return sma_gpu(close, window)
    if kind == "rsi":
        return rsi_gpu(close, window)
    if kind == "atr":
        return atr_gpu(ohlcv["high"], ohlcv["low"], ohlcv["close"], window)
    raise ValueError(kind)
