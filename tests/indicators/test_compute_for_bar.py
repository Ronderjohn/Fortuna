"""Phase 1 completion 2.5: ``IndicatorEngine.compute_for_bar()`` parity + latency."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.schema import IndicatorSpec, IndicatorType, PriceSource


def _make_ohlcv(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-01 09:15", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _basic_indicator_stack() -> list[IndicatorSpec]:
    return [
        IndicatorSpec(
            id="ema_fast",
            type=IndicatorType.EMA,
            source=PriceSource.CLOSE,
            params={"window": 9},
        ),
        IndicatorSpec(
            id="ema_slow",
            type=IndicatorType.EMA,
            source=PriceSource.CLOSE,
            params={"window": 21},
        ),
        IndicatorSpec(
            id="atr",
            type=IndicatorType.ATR,
            source=PriceSource.CLOSE,
            params={"window": 14},
        ),
        IndicatorSpec(
            id="rsi",
            type=IndicatorType.RSI,
            source=PriceSource.CLOSE,
            params={"window": 14},
        ),
    ]


def test_compute_for_bar_matches_last_row_of_full_compute():
    df = _make_ohlcv(600)
    eng = IndicatorEngine()
    specs = _basic_indicator_stack()

    full = eng.compute(df, specs)
    last = eng.compute_for_bar(df, specs, lookback=300)

    # Compare every indicator column at the final bar (allow tiny float noise).
    for col in ["ema_fast", "ema_slow", "atr", "rsi"]:
        assert col in last.index, f"missing {col}"
        if pd.isna(full[col].iloc[-1]) and pd.isna(last[col]):
            continue
        np.testing.assert_allclose(
            float(last[col]),
            float(full[col].iloc[-1]),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{col} parity broken",
        )


def test_compute_for_bar_under_50ms():
    """Hot-path budget: < 50 ms is generous; target is < 10 ms on CPU."""
    df = _make_ohlcv(600)
    eng = IndicatorEngine()
    specs = _basic_indicator_stack()

    eng.compute_for_bar(df, specs, lookback=300)  # warmup

    t0 = time.perf_counter()
    for _ in range(10):
        eng.compute_for_bar(df, specs, lookback=300)
    avg_ms = (time.perf_counter() - t0) / 10 * 1000

    assert avg_ms < 50.0, f"compute_for_bar averaged {avg_ms:.1f}ms (budget 50)"


def test_compute_for_bar_rejects_empty_frame():
    with pytest.raises(ValueError):
        IndicatorEngine().compute_for_bar(pd.DataFrame(), _basic_indicator_stack())
