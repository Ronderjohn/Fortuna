"""TradingView interval specs and OHLCV resampling."""

from __future__ import annotations

import pandas as pd
import pytest

from fortuna.data.timeframes import (
    get_timeframe_spec,
    materialize_timeframe,
    resample_ohlcv,
    TRADINGVIEW_INTERVALS,
)


def test_tradingview_intervals_list() -> None:
    assert "5m" in TRADINGVIEW_INTERVALS
    assert "45m" in TRADINGVIEW_INTERVALS
    assert "1W" in TRADINGVIEW_INTERVALS


def test_45m_resampled_from_15m() -> None:
    idx = pd.date_range("2026-05-01 09:15", periods=30, freq="15min")
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100},
        index=idx,
    )
    out = materialize_timeframe(df, "45m")
    assert 9 <= len(out) <= 12
    spec = get_timeframe_spec("45m")
    assert spec.api_timeframe == "15m"
    assert spec.resample_rule == "45min"


def test_resample_ohlcv_aggregation() -> None:
    idx = pd.date_range("2026-05-01", periods=8, freq="1h")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6, 7, 8],
            "high": [2, 3, 4, 5, 6, 7, 8, 9],
            "low": [0, 1, 2, 3, 4, 5, 6, 7],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
            "volume": [10] * 8,
        },
        index=idx,
    )
    out = resample_ohlcv(df, "2h")
    assert len(out) == 4
    assert float(out.iloc[0]["open"]) == 1.0
    assert float(out.iloc[0]["close"]) == 2.5
