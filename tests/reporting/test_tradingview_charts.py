"""Signal-frame and chart-viewport helper tests.

The live dashboard renders charts via Lightweight Charts (see
``tests/app/test_lightweight_chart.py``); this module covers the
strategy-agnostic helpers in ``chart_viewport.py`` plus
``build_signal_frame`` from the report builder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fortuna.reporting.strategy_tester.signal_overlay import build_signal_frame
from fortuna.strategy.loader import load_strategy


def _ohlcv(n: int = 80) -> pd.DataFrame:
    # 2026-03-02 is a Monday; bars stay inside NSE session so any
    # downstream filtering keeps all rows.
    idx = pd.date_range("2026-03-02 09:15", periods=n, freq="5min")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.05, n),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
        },
        index=idx,
    )


def test_build_signal_frame_macd_strategy() -> None:
    path = Path("strategies/intraday/macd_trend_5m.json")
    if not path.exists():
        pytest.skip("macd strategy json missing")
    strategy = load_strategy(path)
    ohlcv = _ohlcv()
    from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
    from fortuna.config.settings import load_settings

    bt = NumPyBacktestRunner(load_settings()).run(strategy, ohlcv)
    sig = build_signal_frame(strategy, ohlcv, bt.enriched_data)
    assert list(sig.columns) == ["signal", "enter", "exit"]
    assert set(sig["signal"].unique()).issubset({"BUY", "SELL", "HOLD"})


def test_chart_normalizes_mixed_timezone_index() -> None:
    """Regression: tz-aware live forming bar + tz-naive history must merge cleanly."""
    from zoneinfo import ZoneInfo

    from fortuna.reporting.strategy_tester.chart_viewport import normalize_to_naive_ist

    hist = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.date_range("2026-05-18 09:15", periods=10, freq="5min"),
    )
    forming = pd.DataFrame(
        {"open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 2.0},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-05-22 10:30", tz=ZoneInfo("Asia/Kolkata"))]
        ),
    )
    merged = pd.concat([hist, forming])
    assert merged.index.dtype == object  # demonstrates the bug

    normalized = normalize_to_naive_ist(merged)
    assert normalized.index.tz is None
    assert "datetime64" in str(normalized.index.dtype)
    assert len(normalized) == len(merged)


def test_visible_bars_for_timeframe_returns_sensible_defaults() -> None:
    """Smoke test for the timeframe → viewport-size helper."""
    from fortuna.reporting.strategy_tester.chart_viewport import (
        visible_bars_for_timeframe,
        visible_sessions_for_timeframe,
    )

    assert visible_bars_for_timeframe("5m") >= 40
    assert visible_bars_for_timeframe("1d") >= 40
    # Unknown timeframes fall back to a single session (75 5m bars).
    assert visible_sessions_for_timeframe("nonsense") == 1.0


def test_rsi_scalp_entries_not_always_zero() -> None:
    from fortuna.backtesting.compiler import StrategyCompiler
    from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
    from fortuna.config.settings import load_settings
    from fortuna.indicators.engine import IndicatorEngine
    from fortuna.strategy.loader import load_strategy

    path = Path("strategies/intraday/rsi_scalp_5m.json")
    if not path.exists():
        pytest.skip("rsi_scalp_5m missing")
    ohlcv = _ohlcv(800)
    strategy = load_strategy(path)
    enr = IndicatorEngine().compute(ohlcv, strategy.indicators)
    entries, _ = StrategyCompiler().compile(strategy, enr)
    assert int(entries.sum()) > 0
    bt = NumPyBacktestRunner(load_settings()).run(strategy, ohlcv)
    assert bt.metrics.total_trades > 0
