"""Regression: live forming bar must merge with tz-naive history without error."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from fortuna.app.live_session import (
    LiveSessionBridge,
    _in_nse_session,
    _to_naive_ist,
)
from fortuna.data.sources.smartapi_bar_aggregator import OhlcvBar


def _hist() -> pd.DataFrame:
    """Tz-naive IST historical OHLCV (matches what the cache writes)."""
    idx = pd.date_range("2026-05-25 09:15", periods=3, freq="5min")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.5, 100.5, 101.5],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1500, 1200],
        },
        index=idx,
    )


def test_to_naive_ist_strips_timezone() -> None:
    aware = pd.Timestamp("2026-05-25 09:30", tz="Asia/Kolkata")
    naive = _to_naive_ist(aware)
    assert naive.tzinfo is None
    assert str(naive) == "2026-05-25 09:30:00"


def test_in_nse_session_handles_tz_aware_input() -> None:
    aware = pd.Timestamp("2026-05-25 11:40", tz="Asia/Kolkata")
    assert _in_nse_session(aware) is True
    weekend = pd.Timestamp("2026-05-23 11:40", tz="Asia/Kolkata")
    assert _in_nse_session(weekend) is False
    too_early = pd.Timestamp("2026-05-25 08:30", tz="Asia/Kolkata")
    assert _in_nse_session(too_early) is False


def test_appended_bars_are_tz_naive_so_concat_with_history_works() -> None:
    """The historical OHLCV is tz-naive IST. WS bars arrive tz-aware IST.

    After ``_append_bar`` runs, the bridge's in-memory index must remain
    tz-naive so ``pd.concat([hist, live_bar]).sort_index()`` doesn't throw
    "Cannot compare tz-naive and tz-aware timestamps".
    """
    manager = MagicMock()
    manager.append_bar = MagicMock()
    bridge = LiveSessionBridge(manager, "ICICIBANK.NS", "5m", _hist())

    aware_bar = OhlcvBar(
        datetime=pd.Timestamp("2026-05-25 09:30", tz="Asia/Kolkata"),
        open=102.0,
        high=103.0,
        low=101.5,
        close=102.7,
        volume=2000,
    )
    bridge._append_bar(aware_bar)

    idx = bridge.ohlcv.index
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.tz is None  # mixing tz-aware and tz-naive would have left it tz-aware

    # And the merge that the dashboard does every refresh works without error.
    merged = pd.concat([_hist(), bridge.ohlcv]).sort_index()
    assert merged.index.is_monotonic_increasing


def test_forming_bar_df_is_tz_naive_for_chart_concat() -> None:
    manager = MagicMock()
    bridge = LiveSessionBridge(manager, "ICICIBANK.NS", "5m", _hist())

    fake_feed = MagicMock()
    fake_feed.aggregator.current_bar = OhlcvBar(
        datetime=pd.Timestamp("2026-05-25 11:40", tz="Asia/Kolkata"),
        open=1279.0,
        high=1280.0,
        low=1278.5,
        close=1279.6,
        volume=12345,
    )
    bridge._feed = fake_feed

    forming = bridge.forming_bar_df()
    assert forming is not None and not forming.empty
    assert forming.index.tz is None

    merged = pd.concat([_hist(), forming]).sort_index()
    assert merged.index.is_monotonic_increasing
    assert pd.Timestamp("2026-05-25 11:40") in merged.index
