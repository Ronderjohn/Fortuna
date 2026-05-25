"""Timeframe → default viewport sizing + index normalization for charts.

Used by:

- ``app/streamlit_app.py`` to pick the initial visible-bar count per
  timeframe before handing it to :mod:`fortuna.app.lightweight_chart`.
- :mod:`fortuna.app.lightweight_chart` to coerce mixed tz-aware / tz-naive
  indices (history vs. live forming bar) into a single tz-naive IST index.
"""

from __future__ import annotations

import pandas as pd

# NSE cash session ≈ 375 minutes (09:15–15:30)
_BARS_PER_SESSION: dict[str, int] = {
    "1m": 375,
    "5m": 75,
    "15m": 25,
    "30m": 13,
    "45m": 9,
    "60m": 7,
    "1h": 7,
    "2h": 4,
    "3h": 3,
    "4h": 2,
    "1d": 1,
    "1W": 1,
    "1M": 1,
}

# How many sessions to show by default (pan left for older history)
_VISIBLE_SESSIONS: dict[str, float] = {
    "1m": 0.35,
    "5m": 1.0,
    "15m": 3.0,
    "30m": 4.0,
    "45m": 4.0,
    "60m": 6.0,
    "1h": 6.0,
    "2h": 8.0,
    "3h": 10.0,
    "4h": 12.0,
    "1d": 60.0,
    "1W": 26.0,
    "1M": 12.0,
}

_IST_TZ = "Asia/Kolkata"


def visible_sessions_for_timeframe(timeframe: str) -> float:
    """How many NSE sessions to show in the initial chart viewport."""
    return _VISIBLE_SESSIONS.get(timeframe.lower().strip(), 1.0)


def visible_bars_for_timeframe(timeframe: str) -> int:
    """Default candle count in the initial viewport."""
    tf = timeframe.lower().strip()
    per_day = _BARS_PER_SESSION.get(tf, 75)
    sessions = visible_sessions_for_timeframe(tf)
    return max(40, int(per_day * sessions))


def normalize_to_naive_ist(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Strip tz / coerce object index → tz-naive IST wall-clock datetime64.

    Required because live forming bars arrive tz-aware (IST) while historical
    OHLCV is tz-naive — concatenating produces an ``object`` index that breaks
    pandas hour-of-day arithmetic. Naive timestamps are treated as IST
    wall-clock; tz-aware timestamps are converted to IST then de-localized so
    the merged index stays homogeneous.
    """
    if ohlcv.empty:
        return ohlcv
    idx = ohlcv.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_convert(_IST_TZ).tz_localize(None)
    else:
        # Object index from concat of naive + tz-aware — coerce element-wise.
        normalized: list[pd.Timestamp] = []
        for ts in idx:
            t = pd.Timestamp(ts)
            if t.tzinfo is not None:
                t = t.tz_convert(_IST_TZ).tz_localize(None)
            normalized.append(t)
        idx = pd.DatetimeIndex(normalized)
    out = ohlcv.copy()
    out.index = idx
    out = out[~out.index.isna()]
    return out.sort_index()
