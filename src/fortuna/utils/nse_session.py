"""NSE cash session helpers (IST, Mon-Fri 09:15-15:30)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from fortuna.backtesting.standard.config import SessionRules

_SESSION = SessionRules()
_OPEN_MIN = int(_SESSION.market_open.split(":")[0]) * 60 + int(
    _SESSION.market_open.split(":")[1]
)
_CLOSE_MIN = int(_SESSION.market_close.split(":")[0]) * 60 + int(
    _SESSION.market_close.split(":")[1]
)


def in_nse_session(ts: pd.Timestamp | datetime | None) -> bool:
    """True when ``ts`` lies inside an NSE regular session (Mon-Fri, 09:15-15:30 IST)."""
    if ts is None:
        return False
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Kolkata").tz_localize(None)
    if stamp.weekday() >= 5:
        return False
    minutes = stamp.hour * 60 + stamp.minute
    return _OPEN_MIN <= minutes <= _CLOSE_MIN


def nse_session_date(ts: pd.Timestamp | datetime | None) -> date | None:
    """Calendar date for an NSE session timestamp (IST)."""
    if ts is None:
        return None
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Kolkata").tz_localize(None)
    return stamp.date()
