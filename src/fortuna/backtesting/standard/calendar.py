"""NSE intraday session calendar and square-off rules."""

from __future__ import annotations

import pandas as pd

from fortuna.backtesting.standard.config import SessionRules


def _to_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def filter_session_bars(ohlcv: pd.DataFrame, rules: SessionRules) -> pd.DataFrame:
    """Keep only NSE regular session bars (09:15–15:30 IST, Mon–Fri)."""
    if ohlcv.empty:
        return ohlcv
    idx = ohlcv.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    open_m = _to_minutes(rules.market_open)
    close_m = _to_minutes(rules.market_close)
    minutes = idx.hour * 60 + idx.minute
    weekday = idx.weekday
    mask = (minutes >= open_m) & (minutes <= close_m) & (weekday < 5)
    out = ohlcv.copy()
    out.index = idx
    return out.loc[mask]


def apply_square_off(
    entries: pd.Series,
    exits: pd.Series,
    ohlcv: pd.DataFrame,
    rules: SessionRules,
) -> tuple[pd.Series, pd.Series]:
    """
    Force exit before square-off time; block new entries after square-off.
    """
    entries = entries.copy()
    exits = exits.copy()
    if ohlcv.empty:
        return entries, exits

    sq_m = _to_minutes(rules.square_off_time)
    minutes = ohlcv.index.hour * 60 + ohlcv.index.minute

    for i in range(len(ohlcv)):
        if minutes[i] >= sq_m:
            entries.iloc[i] = False
            exits.iloc[i] = True
    return entries, exits


def trading_days_count(ohlcv: pd.DataFrame) -> int:
    if ohlcv.empty:
        return 0
    return int(pd.Series(ohlcv.index).dt.date.nunique())
