"""Map strategy timeframe strings to vectorbt/pandas frequency codes."""

from __future__ import annotations

TIMEFRAME_TO_FREQ: dict[str, str] = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "45m": "45min",
    "1h": "1h",
    "60m": "1h",
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
    "1d": "1d",
    "1W": "1W",
    "1wk": "1W",
    "1M": "1M",
    "1mo": "1M",
}


def timeframe_to_freq(timeframe: str) -> str:
    return TIMEFRAME_TO_FREQ.get(timeframe, "1d")
