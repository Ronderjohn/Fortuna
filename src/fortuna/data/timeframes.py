"""TradingView-style intervals: SmartAPI fetch + optional OHLCV resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Labels shown in the dashboard (TradingView order)
TRADINGVIEW_INTERVALS: list[str] = [
    "5m",
    "15m",
    "30m",
    "45m",
    "1h",
    "2h",
    "3h",
    "4h",
    "1d",
    "1W",
    "1M",
]

_INTERVAL_LABELS: dict[str, str] = {
    "5m": "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "45m": "45 minutes",
    "1h": "1 hour",
    "2h": "2 hours",
    "3h": "3 hours",
    "4h": "4 hours",
    "1d": "1 day",
    "1W": "1 week",
    "1M": "1 month",
}


@dataclass(frozen=True)
class TimeframeSpec:
    """How to obtain bars for a UI timeframe."""

    api_timeframe: str
    resample_rule: Optional[str] = None
    min_history_days: int = 30


_SPECS: dict[str, TimeframeSpec] = {
    "5m": TimeframeSpec("5m", min_history_days=30),
    "15m": TimeframeSpec("15m", min_history_days=45),
    "30m": TimeframeSpec("30m", min_history_days=60),
    "45m": TimeframeSpec("15m", "45min", min_history_days=60),
    "1h": TimeframeSpec("1h", min_history_days=90),
    "60m": TimeframeSpec("1h", min_history_days=90),
    "2h": TimeframeSpec("1h", "2h", min_history_days=120),
    "3h": TimeframeSpec("1h", "3h", min_history_days=180),
    "4h": TimeframeSpec("1h", "4h", min_history_days=180),
    "1d": TimeframeSpec("1d", min_history_days=365),
    "1D": TimeframeSpec("1d", min_history_days=365),
    "1W": TimeframeSpec("1d", "W-FRI", min_history_days=730),
    "1wk": TimeframeSpec("1d", "W-FRI", min_history_days=730),
    "1M": TimeframeSpec("1d", "ME", min_history_days=1095),
    "1mo": TimeframeSpec("1d", "ME", min_history_days=1095),
}


def normalize_timeframe(timeframe: str) -> str:
    tf = timeframe.strip()
    aliases = {"60m": "1h", "1D": "1d", "1wk": "1W", "1mo": "1M"}
    return aliases.get(tf, tf)


def get_timeframe_spec(timeframe: str) -> TimeframeSpec:
    key = normalize_timeframe(timeframe)
    spec = _SPECS.get(key)
    if spec is None:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Choose from: {', '.join(TRADINGVIEW_INTERVALS)}"
        )
    return spec


def is_intraday(timeframe: str) -> bool:
    return normalize_timeframe(timeframe) not in ("1d", "1W", "1M")


def interval_label(timeframe: str) -> str:
    key = normalize_timeframe(timeframe)
    return _INTERVAL_LABELS.get(key, key)


def effective_history_days(timeframe: str, requested_days: int) -> int:
    """Ensure enough calendar days for resampled / higher-TF charts."""
    spec = get_timeframe_spec(timeframe)
    return max(requested_days, spec.min_history_days)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate OHLCV to a higher timeframe (NSE session-aware index preserved)."""
    if df.empty:
        return df
    out = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def materialize_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Apply resampling when the UI timeframe differs from the API interval."""
    spec = get_timeframe_spec(timeframe)
    if not spec.resample_rule:
        return df
    return resample_ohlcv(df, spec.resample_rule)
