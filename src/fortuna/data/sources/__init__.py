"""Pluggable market data sources."""

from fortuna.data.sources.base import DataSource
from fortuna.data.sources.symbols import (
    normalize_for_openchart,
    normalize_for_smartapi,
    normalize_for_yfinance,
)
from fortuna.data.sources.yfinance_source import YFinanceSource

__all__ = [
    "DataSource",
    "YFinanceSource",
    "normalize_for_openchart",
    "normalize_for_smartapi",
    "normalize_for_yfinance",
]


def get_data_source(name: str) -> DataSource:
    """Factory for configured data sources."""
    key = name.lower().strip()
    if key == "yfinance":
        return YFinanceSource()
    if key == "openchart":
        from fortuna.data.sources.openchart_source import OpenChartSource

        return OpenChartSource()
    if key == "smartapi":
        from fortuna.data.sources.smartapi_historical import SmartAPIHistoricalSource

        return SmartAPIHistoricalSource()
    raise ValueError(f"Unknown data source: {name}")


def get_live_feed_class():
    """Lazy import for optional smartapi-python dependency."""
    from fortuna.data.sources.smartapi_live import SmartAPILiveBarFeed

    return SmartAPILiveBarFeed
