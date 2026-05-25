"""Market data ingestion and caching (lazy exports — avoids heavy import on submodules)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DataCache",
    "DataFetcher",
    "DuckDBStore",
    "MarketDataManager",
    "OhlcvBarStream",
    "StreamConfig",
    "stream_from_manager",
]

_LAZY = {
    "DataCache": ("fortuna.data.cache", "DataCache"),
    "DataFetcher": ("fortuna.data.fetcher", "DataFetcher"),
    "DuckDBStore": ("fortuna.data.duckdb_store", "DuckDBStore"),
    "MarketDataManager": ("fortuna.data.manager", "MarketDataManager"),
    "OhlcvBarStream": ("fortuna.data.stream", "OhlcvBarStream"),
    "StreamConfig": ("fortuna.data.stream", "StreamConfig"),
    "stream_from_manager": ("fortuna.data.stream", "stream_from_manager"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = _LAZY[name]
    return getattr(importlib.import_module(mod_name), attr)
