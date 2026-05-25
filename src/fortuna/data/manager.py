"""Orchestrate fetch, cache, and optional DuckDB registration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.backtesting.standard.calendar import filter_session_bars
from fortuna.backtesting.standard.config import SessionRules
from fortuna.config.settings import Settings, get_settings
from fortuna.data.cache import DataCache
from fortuna.data.duckdb_store import DuckDBStore
from fortuna.data.sources import get_data_source
from fortuna.data.timeframes import (
    effective_history_days,
    get_timeframe_spec,
    is_intraday,
    materialize_timeframe,
    normalize_timeframe,
)
from fortuna.utils.logging import get_logger
from fortuna.utils.timing import timed_step

logger = get_logger(__name__)

_INTRADAY_TIMEFRAMES = frozenset(
    {"1m", "2m", "5m", "15m", "30m", "45m", "1h", "60m", "2h", "3h", "4h"}
)
_NSE_SESSION = SessionRules()


class MarketDataManager:
    """High-level API for market data with cache-first semantics."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[DataCache] = None,
        duckdb_store: Optional[DuckDBStore] = None,
        data_source: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        cache_dir = self.settings.resolve_path(self.settings.data_cache_dir)
        self.cache = cache or DataCache(cache_dir)
        source_name = data_source or self.settings.data_source
        self._source = get_data_source(source_name)
        self._source_name = source_name
        self._duckdb: Optional[DuckDBStore] = duckdb_store

    @property
    def duckdb(self) -> DuckDBStore:
        if self._duckdb is None:
            db_path = self.settings.resolve_path(self.settings.duckdb_path)
            self._duckdb = DuckDBStore(db_path)
        return self._duckdb

    def _cache_stale(self, symbol: str, timeframe: str) -> bool:
        if timeframe in _INTRADAY_TIMEFRAMES and self.settings.cache_max_age_hours:
            return self.cache.is_stale(
                symbol,
                timeframe,
                max_age_hours=self.settings.cache_max_age_hours,
            )
        return self.cache.is_stale(symbol, timeframe, max_age_days=self.settings.cache_max_age_days)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        force_refresh: bool = False,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return OHLCV data, fetching and caching if needed."""
        symbol = symbol.upper()
        tf = normalize_timeframe(timeframe)
        spec = get_timeframe_spec(tf)
        fetch_days = effective_history_days(tf, days or self.settings.default_days)
        stale = self._cache_stale(symbol, tf)

        if not force_refresh and self.cache.exists(symbol, tf) and not stale:
            with timed_step(f"data.cache_hit {symbol} {tf}") as d:
                df = self.cache.read(symbol, tf, start=start, end=end)
                if is_intraday(tf):
                    df = filter_session_bars(df, _NSE_SESSION)
                d["rows"] = str(len(df))
            return df

        reason = "force_refresh" if force_refresh else "stale_or_miss"
        logger.info(
            "Fetching %s %s (api %s) via %s (refresh=%s)",
            symbol,
            tf,
            spec.api_timeframe,
            self._source_name,
            force_refresh or stale,
        )
        with timed_step(f"data.fetch {symbol} {tf}") as d:
            d["source"] = self._source_name
            d["reason"] = reason
            raw = self._source.fetch(
                symbol,
                spec.api_timeframe,
                start=start,
                end=end,
                days=fetch_days,
            )
            df = materialize_timeframe(raw, tf)
            if is_intraday(tf):
                df = filter_session_bars(df, _NSE_SESSION)
            d["rows"] = str(len(df))
            self.cache.write(df, symbol, tf)
            path = self.cache.get_path(symbol, tf)
            if path is not None:
                self.duckdb.register_parquet(symbol, tf, path)

        if start or end:
            df = self.cache.read(symbol, tf, start=start, end=end)
            if is_intraday(tf):
                df = filter_session_bars(df, _NSE_SESSION)
            return df
        return df

    def append_bar(self, symbol: str, timeframe: str, bar_df: pd.DataFrame) -> None:
        """Append one or more OHLCV rows to the Parquet cache (live feed)."""
        symbol = symbol.upper()
        if self.cache.exists(symbol, timeframe):
            existing = self.cache.read(symbol, timeframe)
            combined = pd.concat([existing, bar_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = bar_df.sort_index()
        self.cache.write(combined, symbol, timeframe)
        path = self.cache.get_path(symbol, timeframe)
        if path is not None:
            self.duckdb.register_parquet(symbol, timeframe, path)

    def query(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query cached data via DuckDB (loads cache first if missing)."""
        self.get_ohlcv(symbol, timeframe)
        path = self.cache.get_path(symbol, timeframe)
        if path is None:
            raise FileNotFoundError(f"No cache for {symbol} {timeframe}")
        with timed_step(f"data.duckdb_query {symbol} {timeframe}") as d:
            df = self.duckdb.query_ohlcv(path, start=start, end=end)
            d["rows"] = str(len(df))
        return df
