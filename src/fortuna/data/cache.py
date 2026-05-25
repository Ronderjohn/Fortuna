"""Parquet-based local OHLCV cache."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.indicators.registry import OHLCV_COLUMNS
from fortuna.utils.logging import get_logger
from fortuna.utils.timing import timed_step

logger = get_logger(__name__)


class DataCache:
    """Read/write OHLCV data as Parquet files."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_").upper()
        return self.cache_dir / safe_symbol / f"{timeframe}.parquet"

    def exists(self, symbol: str, timeframe: str) -> bool:
        return self._path(symbol, timeframe).exists()

    def is_stale(
        self,
        symbol: str,
        timeframe: str,
        max_age_days: int = 1,
        max_age_hours: Optional[int] = None,
    ) -> bool:
        path = self._path(symbol, timeframe)
        if not path.exists():
            return True
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        if max_age_hours is not None:
            return age > timedelta(hours=max_age_hours)
        return age > timedelta(days=max_age_days)

    def read(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        path = self._path(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(f"Cache miss: {path}")
        with timed_step(f"data.parquet_read {symbol} {timeframe}") as d:
            df = pd.read_parquet(path, columns=list(OHLCV_COLUMNS))
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]
            d["rows"] = str(len(df))
        logger.debug("Cache hit: %s %s (%s rows)", symbol, timeframe, len(df))
        return df

    def write(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        path = self._path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        if out.index.name != "datetime":
            out.index.name = "datetime"
        with timed_step(f"data.parquet_write {symbol} {timeframe}") as d:
            out.to_parquet(path, index=True)
            d["rows"] = str(len(out))
        logger.info("Cached %s %s -> %s", symbol, timeframe, path)
        return path

    def get_path(self, symbol: str, timeframe: str) -> Optional[Path]:
        path = self._path(symbol, timeframe)
        return path if path.exists() else None
