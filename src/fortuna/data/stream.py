"""Stream OHLCV windows from DuckDB / Parquet for rolling evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from fortuna.data.duckdb_store import DuckDBStore
from fortuna.utils.timing import timed_step


@dataclass
class StreamConfig:
    """Bar-window streaming settings."""

    window_bars: int = 78
    step_bars: int = 1
    warmup_bars: int = 0


class OhlcvBarStream:
    """
    Yield rolling OHLCV windows from a Parquet file via DuckDB.

    Loads the series once, then slices in memory (efficient for intraday caches).
    For very large files, use ``chunk_query`` on DuckDBStore directly.
    """

    def __init__(
        self,
        parquet_path: Path,
        config: Optional[StreamConfig] = None,
        store: Optional[DuckDBStore] = None,
    ) -> None:
        self.path = Path(parquet_path)
        self.config = config or StreamConfig()
        self._store = store
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            with timed_step("stream.load_ohlcv") as d:
                if self._store is not None:
                    d["via"] = "duckdb"
                    self._df = self._store.query_ohlcv(self.path)
                else:
                    d["via"] = "parquet"
                    self._df = pd.read_parquet(self.path)
                    if "datetime" in self._df.columns:
                        self._df = self._df.set_index("datetime")
                    self._df.index = pd.to_datetime(self._df.index)
                d["rows"] = str(len(self._df))
        return self._df

    def __len__(self) -> int:
        df = self._load()
        start = self.config.warmup_bars + self.config.window_bars
        if len(df) < start:
            return 0
        return max(0, (len(df) - start) // self.config.step_bars + 1)

    def windows(self) -> Iterator[tuple[int, pd.DataFrame]]:
        """Yield ``(bar_index, window_df)`` for each rolling step."""
        df = self._load()
        cfg = self.config
        start_t = cfg.warmup_bars + cfg.window_bars
        for t in range(start_t, len(df), cfg.step_bars):
            window = df.iloc[t - cfg.window_bars : t].copy()
            yield t, window

    def full_series(self) -> pd.DataFrame:
        """Entire OHLCV series (for full-sample parameter search)."""
        return self._load().copy()


def stream_from_manager(
    manager,
    symbol: str,
    timeframe: str,
    *,
    config: Optional[StreamConfig] = None,
    days: Optional[int] = None,
) -> OhlcvBarStream:
    """Build a bar stream from cached data (fetch if missing)."""
    with timed_step(f"stream.from_manager {symbol} {timeframe}") as d:
        manager.get_ohlcv(symbol, timeframe, days=days)
        path = manager.cache.get_path(symbol, timeframe)
        if path is None:
            raise FileNotFoundError(f"No cache for {symbol} {timeframe}")
        d["path"] = path.name
        return OhlcvBarStream(path, config=config, store=manager.duckdb)
