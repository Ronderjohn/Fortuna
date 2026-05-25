"""DuckDB analytics over cached Parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from fortuna.utils.logging import get_logger
from fortuna.utils.timing import timed_step

logger = get_logger(__name__)


class DuckDBStore:
    """Query OHLCV Parquet cache with DuckDB."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def register_parquet(self, symbol: str, timeframe: str, parquet_path: Path) -> str:
        """Register a parquet file as a DuckDB view."""
        view_name = f"ohlcv_{symbol.replace('.', '_').lower()}_{timeframe.replace(' ', '_')}"
        path_str = str(parquet_path).replace("\\", "/")
        self._conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{path_str}')")
        logger.debug("Registered DuckDB view: %s", view_name)
        return view_name

    def query_ohlcv(
        self,
        parquet_path: Path,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query OHLCV from a parquet file with optional date filters."""
        path_str = str(parquet_path).replace("\\", "/")
        sql = f"SELECT * FROM read_parquet('{path_str}') WHERE 1=1"
        if start:
            sql += f" AND datetime >= '{start}'"
        if end:
            sql += f" AND datetime <= '{end}'"
        sql += " ORDER BY datetime"
        with timed_step("data.duckdb_read_parquet") as d:
            df = self._conn.execute(sql).df()
            d["rows"] = str(len(df))
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        if keep:
            df = df[keep]
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        elif "symbol" in df.columns:
            df = df.drop(columns=["symbol"], errors="ignore")
        df.index = pd.to_datetime(df.index)
        return df

    def close(self) -> None:
        self._conn.close()
