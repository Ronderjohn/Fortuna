"""OpenChart NSE intraday data source (optional dependency)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from fortuna.data.sources.symbols import normalize_for_openchart
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "60m": "1h",
    "1d": "1d",
}


class OpenChartSource:
    """Fetch NSE OHLCV via openchart library."""

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        try:
            from openchart import NSEData
        except ImportError as e:
            raise ImportError(
                "openchart is required for OpenChartSource. Install with: uv sync --group nse"
            ) from e

        interval = _INTERVAL_MAP.get(timeframe, timeframe)
        nse = NSEData()
        sym = normalize_for_openchart(symbol)

        end_dt = datetime.now() if end is None else pd.Timestamp(end).to_pydatetime()
        if start:
            start_dt = pd.Timestamp(start).to_pydatetime()
        elif days:
            start_dt = end_dt - timedelta(days=days)
        else:
            start_dt = end_dt - timedelta(days=30)

        logger.info("OpenChart fetch %s %s %s -> %s", sym, interval, start_dt, end_dt)
        raw = nse.historical(sym, "EQ", start_dt, end_dt, interval)
        return self._normalize(raw, sym)

    @staticmethod
    def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        out = df.copy()
        out.columns = [str(c).lower() for c in out.columns]
        rename = {
            "timestamp": "datetime",
            "date": "datetime",
        }
        out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"])
            out = out.set_index("datetime")
        elif not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)

        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out.index.name = "datetime"

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise ValueError(f"OpenChart missing columns {missing}")

        out = out[required].astype(float)
        out["symbol"] = symbol
        return out.sort_index()
