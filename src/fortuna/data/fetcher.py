"""Fetch OHLCV data from yfinance."""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from fortuna.utils.logging import get_logger
from fortuna.utils.timing import timed_step

logger = get_logger(__name__)

# yfinance interval mapping
TIMEFRAME_MAP = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "60m": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

PERIOD_MAP = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "60m": "730d",
    "1d": "10y",
    "1wk": "10y",
    "1mo": "max",
}


class DataFetcher:
    """Download and normalize OHLCV bars from yfinance."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol."""
        interval = TIMEFRAME_MAP.get(timeframe, timeframe)
        if interval not in TIMEFRAME_MAP.values():
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if period is None and start is None:
            period = PERIOD_MAP.get(interval, "2y")

        import yfinance as yf

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                with timed_step(f"yfinance.fetch {symbol} {interval}") as d:
                    d["attempt"] = str(attempt)
                    if start or end:
                        df = ticker.history(
                            start=start,
                            end=end,
                            interval=interval,
                            auto_adjust=True,
                        )
                    else:
                        df = ticker.history(
                            period=period,
                            interval=interval,
                            auto_adjust=True,
                        )
                    normalized = self._normalize(df, symbol)
                    d["rows"] = str(len(normalized))
                    return normalized
            except Exception as e:
                last_error = e
                logger.warning(
                    "Fetch attempt %s/%s failed for %s: %s",
                    attempt,
                    self.max_retries,
                    symbol,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"Failed to fetch {symbol} after {self.max_retries} attempts") from last_error

    def _normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize columns to lowercase OHLCV with DatetimeIndex."""
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]

        rename_map = {
            "adj close": "adj_close",
        }
        df = df.rename(columns=rename_map)

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} for {symbol}")

        df = df[required].copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.index.name = "datetime"
        df["symbol"] = symbol
        df = df.dropna(subset=["open", "high", "low", "close"])
        logger.info("Fetched %s rows for %s", len(df), symbol)
        return df
