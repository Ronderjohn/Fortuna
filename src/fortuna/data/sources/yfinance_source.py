"""yfinance-backed data source."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from fortuna.data.fetcher import DataFetcher
from fortuna.data.sources.symbols import normalize_for_yfinance


class YFinanceSource:
    """Fetch OHLCV via existing DataFetcher."""

    def __init__(self, fetcher: Optional[DataFetcher] = None) -> None:
        self._fetcher = fetcher or DataFetcher()

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        sym = normalize_for_yfinance(symbol)
        if days and not start:
            start_dt = datetime.now() - timedelta(days=days)
            start = start_dt.strftime("%Y-%m-%d")
        return self._fetcher.fetch(sym, timeframe, start=start, end=end)
