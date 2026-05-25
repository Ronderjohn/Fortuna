"""Data source protocol."""

from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd


class DataSource(Protocol):
    """Fetch OHLCV bars for a symbol and timeframe."""

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        ...
