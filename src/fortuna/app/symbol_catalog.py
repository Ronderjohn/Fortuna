"""Background-loaded NSE equity catalog for TradingView-style symbol search."""

from __future__ import annotations

import threading
from typing import Optional

import pandas as pd

from fortuna.data.instruments import InstrumentRegistry, SymbolSearchHit


class SymbolCatalog:
    """
    Thread-safe wrapper around InstrumentRegistry for UI autocomplete.

    Load once at app startup; search runs in-memory over all NSE cash equities.
    """

    def __init__(self, registry: Optional[InstrumentRegistry] = None) -> None:
        self._registry = registry or InstrumentRegistry()
        self._lock = threading.Lock()
        self._loaded = False
        self._hits: list[SymbolSearchHit] = []
        self._df: Optional[pd.DataFrame] = None

    def ensure_loaded(self, *, force_refresh: bool = False) -> int:
        with self._lock:
            if self._loaded and not force_refresh:
                return len(self._hits)
            if force_refresh:
                self._registry.ensure_loaded(force_refresh=True)
            else:
                self._registry.ensure_loaded()
            self._hits = self._registry.catalog()
            self._df = pd.DataFrame(
                [
                    {
                        "symbol": h.symbol,
                        "name": h.display.split(" — ")[0],
                        "tradingsymbol": h.tradingsymbol,
                        "exchange": "NSE",
                    }
                    for h in self._hits
                ]
            )
            self._loaded = True
            return len(self._hits)

    @property
    def count(self) -> int:
        return len(self._hits)

    @property
    def dataframe(self) -> pd.DataFrame:
        self.ensure_loaded()
        assert self._df is not None
        return self._df

    def search(self, query: str, *, limit: int = 50) -> list[SymbolSearchHit]:
        self.ensure_loaded()
        return self._registry.search(query, limit=limit)

    def resolve_display(self, symbol: str) -> str:
        self.ensure_loaded()
        try:
            ref = self._registry.resolve(symbol)
            return f"{ref.symbol} — NSE"
        except KeyError:
            return symbol
