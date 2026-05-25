"""Symbol catalog tests."""

from __future__ import annotations

from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.data.instruments import InstrumentRef, InstrumentRegistry


class _FakeRegistry(InstrumentRegistry):
    def __init__(self) -> None:
        self._loaded = True
        self._equity_bases = ["RELIANCE", "ICICIBANK", "TCS"]
        refs = [
            InstrumentRef("RELIANCE", "RELIANCE-EQ", "1", "NSE"),
            InstrumentRef("ICICIBANK", "ICICIBANK-EQ", "2", "NSE"),
            InstrumentRef("TCS", "TCS-EQ", "3", "NSE"),
        ]
        self._by_key = {r.symbol: r for r in refs}
        for r in refs:
            self._by_key[r.tradingsymbol] = r

    def catalog(self):
        return [self._hit_from_ref(self._by_key[b]) for b in self._equity_bases]


def test_catalog_load_count() -> None:
    cat = SymbolCatalog(_FakeRegistry())
    assert cat.ensure_loaded() == 3
    assert cat.count == 3
    assert len(cat.dataframe) == 3


def test_catalog_search() -> None:
    cat = SymbolCatalog(_FakeRegistry())
    hits = cat.search("REL", limit=10)
    assert len(hits) == 1
    assert hits[0].symbol == "RELIANCE.NS"
