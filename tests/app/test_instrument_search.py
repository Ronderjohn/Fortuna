"""Instrument registry search tests."""

from __future__ import annotations

from fortuna.data.instruments import InstrumentRegistry, InstrumentRef, SymbolSearchHit


class _FakeRegistry(InstrumentRegistry):
    def __init__(self) -> None:
        self._loaded = True
        self._equity_bases = ["GAIL", "ICICIBANK", "RELIANCE", "TCS"]
        ref = InstrumentRef(
            symbol="RELIANCE",
            tradingsymbol="RELIANCE-EQ",
            symboltoken="2885",
            exchange="NSE",
        )
        self._by_key = {
            "RELIANCE": ref,
            "GAIL": InstrumentRef("GAIL", "GAIL-EQ", "1", "NSE"),
            "ICICIBANK": InstrumentRef("ICICIBANK", "ICICIBANK-EQ", "2", "NSE"),
            "TCS": InstrumentRef("TCS", "TCS-EQ", "3", "NSE"),
        }
        # Empty futures indexes so the dual-segment search still works.
        self._futures_bases = []
        self._futures_by_base = {}
        self._futures_by_symbol = {}


def test_search_prefix() -> None:
    reg = _FakeRegistry()
    hits = reg.search("ICI", limit=5)
    assert len(hits) == 1
    assert hits[0].symbol == "ICICIBANK.NS"


def test_search_empty_returns_popular() -> None:
    reg = _FakeRegistry()
    hits = reg.search("", limit=3)
    assert len(hits) <= 3
