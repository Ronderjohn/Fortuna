"""Instrument registry unit tests (no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from fortuna.config import smartapi_settings
from fortuna.data.instruments import (
    NSE_CM_EXCHANGE_TYPE,
    NSE_FO_EXCHANGE_TYPE,
    InstrumentRegistry,
)


@pytest.fixture
def scrip_fixture(tmp_path: Path) -> Path:
    data = [
        {
            "token": "2885",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "exch_seg": "NSE",
            "instrumenttype": "EQ",
        },
        {
            "token": "999",
            "symbol": "FOO-EQ",
            "name": "FOO",
            "exch_seg": "BSE",
            "instrumenttype": "EQ",
        },
        {
            "token": "25441",
            "symbol": "CROMPTON-EQ",
            "name": "CROMPTON",
            "exch_seg": "NSE",
            "instrumenttype": "EQ",
        },
        # CROMPTON futures chain: a contract already expired in the past,
        # the front-month, the next-month, and a far-month contract.
        {
            "token": "70001",
            "symbol": "CROMPTON29APR26FUT",
            "name": "CROMPTON",
            "expiry": "29APR2026",
            "lotsize": "1800",
            "instrumenttype": "FUTSTK",
            "exch_seg": "NFO",
        },
        {
            "token": "66136",
            "symbol": "CROMPTON26MAY26FUT",
            "name": "CROMPTON",
            "expiry": "26MAY2026",
            "lotsize": "1800",
            "instrumenttype": "FUTSTK",
            "exch_seg": "NFO",
        },
        {
            "token": "62508",
            "symbol": "CROMPTON30JUN26FUT",
            "name": "CROMPTON",
            "expiry": "30JUN2026",
            "lotsize": "1800",
            "instrumenttype": "FUTSTK",
            "exch_seg": "NFO",
        },
        {
            "token": "61150",
            "symbol": "CROMPTON28JUL26FUT",
            "name": "CROMPTON",
            "expiry": "28JUL2026",
            "lotsize": "2150",
            "instrumenttype": "FUTSTK",
            "exch_seg": "NFO",
        },
        # Index future (FUTIDX) — should also be indexed.
        {
            "token": "88001",
            "symbol": "NIFTY28MAY26FUT",
            "name": "NIFTY",
            "expiry": "28MAY2026",
            "lotsize": "75",
            "instrumenttype": "FUTIDX",
            "exch_seg": "NFO",
        },
        # Option row — must be skipped.
        {
            "token": "99001",
            "symbol": "CROMPTON26MAY26P400",
            "name": "CROMPTON",
            "expiry": "26MAY2026",
            "strike": "40000.000000",
            "lotsize": "1800",
            "instrumenttype": "OPTSTK",
            "exch_seg": "NFO",
        },
    ]
    path = tmp_path / "OpenAPIScripMaster.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _registry(scrip_fixture: Path) -> InstrumentRegistry:
    settings = smartapi_settings.SmartAPISettings(scrip_cache_path=scrip_fixture)
    reg = InstrumentRegistry(settings)
    reg.ensure_loaded()
    return reg


def test_resolve_reliance_ns(scrip_fixture: Path) -> None:
    reg = _registry(scrip_fixture)
    ref = reg.resolve("RELIANCE.NS")
    assert ref.symboltoken == "2885"
    assert ref.exchange == "NSE"
    assert ref.tradingsymbol == "RELIANCE-EQ"
    assert ref.exchange_type == NSE_CM_EXCHANGE_TYPE
    assert ref.is_future is False


def test_registry_indexes_nfo_futures_and_skips_options(scrip_fixture: Path) -> None:
    reg = _registry(scrip_fixture)
    # 3 EQ in NSE (RELIANCE, CROMPTON; FOO is BSE so skipped).
    assert reg.equity_count == 2
    # CROMPTON + NIFTY have futures chains.
    assert reg.futures_base_count == 2


def test_resolve_future_picks_front_month(
    scrip_fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CROMPTON.FUT`` must resolve to the nearest non-expired contract.

    Freezing ``date.today()`` to early May 2026: the 29-Apr contract is in
    the past, so the picker should return the 26-May front-month.
    """
    reg = _registry(scrip_fixture)

    class _FrozenDate(date):
        @classmethod
        def today(cls) -> date:  # type: ignore[override]
            return date(2026, 5, 1)

    monkeypatch.setattr("fortuna.data.instruments.date", _FrozenDate)

    ref = reg.resolve("CROMPTON.FUT")
    assert ref.exchange == "NFO"
    assert ref.exchange_type == NSE_FO_EXCHANGE_TYPE
    assert ref.instrumenttype == "FUTSTK"
    assert ref.symboltoken == "66136"
    assert ref.tradingsymbol == "CROMPTON26MAY26FUT"
    assert ref.expiry == date(2026, 5, 26)
    assert ref.lot_size == 1800
    assert ref.is_future is True


def test_resolve_future_with_explicit_expiry(scrip_fixture: Path) -> None:
    reg = _registry(scrip_fixture)
    ref = reg.resolve("CROMPTON.FUT.30JUN2026")
    assert ref.symboltoken == "62508"
    assert ref.tradingsymbol == "CROMPTON30JUN26FUT"
    assert ref.expiry == date(2026, 6, 30)


def test_resolve_future_by_angel_tradingsymbol(scrip_fixture: Path) -> None:
    reg = _registry(scrip_fixture)
    ref = reg.resolve("CROMPTON28JUL26FUT")
    assert ref.symboltoken == "61150"
    assert ref.expiry == date(2026, 7, 28)


def test_resolve_unknown_future_raises(scrip_fixture: Path) -> None:
    reg = _registry(scrip_fixture)
    with pytest.raises(KeyError):
        reg.resolve("NONEXISTENT.FUT")
    with pytest.raises(KeyError):
        reg.resolve("CROMPTON.FUT.01JAN1999")


def test_search_returns_both_equity_and_futures_for_same_base(
    scrip_fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Searching ``CROMPTON`` must yield BOTH the cash equity hit and the
    front-month future hit so the user can pick either one."""
    reg = _registry(scrip_fixture)

    class _FrozenDate(date):
        @classmethod
        def today(cls) -> date:  # type: ignore[override]
            return date(2026, 5, 1)

    monkeypatch.setattr("fortuna.data.instruments.date", _FrozenDate)

    hits = reg.search("CROMPTON", limit=10)
    by_symbol = {h.symbol: h for h in hits}
    assert "CROMPTON.NS" in by_symbol, f"missing equity hit; got {list(by_symbol)}"
    assert "CROMPTON.FUT" in by_symbol, f"missing futures hit; got {list(by_symbol)}"
    assert by_symbol["CROMPTON.NS"].segment == "EQUITY"
    assert by_symbol["CROMPTON.FUT"].segment == "FUTURES"
    # The futures display must surface the resolved expiry + tradingsymbol.
    assert "CROMPTON26MAY26FUT" in by_symbol["CROMPTON.FUT"].display
    assert "26-May-2026" in by_symbol["CROMPTON.FUT"].display


def test_catalog_includes_futures_chain_bases(scrip_fixture: Path) -> None:
    reg = _registry(scrip_fixture)
    hits = reg.catalog()
    segments = {h.segment for h in hits}
    assert segments == {"EQUITY", "FUTURES"}
    fut_symbols = {h.symbol for h in hits if h.segment == "FUTURES"}
    assert fut_symbols == {"CROMPTON.FUT", "NIFTY.FUT"}
