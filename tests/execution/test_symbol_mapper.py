"""Angel tradingsymbol → Fortuna canonical symbol mapping."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from fortuna.data.instruments import (
    NSE_CM_EXCHANGE_TYPE,
    NSE_FO_EXCHANGE_TYPE,
    InstrumentRef,
)
from fortuna.execution.smartapi_account import fortuna_symbol_for


def _eq(base):
    return InstrumentRef(
        symbol=base,
        tradingsymbol=f"{base}-EQ",
        symboltoken="X",
        exchange="NSE",
        exchange_type=NSE_CM_EXCHANGE_TYPE,
        instrumenttype="EQ",
        name=base,
    )


def _fut(base, expiry):
    return InstrumentRef(
        symbol=f"{base}.FUT",
        tradingsymbol=f"{base}25{expiry.strftime('%b').upper()}FUT",
        symboltoken="Y",
        exchange="NFO",
        exchange_type=NSE_FO_EXCHANGE_TYPE,
        instrumenttype="FUTSTK",
        expiry=expiry,
        lot_size=250,
        name=base,
    )


# ============================================================ equity
def test_equity_maps_to_ns():
    ref = _eq("RELIANCE")
    assert fortuna_symbol_for(ref) == "RELIANCE.NS"


def test_equity_does_not_need_registry():
    assert fortuna_symbol_for(_eq("HDFCBANK"), registry=None) == "HDFCBANK.NS"


# ============================================================ futures
def test_future_no_registry_uses_explicit_expiry():
    ref = _fut("RELIANCE", date(2025, 11, 27))
    # No registry → fall back to explicit-expiry form (safe to round-trip).
    assert fortuna_symbol_for(ref, registry=None) == "RELIANCE.FUT.27NOV2025"


def test_future_front_month_strips_expiry_with_registry():
    ref = _fut("RELIANCE", date(2025, 11, 27))
    registry = MagicMock()
    registry._front_future.return_value = ref
    assert fortuna_symbol_for(ref, registry=registry) == "RELIANCE.FUT"


def test_future_back_month_keeps_expiry_with_registry():
    front = _fut("RELIANCE", date(2025, 11, 27))
    back = _fut("RELIANCE", date(2025, 12, 25))
    registry = MagicMock()
    registry._front_future.return_value = front
    assert fortuna_symbol_for(back, registry=registry) == "RELIANCE.FUT.25DEC2025"


def test_future_with_no_expiry_falls_back_to_bare_fut():
    ref = InstrumentRef(
        symbol="ABC.FUT",
        tradingsymbol="ABC25NOVFUT",
        symboltoken="Z",
        exchange="NFO",
        exchange_type=NSE_FO_EXCHANGE_TYPE,
        instrumenttype="FUTSTK",
        expiry=None,
        name="ABC",
    )
    assert fortuna_symbol_for(ref, registry=None) == "ABC.FUT"
