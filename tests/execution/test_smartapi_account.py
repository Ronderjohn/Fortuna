"""SmartAPIAccountReader: parsing real-shaped SmartConnect responses.

Uses a duck-typed ``FakeSmartConnect`` so the SmartApi SDK is not required
to test the parser logic.
"""

from __future__ import annotations

import pytest

from fortuna.execution.smartapi_account import (
    AccountSnapshot,
    SmartAPIAccountReader,
)
from fortuna.execution.types import Side


# --------------------------------------------------------------- fake client
class FakeSmartConnect:
    """Minimum surface a SmartAPIAccountReader needs to operate."""

    def __init__(
        self,
        *,
        rms=None,
        positions=None,
        holdings=None,
        order_book=None,
        trade_book=None,
        raise_on=None,
    ):
        self._rms = rms or {"status": True, "data": {}}
        self._positions = positions or {"status": True, "data": []}
        self._holdings = holdings or {"status": True, "data": []}
        self._order_book = order_book or {"status": True, "data": []}
        self._trade_book = trade_book or {"status": True, "data": []}
        self._raise_on = set(raise_on or [])

    def rmsLimit(self):
        if "rmsLimit" in self._raise_on:
            raise RuntimeError("simulated rmsLimit failure")
        return self._rms

    def position(self):
        if "position" in self._raise_on:
            raise RuntimeError("simulated position failure")
        return self._positions

    def holding(self):
        if "holding" in self._raise_on:
            raise RuntimeError("simulated holding failure")
        return self._holdings

    def orderBook(self):
        if "orderBook" in self._raise_on:
            raise RuntimeError("simulated orderBook failure")
        return self._order_book

    def tradeBook(self):
        if "tradeBook" in self._raise_on:
            raise RuntimeError("simulated tradeBook failure")
        return self._trade_book


# ============================================================ funds
def test_funds_parsed_from_rms_envelope():
    fake = FakeSmartConnect(
        rms={
            "status": True,
            "data": {
                "net": "152345.67",
                "availablecash": "100000.00",
                "utilisedDebits": "12345.00",
                "collateral": "50000.00",
                "m2munrealized": "345.67",
            },
        }
    )
    reader = SmartAPIAccountReader(client=fake)
    funds = reader.snapshot_funds()
    assert funds.net == pytest.approx(152345.67)
    assert funds.available_cash == pytest.approx(100000.0)
    assert funds.used_margin == pytest.approx(12345.0)
    assert funds.collateral == pytest.approx(50000.0)
    assert funds.m2m == pytest.approx(345.67)
    assert funds.raw  # raw envelope preserved for debugging


def test_funds_tolerant_of_missing_keys():
    fake = FakeSmartConnect(rms={"status": True, "data": {}})
    funds = SmartAPIAccountReader(client=fake).snapshot_funds()
    assert funds.net == 0.0
    assert funds.available_cash == 0.0


# ============================================================ positions
def test_positions_parsed_with_sign_mapping():
    fake = FakeSmartConnect(
        positions={
            "status": True,
            "data": [
                {
                    "tradingsymbol": "RELIANCE-EQ",
                    "symboltoken": "2885",
                    "exchange": "NSE",
                    "producttype": "INTRADAY",
                    "netqty": "10",
                    "avgnetprice": "2000.50",
                    "ltp": "2010.75",
                    "unrealised": "102.50",
                    "realised": "0.0",
                },
                {
                    "tradingsymbol": "HDFCBANK-EQ",
                    "symboltoken": "1333",
                    "exchange": "NSE",
                    "producttype": "INTRADAY",
                    "netqty": "-5",  # short
                    "avgnetprice": "1500.0",
                    "ltp": "1490.0",
                },
                {
                    # closed position — should be skipped
                    "tradingsymbol": "TCS-EQ",
                    "symboltoken": "11536",
                    "exchange": "NSE",
                    "producttype": "INTRADAY",
                    "netqty": "0",
                    "avgnetprice": "3500.0",
                },
            ],
        }
    )
    reader = SmartAPIAccountReader(client=fake)
    out = reader.snapshot_positions()
    assert len(out) == 2

    rel = next(p for p in out if p.tradingsymbol == "RELIANCE-EQ")
    assert rel.side is Side.LONG
    assert rel.quantity == 10
    assert rel.avg_price == pytest.approx(2000.50)
    assert rel.last_price == pytest.approx(2010.75)
    assert rel.signed_qty == 10

    hdfc = next(p for p in out if p.tradingsymbol == "HDFCBANK-EQ")
    assert hdfc.side is Side.SHORT
    assert hdfc.quantity == 5
    assert hdfc.signed_qty == -5


def test_positions_handles_empty_list():
    fake = FakeSmartConnect(positions={"status": True, "data": []})
    assert SmartAPIAccountReader(client=fake).snapshot_positions() == []


def test_positions_handles_null_data():
    fake = FakeSmartConnect(positions={"status": True, "data": None})
    assert SmartAPIAccountReader(client=fake).snapshot_positions() == []


def test_positions_carryforward_avg_from_cfbuyavgprice():
    """NRML / carry-forward rows often only populate cfbuy* fields, not avgnetprice."""
    fake = FakeSmartConnect(
        positions={
            "status": True,
            "data": [
                {
                    "tradingsymbol": "CROMPTON25JUNFUT",
                    "symboltoken": "123",
                    "exchange": "NFO",
                    "producttype": "CARRYFORWARD",
                    "netqty": "1800",
                    "avgnetprice": "0",
                    "netavgprice": "0",
                    "buyavgprice": "0",
                    "cfbuyavgprice": "285.50",
                    "cfbuyqty": "1800",
                    "ltp": "291.40",
                    "unrealised": "10620.0",
                }
            ],
        }
    )
    out = SmartAPIAccountReader(client=fake).snapshot_positions()
    assert len(out) == 1
    p = out[0]
    assert p.avg_price == pytest.approx(285.50)
    assert p.quantity == 1800
    assert p.last_price == pytest.approx(291.40)


# ============================================================ holdings
def test_holdings_parsed_with_skip_on_zero_qty():
    fake = FakeSmartConnect(
        holdings={
            "status": True,
            "data": [
                {
                    "tradingsymbol": "ITC-EQ",
                    "symboltoken": "1660",
                    "exchange": "NSE",
                    "quantity": "100",
                    "averageprice": "400.0",
                    "ltp": "415.0",
                    "profitandloss": "1500.0",
                },
                {
                    # phantom zero-qty entry — skip
                    "tradingsymbol": "ZZZ-EQ",
                    "symboltoken": "0",
                    "exchange": "NSE",
                    "quantity": "0",
                    "averageprice": "0.0",
                },
            ],
        }
    )
    out = SmartAPIAccountReader(client=fake).snapshot_holdings()
    assert len(out) == 1
    h = out[0]
    assert h.tradingsymbol == "ITC-EQ"
    assert h.quantity == 100
    assert h.avg_price == pytest.approx(400.0)
    assert h.last_price == pytest.approx(415.0)
    assert h.pnl == pytest.approx(1500.0)
    assert h.market_value == pytest.approx(41500.0)


# ============================================================ orders & trades
def test_orders_parsed():
    fake = FakeSmartConnect(
        order_book={
            "status": True,
            "data": [
                {
                    "orderid": "ORD123",
                    "uniqueorderid": "U-1",
                    "tradingsymbol": "RELIANCE-EQ",
                    "symboltoken": "2885",
                    "exchange": "NSE",
                    "transactiontype": "BUY",
                    "ordertype": "MARKET",
                    "producttype": "INTRADAY",
                    "quantity": "10",
                    "filledshares": "10",
                    "price": "0",
                    "averageprice": "2001.5",
                    "status": "complete",
                    "updatetime": "26-May-2026 09:30:15",
                    "text": "",
                }
            ],
        }
    )
    out = SmartAPIAccountReader(client=fake).snapshot_orders()
    assert len(out) == 1
    o = out[0]
    assert o.order_id == "ORD123"
    assert o.transaction_type == "BUY"
    assert o.quantity == 10
    assert o.filled_qty == 10
    assert o.average_price == pytest.approx(2001.5)
    assert o.status == "complete"


def test_trades_parsed_with_timestamp():
    fake = FakeSmartConnect(
        trade_book={
            "status": True,
            "data": [
                {
                    "orderid": "ORD123",
                    "tradingsymbol": "RELIANCE-EQ",
                    "symboltoken": "2885",
                    "exchange": "NSE",
                    "transactiontype": "BUY",
                    "fillsize": "10",
                    "fillprice": "2001.5",
                    "filltime": "26-May-2026 09:30:15",
                    "producttype": "INTRADAY",
                }
            ],
        }
    )
    out = SmartAPIAccountReader(client=fake).snapshot_trades()
    assert len(out) == 1
    t = out[0]
    assert t.order_id == "ORD123"
    assert t.transaction_type == "BUY"
    assert t.quantity == 10
    assert t.fill_price == pytest.approx(2001.5)
    assert t.fill_time is not None
    assert t.fill_time.year == 2026 and t.fill_time.month == 5


# ============================================================ full_snapshot
def test_full_snapshot_degrades_gracefully_per_endpoint():
    fake = FakeSmartConnect(
        rms={"status": True, "data": {"net": 10000.0, "availablecash": 9000.0}},
        positions={"status": True, "data": []},
        raise_on=["holding", "tradeBook"],
    )
    snap = SmartAPIAccountReader(client=fake).full_snapshot()
    assert isinstance(snap, AccountSnapshot)
    assert snap.funds.net == 10000.0
    assert snap.positions == []
    assert snap.holdings == []
    assert snap.trades == []
    assert any("holding" in e for e in snap.errors)
    assert any("trades" in e for e in snap.errors)
    assert snap.has_errors()


# ============================================================ constructor guards
def test_constructor_requires_session_or_client():
    with pytest.raises(ValueError, match="session= or client="):
        SmartAPIAccountReader()


def test_constructor_rejects_both():
    fake = FakeSmartConnect()
    with pytest.raises(ValueError, match="session= OR client="):
        SmartAPIAccountReader(session=object(), client=fake)
