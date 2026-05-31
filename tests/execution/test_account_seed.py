"""LiveAccount.seed_from_broker + reconcile_with_broker."""

from __future__ import annotations

from datetime import datetime

from fortuna.execution.account import LiveAccount
from fortuna.execution.smartapi_account import (
    AccountSnapshot,
    BrokerHolding,
    BrokerPosition,
    FundsSnapshot,
)
from fortuna.execution.types import Side


def _make_snapshot(
    *,
    cash=100_000.0,
    positions=None,
    holdings=None,
    ts=None,
):
    return AccountSnapshot(
        funds=FundsSnapshot(
            net=cash + 5_000.0,
            available_cash=cash,
            available_margin=cash,
            used_margin=0.0,
            raw={"availablecash": cash},
        ),
        positions=list(positions or []),
        holdings=list(holdings or []),
        orders=[],
        trades=[],
        taken_at=ts or datetime(2026, 5, 26, 9, 15),
    )


def test_seed_sets_init_cash_from_funds():
    acct = LiveAccount(init_cash=50_000.0)
    snap = _make_snapshot(cash=87_654.32)
    applied = acct.seed_from_broker(snap)
    assert acct.init_cash == 87_654.32
    assert applied["init_cash_before"] == 50_000.0
    assert applied["init_cash_after"] == 87_654.32
    assert acct.broker_synced_at == snap.taken_at


def test_seed_skips_init_cash_when_reset_false():
    acct = LiveAccount(init_cash=50_000.0)
    snap = _make_snapshot(cash=87_654.32)
    acct.seed_from_broker(snap, reset_init_cash=False)
    assert acct.init_cash == 50_000.0


def test_seed_inserts_positions_with_broker_tag():
    acct = LiveAccount(init_cash=100_000.0)
    snap = _make_snapshot(
        positions=[
            BrokerPosition(
                tradingsymbol="RELIANCE-EQ",
                symboltoken="2885",
                exchange="NSE",
                producttype="INTRADAY",
                side=Side.LONG,
                quantity=10,
                avg_price=2_000.0,
                last_price=2_010.0,
                fortuna_symbol="RELIANCE.NS",
            ),
            BrokerPosition(
                tradingsymbol="HDFCBANK-EQ",
                symboltoken="1333",
                exchange="NSE",
                producttype="INTRADAY",
                side=Side.SHORT,
                quantity=5,
                avg_price=1_500.0,
                fortuna_symbol="HDFCBANK.NS",
            ),
        ]
    )
    acct.seed_from_broker(snap)
    assert "RELIANCE.NS" in acct.positions
    assert "HDFCBANK.NS" in acct.positions
    rel = acct.positions["RELIANCE.NS"]
    assert rel.qty == 10
    assert rel.side is Side.LONG
    assert rel.entry_tag is not None and rel.entry_tag.startswith("broker")
    assert rel.last_mtm_price == 2_010.0


def test_seed_inserts_holdings_separate_from_positions():
    acct = LiveAccount(init_cash=100_000.0)
    snap = _make_snapshot(
        holdings=[
            BrokerHolding(
                tradingsymbol="ITC-EQ",
                symboltoken="1660",
                exchange="NSE",
                quantity=100,
                avg_price=400.0,
                last_price=415.0,
                fortuna_symbol="ITC.NS",
                pnl=1500.0,
            )
        ]
    )
    acct.seed_from_broker(snap)
    assert acct.has_holding("ITC.NS")
    h = acct.get_holding("ITC.NS")
    assert h.qty == 100
    assert h.avg_price == 400.0
    assert h.last_price == 415.0
    assert h.source == "broker"


def test_seed_default_replaces_paper_positions():
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend(
        "FAKE.NS", Side.LONG, qty=5, fill_price=100.0, ts=datetime.now(), tag="paper",
    )
    snap = _make_snapshot(
        positions=[
            BrokerPosition(
                tradingsymbol="RELIANCE-EQ",
                symboltoken="2885",
                exchange="NSE",
                producttype="INTRADAY",
                side=Side.LONG,
                quantity=10,
                avg_price=2_000.0,
                fortuna_symbol="RELIANCE.NS",
            )
        ]
    )
    acct.seed_from_broker(snap)
    assert "FAKE.NS" not in acct.positions  # default replaces
    assert "RELIANCE.NS" in acct.positions


def test_seed_merge_keeps_existing_paper():
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend(
        "FAKE.NS", Side.LONG, qty=5, fill_price=100.0, ts=datetime.now(), tag="paper",
    )
    snap = _make_snapshot(
        positions=[
            BrokerPosition(
                tradingsymbol="RELIANCE-EQ",
                symboltoken="2885",
                exchange="NSE",
                producttype="INTRADAY",
                side=Side.LONG,
                quantity=10,
                avg_price=2_000.0,
                fortuna_symbol="RELIANCE.NS",
            )
        ]
    )
    acct.seed_from_broker(snap, merge=True)
    assert "FAKE.NS" in acct.positions
    assert "RELIANCE.NS" in acct.positions


def test_held_or_open_covers_positions_and_holdings():
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE.NS", Side.LONG, qty=10, fill_price=2_000.0, ts=datetime.now())
    snap = _make_snapshot(
        holdings=[
            BrokerHolding(
                tradingsymbol="ITC-EQ", symboltoken="1660", exchange="NSE",
                quantity=100, avg_price=400.0, fortuna_symbol="ITC.NS",
            )
        ]
    )
    # merge=True so the paper RELIANCE position survives alongside the
    # broker holding we're seeding.
    acct.seed_from_broker(snap, reset_init_cash=False, merge=True)
    assert acct.held_or_open("RELIANCE.NS")
    assert acct.held_or_open("ITC.NS")
    assert not acct.held_or_open("TCS.NS")


def test_reconcile_detects_drift():
    acct = LiveAccount(init_cash=100_000.0)
    # Local says we hold 10 RELIANCE, broker says 8 → qty mismatch.
    acct.open_or_extend(
        "RELIANCE.NS", Side.LONG, qty=10, fill_price=2_000.0,
        ts=datetime.now(), tag="broker:intraday",
    )
    # And local has a stale position broker doesn't carry.
    acct.open_or_extend(
        "STALE.NS", Side.LONG, qty=5, fill_price=100.0,
        ts=datetime.now(), tag="broker:intraday",
    )
    snap = _make_snapshot(
        positions=[
            BrokerPosition(
                tradingsymbol="RELIANCE-EQ", symboltoken="2885", exchange="NSE",
                producttype="INTRADAY", side=Side.LONG, quantity=8, avg_price=2_000.0,
                fortuna_symbol="RELIANCE.NS",
            ),
            BrokerPosition(
                tradingsymbol="NEW-EQ", symboltoken="999", exchange="NSE",
                producttype="INTRADAY", side=Side.LONG, quantity=3, avg_price=1_000.0,
                fortuna_symbol="NEW.NS",
            ),
        ]
    )
    drift = acct.reconcile_with_broker(snap)
    assert "NEW.NS" in drift["missing_in_local"]
    assert "STALE.NS" in drift["missing_in_broker"]
    mismatch_symbols = [m["symbol"] for m in drift["qty_mismatch"]]
    assert "RELIANCE.NS" in mismatch_symbols
