"""Paper journal vs broker tradeBook reconciliation (Phase 2)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fortuna.execution.reconciliation import (
    paper_orders_from_journal,
    reconcile_paper_vs_real,
    render_reconciliation_markdown,
)
from fortuna.execution.smartapi_account import BrokerTrade
from fortuna.execution.types import Side


def _journal_fill(symbol, side, qty, price, ts, strategy="orb"):
    return {
        "ts": ts.isoformat(),
        "type": "order_filled",
        "ack": {
            "order_id": f"PAPER-{symbol}-{ts:%H%M%S}",
            "symbol": symbol,
            "side": side.value,
            "status": "FILLED",
            "qty": qty,
            "filled_qty": qty,
            "filled_price": price,
            "ts": ts.isoformat(),
            "tag": strategy,
        },
    }


def _real_trade(symbol, txn, qty, price, ts, fortuna_symbol=None):
    return BrokerTrade(
        order_id=f"REAL-{symbol}-{ts:%H%M%S}",
        tradingsymbol=symbol,
        symboltoken="X",
        exchange="NSE",
        transaction_type=txn,
        quantity=qty,
        fill_price=price,
        fill_time=ts,
        producttype="INTRADAY",
        fortuna_symbol=fortuna_symbol,
    )


# ============================================================ extraction
def test_paper_orders_from_journal_filters_to_fills_only():
    ts = datetime(2026, 1, 6, 9, 30)
    rows = [
        {"type": "order_placed", "ack": {}},
        _journal_fill("RELIANCE.NS", Side.LONG, 10, 2_000.5, ts),
        {"type": "order_rejected", "ack": {}},
    ]
    orders = paper_orders_from_journal(rows)
    assert len(orders) == 1
    o = orders[0]
    assert o.symbol == "RELIANCE.NS"
    assert o.side is Side.LONG
    assert o.qty == 10
    assert o.fill_price == pytest.approx(2_000.5)
    assert o.fill_ts == ts


def test_paper_orders_ignores_malformed_side():
    rows = [{"type": "order_filled", "ack": {"side": "UNKNOWN", "symbol": "X", "qty": 1}}]
    assert paper_orders_from_journal(rows) == []


# ============================================================ matching
def test_match_pairs_paper_and_real_within_window():
    ts = datetime(2026, 1, 6, 9, 30)
    journal = [_journal_fill("RELIANCE.NS", Side.LONG, 10, 2_000.0, ts)]
    real = [_real_trade("RELIANCE-EQ", "BUY", 10, 2_001.5, ts + timedelta(seconds=30),
                       fortuna_symbol="RELIANCE.NS")]
    report = reconcile_paper_vs_real(journal, real)
    assert len(report.matched) == 1
    m = report.matched[0]
    assert m.symbol == "RELIANCE.NS"
    assert m.side is Side.LONG
    assert m.qty == 10
    assert m.paper_price == 2_000.0
    assert m.real_price == 2_001.5
    assert m.slippage_pct == pytest.approx(0.075, abs=1e-3)


def test_paper_only_when_no_real_match():
    ts = datetime(2026, 1, 6, 9, 30)
    journal = [_journal_fill("RELIANCE.NS", Side.LONG, 10, 2_000.0, ts)]
    report = reconcile_paper_vs_real(journal, [])
    assert len(report.paper_only) == 1
    assert report.matched == []
    assert report.coverage_pct == 100.0  # no real fills to cover
    assert report.precision_pct == 0.0


def test_real_only_when_paper_missing():
    ts = datetime(2026, 1, 6, 9, 30)
    real = [_real_trade("RELIANCE-EQ", "BUY", 10, 2_000.0, ts, fortuna_symbol="RELIANCE.NS")]
    report = reconcile_paper_vs_real([], real)
    assert len(report.real_only) == 1
    assert report.matched == []
    assert report.coverage_pct == 0.0


def test_time_window_rejects_distant_matches():
    ts = datetime(2026, 1, 6, 9, 30)
    journal = [_journal_fill("RELIANCE.NS", Side.LONG, 10, 2_000.0, ts)]
    real = [_real_trade(
        "RELIANCE-EQ", "BUY", 10, 2_000.0, ts + timedelta(hours=2),
        fortuna_symbol="RELIANCE.NS",
    )]
    report = reconcile_paper_vs_real(journal, real, time_window=timedelta(minutes=15))
    assert report.matched == []
    assert len(report.paper_only) == 1
    assert len(report.real_only) == 1


def test_short_side_maps_from_sell_txn():
    ts = datetime(2026, 1, 6, 9, 30)
    journal = [_journal_fill("HDFCBANK.NS", Side.SHORT, 5, 1_500.0, ts)]
    real = [_real_trade(
        "HDFCBANK-EQ", "SELL", 5, 1_499.5, ts + timedelta(seconds=10),
        fortuna_symbol="HDFCBANK.NS",
    )]
    report = reconcile_paper_vs_real(journal, real)
    assert len(report.matched) == 1
    assert report.matched[0].side is Side.SHORT


def test_metrics_aggregate_correctly():
    ts = datetime(2026, 1, 6, 9, 30)
    journal = [
        _journal_fill("RELIANCE.NS", Side.LONG, 10, 2_000.0, ts),
        _journal_fill("ITC.NS", Side.LONG, 50, 400.0, ts + timedelta(minutes=5)),
    ]
    real = [
        _real_trade("RELIANCE-EQ", "BUY", 10, 2_002.0, ts + timedelta(seconds=10),
                   fortuna_symbol="RELIANCE.NS"),
        # No real trade for ITC → paper_only
        # Plus a manual trade with no paper counterpart → real_only
        _real_trade("WIPRO-EQ", "BUY", 20, 500.0, ts + timedelta(minutes=10),
                   fortuna_symbol="WIPRO.NS"),
    ]
    report = reconcile_paper_vs_real(journal, real)
    assert report.paper_fills == 2
    assert report.real_fills == 2
    assert len(report.matched) == 1
    assert len(report.paper_only) == 1  # ITC
    assert len(report.real_only) == 1   # WIPRO
    summary = report.to_dict()
    assert summary["coverage_pct"] == 50.0  # 1 of 2 real
    assert summary["precision_pct"] == 50.0  # 1 of 2 paper


# ============================================================ markdown render
def test_markdown_includes_all_sections(tmp_path):
    ts = datetime(2026, 1, 6, 9, 30)
    journal = [_journal_fill("RELIANCE.NS", Side.LONG, 10, 2_000.0, ts, strategy="orb")]
    real = [
        _real_trade("RELIANCE-EQ", "BUY", 10, 2_001.0, ts + timedelta(seconds=15),
                   fortuna_symbol="RELIANCE.NS"),
        _real_trade("WIPRO-EQ", "BUY", 20, 500.0, ts + timedelta(minutes=10),
                   fortuna_symbol="WIPRO.NS"),
    ]
    journal.append(_journal_fill("ITC.NS", Side.LONG, 100, 400.0, ts + timedelta(minutes=5)))

    report = reconcile_paper_vs_real(journal, real)
    md = render_reconciliation_markdown(report)
    assert "## Reconciliation (paper vs broker)" in md
    assert "Matched" in md and "RELIANCE.NS" in md
    assert "Paper-only" in md and "ITC.NS" in md
    assert "Real-only" in md and "WIPRO" in md


def test_append_to_eod(tmp_path):
    from fortuna.execution.reconciliation import append_reconciliation_to_eod

    eod = tmp_path / "2026-01-06.md"
    eod.write_text("# original EOD\n", encoding="utf-8")

    ts = datetime(2026, 1, 6, 9, 30)
    journal = [_journal_fill("RELIANCE.NS", Side.LONG, 10, 2_000.0, ts)]
    real = [_real_trade("RELIANCE-EQ", "BUY", 10, 2_000.5, ts + timedelta(seconds=5),
                       fortuna_symbol="RELIANCE.NS")]
    report = reconcile_paper_vs_real(journal, real)
    out = append_reconciliation_to_eod(eod, report)
    assert out == eod
    content = out.read_text(encoding="utf-8")
    assert "# original EOD" in content
    assert "## Reconciliation (paper vs broker)" in content
