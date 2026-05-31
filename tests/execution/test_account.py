"""LiveAccount: P&L math, MTM, drawdown, multiple positions."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fortuna.execution.account import LiveAccount
from fortuna.execution.types import Side


@pytest.fixture
def base_ts() -> datetime:
    return datetime(2026, 1, 6, 9, 30)


def test_round_trip_long_realizes_correct_pnl(base_ts):
    acct = LiveAccount(init_cash=100_000.0)

    acct.open_or_extend("RELIANCE", Side.LONG, qty=10, fill_price=2_000.0, ts=base_ts, tag="orb")
    acct.mark_to_market("RELIANCE", 2_010.0)
    assert pytest.approx(acct.unrealized_pnl(), rel=1e-6) == 100.0  # 10 * (2010 - 2000)

    trade = acct.close("RELIANCE", qty=10, fill_price=2_010.0, ts=base_ts + timedelta(hours=2), cost=2.0)
    assert trade is not None
    assert trade.gross_pnl == pytest.approx(100.0)
    assert trade.net_pnl == pytest.approx(98.0)
    assert acct.realized_pnl == pytest.approx(98.0)
    assert acct.cash == pytest.approx(100_098.0)
    assert "RELIANCE" not in acct.positions


def test_short_position_pnl_inverts(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("HDFC", Side.SHORT, qty=5, fill_price=1_500.0, ts=base_ts)

    acct.mark_to_market("HDFC", 1_490.0)
    assert acct.unrealized_pnl() == pytest.approx(50.0)  # short profit when price drops

    trade = acct.close("HDFC", qty=5, fill_price=1_490.0, ts=base_ts + timedelta(hours=1), cost=1.0)
    assert trade is not None
    assert trade.gross_pnl == pytest.approx(50.0)
    assert trade.net_pnl == pytest.approx(49.0)
    assert acct.realized_pnl == pytest.approx(49.0)


def test_extending_same_side_averages_price(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("INFY", Side.LONG, qty=10, fill_price=1_000.0, ts=base_ts)
    acct.open_or_extend("INFY", Side.LONG, qty=10, fill_price=1_100.0, ts=base_ts + timedelta(minutes=5))
    pos = acct.positions["INFY"]
    assert pos.qty == 20
    assert pos.avg_price == pytest.approx(1_050.0)  # (10*1000 + 10*1100) / 20


def test_opposite_side_open_raises(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("TCS", Side.LONG, qty=5, fill_price=3_000.0, ts=base_ts)
    with pytest.raises(ValueError, match="close first"):
        acct.open_or_extend("TCS", Side.SHORT, qty=5, fill_price=3_000.0, ts=base_ts)


def test_partial_close_keeps_remainder(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("WIPRO", Side.LONG, qty=20, fill_price=500.0, ts=base_ts)
    trade = acct.close("WIPRO", qty=10, fill_price=510.0, ts=base_ts + timedelta(hours=1), cost=1.0)
    assert trade is not None
    assert trade.qty == 10
    assert "WIPRO" in acct.positions
    remaining = acct.positions["WIPRO"]
    assert remaining.qty == 10
    assert remaining.avg_price == pytest.approx(500.0)


def test_concurrent_positions_aggregate_in_equity(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE", Side.LONG, qty=10, fill_price=2_000.0, ts=base_ts)
    acct.open_or_extend("HDFC", Side.SHORT, qty=5, fill_price=1_500.0, ts=base_ts)

    acct.mark_to_market("RELIANCE", 2_010.0)
    acct.mark_to_market("HDFC", 1_490.0)

    assert acct.unrealized_pnl() == pytest.approx(100.0 + 50.0)
    assert acct.equity == pytest.approx(100_000.0 + 150.0)
    assert acct.gross_exposure == pytest.approx(2_010.0 * 10 + 1_490.0 * 5)


def test_snapshot_appends_equity_point(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE", Side.LONG, qty=10, fill_price=2_000.0, ts=base_ts)
    acct.mark_to_market("RELIANCE", 2_050.0)
    point = acct.snapshot(base_ts + timedelta(minutes=5))
    assert point.unrealized == pytest.approx(500.0)
    assert point.equity == pytest.approx(100_500.0)
    assert len(acct.equity_curve) == 1


def test_drawdown_tracks_peak(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE", Side.LONG, qty=10, fill_price=2_000.0, ts=base_ts)

    acct.mark_to_market("RELIANCE", 2_200.0)
    acct.snapshot(base_ts + timedelta(minutes=5))
    assert acct.peak_equity == pytest.approx(102_000.0)

    acct.mark_to_market("RELIANCE", 2_100.0)
    acct.snapshot(base_ts + timedelta(minutes=10))
    # Equity = 101_000; peak = 102_000 → DD ≈ 0.98%
    assert acct.drawdown_pct == pytest.approx(0.98039, abs=1e-3)


def test_close_with_no_position_returns_none(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    assert acct.close("UNKNOWN", qty=1, fill_price=10.0, ts=base_ts) is None


def test_to_summary_shapes_match(base_ts):
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE", Side.LONG, qty=10, fill_price=2_000.0, ts=base_ts)
    acct.close("RELIANCE", qty=10, fill_price=2_020.0, ts=base_ts + timedelta(minutes=5), cost=1.0)
    s = acct.to_summary()
    assert {"init_cash", "equity", "realized_pnl", "total_trades", "win_rate_pct"} <= s.keys()
    assert s["total_trades"] == 1
    assert s["wins"] == 1
