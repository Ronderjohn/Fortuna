"""PositionSizer: fixed-fractional + kelly-capped math, lot rounding."""

from __future__ import annotations

from fortuna.execution.account import LiveAccount
from fortuna.execution.risk import PositionSizer, RiskConfig
from fortuna.execution.types import Side


def test_fixed_fractional_uses_mark_when_no_stop_distance():
    sizer = PositionSizer(RiskConfig(risk_per_trade_pct=1.0, init_cash=100_000.0))
    acct = LiveAccount(init_cash=100_000.0)
    qty = sizer.size("RELIANCE", Side.LONG, mark_price=2_000.0, account=acct)
    # 1% of 100k = 1000; 1000 / 2000 = 0 (integer); test should pick a non-trivial
    # mark — try a smaller price.
    assert qty == 0  # 1000 // 2000 == 0; sizer correctly says "too small"


def test_fixed_fractional_with_explicit_stop_distance():
    sizer = PositionSizer(RiskConfig(risk_per_trade_pct=1.0, init_cash=100_000.0))
    acct = LiveAccount(init_cash=100_000.0)
    qty = sizer.size(
        "RELIANCE", Side.LONG, mark_price=2_000.0, account=acct, stop_distance=20.0
    )
    assert qty == 50  # 1000 risk budget / 20 stop = 50 shares


def test_fixed_fractional_low_priced_share():
    """Cheap shares should yield non-zero quantity without explicit stop."""
    sizer = PositionSizer(RiskConfig(risk_per_trade_pct=1.0, init_cash=100_000.0))
    acct = LiveAccount(init_cash=100_000.0)
    qty = sizer.size("PENNY", Side.LONG, mark_price=10.0, account=acct)
    assert qty == 100  # 1000 budget / 10 price = 100


def test_lot_size_rounds_down_to_multiple():
    sizer = PositionSizer(
        RiskConfig(risk_per_trade_pct=10.0, init_cash=100_000.0, lot_sizes={"RELIANCE.FUT": 500})
    )
    acct = LiveAccount(init_cash=100_000.0)
    # Budget = 10_000; mark = 2_000 → raw qty = 5; lot = 500 → rounded to 0
    qty = sizer.size("RELIANCE.FUT", Side.LONG, mark_price=2_000.0, account=acct)
    assert qty == 0

    # With a larger budget the lot rounds DOWN to a multiple of 500.
    sizer_big = PositionSizer(
        RiskConfig(risk_per_trade_pct=100.0, init_cash=10_000_000.0, lot_sizes={"RELIANCE.FUT": 500})
    )
    acct_big = LiveAccount(init_cash=10_000_000.0)
    qty_big = sizer_big.size("RELIANCE.FUT", Side.LONG, mark_price=2_000.0, account=acct_big)
    # 10_000_000 budget / 2_000 = 5_000 → already a multiple of 500
    assert qty_big == 5_000
    assert qty_big % 500 == 0


def test_kelly_falls_back_to_fixed_when_too_few_trades():
    cfg = RiskConfig(position_size_method="kelly_capped", risk_per_trade_pct=1.0, init_cash=100_000.0)
    sizer = PositionSizer(cfg)
    acct = LiveAccount(init_cash=100_000.0)
    qty = sizer.size("PENNY", Side.LONG, mark_price=10.0, account=acct, recent_wins=2, recent_losses=1)
    assert qty == 100  # fallback to fixed-fractional


def test_kelly_capped_above_threshold():
    """f* = p - q/b. With p=0.6, b=2 → f* = 0.6 - 0.4/2 = 0.4 → capped at 0.25."""
    cfg = RiskConfig(position_size_method="kelly_capped", kelly_cap_pct=25.0, init_cash=100_000.0)
    sizer = PositionSizer(cfg)
    acct = LiveAccount(init_cash=100_000.0)
    qty = sizer.size(
        "PENNY",
        Side.LONG,
        mark_price=10.0,
        account=acct,
        recent_wins=12,
        recent_losses=8,
        avg_win=200.0,
        avg_loss=100.0,
    )
    # f* = 0.4 → capped at 0.25 → risk budget = 25_000 → qty = 25_000 / 10 = 2500
    assert qty == 2500


def test_kelly_returns_zero_on_negative_edge():
    """f* negative when winrate too low; sizer returns 0 (no trade)."""
    cfg = RiskConfig(position_size_method="kelly_capped", kelly_cap_pct=25.0, init_cash=100_000.0)
    sizer = PositionSizer(cfg)
    acct = LiveAccount(init_cash=100_000.0)
    qty = sizer.size(
        "PENNY",
        Side.LONG,
        mark_price=10.0,
        account=acct,
        recent_wins=3,
        recent_losses=17,
        avg_win=100.0,
        avg_loss=100.0,
    )
    # f* = 0.15 - 0.85/1 = -0.7 → clamped to 0
    assert qty == 0


def test_size_zero_on_invalid_mark_price():
    sizer = PositionSizer(RiskConfig())
    acct = LiveAccount(init_cash=100_000.0)
    assert sizer.size("X", Side.LONG, mark_price=0.0, account=acct) == 0
    assert sizer.size("X", Side.LONG, mark_price=-1.0, account=acct) == 0
