"""RiskGate: per-symbol cap, gross exposure, daily-loss halt, cooldowns."""

from __future__ import annotations

from datetime import datetime

import pytest

from fortuna.execution.account import LiveAccount
from fortuna.execution.risk import RiskConfig, RiskGate, load_risk_config
from fortuna.execution.types import OrderIntent, Side


def _intent(symbol="RELIANCE", side=Side.LONG, qty=10, tag="orb", close=False):
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=qty,
        tag=tag,
        close_position=close,
        intent_ts=datetime(2026, 1, 6, 10, 0),
    )


def test_default_config_allows_clean_order():
    gate = RiskGate()
    acct = LiveAccount(init_cash=100_000.0)
    verdict = gate.allow_order(_intent(qty=10), acct, mark_price=2_000.0)
    assert verdict.allowed


def test_per_symbol_max_notional_blocks_oversized_order():
    gate = RiskGate(RiskConfig(per_symbol_max_notional=10_000.0))
    acct = LiveAccount(init_cash=100_000.0)
    # 10 shares * 2000 = 20_000 > 10_000 cap
    verdict = gate.allow_order(_intent(qty=10), acct, mark_price=2_000.0)
    assert not verdict.allowed
    assert verdict.rule == "per_symbol_max_notional"


def test_gross_exposure_cap_blocks_when_already_loaded():
    gate = RiskGate(RiskConfig(gross_exposure_cap_pct=50.0, init_cash=100_000.0))
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("HDFC", Side.LONG, qty=25, fill_price=1_500.0, ts=datetime.now())  # 37_500
    acct.mark_to_market("HDFC", 1_500.0)
    # cap = 50_000; existing = 37_500; new = 20_000 → total 57_500 > cap
    verdict = gate.allow_order(_intent(symbol="RELIANCE", qty=10), acct, mark_price=2_000.0)
    assert not verdict.allowed
    assert verdict.rule == "gross_exposure_cap"


def test_symbol_in_use_blocks_second_open():
    gate = RiskGate()
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE", Side.LONG, qty=5, fill_price=2_000.0, ts=datetime.now())
    verdict = gate.allow_order(_intent(qty=5), acct, mark_price=2_010.0)
    assert not verdict.allowed
    assert verdict.rule == "symbol_in_use"


def test_opposite_side_open_blocked():
    gate = RiskGate()
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE", Side.LONG, qty=5, fill_price=2_000.0, ts=datetime.now())
    verdict = gate.allow_order(_intent(side=Side.SHORT, qty=5), acct, mark_price=2_010.0)
    assert not verdict.allowed
    assert verdict.rule == "opposite_side_open"


def test_close_is_always_allowed_even_when_halted():
    gate = RiskGate()
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE", Side.LONG, qty=5, fill_price=2_000.0, ts=datetime.now())
    # Manually trip the breaker.
    gate._halted = True
    gate._halted_reason = "test trip"
    verdict = gate.allow_order(_intent(qty=5, close=True), acct, mark_price=2_010.0)
    assert verdict.allowed


def test_session_window_blocks_late_entries():
    gate = RiskGate(RiskConfig(session_window=("09:20", "15:10")))
    acct = LiveAccount(init_cash=100_000.0)
    # 15:30 IST → outside the window
    late = datetime(2026, 1, 6, 15, 30)
    verdict = gate.allow_order(_intent(), acct, now=late, mark_price=2_000.0)
    assert not verdict.allowed
    assert verdict.rule == "session_window"


def test_daily_loss_halt_trips_on_dd_threshold():
    gate = RiskGate(RiskConfig(daily_loss_halt_pct=2.0))
    acct = LiveAccount(init_cash=100_000.0)
    acct.open_or_extend("RELIANCE", Side.LONG, qty=10, fill_price=2_000.0, ts=datetime.now())

    acct.mark_to_market("RELIANCE", 2_100.0)
    gate.update_on_bar(acct)
    assert not gate.is_halted

    acct.mark_to_market("RELIANCE", 1_960.0)  # equity ≈ 99_600; peak ≈ 101_000 → DD ≈ 1.39%
    gate.update_on_bar(acct)
    assert not gate.is_halted

    acct.mark_to_market("RELIANCE", 1_900.0)  # equity = 99_000; peak = 101_000 → DD ≈ 1.98% (just under)
    gate.update_on_bar(acct)

    acct.mark_to_market("RELIANCE", 1_880.0)  # equity = 98_800; DD ≈ 2.18% — trips
    gate.update_on_bar(acct)
    assert gate.is_halted

    verdict = gate.allow_order(_intent(), acct, mark_price=2_000.0)
    assert not verdict.allowed
    assert verdict.rule == "halted"


def test_strategy_cooldown_after_consecutive_losses():
    gate = RiskGate(RiskConfig(per_strategy_cooldown_losses=3))
    acct = LiveAccount(init_cash=100_000.0)
    for _ in range(3):
        gate.update_on_trade("orb", -100.0)
    assert gate.strategy_on_cooldown("orb")
    verdict = gate.allow_order(_intent(tag="orb"), acct, mark_price=2_000.0)
    assert not verdict.allowed
    assert verdict.rule == "strategy_cooldown"

    # A different strategy is unaffected.
    verdict_other = gate.allow_order(_intent(tag="vwap"), acct, mark_price=2_000.0)
    assert verdict_other.allowed


def test_winning_trade_resets_consecutive_losses():
    gate = RiskGate(RiskConfig(per_strategy_cooldown_losses=3))
    gate.update_on_trade("orb", -100.0)
    gate.update_on_trade("orb", -100.0)
    gate.update_on_trade("orb", +50.0)
    assert not gate.strategy_on_cooldown("orb")


def test_reset_for_new_session_clears_halt_and_cooldowns():
    gate = RiskGate(RiskConfig(per_strategy_cooldown_losses=2))
    acct = LiveAccount(init_cash=100_000.0)
    gate.update_on_trade("orb", -100.0)
    gate.update_on_trade("orb", -100.0)
    assert gate.strategy_on_cooldown("orb")
    gate._halted = True
    gate.reset_for_new_session()
    assert not gate.is_halted
    assert not gate.strategy_on_cooldown("orb")


def test_load_risk_config_falls_back_to_defaults_when_missing(tmp_path):
    cfg = load_risk_config(tmp_path / "missing.yaml")
    assert cfg.init_cash == 100_000.0


def test_load_risk_config_parses_yaml(tmp_path):
    f = tmp_path / "exec.yaml"
    f.write_text(
        "init_cash: 200000.0\n"
        "per_symbol_max_notional: 50000.0\n"
        "daily_loss_halt_pct: 3.0\n"
        "session_window: ['09:25', '15:00']\n"
        "lot_sizes:\n  RELIANCE.FUT: 500\n",
        encoding="utf-8",
    )
    cfg = load_risk_config(f)
    assert cfg.init_cash == pytest.approx(200_000.0)
    assert cfg.per_symbol_max_notional == pytest.approx(50_000.0)
    assert cfg.daily_loss_halt_pct == pytest.approx(3.0)
    assert cfg.session_window == ("09:25", "15:00")
    assert cfg.lot_sizes == {"RELIANCE.FUT": 500}
