"""ExecutionRouter: end-to-end signal -> order -> fill cycle.

Uses :class:`PaperBroker` (deterministic next-bar-open fills) so the
router's behaviour can be asserted without mocking the broker layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from fortuna.app.live_signals import LiveSignal
from fortuna.execution.account import LiveAccount
from fortuna.execution.monitor import ExecutionMonitor
from fortuna.execution.paper_broker import PaperBroker
from fortuna.execution.risk import PositionSizer, RiskConfig, RiskGate
from fortuna.execution.router import ExecutionRouter


def _build_router(*, init_cash=100_000.0, cfg_overrides: dict | None = None):
    overrides = {"risk_per_trade_pct": 10.0}
    overrides.update(cfg_overrides or {})
    cfg = RiskConfig(init_cash=init_cash, **overrides)
    account = LiveAccount(init_cash=init_cash)
    broker = PaperBroker(account=account)
    monitor = ExecutionMonitor()
    router = ExecutionRouter(
        broker=broker,
        account=account,
        risk_gate=RiskGate(cfg),
        sizer=PositionSizer(cfg),
        monitor=monitor,
        config=cfg,
    )
    return router, account, broker, monitor


def _buy_signal(ts, close, strategy="orb"):
    return LiveSignal(
        strategy_name=strategy,
        action="BUY",
        label="BUY",
        bar_time=pd.Timestamp(ts),
        bar_close=close,
        enter=True,
        exit=False,
        in_position=False,
        side="long",
    )


def _exit_signal(ts, close, strategy="orb"):
    return LiveSignal(
        strategy_name=strategy,
        action="EXIT_LONG",
        label="EXIT",
        bar_time=pd.Timestamp(ts),
        bar_close=close,
        enter=False,
        exit=True,
        in_position=True,
        side="long",
    )


def _bar(ts, open_, close, *, high=None, low=None, volume=1000.0):
    return {
        "ts": ts,
        "open": float(open_),
        "high": float(high if high is not None else max(open_, close)),
        "low": float(low if low is not None else min(open_, close)),
        "close": float(close),
        "volume": volume,
    }


def test_signal_routes_through_to_pending_order():
    router, account, broker, monitor = _build_router()
    ts = datetime(2026, 1, 6, 10, 0)
    router.on_bar_closed(
        "RELIANCE",
        _bar(ts, 1_995.0, 2_000.0),
        {"orb": _buy_signal(ts, 2_000.0)},
    )

    assert router.stats.signals_seen == 1
    assert router.stats.orders_placed == 1
    # Order is pending — fills only on the NEXT bar.
    assert not account.has_position("RELIANCE")

    router.on_bar_closed(
        "RELIANCE",
        _bar(ts + timedelta(minutes=5), 2_005.0, 2_010.0),
        {},  # no new signal this bar
    )
    assert account.has_position("RELIANCE")
    assert router.stats.fills == 1


def test_full_round_trip_records_trade():
    router, account, broker, monitor = _build_router()
    ts = datetime(2026, 1, 6, 10, 0)

    # Enter on bar 1.
    router.on_bar_closed("RELIANCE", _bar(ts, 1_995.0, 2_000.0), {"orb": _buy_signal(ts, 2_000.0)})
    # Fill happens on bar 2 open.
    router.on_bar_closed("RELIANCE", _bar(ts + timedelta(minutes=5), 2_005.0, 2_010.0), {})
    assert account.has_position("RELIANCE")

    # Exit on bar 3.
    router.on_bar_closed(
        "RELIANCE",
        _bar(ts + timedelta(minutes=10), 2_010.0, 2_015.0),
        {"orb": _exit_signal(ts + timedelta(minutes=10), 2_015.0)},
    )
    # Exit fills on bar 4 open.
    router.on_bar_closed("RELIANCE", _bar(ts + timedelta(minutes=15), 2_020.0, 2_018.0), {})
    assert not account.has_position("RELIANCE")
    assert len(account.trades) == 1


def test_rl_suppressed_signal_skipped():
    router, account, _, monitor = _build_router()
    ts = datetime(2026, 1, 6, 10, 0)
    sig = _buy_signal(ts, 2_000.0)
    sig.rl_suppressed = True
    sig.rl_action = "HOLD"
    router.on_bar_closed("RELIANCE", _bar(ts, 1_995.0, 2_000.0), {"orb": sig})
    assert router.stats.rl_suppressed == 1
    assert router.stats.orders_placed == 0
    assert any(e.event_type == "signal_rl_suppressed" for e in monitor.snapshot())


def test_circuit_breaker_blocks_new_entries():
    router, account, broker, monitor = _build_router(
        cfg_overrides={
            "daily_loss_halt_pct": 0.5,
            "risk_per_trade_pct": 50.0,
            "per_symbol_max_notional": 200_000.0,
            "gross_exposure_cap_pct": 200.0,
        }
    )
    ts = datetime(2026, 1, 6, 10, 0)
    # Enter on bar 1, fill on bar 2.
    router.on_bar_closed("RELIANCE", _bar(ts, 1_990.0, 2_000.0), {"orb": _buy_signal(ts, 2_000.0)})
    router.on_bar_closed("RELIANCE", _bar(ts + timedelta(minutes=5), 2_000.0, 2_000.0), {})
    assert account.has_position("RELIANCE")

    # Crash 5% to trip the 0.5% breaker.
    router.on_bar_closed("RELIANCE", _bar(ts + timedelta(minutes=10), 2_000.0, 1_900.0), {})
    assert router.risk_gate.is_halted
    assert router.stats.circuit_breaker_trips == 1

    # New BUY signal on a different symbol is now blocked.
    router.on_bar_closed(
        "HDFC",
        _bar(ts + timedelta(minutes=15), 1_500.0, 1_510.0),
        {"orb": _buy_signal(ts + timedelta(minutes=15), 1_510.0, strategy="vwap")},
    )
    assert router.stats.risk_blocks >= 1
    halt_events = [
        e for e in monitor.snapshot()
        if e.event_type == "risk_breach" and e.payload.get("rule") == "halted"
    ]
    assert halt_events


def test_strategy_cooldown_after_losses_blocks_strategy():
    router, account, broker, monitor = _build_router(
        cfg_overrides={
            "per_strategy_cooldown_losses": 2,
            "risk_per_trade_pct": 50.0,
            "per_symbol_max_notional": 200_000.0,
        }
    )
    ts = datetime(2026, 1, 6, 10, 0)

    # Trip the strategy directly (faster than simulating two losing trades).
    router.risk_gate.update_on_trade("orb", -100.0)
    router.risk_gate.update_on_trade("orb", -100.0)
    assert router.risk_gate.strategy_on_cooldown("orb")

    router.on_bar_closed(
        "RELIANCE",
        _bar(ts, 1_990.0, 2_000.0),
        {"orb": _buy_signal(ts, 2_000.0)},
    )
    assert router.stats.cooldown_skips == 1
    assert router.stats.orders_placed == 0


def test_zero_size_skipped_silently():
    # 1% risk on 100k = 1000; mark = 1_000_000 → qty = 0 → router emits size_zero.
    router, account, _, monitor = _build_router(
        cfg_overrides={"risk_per_trade_pct": 1.0, "per_symbol_max_notional": 10_000_000.0}
    )
    ts = datetime(2026, 1, 6, 10, 0)
    router.on_bar_closed(
        "EXPENSIVE",
        _bar(ts, 999_999.0, 1_000_000.0),
        {"orb": _buy_signal(ts, 1_000_000.0)},
    )
    assert router.stats.orders_placed == 0
    assert any(e.event_type == "size_zero" for e in monitor.snapshot())


def test_equity_curve_grows_with_each_bar():
    router, account, _, _ = _build_router()
    ts = datetime(2026, 1, 6, 10, 0)
    for i in range(5):
        router.on_bar_closed(
            "RELIANCE",
            _bar(ts + timedelta(minutes=5 * i), 2_000.0, 2_000.0 + i),
            {},
        )
    assert len(account.equity_curve) == 5


def test_reset_for_new_session_clears_halt():
    router, account, broker, _ = _build_router(
        cfg_overrides={
            "daily_loss_halt_pct": 0.5,
            "risk_per_trade_pct": 50.0,
            "per_symbol_max_notional": 200_000.0,
            "gross_exposure_cap_pct": 200.0,
        }
    )
    ts = datetime(2026, 1, 6, 10, 0)
    router.on_bar_closed("RELIANCE", _bar(ts, 1_990.0, 2_000.0), {"orb": _buy_signal(ts, 2_000.0)})
    router.on_bar_closed("RELIANCE", _bar(ts + timedelta(minutes=5), 2_000.0, 2_000.0), {})
    router.on_bar_closed("RELIANCE", _bar(ts + timedelta(minutes=10), 2_000.0, 1_900.0), {})
    assert router.risk_gate.is_halted

    router.reset_for_new_session()
    assert not router.risk_gate.is_halted


def test_eod_summary_emits_event_and_writes_file(tmp_path, monkeypatch):
    router, account, broker, monitor = _build_router()
    ts = datetime(2026, 1, 6, 10, 0)
    router.on_bar_closed("RELIANCE", _bar(ts, 1_995.0, 2_000.0), {"orb": _buy_signal(ts, 2_000.0)})
    router.on_bar_closed("RELIANCE", _bar(ts + timedelta(minutes=5), 2_005.0, 2_010.0), {})

    # Redirect write_eod_summary to the tmp_path so we don't pollute the repo.
    import fortuna.execution.router as router_mod
    monkeypatch.setattr(
        router_mod, "write_eod_summary",
        lambda acct, mon: __import__("fortuna.execution.monitor", fromlist=["write_eod_summary"])
            .write_eod_summary(acct, mon, output_dir=tmp_path),
    )
    out = router.write_eod_summary()
    assert out is not None
    assert any(e.event_type == "eod_summary" for e in monitor.snapshot())
