"""End-to-end smoke for the Tier 1 execution stack.

Builds a fresh ExecutionRouter (paper broker + risk gate + sizer + monitor +
journal), feeds it a sequence of synthetic 5-minute bars + LiveSignals, and
asserts the acceptance criteria from the plan:

* Paper orders appear in ``logs/execution/<DATE>.jsonl``
* Account equity curve grows in lockstep with bar closes
* Positions open/close as signals fire
* When intraday P&L crosses ``-2%`` (daily_loss_halt_pct), no more entries are
  placed and the Monitor receives a ``circuit_breaker_tripped`` event
* The EOD summary file is written to ``reports/execution/<DATE>.md``

Run with::

    uv run python scripts/smoke_execution.py

Exits 0 on success; non-zero on assertion failure or import problem.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fortuna.app.live_signals import LiveSignal  # noqa: E402
from fortuna.execution import (  # noqa: E402
    ExecutionMonitor,
    ExecutionRouter,
    EventSeverity,
    LiveAccount,
    OrderJournal,
    PaperBroker,
    PositionSizer,
    RiskConfig,
    RiskGate,
)
from fortuna.execution.monitor import write_eod_summary  # noqa: E402


def banner(msg: str) -> None:
    print(f"\n========== {msg} ==========")


def make_bar(ts: datetime, open_: float, close: float) -> dict:
    return {
        "ts": ts,
        "open": float(open_),
        "high": float(max(open_, close)),
        "low": float(min(open_, close)),
        "close": float(close),
        "volume": 10_000.0,
    }


def buy_signal(ts: datetime, close: float, strategy: str = "orb_5m") -> LiveSignal:
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


def exit_signal(ts: datetime, close: float, strategy: str = "orb_5m") -> LiveSignal:
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


def main() -> int:
    banner("BUILD ROUTER")
    # Slightly aggressive sizing so the deliberate ~4% adverse move in
    # scenario 3 actually carries enough equity damage to trip the 2%
    # breaker. Production defaults (10% risk, 25k symbol cap) are much
    # tighter — see config/execution.yaml.
    cfg = RiskConfig(
        init_cash=100_000.0,
        risk_per_trade_pct=100.0,
        daily_loss_halt_pct=2.0,
        per_symbol_max_notional=200_000.0,
        gross_exposure_cap_pct=200.0,
    )
    account = LiveAccount(init_cash=cfg.init_cash)
    monitor = ExecutionMonitor()
    journal = OrderJournal(directory=ROOT / "logs" / "execution_smoke")
    broker = PaperBroker(account=account, journal=journal)
    router = ExecutionRouter(
        broker=broker,
        account=account,
        risk_gate=RiskGate(cfg),
        sizer=PositionSizer(cfg),
        monitor=monitor,
        journal=journal,
        config=cfg,
    )
    print(f"  init_cash={cfg.init_cash}  cap={cfg.per_symbol_max_notional}")

    base = datetime(2026, 1, 6, 9, 30)
    symbol = "RELIANCE.NS"

    banner("SCENARIO 1: round-trip on RELIANCE")
    # Bar 1: BUY signal at 2000 close
    router.on_bar_closed(symbol, make_bar(base, 1_998.0, 2_000.0), {"orb_5m": buy_signal(base, 2_000.0)})
    # Bar 2: order fills at open ~2005 (no new signal)
    router.on_bar_closed(symbol, make_bar(base + timedelta(minutes=5), 2_005.0, 2_010.0), {})
    assert account.has_position(symbol), "position should have opened after fill"
    pos = account.get_position(symbol)
    print(f"  opened {pos.side.value} {pos.qty}@{pos.avg_price:.2f}")
    # Bar 3: EXIT_LONG signal
    router.on_bar_closed(
        symbol,
        make_bar(base + timedelta(minutes=10), 2_010.0, 2_015.0),
        {"orb_5m": exit_signal(base + timedelta(minutes=10), 2_015.0)},
    )
    # Bar 4: exit fills
    router.on_bar_closed(symbol, make_bar(base + timedelta(minutes=15), 2_020.0, 2_018.0), {})
    assert not account.has_position(symbol), "position should have closed"
    assert len(account.trades) == 1
    trade = account.trades[0]
    print(f"  closed: net_pnl={trade.net_pnl:+.2f} cost={trade.cost:.2f}")

    banner("SCENARIO 2: equity curve growth")
    assert len(account.equity_curve) == 4, f"expected 4 equity points, got {len(account.equity_curve)}"
    for p in account.equity_curve:
        print(f"  {p.ts:%H:%M}  equity={p.equity:,.2f}  unreal={p.unrealized:+.2f}  realized={p.realized:+.2f}")

    banner("SCENARIO 3: trip the daily-loss circuit breaker")
    # New entry
    base2 = base + timedelta(hours=1)
    router.on_bar_closed(symbol, make_bar(base2, 2_018.0, 2_020.0), {"orb_5m": buy_signal(base2, 2_020.0)})
    router.on_bar_closed(symbol, make_bar(base2 + timedelta(minutes=5), 2_020.0, 2_025.0), {})
    assert account.has_position(symbol)
    print(f"  re-entered LONG; peak equity={account.peak_equity:.2f}")
    # Hammer the price down 3% to trip the 2% halt.
    for i, drop in enumerate([2_020.0, 1_980.0, 1_950.0, 1_950.0]):
        router.on_bar_closed(
            symbol,
            make_bar(base2 + timedelta(minutes=10 + 5 * i), drop, drop),
            {},
        )
        if router.risk_gate.is_halted:
            print(f"  breaker tripped on bar {i+1}: {router.risk_gate.halted_reason}")
            break
    assert router.risk_gate.is_halted, "expected daily-loss halt to trip"
    breaker_events = [
        e for e in monitor.snapshot() if e.event_type == "circuit_breaker_tripped"
    ]
    assert breaker_events, "monitor should have a circuit_breaker_tripped event"

    banner("SCENARIO 4: post-halt entries blocked")
    base3 = base2 + timedelta(hours=2)
    router.on_bar_closed(
        "HDFC",
        make_bar(base3, 1_500.0, 1_510.0),
        {"vwap": buy_signal(base3, 1_510.0, strategy="vwap")},
    )
    risk_blocks = [
        e for e in monitor.snapshot()
        if e.event_type == "risk_breach" and e.payload.get("rule") == "halted"
    ]
    assert risk_blocks, "expected at least one halted risk_breach"
    print(f"  {len(risk_blocks)} order(s) blocked post-halt")

    banner("SCENARIO 5: journal + EOD summary")
    journal.close()
    journal_rows = journal.read_today()
    types = sorted({r["type"] for r in journal_rows})
    print(f"  journal rows: {len(journal_rows)}  types: {types}")
    assert "order_placed" in types
    assert "order_filled" in types

    eod_path = write_eod_summary(account, monitor, output_dir=ROOT / "reports" / "execution_smoke")
    print(f"  EOD summary: {eod_path}")
    assert eod_path.exists()

    banner("ROUTER STATS")
    stats = router.stats
    print(f"  bars_processed:     {stats.bars_processed}")
    print(f"  signals_seen:       {stats.signals_seen}")
    print(f"  orders_placed:      {stats.orders_placed}")
    print(f"  orders_rejected:    {stats.orders_rejected}")
    print(f"  fills:              {stats.fills}")
    print(f"  risk_blocks:        {stats.risk_blocks}")
    print(f"  circuit_breaker:    {stats.circuit_breaker_trips}")

    banner("ACCOUNT SUMMARY")
    for k, v in account.to_summary().items():
        print(f"  {k:>20s}: {v}")

    print("\nSMOKE PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
