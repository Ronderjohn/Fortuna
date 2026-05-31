"""End-to-end smoke for Phase 1 + Phase 2 (portfolio-aware paper sim).

Wires together:
1. A fake SmartConnect client → SmartAPIAccountReader → full_snapshot
2. LiveAccount.seed_from_broker → real-shaped cash + positions + holdings
3. ExecutionRouter consuming a Fortuna LiveSignal that closes the seeded
   broker position (paper-side exit at next bar open)
4. reconcile_paper_vs_real diff between the paper journal and a fake
   tradeBook → ReconciliationReport with one matched pair

Run::

    uv run python scripts/smoke_account_sync.py

The script writes nothing to ``logs/`` by default; everything lands in a
``tmp_smoke/`` directory next to it and is cleaned up on exit.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------- imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.live_signals import LiveSignal  # noqa: E402
from fortuna.execution.account import LiveAccount  # noqa: E402
from fortuna.execution.journal import OrderJournal  # noqa: E402
from fortuna.execution.monitor import ExecutionMonitor  # noqa: E402
from fortuna.execution.paper_broker import PaperBroker  # noqa: E402
from fortuna.execution.reconciliation import (  # noqa: E402
    append_reconciliation_to_eod,
    reconcile_paper_vs_real,
)
from fortuna.execution.risk import (  # noqa: E402
    PositionSizer,
    RiskConfig,
    RiskGate,
)
from fortuna.execution.router import ExecutionRouter  # noqa: E402
from fortuna.execution.smartapi_account import SmartAPIAccountReader  # noqa: E402


def _bar(ts: datetime, open_: float, close: float) -> dict:
    return {
        "ts": ts,
        "open": float(open_),
        "high": float(max(open_, close)),
        "low": float(min(open_, close)),
        "close": float(close),
        "volume": 10_000.0,
    }


def _exit_signal(ts: datetime, close: float, strategy: str = "orb") -> LiveSignal:
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


# ----------------------------------------------------------------- fake client
class FakeSmartConnect:
    """Synthetic Angel One responses so the smoke doesn't need real creds."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def rmsLimit(self):
        self.calls.append("rmsLimit")
        return {
            "status": True,
            "data": {
                "net": "152345.67",
                "availablecash": "150000.00",
                "utilisedDebits": "2345.67",
                "collateral": "0.00",
                "m2munrealized": "0.0",
            },
        }

    def position(self):
        self.calls.append("position")
        return {
            "status": True,
            "data": [
                {
                    "tradingsymbol": "RELIANCE-EQ",
                    "symboltoken": "2885",
                    "exchange": "NSE",
                    "producttype": "INTRADAY",
                    "netqty": "10",
                    "avgnetprice": "2000.0",
                    "ltp": "2005.0",
                    "unrealised": "50.0",
                    "realised": "0.0",
                }
            ],
        }

    def holding(self):
        self.calls.append("holding")
        return {
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
                }
            ],
        }

    def orderBook(self):
        self.calls.append("orderBook")
        return {"status": True, "data": []}

    def tradeBook(self):
        self.calls.append("tradeBook")
        # One real fill that matches the paper EXIT we'll generate below.
        return {
            "status": True,
            "data": [
                {
                    "orderid": "REAL-1",
                    "tradingsymbol": "RELIANCE-EQ",
                    "symboltoken": "2885",
                    "exchange": "NSE",
                    "transactiontype": "SELL",
                    "fillsize": "10",
                    "fillprice": "2005.5",
                    "filltime": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
                    "producttype": "INTRADAY",
                }
            ],
        }


# ----------------------------------------------------------------- helpers
def assert_(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="fortuna_account_smoke_"))
    print(f"[smoke] tmp dir: {tmp_root}")

    try:
        # ============================================ 1. account reader
        print("\n[1/4] SmartAPIAccountReader -> full_snapshot")
        reader = SmartAPIAccountReader(client=FakeSmartConnect())
        snap = reader.full_snapshot()
        assert_(not snap.has_errors(), "snapshot has no errors")
        assert_(snap.funds.available_cash == 150_000.0, "funds parsed")
        assert_(len(snap.positions) == 1, "1 broker position")
        assert_(len(snap.holdings) == 1, "1 broker holding")
        assert_(snap.positions[0].fortuna_symbol is None,
                "no registry -> fortuna_symbol is None (expected)")

        # ============================================ 2. seed LiveAccount
        print("\n[2/4] LiveAccount.seed_from_broker (broker -> paper)")
        # Hack: assign fortuna_symbol manually since we don't have a registry.
        snap.positions[0] = replace(snap.positions[0], fortuna_symbol="RELIANCE.NS")
        snap.holdings[0] = replace(snap.holdings[0], fortuna_symbol="ITC.NS")

        account = LiveAccount(init_cash=10_000.0)  # will be overwritten by seed
        applied = account.seed_from_broker(snap)
        assert_(account.init_cash == 150_000.0, "init_cash replaced by broker cash")
        assert_("RELIANCE.NS" in account.positions, "RELIANCE position seeded")
        assert_("ITC.NS" in account.holdings, "ITC holding seeded")
        assert_(applied["positions_added"] == 1, "1 position counted as added")
        assert_(applied["holdings_added"] == 1, "1 holding counted as added")

        # ============================================ 3. router exits the position
        print("\n[3/4] ExecutionRouter exits the seeded broker position")
        journal = OrderJournal(directory=tmp_root / "journal")
        monitor = ExecutionMonitor()
        cfg = RiskConfig(
            init_cash=account.init_cash,
            per_symbol_max_notional=10_000_000.0,
            gross_exposure_cap_pct=500.0,
            daily_loss_halt_pct=100.0,
            risk_per_trade_pct=1.0,
            session_window=None,
        )
        broker = PaperBroker(account=account, journal=journal)
        risk_gate = RiskGate(cfg)
        sizer = PositionSizer(cfg)
        router = ExecutionRouter(
            broker=broker,
            account=account,
            risk_gate=risk_gate,
            sizer=sizer,
            monitor=monitor,
            journal=journal,
            config=cfg,
        )

        # Two bars: t0 emit EXIT_LONG, t1 fills at next open.
        # Use "now" so the paper journal timestamp lines up with the real
        # trade's fill_time (which the FakeSmartConnect emits as now()).
        t0 = datetime.now().replace(microsecond=0) - timedelta(minutes=5)
        t1 = t0 + timedelta(minutes=5)
        router.on_bar_closed(
            "RELIANCE.NS", _bar(t0, 2005.0, 2005.0), {"orb": _exit_signal(t0, 2005.0)},
        )
        router.on_bar_closed("RELIANCE.NS", _bar(t1, 2005.5, 2007.0), {})

        assert_("RELIANCE.NS" not in account.positions, "position flattened")
        assert_(account.realized_pnl != 0.0, "realized P&L recorded")
        print(f"     realized P&L = INR {account.realized_pnl:.2f}")

        # ============================================ 4. reconciliation
        print("\n[4/4] reconcile_paper_vs_real")
        # Pull "real" trades from the fake client; tag with fortuna_symbol manually.
        real_trades_raw = SmartAPIAccountReader(client=FakeSmartConnect()).snapshot_trades()
        real_trades = [replace(rt, fortuna_symbol="RELIANCE.NS") for rt in real_trades_raw]
        rows = journal.read_today()
        report = reconcile_paper_vs_real(rows, real_trades)
        print(f"     paper={report.paper_fills} real={report.real_fills} "
              f"matched={len(report.matched)} paper_only={len(report.paper_only)} "
              f"real_only={len(report.real_only)}")
        assert_(report.paper_fills == 1, "1 paper fill")
        assert_(report.real_fills == 1, "1 real fill")
        assert_(len(report.matched) == 1, "1 matched pair")

        eod_path = tmp_root / "eod.md"
        eod_path.write_text("# EOD smoke summary\n", encoding="utf-8")
        append_reconciliation_to_eod(eod_path, report)
        assert_("Reconciliation" in eod_path.read_text(encoding="utf-8"),
                "reconciliation block appended to EOD markdown")

        print("\n[smoke] PASS - all assertions passed.")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
