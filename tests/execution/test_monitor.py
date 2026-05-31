"""ExecutionMonitor: ordering, capacity, snapshot consistency, EOD writer."""

from __future__ import annotations

import threading
from datetime import datetime

from fortuna.execution.account import LiveAccount
from fortuna.execution.monitor import EventSeverity, ExecutionMonitor, write_eod_summary
from fortuna.execution.types import Side


def test_publish_preserves_order():
    m = ExecutionMonitor()
    for i in range(5):
        m.publish("signal", message=str(i))
    snap = m.snapshot()
    assert [e.message for e in snap] == ["0", "1", "2", "3", "4"]


def test_capacity_overflow_drops_oldest():
    m = ExecutionMonitor(capacity=10)
    for i in range(25):
        m.publish("evt", message=str(i))
    snap = m.snapshot()
    assert len(snap) == 10
    # Newest 10: messages 15..24
    assert [e.message for e in snap] == [str(i) for i in range(15, 25)]


def test_filter_by_severity_symbol_strategy():
    m = ExecutionMonitor()
    m.publish("signal", severity=EventSeverity.INFO, symbol="A", strategy="orb")
    m.publish("risk_breach", severity=EventSeverity.WARN, symbol="A", strategy="vwap")
    m.publish("circuit_breaker_tripped", severity=EventSeverity.CRITICAL, symbol="B")
    m.publish("signal", severity=EventSeverity.INFO, symbol="B", strategy="orb")

    crit = m.filter(severities=[EventSeverity.CRITICAL])
    assert len(crit) == 1
    assert crit[0].symbol == "B"

    a_only = m.filter(symbol="A")
    assert len(a_only) == 2

    orb_only = m.filter(strategy="orb")
    assert len(orb_only) == 2

    sig_only = m.filter(event_types=["signal"])
    assert all(e.event_type == "signal" for e in sig_only)


def test_concurrent_writes_do_not_corrupt(tmp_path):
    """Stress: 8 threads writing 1000 events each into a 5000-capacity buffer."""
    m = ExecutionMonitor(capacity=5000)

    def worker(tid: int) -> None:
        for i in range(1000):
            m.publish("evt", message=f"t{tid}-{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = m.snapshot()
    # Total writes = 8000; capacity = 5000 → exactly 5000 survive
    assert len(snap) == 5000


def test_tail_returns_recent_events():
    m = ExecutionMonitor()
    for i in range(20):
        m.publish("evt", message=str(i))
    assert [e.message for e in m.tail(5)] == ["15", "16", "17", "18", "19"]
    # Asking for more than buffered returns everything.
    assert len(m.tail(50)) == 20


def test_clear_removes_all_events():
    m = ExecutionMonitor()
    m.publish("evt", message="x")
    m.clear()
    assert m.snapshot() == []
    assert len(m) == 0


def test_event_to_dict_is_json_safe():
    m = ExecutionMonitor()
    m.publish("evt", severity=EventSeverity.WARN, message="hi", symbol="X")
    d = m.snapshot()[0].to_dict()
    assert d["severity"] == "WARN"
    assert isinstance(d["ts"], str)
    assert d["message"] == "hi"


def test_eod_summary_writes_markdown(tmp_path):
    account = LiveAccount(init_cash=100_000.0)
    monitor = ExecutionMonitor()
    monitor.publish("risk_breach", severity=EventSeverity.WARN, message="test breach")
    ts = datetime(2026, 1, 6, 9, 30)
    account.open_or_extend("RELIANCE", Side.LONG, qty=10, fill_price=2_000.0, ts=ts)
    account.close("RELIANCE", qty=10, fill_price=2_020.0, ts=ts, cost=1.0)
    path = write_eod_summary(account, monitor, output_dir=tmp_path)
    content = path.read_text(encoding="utf-8")
    assert "# Fortuna execution" in content
    assert "RELIANCE" in content
    assert "test breach" in content
