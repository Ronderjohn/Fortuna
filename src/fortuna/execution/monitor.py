"""In-memory execution event feed for the Streamlit Monitor tab.

Lightweight by design: a bounded ``collections.deque`` of typed events,
guarded by a ``threading.Lock``. The router pushes events; the dashboard
reads snapshots. No third-party push notifications (Telegram / email /
etc) — per the answered design question, Tier 1 monitoring is
dashboard-only.

Also exposes :func:`write_eod_summary` which formats a Markdown report
under ``reports/execution/<DATE>.md`` for next-day reconciliation.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from fortuna.execution.account import LiveAccount


class EventSeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ExecutionEvent:
    """One typed event in the live feed."""

    ts: datetime
    event_type: str  # signal_fired / order_placed / order_filled / risk_breach / ...
    severity: EventSeverity
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        d["severity"] = self.severity.value
        return d


class ExecutionMonitor:
    """Thread-safe ring buffer of :class:`ExecutionEvent`.

    Capacity defaults to 1000 events; older events are silently dropped
    once the buffer overflows. That's more than enough for one session
    even at high signal density (10 symbols x ~80 bars/day x a handful
    of events per bar ~ 8000 ceiling).
    """

    DEFAULT_CAPACITY = 1000

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._events: deque[ExecutionEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- writes
    def publish(
        self,
        event_type: str,
        *,
        severity: EventSeverity = EventSeverity.INFO,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        message: str = "",
        **payload: Any,
    ) -> ExecutionEvent:
        evt = ExecutionEvent(
            ts=datetime.now(),
            event_type=event_type,
            severity=severity,
            symbol=symbol,
            strategy=strategy,
            message=message,
            payload=payload,
        )
        with self._lock:
            self._events.append(evt)
        return evt

    # ------------------------------------------------------------------ reads
    def snapshot(self) -> list[ExecutionEvent]:
        """Return a list copy of every event currently buffered (oldest first)."""
        with self._lock:
            return list(self._events)

    def tail(self, n: int = 100) -> list[ExecutionEvent]:
        """Return the most recent ``n`` events (newest last)."""
        with self._lock:
            if n >= len(self._events):
                return list(self._events)
            return list(self._events)[-n:]

    def filter(
        self,
        *,
        severities: Optional[Iterable[EventSeverity]] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        event_types: Optional[Iterable[str]] = None,
    ) -> list[ExecutionEvent]:
        sev_set = set(severities) if severities else None
        type_set = set(event_types) if event_types else None
        with self._lock:
            events = list(self._events)
        return [
            e
            for e in events
            if (sev_set is None or e.severity in sev_set)
            and (symbol is None or e.symbol == symbol)
            and (strategy is None or e.strategy == strategy)
            and (type_set is None or e.event_type in type_set)
        ]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


def write_eod_summary(
    account: LiveAccount,
    monitor: ExecutionMonitor,
    output_dir: str | Path = "reports/execution",
    *,
    session_date: Optional[datetime] = None,
) -> Path:
    """Render an EOD Markdown summary of the day's trading.

    Returns the path of the file written. Safe to call multiple times in
    one session; each call overwrites the same dated file so the most
    recent snapshot wins.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    when = session_date or datetime.now()
    path = out_dir / f"{when.date().isoformat()}.md"

    summary = account.to_summary()
    trades = list(account.trades)
    breaches = monitor.filter(severities=[EventSeverity.WARN, EventSeverity.ERROR, EventSeverity.CRITICAL])

    lines: list[str] = []
    lines.append(f"# Fortuna execution — {when.date().isoformat()}")
    lines.append("")
    lines.append("## Portfolio")
    lines.append("")
    lines.append(f"- init_cash: {summary['init_cash']:,.0f}")
    lines.append(f"- equity:    {summary['equity']:,.0f}")
    lines.append(f"- realized:  {summary['realized_pnl']:+,.2f}")
    lines.append(f"- unreal:    {summary['unrealized_pnl']:+,.2f}")
    lines.append(f"- peak:      {summary['peak_equity']:,.0f}")
    lines.append(f"- max DD:    {summary['max_drawdown_pct']:.2f}%")
    lines.append("")
    lines.append("## Trade ledger")
    lines.append("")
    if not trades:
        lines.append("_no trades today_")
    else:
        lines.append(
            f"- total: {summary['total_trades']} | wins: {summary['wins']} | losses: {summary['losses']} | "
            f"win%: {summary['win_rate_pct']:.2f}% | PF: {summary['profit_factor']}"
        )
        lines.append("")
        lines.append("| symbol | side | qty | entry | exit | net P&L | return % | tag |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for t in trades:
            lines.append(
                f"| {t.symbol} | {t.side.value} | {t.qty} | "
                f"{t.entry_price:.2f} | {t.exit_price:.2f} | "
                f"{t.net_pnl:+,.2f} | {t.return_pct:+.2f}% | {t.tag or ''} |"
            )

    if breaches:
        lines.append("")
        lines.append("## Risk / monitor events")
        lines.append("")
        for evt in breaches:
            lines.append(
                f"- [{evt.severity.value}] {evt.ts.strftime('%H:%M:%S')} {evt.event_type} "
                f"{evt.symbol or ''} {evt.strategy or ''} — {evt.message}"
            )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["EventSeverity", "ExecutionEvent", "ExecutionMonitor", "write_eod_summary"]
