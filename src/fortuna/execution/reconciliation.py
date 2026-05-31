"""Phase 2: diff paper Fortuna activity against real broker activity.

Given:

- The day's :class:`fortuna.execution.journal.OrderJournal` events (what Fortuna
  *would have* placed / filled, in paper)
- The day's real :class:`BrokerTrade` rows + open :class:`BrokerPosition`s
  pulled from SmartAPI

…produce a per-symbol diff that answers two questions you'll ask every
session:

1. **Coverage** — for every real trade I executed, did Fortuna also signal
   it (and roughly when)? If not, that's manual discretion the simulator
   is missing.
2. **Quality** — for every paper trade Fortuna placed, did the real
   account follow? When it didn't, what would the P&L have been? Stack
   these up over a week and you have an empirical "advisor accuracy"
   curve.

The reconciliation is best-effort: timestamp formats and symbol aliasing
between Fortuna and Angel are messy, so we lean on (symbol, side, qty,
fill-time window) heuristic matching. Pairs and orphans are returned
separately so the dashboard can colour-code each row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from fortuna.execution.types import Side
from fortuna.utils.logging import get_logger

if TYPE_CHECKING:
    from fortuna.execution.smartapi_account import BrokerTrade

logger = get_logger(__name__)


# ============================================================ dataclasses


@dataclass(frozen=True)
class PaperOrder:
    """Flattened ``order_filled`` event from the paper journal."""

    order_id: str
    symbol: str
    side: Side
    qty: int
    fill_price: float
    fill_ts: Optional[datetime]
    strategy: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MatchedPair:
    """A paper-vs-real trade match (symbol + side aligned)."""

    symbol: str
    side: Side
    qty: int
    paper_price: float
    real_price: float
    paper_ts: Optional[datetime]
    real_ts: Optional[datetime]
    strategy: Optional[str] = None
    price_diff: float = 0.0  # real - paper (negative => paper got the better fill)

    @property
    def slippage_pct(self) -> float:
        if self.paper_price == 0:
            return 0.0
        return (self.real_price - self.paper_price) / self.paper_price * 100.0


@dataclass
class ReconciliationReport:
    """Output of :func:`reconcile_paper_vs_real`."""

    session_date: datetime
    matched: list[MatchedPair] = field(default_factory=list)
    paper_only: list[PaperOrder] = field(default_factory=list)
    real_only: list["BrokerTrade"] = field(default_factory=list)
    paper_fills: int = 0
    real_fills: int = 0

    @property
    def coverage_pct(self) -> float:
        """Fraction of real fills that had a matching paper signal."""
        if self.real_fills == 0:
            return 100.0
        return len(self.matched) / self.real_fills * 100.0

    @property
    def precision_pct(self) -> float:
        """Fraction of paper signals that actually got executed in real."""
        if self.paper_fills == 0:
            return 0.0
        return len(self.matched) / self.paper_fills * 100.0

    @property
    def avg_slippage_pct(self) -> float:
        if not self.matched:
            return 0.0
        return sum(m.slippage_pct for m in self.matched) / len(self.matched)

    def to_dict(self) -> dict:
        return {
            "session_date": self.session_date.isoformat(),
            "paper_fills": self.paper_fills,
            "real_fills": self.real_fills,
            "matched": len(self.matched),
            "paper_only": len(self.paper_only),
            "real_only": len(self.real_only),
            "coverage_pct": round(self.coverage_pct, 2),
            "precision_pct": round(self.precision_pct, 2),
            "avg_slippage_pct": round(self.avg_slippage_pct, 4),
        }


# ============================================================ paper extraction


def paper_orders_from_journal(rows: Iterable[dict]) -> list[PaperOrder]:
    """Walk a list of OrderJournal events and emit one PaperOrder per fill.

    The journal records all event types (placed / filled / rejected /
    cancelled); we only care about ``order_filled`` for reconciliation.

    Side mapping: for fills that *close* an existing position, the actual
    market direction is opposite to the position direction (an EXIT_LONG
    is a SELL). We pull ``intent.close_position`` from the journal payload
    so the reconciliation buckets line up with the broker's BUY/SELL labels.
    """
    out: list[PaperOrder] = []
    for r in rows:
        if r.get("type") != "order_filled":
            continue
        ack = r.get("ack") or {}
        intent = r.get("intent") or {}
        side_raw = (ack.get("side") or "").upper()
        if side_raw not in {"LONG", "SHORT"}:
            continue
        side = Side(side_raw)
        # If the journal includes the underlying intent flag, flip the side
        # for closes so SELL-to-close (paper) matches SELL fills (broker).
        if bool(intent.get("close_position", False)):
            side = Side.SHORT if side is Side.LONG else Side.LONG
        ts = _parse_iso(ack.get("ts") or r.get("ts"))
        out.append(
            PaperOrder(
                order_id=str(ack.get("order_id", "")),
                symbol=str(ack.get("symbol", "")).upper(),
                side=side,
                qty=int(ack.get("filled_qty") or ack.get("qty") or 0),
                fill_price=float(ack.get("filled_price") or 0.0),
                fill_ts=ts,
                strategy=ack.get("tag"),
                raw=r,
            )
        )
    return out


# ============================================================ matching


def reconcile_paper_vs_real(
    journal_rows: Iterable[dict],
    real_trades: Iterable["BrokerTrade"],
    *,
    session_date: Optional[datetime] = None,
    time_window: timedelta = timedelta(minutes=15),
) -> ReconciliationReport:
    """Match paper fills against real trades; return the diff report.

    Matching policy (greedy, deterministic):

    1. Index real trades by ``(fortuna_symbol or tradingsymbol, mapped_side)``.
    2. For each paper fill in fill-time order, take the closest real trade
       in time *within ``time_window``*. If none, mark paper-only.
    3. Remaining real trades become real-only.

    Side mapping for real trades: SmartAPI uses ``transaction_type ∈ {BUY,
    SELL}``. We map ``BUY → LONG`` and ``SELL → SHORT`` only for the
    purpose of grouping; same-side longs that close shorts (and vice versa)
    end up in real-only, which is fine — they show up as "you did this
    Fortuna didn't recommend".
    """
    session_date = session_date or datetime.now()
    paper_orders = paper_orders_from_journal(journal_rows)
    real_list = list(real_trades)

    # Index real by (symbol, mapped side) for fast pop().
    real_by_bucket: dict[tuple[str, Side], list["BrokerTrade"]] = {}
    for rt in real_list:
        sym = (rt.fortuna_symbol or rt.tradingsymbol or "").upper()
        if not sym:
            continue
        side = _side_from_txn_type(rt.transaction_type)
        real_by_bucket.setdefault((sym, side), []).append(rt)

    matched: list[MatchedPair] = []
    paper_only: list[PaperOrder] = []

    for paper in sorted(paper_orders, key=lambda p: p.fill_ts or datetime.min):
        bucket = real_by_bucket.get((paper.symbol, paper.side), [])
        if not bucket:
            paper_only.append(paper)
            continue
        candidate = _closest_within(bucket, paper.fill_ts, time_window)
        if candidate is None:
            paper_only.append(paper)
            continue
        bucket.remove(candidate)
        matched.append(
            MatchedPair(
                symbol=paper.symbol,
                side=paper.side,
                qty=min(paper.qty, candidate.quantity),
                paper_price=paper.fill_price,
                real_price=candidate.fill_price,
                paper_ts=paper.fill_ts,
                real_ts=candidate.fill_time,
                strategy=paper.strategy,
                price_diff=candidate.fill_price - paper.fill_price,
            )
        )

    real_only: list["BrokerTrade"] = []
    for bucket in real_by_bucket.values():
        real_only.extend(bucket)

    return ReconciliationReport(
        session_date=session_date,
        matched=matched,
        paper_only=paper_only,
        real_only=real_only,
        paper_fills=len(paper_orders),
        real_fills=len(real_list),
    )


# ============================================================ markdown render


def render_reconciliation_markdown(report: ReconciliationReport) -> str:
    """Produce the Markdown block appended to the EOD summary."""
    lines: list[str] = []
    lines.append("## Reconciliation (paper vs broker)")
    lines.append("")
    summary = report.to_dict()
    lines.append(f"- paper fills:      {summary['paper_fills']}")
    lines.append(f"- real fills:       {summary['real_fills']}")
    lines.append(f"- matched:          {summary['matched']}")
    lines.append(
        f"- paper-only:       {summary['paper_only']}  (signals you didn't act on)"
    )
    lines.append(
        f"- real-only:        {summary['real_only']}   "
        "(trades you placed that Fortuna didn't recommend)"
    )
    lines.append(
        f"- coverage:         {summary['coverage_pct']:.2f}%  "
        "(real fills with a paper match)"
    )
    lines.append(
        f"- precision:        {summary['precision_pct']:.2f}%  "
        "(paper fills you actually placed)"
    )
    lines.append(f"- avg slippage:     {summary['avg_slippage_pct']:+.4f}%")
    lines.append("")

    if report.matched:
        lines.append("### Matched")
        lines.append("")
        lines.append(
            "| symbol | side | qty | paper px | real px | slippage % | "
            "strategy | paper ts | real ts |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|---|---|")
        for m in report.matched:
            lines.append(
                f"| {m.symbol} | {m.side.value} | {m.qty} | "
                f"{m.paper_price:.2f} | {m.real_price:.2f} | "
                f"{m.slippage_pct:+.2f}% | {m.strategy or ''} | "
                f"{_fmt_ts(m.paper_ts)} | {_fmt_ts(m.real_ts)} |"
            )
        lines.append("")

    if report.paper_only:
        lines.append("### Paper-only (you missed these)")
        lines.append("")
        lines.append("| symbol | side | qty | paper px | strategy | paper ts |")
        lines.append("|---|---|---:|---:|---|---|")
        for p in report.paper_only:
            lines.append(
                f"| {p.symbol} | {p.side.value} | {p.qty} | {p.fill_price:.2f} | "
                f"{p.strategy or ''} | {_fmt_ts(p.fill_ts)} |"
            )
        lines.append("")

    if report.real_only:
        lines.append("### Real-only (manual, no Fortuna recommendation)")
        lines.append("")
        lines.append("| symbol | side | qty | real px | producttype | real ts |")
        lines.append("|---|---|---:|---:|---|---|")
        for rt in report.real_only:
            sym = rt.fortuna_symbol or rt.tradingsymbol
            lines.append(
                f"| {sym} | {rt.transaction_type} | {rt.quantity} | {rt.fill_price:.2f} | "
                f"{rt.producttype or ''} | {_fmt_ts(rt.fill_time)} |"
            )
        lines.append("")

    return "\n".join(lines)


def append_reconciliation_to_eod(
    eod_path: str | Path,
    report: ReconciliationReport,
) -> Path:
    """Append a reconciliation block to an existing EOD Markdown file."""
    path = Path(eod_path)
    block = render_reconciliation_markdown(report)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + "\n" + block + "\n", encoding="utf-8")
    return path


# ============================================================ helpers


def _side_from_txn_type(txn: str) -> Side:
    txn = (txn or "").upper().strip()
    if txn == "SELL":
        return Side.SHORT
    return Side.LONG


def _closest_within(
    bucket: list["BrokerTrade"],
    when: Optional[datetime],
    window: timedelta,
) -> Optional["BrokerTrade"]:
    if when is None:
        return bucket[0] if bucket else None
    best = None
    best_delta: Optional[timedelta] = None
    for rt in bucket:
        if rt.fill_time is None:
            continue
        delta = abs(rt.fill_time - when)
        if delta > window:
            continue
        if best_delta is None or delta < best_delta:
            best = rt
            best_delta = delta
    return best


def _parse_iso(s) -> Optional[datetime]:
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(str(s))
    except ValueError:
        return None


def _fmt_ts(ts: Optional[datetime]) -> str:
    if ts is None:
        return "—"
    return ts.strftime("%H:%M:%S")


__all__ = [
    "MatchedPair",
    "PaperOrder",
    "ReconciliationReport",
    "append_reconciliation_to_eod",
    "paper_orders_from_journal",
    "reconcile_paper_vs_real",
    "render_reconciliation_markdown",
]
