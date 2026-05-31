"""Common dataclasses shared across the execution layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    """Position / order side. ``LONG`` = buy-to-open, ``SHORT`` = sell-to-open."""

    LONG = "LONG"
    SHORT = "SHORT"

    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class OrderType(str, Enum):
    """Order entry type. v1 supports MARKET only; LIMIT/SL are reserved."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"  # reserved; not honoured by PaperBroker yet
    STOP = "STOP"    # reserved


class OrderStatus(str, Enum):
    PENDING = "PENDING"     # accepted, awaiting fill on next bar
    FILLED = "FILLED"       # fully filled
    REJECTED = "REJECTED"   # broker rejected (e.g. insufficient cash, invalid)
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class OrderIntent:
    """Caller's request to enter or exit a position.

    The router builds these from ``LiveSignal``s and hands them to the broker.
    """

    symbol: str
    side: Side
    qty: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    tag: Optional[str] = None  # strategy name or any router-defined tag
    intent_ts: Optional[datetime] = None
    # ``close_position`` marks the intent as an exit on an existing position
    # (no new cash committed beyond margin); the broker uses this to compute
    # realized PnL on fill.
    close_position: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        if self.intent_ts is not None:
            d["intent_ts"] = self.intent_ts.isoformat()
        return d


@dataclass(frozen=True)
class OrderAck:
    """Broker's response to ``place_order``.

    ``PENDING`` means accepted but not yet filled. ``FILLED`` carries the
    actual ``filled_price`` and ``filled_qty``. ``REJECTED`` carries a
    ``reject_reason``.
    """

    order_id: str
    symbol: str
    side: Side
    status: OrderStatus
    qty: int
    filled_qty: int = 0
    filled_price: Optional[float] = None
    reject_reason: Optional[str] = None
    ts: Optional[datetime] = None
    tag: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["status"] = self.status.value
        if self.ts is not None:
            d["ts"] = self.ts.isoformat()
        return d


@dataclass
class Position:
    """One open position tracked by :class:`LiveAccount`.

    v1 holds at most one position per symbol (no concurrent long+short, no
    multi-strategy stacking). The owning strategy is recorded in
    ``entry_tag`` for per-strategy P&L attribution downstream.
    """

    symbol: str
    side: Side
    qty: int
    avg_price: float
    entry_ts: datetime
    entry_tag: Optional[str] = None
    last_mtm_price: Optional[float] = None

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side is Side.LONG:
            return (mark_price - self.avg_price) * self.qty
        return (self.avg_price - mark_price) * self.qty

    def notional(self, mark_price: float) -> float:
        return mark_price * self.qty


@dataclass
class Holding:
    """Long-term DEMAT holding mirrored from the broker.

    Distinct from :class:`Position` (which is an *intraday* tradeable book
    the paper broker can flatten). Holdings are display-only by default —
    they participate in the dashboard's "What to do now" join with live
    signals but the router will not auto-place exit orders against them.
    """

    symbol: str
    tradingsymbol: str
    qty: int
    avg_price: float
    last_price: Optional[float] = None
    pnl: Optional[float] = None
    exchange: str = "NSE"
    source: str = "broker"  # "broker" for snapshotted; "manual" if entered locally

    def market_value(self) -> float:
        return (self.last_price or self.avg_price) * self.qty


@dataclass(frozen=True)
class Trade:
    """A completed round-trip recorded by :class:`LiveAccount` on close."""

    symbol: str
    side: Side
    qty: int
    entry_price: float
    entry_ts: datetime
    exit_price: float
    exit_ts: datetime
    gross_pnl: float
    cost: float
    tag: Optional[str] = None

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.cost

    @property
    def return_pct(self) -> float:
        notional = self.entry_price * self.qty
        if notional == 0:
            return 0.0
        return self.net_pnl / notional * 100.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "entry_ts": self.entry_ts.isoformat(),
            "exit_price": self.exit_price,
            "exit_ts": self.exit_ts.isoformat(),
            "gross_pnl": self.gross_pnl,
            "cost": self.cost,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "tag": self.tag,
        }


__all__ = [
    "Holding",
    "OrderAck",
    "OrderIntent",
    "OrderStatus",
    "OrderType",
    "Position",
    "Side",
    "Trade",
]
