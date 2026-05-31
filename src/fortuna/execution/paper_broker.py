"""Paper broker: virtual portfolio with next-bar-open fills.

Why next-bar-open: in live trading, a market order placed at bar-N close
gets filled at the *next* tick — which is approximately bar-N+1's open.
Filling at bar-N close would give us a small look-ahead advantage the
real broker would never reproduce, so paper fills are deliberately one
bar delayed.

Flow per ``ExecutionRouter.on_bar_closed(symbol, bar)`` invocation:

1. Drain the pending-fill queue for ``symbol`` using ``bar.open`` (+/-
   slippage from ``MarketCostModel``). Each fill updates ``LiveAccount``
   (open_or_extend for entries, close for exits) and writes an
   ``order_filled`` event to the journal.
2. Mark all open positions on this symbol to ``bar.close``.
3. New ``place_order`` calls during the same bar enqueue PENDING orders
   that will fill on the *next* ``on_new_bar`` call.

Slippage is applied symmetrically: a LONG entry pays ``open * (1 + s)``,
a SHORT entry receives ``open * (1 - s)``. Brokerage + exchange-fee +
spread are tracked as ``cost`` on the resulting :class:`Trade` so the
account's realized P&L is net.
"""

from __future__ import annotations

import itertools
import threading
from datetime import datetime
from typing import Iterable, Mapping, Optional

from fortuna.backtesting.standard.config import MarketCostModel
from fortuna.execution.account import LiveAccount
from fortuna.execution.broker import Broker
from fortuna.execution.journal import OrderJournal
from fortuna.execution.types import (
    OrderAck,
    OrderIntent,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class PaperBroker(Broker):
    """In-process paper broker; binds to one :class:`LiveAccount`.

    Thread-safety: ``place_order``, ``cancel``, and ``on_new_bar`` all
    serialize through ``self._lock``. The bound ``LiveAccount`` has its
    own lock so concurrent reads from the dashboard remain safe.
    """

    def __init__(
        self,
        account: LiveAccount,
        *,
        cost_model: Optional[MarketCostModel] = None,
        journal: Optional[OrderJournal] = None,
    ) -> None:
        self.account = account
        self.cost_model = cost_model or MarketCostModel()
        self.journal = journal
        self._id_seq = itertools.count(1)
        # ``_pending[symbol]`` is a FIFO list of (intent, ack) tuples
        # awaiting fill on the next bar of that symbol.
        self._pending: dict[str, list[tuple[OrderIntent, OrderAck]]] = {}
        self._acks: dict[str, OrderAck] = {}
        self._lock = threading.Lock()

    # ==================================================================== api
    def place_order(self, intent: OrderIntent) -> OrderAck:
        order_id = f"PAPER-{next(self._id_seq):06d}"
        ts = intent.intent_ts or datetime.now()

        reject_reason = self._pretrade_validate(intent)
        if reject_reason is not None:
            ack = OrderAck(
                order_id=order_id,
                symbol=intent.symbol,
                side=intent.side,
                status=OrderStatus.REJECTED,
                qty=intent.qty,
                reject_reason=reject_reason,
                ts=ts,
                tag=intent.tag,
            )
            with self._lock:
                self._acks[order_id] = ack
            self._journal("order_rejected", {"ack": ack.to_dict(), "intent": intent.to_dict()})
            return ack

        ack = OrderAck(
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            status=OrderStatus.PENDING,
            qty=intent.qty,
            ts=ts,
            tag=intent.tag,
        )
        with self._lock:
            self._pending.setdefault(intent.symbol, []).append((intent, ack))
            self._acks[order_id] = ack
        self._journal("order_placed", {"ack": ack.to_dict(), "intent": intent.to_dict()})
        return ack

    def cancel(self, order_id: str) -> bool:
        with self._lock:
            ack = self._acks.get(order_id)
            if ack is None or ack.status is not OrderStatus.PENDING:
                return False
            for symbol, queue in self._pending.items():
                for idx, (_, qa) in enumerate(queue):
                    if qa.order_id == order_id:
                        queue.pop(idx)
                        cancelled = OrderAck(
                            order_id=order_id,
                            symbol=symbol,
                            side=ack.side,
                            status=OrderStatus.CANCELLED,
                            qty=ack.qty,
                            ts=datetime.now(),
                            tag=ack.tag,
                        )
                        self._acks[order_id] = cancelled
                        self._journal("order_cancelled", {"ack": cancelled.to_dict()})
                        return True
        return False

    def positions(self) -> Mapping[str, Position]:
        return dict(self.account.positions)

    def cash(self) -> float:
        return self.account.cash

    def order_status(self, order_id: str) -> OrderAck | None:
        return self._acks.get(order_id)

    # ============================================================ bar driver
    def on_new_bar(
        self,
        symbol: str,
        bar: Mapping[str, float],
    ) -> Iterable[OrderAck]:
        """Fill queued orders @ bar.open, then mark-to-market @ bar.close.

        ``bar`` must contain ``ts`` (datetime), ``open``, and ``close``.
        Additional fields are ignored.
        """
        ts = _coerce_ts(bar.get("ts"))
        open_price = float(bar.get("open"))
        close_price = float(bar.get("close"))
        fills: list[OrderAck] = []

        with self._lock:
            queue = self._pending.pop(symbol, [])

        for intent, pending_ack in queue:
            try:
                fill_ack = self._fill_one(intent, pending_ack, open_price, ts)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[paper_broker] fill failed for %s order_id=%s: %s",
                    intent.symbol,
                    pending_ack.order_id,
                    exc,
                )
                rejected = OrderAck(
                    order_id=pending_ack.order_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    status=OrderStatus.REJECTED,
                    qty=intent.qty,
                    reject_reason=str(exc),
                    ts=ts,
                    tag=intent.tag,
                )
                with self._lock:
                    self._acks[pending_ack.order_id] = rejected
                self._journal("order_rejected", {"ack": rejected.to_dict()})
                fills.append(rejected)
                continue
            fills.append(fill_ack)

        self.account.mark_to_market(symbol, close_price)
        return fills

    # ================================================================ helpers
    def _fill_one(
        self,
        intent: OrderIntent,
        pending_ack: OrderAck,
        open_price: float,
        ts: datetime,
    ) -> OrderAck:
        slip = self.cost_model.slippage_rate + self.cost_model.spread_rate
        # Buyers pay up; sellers receive less. ``close_position`` flips
        # the friction direction because a long-exit is a sell and a
        # short-exit is a buy.
        if intent.side is Side.LONG and not intent.close_position:
            fill_price = open_price * (1.0 + slip)
        elif intent.side is Side.SHORT and not intent.close_position:
            fill_price = open_price * (1.0 - slip)
        elif intent.side is Side.LONG and intent.close_position:
            # closing a LONG = selling
            fill_price = open_price * (1.0 - slip)
        else:
            # closing a SHORT = buying back
            fill_price = open_price * (1.0 + slip)

        if intent.close_position:
            cost_per_round_trip_pct = self.cost_model.brokerage_rate + self.cost_model.exchange_fee_rate
            cost_abs = fill_price * intent.qty * cost_per_round_trip_pct * 2.0
            trade = self.account.close(
                symbol=intent.symbol,
                qty=intent.qty,
                fill_price=fill_price,
                ts=ts,
                cost=cost_abs,
            )
            if trade is None:
                raise RuntimeError(
                    f"close called but no position open on {intent.symbol}"
                )
        else:
            self.account.open_or_extend(
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                fill_price=fill_price,
                ts=ts,
                tag=intent.tag,
            )

        ack = OrderAck(
            order_id=pending_ack.order_id,
            symbol=intent.symbol,
            side=intent.side,
            status=OrderStatus.FILLED,
            qty=intent.qty,
            filled_qty=intent.qty,
            filled_price=fill_price,
            ts=ts,
            tag=intent.tag,
        )
        with self._lock:
            self._acks[pending_ack.order_id] = ack
        self._journal(
            "order_filled",
            {
                "ack": ack.to_dict(),
                "intent": intent.to_dict(),
                "open_price": open_price,
                "slip_rate": slip,
            },
        )
        return ack

    def _pretrade_validate(self, intent: OrderIntent) -> Optional[str]:
        if intent.qty <= 0:
            return "qty must be > 0"
        if intent.order_type is not OrderType.MARKET:
            return f"order_type {intent.order_type.value} not yet supported by PaperBroker"
        if intent.close_position:
            pos = self.account.get_position(intent.symbol)
            if pos is None:
                return "no open position to close"
            if pos.qty < intent.qty:
                return f"close qty {intent.qty} exceeds open qty {pos.qty}"
            return None
        existing = self.account.get_position(intent.symbol)
        if existing is not None and existing.side is not intent.side:
            return (
                f"symbol {intent.symbol} already has {existing.side.value} position; "
                "close before opening opposite side"
            )
        return None

    def _journal(self, event_type: str, payload: dict) -> None:
        if self.journal is None:
            return
        try:
            self.journal.write(event_type, payload)
        except Exception:  # noqa: BLE001
            logger.exception("[paper_broker] journal write failed for %s", event_type)


def _coerce_ts(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if ts is None:
        return datetime.now()
    try:
        import pandas as pd
        return pd.Timestamp(ts).to_pydatetime()
    except Exception:  # noqa: BLE001
        return datetime.now()


__all__ = ["PaperBroker"]
