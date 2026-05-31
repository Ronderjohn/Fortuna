"""Broker interface.

Implementations:

- :class:`PaperBroker` — virtual portfolio, next-bar-open fills.
- :class:`SmartAPIBroker` — stub for real Angel One trading (Tier 1
  follow-up; raises ``NotImplementedError`` until enabled).

The interface deliberately stays tiny — placing, cancelling, querying.
Per-bar bookkeeping (queued fills, mark-to-market) lives on concrete
implementations because real brokers don't expose those hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from fortuna.execution.types import OrderAck, OrderIntent, Position


class Broker(ABC):
    """Abstract broker. Concrete implementations bind a ``LiveAccount``."""

    @abstractmethod
    def place_order(self, intent: OrderIntent) -> OrderAck:
        """Submit an order; return the broker's acknowledgement.

        ``OrderStatus.PENDING`` means accepted but not yet filled (live
        and paper brokers both behave this way). ``REJECTED`` means the
        broker refused the order (validation failure, insufficient cash,
        etc.) — the router should not retry.
        """

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if it was found + cancelled."""

    @abstractmethod
    def positions(self) -> Mapping[str, Position]:
        """Snapshot of currently-open positions, keyed by symbol."""

    @abstractmethod
    def cash(self) -> float:
        """Currently available cash for new positions."""

    @abstractmethod
    def order_status(self, order_id: str) -> OrderAck | None:
        """Latest known ack for an order (None if unknown)."""

    # Optional hook — only paper-style brokers implement it. Real brokers
    # receive fills asynchronously via websockets, so the default does
    # nothing.
    def on_new_bar(
        self,
        symbol: str,
        bar: Mapping[str, float],
    ) -> Iterable[OrderAck]:
        """Notify the broker that a new bar has closed for ``symbol``.

        Paper brokers use this to (a) fill any queued pending orders at
        the new bar's ``open`` and (b) mark-to-market open positions at
        the new bar's ``close``. Real brokers ignore it (they're driven
        by the websocket fill stream).

        Returns an iterable of ``OrderAck``s describing any fills that
        occurred during this bar.
        """
        return ()


__all__ = ["Broker"]
