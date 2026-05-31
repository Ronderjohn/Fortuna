"""Stub for the real Angel One broker — intentionally not yet implemented.

The interface stays in place so the router can be swapped onto a live
broker by a single config flag once real-money trading is enabled. Until
then every method raises ``NotImplementedError`` with an explicit message
so it can never silently route real orders.

When this is implemented in a follow-up plan it will:

- Use the existing :class:`~fortuna.data.sources.smartapi_session.SmartAPISession`
  for authenticated SmartConnect calls.
- Map :class:`OrderIntent` to ``placeOrder({variety, tradingsymbol,
  symboltoken, exchange, transactiontype, ordertype, producttype,
  quantity, price, ...})``.
- Consume fills via SmartWebSocketV2's order-update stream instead of
  the synchronous ``on_new_bar`` polling the paper broker uses.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from fortuna.execution.broker import Broker
from fortuna.execution.types import OrderAck, OrderIntent, Position

_DISABLED_MSG = (
    "live SmartAPI trading is intentionally disabled in this build; "
    "use PaperBroker (settings.execution_enabled with broker='paper')"
)


class SmartAPIBroker(Broker):
    """Placeholder concrete broker. Every call raises ``NotImplementedError``."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(_DISABLED_MSG)

    def place_order(self, intent: OrderIntent) -> OrderAck:  # pragma: no cover
        raise NotImplementedError(_DISABLED_MSG)

    def cancel(self, order_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError(_DISABLED_MSG)

    def positions(self) -> Mapping[str, Position]:  # pragma: no cover
        raise NotImplementedError(_DISABLED_MSG)

    def cash(self) -> float:  # pragma: no cover
        raise NotImplementedError(_DISABLED_MSG)

    def order_status(self, order_id: str) -> OrderAck | None:  # pragma: no cover
        raise NotImplementedError(_DISABLED_MSG)

    def on_new_bar(  # pragma: no cover
        self,
        symbol: str,
        bar: Mapping[str, float],
    ) -> Iterable[OrderAck]:
        raise NotImplementedError(_DISABLED_MSG)


__all__ = ["SmartAPIBroker"]
