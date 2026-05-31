"""Per-bar execution orchestrator.

Sits between :func:`fortuna.app.live_signals.compute_live_signals_with_rl`
(which produces ``{strategy_name: LiveSignal}`` on every bar close) and the
broker. The flow is deliberately linear:

1. Process broker fills + MTM from the new bar (``broker.on_new_bar``).
2. Update the risk gate's halt state from the new equity reading.
3. For each ``LiveSignal`` whose ``action`` is actionable (BUY / SELL /
   EXIT_*) and that isn't ``rl_suppressed``:
   a. Build an :class:`OrderIntent` (entry or close depending on action).
   b. Ask :class:`RiskGate` for permission.
   c. Ask :class:`PositionSizer` to compute qty (entries only).
   d. Hand to broker.
   e. Emit a monitor event.
4. Snapshot the equity curve for the dashboard.

The router doesn't know about the strategy DSL; everything it needs comes
from the ``LiveSignal`` payload and the latest bar OHLCV dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Mapping, Optional

from fortuna.execution.account import LiveAccount
from fortuna.execution.broker import Broker
from fortuna.execution.journal import OrderJournal
from fortuna.execution.monitor import EventSeverity, ExecutionMonitor, write_eod_summary
from fortuna.execution.risk import PositionSizer, RiskConfig, RiskGate
from fortuna.execution.types import OrderIntent, OrderType, Side
from fortuna.utils.logging import get_logger

if TYPE_CHECKING:
    from fortuna.app.live_signals import LiveSignal


logger = get_logger(__name__)


_ACTIONABLE = {"BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"}


@dataclass
class RouterStats:
    """Aggregate counters surfaced to the dashboard / tests."""

    bars_processed: int = 0
    signals_seen: int = 0
    orders_placed: int = 0
    orders_rejected: int = 0
    risk_blocks: int = 0
    rl_suppressed: int = 0
    circuit_breaker_trips: int = 0
    fills: int = 0
    cooldown_skips: int = 0


class ExecutionRouter:
    """The orchestrator. One per ``LiveAccount``.

    Typical lifecycle: instantiated once by the session engine when
    ``settings.execution_enabled=True``, then called once per bar close
    via :meth:`on_bar_closed`.
    """

    def __init__(
        self,
        *,
        broker: Broker,
        account: LiveAccount,
        risk_gate: RiskGate,
        sizer: PositionSizer,
        monitor: ExecutionMonitor,
        journal: Optional[OrderJournal] = None,
        config: Optional[RiskConfig] = None,
    ) -> None:
        self.broker = broker
        self.account = account
        self.risk_gate = risk_gate
        self.sizer = sizer
        self.monitor = monitor
        self.journal = journal
        self.config = config or risk_gate.cfg
        self.stats = RouterStats()
        # Track which strategy currently owns the position on each symbol
        # so per-strategy cooldown counters fire on the right close.
        self._owners: dict[str, str] = {}

    # =================================================================== api
    def on_bar_closed(
        self,
        symbol: str,
        bar: Mapping[str, float],
        signals: Mapping[str, "LiveSignal"],
    ) -> None:
        """Drive the full per-bar pipeline.

        ``bar`` must include ``ts`` (datetime), ``open``, ``high``, ``low``,
        ``close``, ``volume``. Anything extra is ignored.
        """
        self.stats.bars_processed += 1
        bar_ts = _coerce_ts(bar.get("ts"))
        bar_close = float(bar.get("close"))

        # 1. Fill pending orders + MTM from this bar.
        previously_halted = self.risk_gate.is_halted
        fills = list(self.broker.on_new_bar(symbol, bar))
        for ack in fills:
            if ack.status.value == "FILLED":
                self.stats.fills += 1
                self._emit_fill(ack, bar_ts)
                # If this fill closed a position, bump the strategy's
                # cooldown counter using the most recent trade.
                if self.account.trades:
                    last_trade = self.account.trades[-1]
                    if last_trade.exit_ts == ack.ts:
                        self.risk_gate.update_on_trade(last_trade.tag, last_trade.net_pnl)
                        self._owners.pop(ack.symbol, None)
            elif ack.status.value == "REJECTED":
                self.stats.orders_rejected += 1
                self._emit_rejection(ack, bar_ts)

        # 2. Update halt state from the freshly-MTM'd equity.
        self.risk_gate.update_on_bar(self.account)
        if self.risk_gate.is_halted and not previously_halted:
            self.stats.circuit_breaker_trips += 1
            self._emit_breaker_tripped(bar_ts)

        # 3. Process each strategy's LiveSignal.
        for strategy_name, signal in signals.items():
            self.stats.signals_seen += 1

            if signal.rl_suppressed:
                self.stats.rl_suppressed += 1
                self.monitor.publish(
                    "signal_rl_suppressed",
                    severity=EventSeverity.INFO,
                    symbol=symbol,
                    strategy=strategy_name,
                    message=f"RL veto on {signal.action}",
                    rl_action=signal.rl_action,
                    rl_run_id=signal.rl_run_id,
                )
                continue

            if signal.action not in _ACTIONABLE:
                continue

            self._handle_actionable_signal(
                strategy_name=strategy_name,
                symbol=symbol,
                signal=signal,
                bar_ts=bar_ts,
                bar_close=bar_close,
            )

        # 4. Snapshot equity for the dashboard.
        self.account.snapshot(bar_ts)

    def reset_for_new_session(self) -> None:
        """Clear halt + cooldowns; called by the session engine at open."""
        self.risk_gate.reset_for_new_session()
        self._owners.clear()

    def write_eod_summary(self) -> Optional[str]:
        """Persist the Markdown EOD report; returns the file path (or None)."""
        try:
            path = write_eod_summary(self.account, self.monitor)
        except Exception:  # noqa: BLE001
            logger.exception("[router] failed to write EOD summary")
            return None
        self.monitor.publish(
            "eod_summary",
            severity=EventSeverity.INFO,
            message=f"EOD summary written to {path}",
            **self.account.to_summary(),
        )
        return str(path)

    # =============================================================== handlers
    def _handle_actionable_signal(
        self,
        *,
        strategy_name: str,
        symbol: str,
        signal: "LiveSignal",
        bar_ts: datetime,
        bar_close: float,
    ) -> None:
        is_exit = signal.action.startswith("EXIT_")
        side = self._signal_side(signal, is_exit=is_exit)
        if side is None:
            return

        self.monitor.publish(
            "signal_fired",
            severity=EventSeverity.INFO,
            symbol=symbol,
            strategy=strategy_name,
            message=f"{signal.action} @ {bar_close:.2f}",
            bar_close=bar_close,
            regime=signal.regime,
            rl_confidence=signal.rl_confidence,
        )

        intent = OrderIntent(
            symbol=symbol,
            side=side,
            qty=0,  # filled in below for entries; closes use position qty
            order_type=OrderType.MARKET,
            tag=strategy_name,
            intent_ts=bar_ts,
            close_position=is_exit,
        )

        if is_exit:
            existing = self.account.get_position(symbol)
            if existing is None:
                logger.debug(
                    "[router] EXIT signal but no position on %s (strategy=%s) - skipping",
                    symbol, strategy_name,
                )
                return
            intent = OrderIntent(
                symbol=symbol,
                side=existing.side,  # close intent's side = original position side
                qty=existing.qty,
                order_type=OrderType.MARKET,
                tag=strategy_name,
                intent_ts=bar_ts,
                close_position=True,
            )
        else:
            qty = self.sizer.size(
                symbol=symbol,
                side=side,
                mark_price=bar_close,
                account=self.account,
            )
            if qty <= 0:
                self.monitor.publish(
                    "size_zero",
                    severity=EventSeverity.WARN,
                    symbol=symbol,
                    strategy=strategy_name,
                    message="position size resolved to 0",
                )
                return
            intent = OrderIntent(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=OrderType.MARKET,
                tag=strategy_name,
                intent_ts=bar_ts,
                close_position=False,
            )

        verdict = self.risk_gate.allow_order(
            intent,
            self.account,
            now=bar_ts,
            mark_price=bar_close,
        )
        if not verdict.allowed:
            self.stats.risk_blocks += 1
            severity = (
                EventSeverity.CRITICAL
                if verdict.rule == "halted"
                else EventSeverity.WARN
            )
            self.monitor.publish(
                "risk_breach",
                severity=severity,
                symbol=symbol,
                strategy=strategy_name,
                message=verdict.reason or "blocked",
                rule=verdict.rule,
            )
            if verdict.rule == "strategy_cooldown":
                self.stats.cooldown_skips += 1
            if self.journal is not None:
                self.journal.write(
                    "risk_breach",
                    {
                        "intent": intent.to_dict(),
                        "reason": verdict.reason,
                        "rule": verdict.rule,
                    },
                )
            return

        ack = self.broker.place_order(intent)
        self.stats.orders_placed += 1
        if not is_exit:
            self._owners[symbol] = strategy_name
        self.monitor.publish(
            "order_placed",
            severity=EventSeverity.INFO,
            symbol=symbol,
            strategy=strategy_name,
            message=f"{intent.side.value} {intent.qty}",
            order_id=ack.order_id,
            status=ack.status.value,
            close_position=intent.close_position,
        )

    # ----------------------------------------------------------------- helpers
    def _signal_side(self, signal: "LiveSignal", *, is_exit: bool) -> Optional[Side]:
        if is_exit:
            if signal.action == "EXIT_LONG":
                return Side.LONG
            if signal.action == "EXIT_SHORT":
                return Side.SHORT
            return None
        if signal.action == "BUY":
            return Side.LONG
        if signal.action == "SELL":
            return Side.SHORT
        return None

    def _emit_fill(self, ack, ts: datetime) -> None:
        self.monitor.publish(
            "order_filled",
            severity=EventSeverity.INFO,
            symbol=ack.symbol,
            strategy=ack.tag,
            message=f"{ack.side.value} {ack.filled_qty} @ {ack.filled_price:.2f}",
            order_id=ack.order_id,
            filled_price=ack.filled_price,
        )

    def _emit_rejection(self, ack, ts: datetime) -> None:
        self.monitor.publish(
            "order_rejected",
            severity=EventSeverity.WARN,
            symbol=ack.symbol,
            strategy=ack.tag,
            message=ack.reject_reason or "rejected",
            order_id=ack.order_id,
        )

    def _emit_breaker_tripped(self, ts: datetime) -> None:
        self.monitor.publish(
            "circuit_breaker_tripped",
            severity=EventSeverity.CRITICAL,
            message=self.risk_gate.halted_reason or "daily loss halt",
            **self.account.to_summary(),
        )


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


__all__ = ["ExecutionRouter", "RouterStats"]
