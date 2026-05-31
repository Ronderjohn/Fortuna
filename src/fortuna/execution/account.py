"""Live portfolio tracking: cash, positions, realized/unrealized P&L, max-DD.

Modelled on the existing Phase-1 :class:`fortuna.paper.account.PaperAccount`
but evaluated on each new bar (not just at trade close) so the dashboard
can show a live equity curve.

Cash model: simplified Indian-intraday-style. We track ``init_cash`` and
``realized_pnl`` separately; ``cash`` available for new positions equals
``init_cash + realized_pnl``. Entries/exits don't change ``cash`` directly
(broker margin handles position commitments). Unrealized P&L is computed
on-demand from the latest mark-to-market price.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from fortuna.execution.types import Holding, Position, Side, Trade

if TYPE_CHECKING:
    from fortuna.execution.smartapi_account import (
        AccountSnapshot,
    )


@dataclass(frozen=True)
class EquityPoint:
    """One sample of the intraday equity curve."""

    ts: datetime
    cash: float
    unrealized: float
    realized: float
    equity: float
    drawdown_pct: float


class LiveAccount:
    """Thread-safe portfolio state for the live (paper) execution layer.

    Designed to be shared between the broker (writes on fills + MTM) and
    the dashboard (reads positions + equity curve). Both touch points
    take the same lock.
    """

    def __init__(self, init_cash: float) -> None:
        self.init_cash = float(init_cash)
        self.realized_pnl = 0.0
        self.positions: dict[str, Position] = {}
        # Long-term DEMAT holdings mirrored from the broker; display-only by
        # default (the paper router will not auto-flatten them unless
        # ``allow_holding_exits`` is set in the router config).
        self.holdings: dict[str, Holding] = {}
        # Provenance of the most recent broker sync (None if never synced).
        self.broker_synced_at: Optional[datetime] = None
        self.broker_funds_raw: Optional[dict] = None
        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []
        self.peak_equity = float(init_cash)
        self._mtm_prices: dict[str, float] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- cash
    @property
    def cash(self) -> float:
        """Cash available for new positions = init_cash + realized P&L."""
        return self.init_cash + self.realized_pnl

    def unrealized_pnl(self) -> float:
        """Sum of MTM gains/losses on all open positions."""
        total = 0.0
        for pos in self.positions.values():
            mark = self._mtm_prices.get(pos.symbol, pos.avg_price)
            total += pos.unrealized_pnl(mark)
        return total

    @property
    def equity(self) -> float:
        return self.cash + self.unrealized_pnl()

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        eq = self.equity
        return max(0.0, (self.peak_equity - eq) / self.peak_equity * 100.0)

    @property
    def gross_exposure(self) -> float:
        """Sum of |notional| across all open positions, MTM."""
        total = 0.0
        for pos in self.positions.values():
            mark = self._mtm_prices.get(pos.symbol, pos.avg_price)
            total += abs(pos.notional(mark))
        return total

    # -------------------------------------------------------------- position
    def open_or_extend(
        self,
        symbol: str,
        side: Side,
        qty: int,
        fill_price: float,
        ts: datetime,
        tag: Optional[str] = None,
    ) -> Position:
        """Add a new position or pyramid into an existing same-side one.

        Raises ``ValueError`` if the symbol already holds a position on the
        opposite side — callers must close first. v1 keeps the semantics
        intentionally tight; we can relax this later for pyramid / hedge
        strategies.
        """
        with self._lock:
            existing = self.positions.get(symbol)
            if existing is None:
                pos = Position(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    avg_price=fill_price,
                    entry_ts=ts,
                    entry_tag=tag,
                    last_mtm_price=fill_price,
                )
                self.positions[symbol] = pos
                self._mtm_prices[symbol] = fill_price
                return pos
            if existing.side is side:
                new_qty = existing.qty + qty
                new_avg = (
                    existing.avg_price * existing.qty + fill_price * qty
                ) / new_qty
                pos = Position(
                    symbol=symbol,
                    side=side,
                    qty=new_qty,
                    avg_price=new_avg,
                    entry_ts=existing.entry_ts,
                    entry_tag=existing.entry_tag,
                    last_mtm_price=fill_price,
                )
                self.positions[symbol] = pos
                return pos
            raise ValueError(
                f"cannot extend {existing.side.value} position with "
                f"{side.value} order on {symbol}; close first"
            )

    def close(
        self,
        symbol: str,
        qty: int,
        fill_price: float,
        ts: datetime,
        cost: float = 0.0,
    ) -> Optional[Trade]:
        """Close (or partially close) an existing position; return the trade.

        ``cost`` is round-trip transaction friction in absolute currency
        units; subtracted from gross P&L to compute ``net_pnl`` on the
        trade and to update ``realized_pnl`` on the account.

        Returns ``None`` if no position exists or ``qty`` exceeds the open
        position; the caller is expected to validate beforehand via the
        risk gate.
        """
        with self._lock:
            existing = self.positions.get(symbol)
            if existing is None or qty <= 0 or qty > existing.qty:
                return None

            if existing.side is Side.LONG:
                gross = (fill_price - existing.avg_price) * qty
            else:
                gross = (existing.avg_price - fill_price) * qty

            trade = Trade(
                symbol=symbol,
                side=existing.side,
                qty=qty,
                entry_price=existing.avg_price,
                entry_ts=existing.entry_ts,
                exit_price=fill_price,
                exit_ts=ts,
                gross_pnl=gross,
                cost=cost,
                tag=existing.entry_tag,
            )
            self.trades.append(trade)
            self.realized_pnl += trade.net_pnl

            remaining = existing.qty - qty
            if remaining == 0:
                self.positions.pop(symbol, None)
                self._mtm_prices.pop(symbol, None)
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=existing.side,
                    qty=remaining,
                    avg_price=existing.avg_price,
                    entry_ts=existing.entry_ts,
                    entry_tag=existing.entry_tag,
                    last_mtm_price=fill_price,
                )
            self._refresh_peak_unlocked()
            return trade

    def mark_to_market(self, symbol: str, mark_price: float) -> None:
        with self._lock:
            self._mtm_prices[symbol] = float(mark_price)
            pos = self.positions.get(symbol)
            if pos is not None:
                self.positions[symbol] = Position(
                    symbol=pos.symbol,
                    side=pos.side,
                    qty=pos.qty,
                    avg_price=pos.avg_price,
                    entry_ts=pos.entry_ts,
                    entry_tag=pos.entry_tag,
                    last_mtm_price=float(mark_price),
                )
            self._refresh_peak_unlocked()

    def snapshot(self, ts: datetime) -> EquityPoint:
        """Append + return the current equity-curve sample."""
        with self._lock:
            unrealized = self.unrealized_pnl()
            equity = self.cash + unrealized
            self.peak_equity = max(self.peak_equity, equity)
            dd = (
                max(0.0, (self.peak_equity - equity) / self.peak_equity * 100.0)
                if self.peak_equity > 0
                else 0.0
            )
            point = EquityPoint(
                ts=ts,
                cash=self.cash,
                unrealized=unrealized,
                realized=self.realized_pnl,
                equity=equity,
                drawdown_pct=dd,
            )
            self.equity_curve.append(point)
            return point

    # ----------------------------------------------------------------- introspection
    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def has_holding(self, symbol: str) -> bool:
        return symbol in self.holdings

    def get_holding(self, symbol: str) -> Optional[Holding]:
        return self.holdings.get(symbol)

    def held_or_open(self, symbol: str) -> bool:
        """True if we already hold the symbol (either intraday or DEMAT)."""
        return symbol in self.positions or symbol in self.holdings

    # ----------------------------------------------------------------- broker sync
    def seed_from_broker(
        self,
        snapshot: "AccountSnapshot",
        *,
        reset_init_cash: bool = True,
        merge: bool = False,
    ) -> dict:
        """Hydrate the account from a fresh :class:`AccountSnapshot`.

        Parameters
        ----------
        snapshot
            Result of :meth:`SmartAPIAccountReader.full_snapshot`. Only the
            ``funds``, ``positions``, and ``holdings`` slices are consumed
            here; orders/trades feed reconciliation downstream.
        reset_init_cash
            When True (default) replaces ``init_cash`` with the broker's
            available cash + collateral. Set to False to keep an existing
            simulated cash baseline (useful when running pure stress tests).
        merge
            When True, broker positions extend any existing simulated
            positions of the same side rather than replacing them. Default
            False — broker state wins on conflict, which is what an "advisor"
            mode wants.

        Returns a small dict summarising what was applied (for logging /
        toast notifications in the dashboard).
        """
        from fortuna.execution.smartapi_account import AccountSnapshot  # local to dodge cycle

        if not isinstance(snapshot, AccountSnapshot):  # pragma: no cover — defensive
            raise TypeError("seed_from_broker expects an AccountSnapshot")

        applied = {
            "positions_added": 0,
            "positions_skipped": 0,
            "holdings_added": 0,
            "init_cash_before": self.init_cash,
            "init_cash_after": self.init_cash,
            "errors": list(snapshot.errors),
        }
        with self._lock:
            if reset_init_cash:
                # ``available_cash`` is the most defensible "buying power"
                # baseline; collateral is intentionally excluded so the risk
                # gate doesn't accidentally size on pledged stock.
                self.init_cash = float(
                    snapshot.funds.available_cash
                    if snapshot.funds.available_cash > 0
                    else snapshot.funds.net
                )
                self.peak_equity = max(self.peak_equity, self.init_cash)
                applied["init_cash_after"] = self.init_cash
            self.broker_synced_at = snapshot.taken_at
            self.broker_funds_raw = dict(snapshot.funds.raw) if snapshot.funds.raw else None

            # Holdings — always idempotent (key = fortuna_symbol or tradingsymbol).
            new_holdings: dict[str, Holding] = {}
            for h in snapshot.holdings:
                key = h.fortuna_symbol or h.tradingsymbol
                if not key:
                    continue
                new_holdings[key] = Holding(
                    symbol=key,
                    tradingsymbol=h.tradingsymbol,
                    qty=int(h.quantity),
                    avg_price=float(h.avg_price),
                    last_price=h.last_price,
                    pnl=h.pnl,
                    exchange=h.exchange or "NSE",
                    source="broker",
                )
                applied["holdings_added"] += 1
            self.holdings = new_holdings

            # Positions — careful here, we may already have simulated paper
            # positions in play. Default policy: clear paper positions then
            # mirror broker. Merge=True keeps both (broker overrides on conflict).
            if not merge:
                self.positions.clear()
                self._mtm_prices.clear()

            for bp in snapshot.positions:
                key = bp.fortuna_symbol or bp.tradingsymbol
                if not key or bp.quantity <= 0:
                    applied["positions_skipped"] += 1
                    continue
                existing = self.positions.get(key)
                if existing is not None and merge:
                    if existing.side is not bp.side:
                        applied["positions_skipped"] += 1
                        continue
                    new_qty = existing.qty + bp.quantity
                    new_avg = (
                        existing.avg_price * existing.qty + bp.avg_price * bp.quantity
                    ) / new_qty
                    self.positions[key] = Position(
                        symbol=key,
                        side=bp.side,
                        qty=new_qty,
                        avg_price=new_avg,
                        entry_ts=existing.entry_ts,
                        entry_tag=existing.entry_tag,
                        last_mtm_price=bp.last_price or bp.avg_price,
                    )
                else:
                    self.positions[key] = Position(
                        symbol=key,
                        side=bp.side,
                        qty=int(bp.quantity),
                        avg_price=float(bp.avg_price),
                        entry_ts=snapshot.taken_at,
                        entry_tag=(
                            f"broker:{bp.producttype.lower()}"
                            if bp.producttype else "broker"
                        ),
                        last_mtm_price=bp.last_price or bp.avg_price,
                    )
                if bp.last_price is not None:
                    self._mtm_prices[key] = float(bp.last_price)
                applied["positions_added"] += 1

            self._refresh_peak_unlocked()
        return applied

    def reconcile_with_broker(self, snapshot: "AccountSnapshot") -> dict:
        """Compare local state to a fresh snapshot; return a drift report.

        Read-only. Useful for the dashboard's "Sync status" badge to alert
        when the simulator has drifted from reality (e.g. you placed a
        manual order outside Fortuna).
        """
        drift = {
            "missing_in_local": [],   # broker has it, we don't
            "missing_in_broker": [],  # we have it, broker doesn't
            "qty_mismatch": [],
            "synced_at": snapshot.taken_at.isoformat(),
        }
        with self._lock:
            broker_by_symbol = {
                (bp.fortuna_symbol or bp.tradingsymbol): bp for bp in snapshot.positions
            }
            for key, pos in self.positions.items():
                bp = broker_by_symbol.get(key)
                if bp is None:
                    if pos.entry_tag and pos.entry_tag.startswith("broker"):
                        drift["missing_in_broker"].append(key)
                    continue
                if bp.quantity != pos.qty or bp.side is not pos.side:
                    drift["qty_mismatch"].append({
                        "symbol": key,
                        "local_qty": pos.qty,
                        "local_side": pos.side.value,
                        "broker_qty": bp.quantity,
                        "broker_side": bp.side.value,
                    })
            for key in broker_by_symbol:
                if key not in self.positions:
                    drift["missing_in_local"].append(key)
        return drift

    def to_summary(self) -> dict:
        """Compact dict for the dashboard / EOD report."""
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        losses = sum(1 for t in self.trades if t.net_pnl < 0)
        total = len(self.trades)
        win_rate = (wins / total * 100.0) if total else 0.0
        gross_profit = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_loss = -sum(t.net_pnl for t in self.trades if t.net_pnl < 0)
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        return {
            "init_cash": self.init_cash,
            "equity": round(self.equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl(), 2),
            "peak_equity": round(self.peak_equity, 2),
            "max_drawdown_pct": round(self.drawdown_pct, 4),
            "open_positions": len(self.positions),
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": (
                round(pf, 4) if pf != float("inf") else None
            ),
            "gross_exposure": round(self.gross_exposure, 2),
        }

    # ------------------------------------------------------------------ private
    def _refresh_peak_unlocked(self) -> None:
        """Recompute peak equity; caller must already hold ``self._lock``."""
        equity = self.cash + sum(
            pos.unrealized_pnl(self._mtm_prices.get(sym, pos.avg_price))
            for sym, pos in self.positions.items()
        )
        if equity > self.peak_equity:
            self.peak_equity = equity


__all__ = ["EquityPoint", "LiveAccount"]
