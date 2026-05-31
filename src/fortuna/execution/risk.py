"""Pre-trade risk gate + position sizer.

Two layers, both pure functions of ``(intent, account, state)``:

- :class:`RiskGate` — composable yes/no checks.
- :class:`PositionSizer` — turns a risk verdict into a share count.

Both are configured by :class:`RiskConfig`, which is loaded from
``config/execution.yaml`` at orchestrator startup. Defaults are
deliberately conservative.

Key invariant: once a circuit-breaker (e.g. daily loss halt) trips, all
subsequent ``allow_order`` calls return ``BLOCK`` until the next session
boundary explicitly calls :meth:`RiskGate.reset_for_new_session`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml

from fortuna.execution.account import LiveAccount
from fortuna.execution.types import OrderIntent, Side


class RiskAction(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class RiskVerdict:
    """Result of :meth:`RiskGate.allow_order`."""

    action: RiskAction
    reason: Optional[str] = None
    rule: Optional[str] = None  # which rule fired (for telemetry)

    @property
    def allowed(self) -> bool:
        return self.action is RiskAction.PASS

    @classmethod
    def pass_(cls) -> "RiskVerdict":
        return cls(action=RiskAction.PASS)

    @classmethod
    def block(cls, reason: str, rule: str) -> "RiskVerdict":
        return cls(action=RiskAction.BLOCK, reason=reason, rule=rule)


@dataclass
class RiskConfig:
    """Configuration knobs for the risk gate + sizer.

    Loaded from ``config/execution.yaml``; defaults reflect a conservative
    100k-INR retail intraday account.
    """

    init_cash: float = 100_000.0
    per_symbol_max_notional: float = 25_000.0
    gross_exposure_cap_pct: float = 100.0
    daily_loss_halt_pct: float = 2.0
    per_strategy_cooldown_losses: int = 3
    position_size_method: Literal["fixed_fractional", "kelly_capped"] = (
        "fixed_fractional"
    )
    risk_per_trade_pct: float = 1.0
    kelly_cap_pct: float = 25.0
    session_window: tuple[str, str] = ("09:20", "15:10")
    min_order_qty: int = 1
    # Optional per-symbol lot-size override (e.g. {"RELIANCE.FUT": 500}).
    # Used by PositionSizer to round quantity to a multiple of lot size for
    # F&O. Unknown symbols default to 1 (cash equity).
    lot_sizes: dict[str, int] = field(default_factory=dict)


def load_risk_config(path: str | Path = "config/execution.yaml") -> RiskConfig:
    """Load :class:`RiskConfig` from a YAML file. Missing file = defaults."""
    p = Path(path)
    if not p.exists():
        return RiskConfig()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    sw = raw.get("session_window")
    if isinstance(sw, list) and len(sw) == 2:
        raw["session_window"] = (str(sw[0]), str(sw[1]))
    # Filter to known fields only so a stray YAML key doesn't blow up.
    known = {f.name for f in RiskConfig.__dataclass_fields__.values()}
    cleaned = {k: v for k, v in raw.items() if k in known}
    return RiskConfig(**cleaned)


@dataclass
class _StrategyState:
    """Per-strategy mutable book-keeping (kept inside :class:`RiskGate`)."""

    consecutive_losses: int = 0
    on_cooldown: bool = False


class RiskGate:
    """Composable pre-trade checks + circuit breakers.

    The gate is stateful: it tracks per-strategy consecutive losses, peak
    equity, and the "halted" flag once the daily-loss trigger fires.
    """

    def __init__(self, cfg: Optional[RiskConfig] = None) -> None:
        self.cfg = cfg or RiskConfig()
        self._halted: bool = False
        self._halted_reason: Optional[str] = None
        self._session_peak_equity: Optional[float] = None
        self._strategies: dict[str, _StrategyState] = {}

    # ---------------------------------------------------------------- queries
    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halted_reason(self) -> Optional[str]:
        return self._halted_reason

    def strategy_on_cooldown(self, strategy: str) -> bool:
        state = self._strategies.get(strategy)
        return bool(state and state.on_cooldown)

    # ---------------------------------------------------------------- gate
    def allow_order(
        self,
        intent: OrderIntent,
        account: LiveAccount,
        *,
        now: Optional[datetime] = None,
        mark_price: Optional[float] = None,
    ) -> RiskVerdict:
        """Composed yes/no decision. Returns the first failing rule.

        ``now`` and ``mark_price`` are explicit so callers can drive the
        gate from the bar timestamp without relying on wall-clock or a
        live tick.
        """
        # 0. Closing intents are always allowed once the position exists —
        #    we WANT to be able to flatten under any circumstance.
        if intent.close_position:
            if not account.has_position(intent.symbol):
                return RiskVerdict.block(
                    f"no open position on {intent.symbol} to close",
                    rule="no_open_position",
                )
            return RiskVerdict.pass_()

        # 1. Circuit breaker overrides everything else for new entries.
        if self._halted:
            return RiskVerdict.block(
                self._halted_reason or "trading halted for the session",
                rule="halted",
            )

        # 2. Per-strategy cooldown.
        if intent.tag and self.strategy_on_cooldown(intent.tag):
            return RiskVerdict.block(
                f"strategy {intent.tag!r} on cooldown after "
                f"{self.cfg.per_strategy_cooldown_losses} consecutive losses",
                rule="strategy_cooldown",
            )

        # 3. Session window — block new entries outside the configured
        #    open/close times. ``close_position`` already short-circuited
        #    so flattens are always permitted.
        if now is not None and not self._in_session_window(now):
            return RiskVerdict.block(
                f"outside session window {self.cfg.session_window[0]}-{self.cfg.session_window[1]}",
                rule="session_window",
            )

        # 4. Already holding a position on the symbol.
        if account.has_position(intent.symbol):
            existing = account.get_position(intent.symbol)
            assert existing is not None
            if existing.side is not intent.side:
                return RiskVerdict.block(
                    f"{intent.symbol} already has {existing.side.value} position; "
                    "close before opening opposite side",
                    rule="opposite_side_open",
                )
            return RiskVerdict.block(
                f"{intent.symbol} already has open {existing.side.value} position",
                rule="symbol_in_use",
            )

        # 5. Per-symbol notional cap.
        if mark_price is not None:
            notional = mark_price * intent.qty
            if notional > self.cfg.per_symbol_max_notional:
                return RiskVerdict.block(
                    f"notional {notional:,.0f} exceeds per-symbol cap "
                    f"{self.cfg.per_symbol_max_notional:,.0f}",
                    rule="per_symbol_max_notional",
                )

        # 6. Gross-exposure cap (as % of init cash).
        cap = self.cfg.gross_exposure_cap_pct / 100.0 * self.cfg.init_cash
        projected = account.gross_exposure + (
            mark_price * intent.qty if mark_price is not None else 0.0
        )
        if projected > cap:
            return RiskVerdict.block(
                f"projected gross exposure {projected:,.0f} exceeds cap "
                f"{cap:,.0f} ({self.cfg.gross_exposure_cap_pct:.1f}% of init cash)",
                rule="gross_exposure_cap",
            )

        # 7. Minimum quantity.
        if intent.qty < self.cfg.min_order_qty:
            return RiskVerdict.block(
                f"qty {intent.qty} below min {self.cfg.min_order_qty}",
                rule="min_order_qty",
            )

        return RiskVerdict.pass_()

    # ---------------------------------------------------------------- updates
    def update_on_bar(self, account: LiveAccount) -> None:
        """Recompute halt state from the latest equity reading.

        Called by the router after each bar's MTM. The halt-trigger uses
        intraday peak equity, *not* ``init_cash``, so a profitable day that
        gives back 2% from the high still trips.
        """
        equity = account.equity
        if self._session_peak_equity is None or equity > self._session_peak_equity:
            self._session_peak_equity = equity
        peak = self._session_peak_equity or self.cfg.init_cash
        if peak <= 0:
            return
        intraday_dd_pct = max(0.0, (peak - equity) / peak * 100.0)
        if intraday_dd_pct >= self.cfg.daily_loss_halt_pct and not self._halted:
            self._halted = True
            self._halted_reason = (
                f"daily-loss circuit breaker tripped: "
                f"DD {intraday_dd_pct:.2f}% >= {self.cfg.daily_loss_halt_pct:.2f}%"
            )

    def update_on_trade(self, strategy: Optional[str], net_pnl: float) -> None:
        """Bump per-strategy loss counter; trip cooldown at threshold."""
        if not strategy:
            return
        state = self._strategies.setdefault(strategy, _StrategyState())
        if net_pnl < 0:
            state.consecutive_losses += 1
            if state.consecutive_losses >= self.cfg.per_strategy_cooldown_losses:
                state.on_cooldown = True
        else:
            state.consecutive_losses = 0

    def reset_for_new_session(self) -> None:
        """Clear halt + cooldowns; called by orchestrator at session start."""
        self._halted = False
        self._halted_reason = None
        self._session_peak_equity = None
        for s in self._strategies.values():
            s.consecutive_losses = 0
            s.on_cooldown = False

    # ---------------------------------------------------------------- helpers
    def _in_session_window(self, now: datetime) -> bool:
        start = _parse_hhmm(self.cfg.session_window[0])
        end = _parse_hhmm(self.cfg.session_window[1])
        t = now.time()
        return start <= t <= end


class PositionSizer:
    """Turn an :class:`OrderIntent` (qty unknown) into a sized order.

    The router instantiates intents with ``qty=0`` and asks the sizer to
    populate it. Returns ``0`` when math says the position is too small to
    be worth trading (so the router skips placement).
    """

    def __init__(self, cfg: Optional[RiskConfig] = None) -> None:
        self.cfg = cfg or RiskConfig()

    def size(
        self,
        symbol: str,
        side: Side,
        mark_price: float,
        account: LiveAccount,
        *,
        stop_distance: Optional[float] = None,
        recent_wins: int = 0,
        recent_losses: int = 0,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
    ) -> int:
        if mark_price <= 0:
            return 0
        if self.cfg.position_size_method == "kelly_capped":
            qty_units = self._kelly_qty(
                account.equity, mark_price, recent_wins, recent_losses, avg_win, avg_loss
            )
        else:
            qty_units = self._fixed_fractional_qty(
                account.equity, mark_price, stop_distance
            )

        # Round to a multiple of the symbol's lot size (1 for equity).
        lot = self.cfg.lot_sizes.get(symbol, 1)
        if lot > 1:
            qty_units = (qty_units // lot) * lot
        return max(0, int(qty_units))

    # ---------------------------------------------------------------- methods
    def _fixed_fractional_qty(
        self,
        equity: float,
        mark_price: float,
        stop_distance: Optional[float],
    ) -> int:
        """``risk_per_trade_pct`` of equity risked per trade.

        If ``stop_distance`` is provided, qty = risk_budget / stop_distance.
        Otherwise falls back to qty = risk_budget / mark_price (treats the
        whole position as the risk budget).
        """
        risk_budget = equity * (self.cfg.risk_per_trade_pct / 100.0)
        if stop_distance is not None and stop_distance > 0:
            return int(risk_budget // stop_distance)
        return int(risk_budget // mark_price)

    def _kelly_qty(
        self,
        equity: float,
        mark_price: float,
        wins: int,
        losses: int,
        avg_win: float,
        avg_loss: float,
    ) -> int:
        """Kelly fraction f* = p - q/b, capped at ``kelly_cap_pct``.

        ``b`` is win/loss ratio; ``p`` is win probability, ``q`` is loss
        probability. Falls back to fixed-fractional when there's no
        reliable history (e.g. < 10 trades).
        """
        total = wins + losses
        if total < 10 or avg_loss <= 0:
            return self._fixed_fractional_qty(equity, mark_price, None)
        p = wins / total
        q = 1.0 - p
        b = avg_win / avg_loss
        if b <= 0:
            return 0
        f_star = p - q / b
        f_capped = max(0.0, min(f_star, self.cfg.kelly_cap_pct / 100.0))
        risk_budget = equity * f_capped
        return int(risk_budget // mark_price)


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


__all__ = [
    "PositionSizer",
    "RiskAction",
    "RiskConfig",
    "RiskGate",
    "RiskVerdict",
    "load_risk_config",
]
