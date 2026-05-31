"""Throttle and quiet-hours policy for advisory notifications."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from fortuna.utils.nse_session import in_nse_session, nse_session_date


@dataclass
class NotificationPolicy:
    max_per_symbol_per_session: int = 10
    min_interval_seconds: int = 300
    quiet_hours_enabled: bool = True


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""


@dataclass
class _SymbolState:
    count: int = 0
    last_sent_monotonic: float = 0.0
    session_date: Optional[date] = None


@dataclass
class NotificationPolicyEngine:
    """Per-symbol throttle state for advisory notifications."""

    policy: NotificationPolicy = field(default_factory=NotificationPolicy)
    _symbols: dict[str, _SymbolState] = field(default_factory=dict)
    _current_session_date: Optional[date] = None

    def evaluate(
        self,
        *,
        symbol: str,
        bar_time: Optional[datetime],
        now: Optional[float] = None,
    ) -> PolicyDecision:
        if self.policy.quiet_hours_enabled and not in_nse_session(bar_time):
            return PolicyDecision(allowed=False, reason="quiet_hours")

        session_date = nse_session_date(bar_time)
        self._maybe_reset_session(session_date)

        sym_key = symbol.upper().strip()
        state = self._symbols.setdefault(sym_key, _SymbolState(session_date=session_date))
        if session_date is not None:
            state.session_date = session_date

        if state.count >= self.policy.max_per_symbol_per_session:
            return PolicyDecision(allowed=False, reason="throttled_count")

        clock = now if now is not None else time.monotonic()
        if state.last_sent_monotonic > 0:
            elapsed = clock - state.last_sent_monotonic
            if elapsed < self.policy.min_interval_seconds:
                return PolicyDecision(allowed=False, reason="throttled_interval")

        return PolicyDecision(allowed=True)

    def record_sent(
        self,
        symbol: str,
        bar_time: Optional[datetime],
        *,
        now: Optional[float] = None,
    ) -> None:
        session_date = nse_session_date(bar_time)
        self._maybe_reset_session(session_date)

        sym_key = symbol.upper().strip()
        state = self._symbols.setdefault(sym_key, _SymbolState(session_date=session_date))
        if session_date is not None:
            state.session_date = session_date
        state.count += 1
        state.last_sent_monotonic = now if now is not None else time.monotonic()

    def reset_session(self, session_date: Optional[date]) -> None:
        self._current_session_date = session_date
        self._symbols.clear()

    def _maybe_reset_session(self, session_date: Optional[date]) -> None:
        if session_date is None:
            return
        if self._current_session_date is None:
            self._current_session_date = session_date
            return
        if session_date != self._current_session_date:
            self.reset_session(session_date)
