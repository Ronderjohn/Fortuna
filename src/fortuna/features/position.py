"""Lightweight position state for the RL trading environment.

Distinct from Phase 1's vectorbt/NumPyBacktestRunner trade ledgers — those
operate post-hoc on signal series, while this object lives inside the env
loop where ``step()`` mutates it bar-by-bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class PositionState:
    """One open or flat intraday position.

    All PnL is expressed in percent (matches the Phase 2 reward function units).
    """

    side: str = "FLAT"  # "FLAT" | "LONG" | "SHORT"
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    bars_held: int = 0
    size: float = 1.0
    realized_pnl_pct: float = 0.0
    last_action_bar: Optional[pd.Timestamp] = None
    history: list[dict] = field(default_factory=list)

    @property
    def is_flat(self) -> bool:
        return self.side == "FLAT"

    @property
    def is_long(self) -> bool:
        return self.side == "LONG"

    @property
    def is_short(self) -> bool:
        return self.side == "SHORT"

    @property
    def is_open(self) -> bool:
        return self.side != "FLAT"

    @property
    def size_fraction(self) -> float:
        """Position size as a fraction of full risk (0.0 when flat)."""
        return self.size if self.is_open else 0.0

    def unrealized_pct(self, current_price: float) -> float:
        """Mark-to-market PnL in percent vs entry."""
        if not self.is_open or self.entry_price <= 0:
            return 0.0
        if self.is_long:
            return (current_price - self.entry_price) / self.entry_price * 100.0
        return (self.entry_price - current_price) / self.entry_price * 100.0

    def open_long(self, price: float, timestamp: pd.Timestamp, size: float = 1.0) -> None:
        self.side = "LONG"
        self.entry_price = float(price)
        self.entry_time = timestamp
        self.bars_held = 0
        self.size = float(size)
        self.last_action_bar = timestamp

    def open_short(self, price: float, timestamp: pd.Timestamp, size: float = 1.0) -> None:
        self.side = "SHORT"
        self.entry_price = float(price)
        self.entry_time = timestamp
        self.bars_held = 0
        self.size = float(size)
        self.last_action_bar = timestamp

    def close(self, price: float, timestamp: pd.Timestamp) -> float:
        """Close the open position. Returns realized PnL in percent."""
        if not self.is_open:
            return 0.0
        pnl_pct = self.unrealized_pct(price)
        self.history.append(
            {
                "side": self.side,
                "entry_price": self.entry_price,
                "exit_price": float(price),
                "entry_time": self.entry_time,
                "exit_time": timestamp,
                "pnl_pct": pnl_pct,
                "bars_held": self.bars_held,
                "size": self.size,
            }
        )
        self.realized_pnl_pct += pnl_pct
        self.side = "FLAT"
        self.entry_price = 0.0
        self.entry_time = None
        self.bars_held = 0
        self.size = 1.0
        self.last_action_bar = timestamp
        return pnl_pct

    def step(self) -> None:
        """Tick the bars-held counter at every env step where position is open."""
        if self.is_open:
            self.bars_held += 1
