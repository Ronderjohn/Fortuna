"""Simulated portfolio for paper trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PaperFill:
    timestamp: str
    side: str
    price: float
    quantity: float
    fees: float
    pnl_pct: Optional[float] = None


@dataclass
class PaperPortfolio:
    """Long-only paper portfolio with fill log."""

    init_cash: float
    cash: float = 0.0
    shares: float = 0.0
    entry_price: float = 0.0
    fills: list[PaperFill] = field(default_factory=list)
    trade_returns: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.init_cash

    @property
    def has_position(self) -> bool:
        return self.shares > 0

    def equity(self, mark_price: float) -> float:
        return self.cash + self.shares * mark_price

    def open_long(self, timestamp: str, price: float, size_frac: float, fees: float) -> bool:
        if self.has_position or price <= 0:
            return False
        invest = self.cash * size_frac
        if invest <= 0:
            return False
        qty = (invest * (1.0 - fees)) / price
        self.cash -= invest
        self.shares = qty
        self.entry_price = price
        self.fills.append(PaperFill(timestamp, "BUY", price, qty, invest * fees))
        return True

    def close_long(self, timestamp: str, price: float, fees: float) -> bool:
        if not self.has_position or price <= 0:
            return False
        ret = (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0
        proceeds = self.shares * price * (1.0 - fees)
        fee_paid = self.shares * price * fees
        self.cash += proceeds
        self.fills.append(PaperFill(timestamp, "SELL", price, self.shares, fee_paid, pnl_pct=ret))
        self.trade_returns.append(ret)
        self.shares = 0.0
        self.entry_price = 0.0
        return True
