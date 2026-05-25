"""Executed trade ledger for strategy tester reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradeRecord:
    """Single completed round-trip trade (TradingView-style)."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    side: TradeSide
    qty: float
    pnl: float
    pnl_percent: float
    holding_time: pd.Timedelta
    commission: float = 0.0
    slippage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gross_pnl(self) -> float:
        return self.pnl + self.commission + self.slippage

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "side": self.side.value,
            "qty": self.qty,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "holding_time": str(self.holding_time),
            "commission": self.commission,
            "slippage": self.slippage,
            **self.metadata,
        }


def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "side",
                "qty",
                "pnl",
                "pnl_percent",
                "holding_time",
                "commission",
                "slippage",
            ]
        )
    return pd.DataFrame([t.to_dict() for t in trades])
