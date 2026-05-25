"""Commission and slippage configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    """
    Trading cost model (mirrors TradingView percent commission + slippage).

    ``commission_rate``: fraction of notional per side (e.g. 0.0005 = 0.05%).
    ``slippage_rate``: fraction of notional per side.
    """

    commission_rate: float = 0.0005
    slippage_rate: float = 0.0005

    def commission_on_notional(self, notional: float) -> float:
        return abs(notional) * self.commission_rate

    def slippage_on_notional(self, notional: float) -> float:
        return abs(notional) * self.slippage_rate

    def round_trip_cost(self, entry_notional: float, exit_notional: float) -> tuple[float, float]:
        comm = self.commission_on_notional(entry_notional) + self.commission_on_notional(exit_notional)
        slip = self.slippage_on_notional(entry_notional) + self.slippage_on_notional(exit_notional)
        return comm, slip
