"""Per-strategy paper account with explicit win/loss ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from fortuna.backtesting.metrics import BacktestMetrics, metrics_from_equity, trade_stats_from_returns


@dataclass
class PaperAccount:
    """
    Independent paper account for one strategy.

    Each competitor starts with the same ``init_cash``; balances never mix.
    """

    strategy_name: str
    symbol: str
    init_cash: float
    final_equity: float = 0.0
    profit_pct: float = 0.0
    profit_amount: float = 0.0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    total_trades: int = 0
    win_ratio_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_returns: list[float] = field(default_factory=list)
    metrics: BacktestMetrics | None = None

    @classmethod
    def from_backtest(
        cls,
        *,
        strategy_name: str,
        symbol: str,
        init_cash: float,
        trade_returns: Sequence[float],
        equity_curve: np.ndarray,
    ) -> PaperAccount:
        returns = list(trade_returns)
        metrics = metrics_from_equity(
            np.asarray(equity_curve, dtype=np.float64),
            init_cash,
            len(returns),
            returns,
        )
        wins = sum(1 for r in returns if r > 0)
        losses = sum(1 for r in returns if r < 0)
        breakeven = sum(1 for r in returns if r == 0)
        win_rate, _, pf, _, _ = trade_stats_from_returns(returns)
        final = float(equity_curve[-1]) if len(equity_curve) else init_cash
        profit_amt = final - init_cash
        return cls(
            strategy_name=strategy_name,
            symbol=symbol,
            init_cash=init_cash,
            final_equity=final,
            profit_pct=metrics.profit_pct,
            profit_amount=profit_amt,
            wins=wins,
            losses=losses,
            breakeven=breakeven,
            total_trades=len(returns),
            win_ratio_pct=win_rate * 100.0,
            profit_factor=pf,
            max_drawdown_pct=metrics.max_drawdown * 100.0,
            trade_returns=returns,
            metrics=metrics,
        )

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "init_cash": self.init_cash,
            "final_equity": round(self.final_equity, 2),
            "profit_amount": round(self.profit_amount, 2),
            "profit_pct": round(self.profit_pct, 4),
            "wins": self.wins,
            "losses": self.losses,
            "breakeven": self.breakeven,
            "total_trades": self.total_trades,
            "win_ratio_pct": round(self.win_ratio_pct, 2),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
        }

    def format_row(self) -> str:
        return (
            f"{self.strategy_name:22} | "
            f"Start {self.init_cash:>10,.0f} -> End {self.final_equity:>10,.0f} | "
            f"P&L {self.profit_pct:+7.2f}% ({self.profit_amount:+,.0f}) | "
            f"W/L {self.wins}/{self.losses} ({self.win_ratio_pct:.1f}%) | "
            f"Trades {self.total_trades:4d} | PF {self.profit_factor:.2f}"
        )
