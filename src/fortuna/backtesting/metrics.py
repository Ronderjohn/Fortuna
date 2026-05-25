"""Extract backtest performance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Standard performance metrics for strategy evaluation."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    expectancy: float
    profit_factor: float
    total_trades: int
    final_value: float
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_trade_return: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "final_value": self.final_value,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "avg_trade_return": self.avg_trade_return,
        }

    @property
    def profit_pct(self) -> float:
        """Net profit as percentage (TradingView-style)."""
        return self.total_return * 100.0

    @property
    def win_ratio_pct(self) -> float:
        """Win rate as percentage."""
        return self.win_rate * 100.0


def trade_stats_from_returns(trade_returns: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Win rate, expectancy, profit factor, gross profit/loss from per-trade returns."""
    if not trade_returns:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(trade_returns, dtype=np.float64)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float(len(wins) / len(arr)) if len(arr) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = float("inf") if gross_profit > 0 else 0.0
    pf = float(min(pf, 999.0))
    return win_rate, expectancy, pf, gross_profit, gross_loss


def metrics_from_equity(
    equity: np.ndarray,
    init_cash: float,
    total_trades: int,
    trade_returns: Sequence[float] | None = None,
) -> BacktestMetrics:
    """Compute core metrics from an equity curve (NumPy backtest path)."""
    final_value = float(equity[-1])
    total_return = (final_value / init_cash) - 1.0 if init_cash > 0 else 0.0

    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) > 1 and np.std(returns) > 1e-12:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(abs(np.min(dd))) if len(dd) else 0.0

    if trade_returns:
        win_rate, expectancy, pf, gross_profit, gross_loss = trade_stats_from_returns(trade_returns)
        avg_trade = float(np.mean(np.asarray(trade_returns, dtype=np.float64)))
    else:
        win_rate, expectancy, pf, gross_profit, gross_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        avg_trade = 0.0

    return BacktestMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe if np.isfinite(sharpe) else 0.0,
        max_drawdown=max_dd,
        win_rate=win_rate,
        expectancy=expectancy,
        profit_factor=pf,
        total_trades=total_trades,
        final_value=final_value,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        avg_trade_return=avg_trade,
    )


class MetricsExtractor:
    """Extract metrics from a vectorbt Portfolio."""

    def extract(
        self,
        portfolio: Any,
        init_cash: float,
        *,
        fast: bool = False,
    ) -> BacktestMetrics:
        if fast:
            metrics = self._extract_fast(portfolio, init_cash)
            if metrics is not None:
                return metrics
        return self._extract_full(portfolio, init_cash)

    def _extract_fast(self, portfolio: Any, init_cash: float) -> BacktestMetrics | None:
        try:
            values = portfolio.value()
            final_value = float(values.iloc[-1])
            total_return = (final_value / init_cash) - 1.0

            sharpe = float(portfolio.sharpe_ratio())
            if not np.isfinite(sharpe):
                sharpe = 0.0

            max_dd = abs(float(portfolio.max_drawdown()))
            if max_dd > 1.0:
                max_dd /= 100.0

            trades = portfolio.trades
            total_trades = int(trades.count())
            win_rate = float(trades.win_rate()) if total_trades else 0.0
            if win_rate > 1.0:
                win_rate /= 100.0

            expectancy, profit_factor, gross_profit, gross_loss, avg_trade = self._trade_stats(trades)
            return BacktestMetrics(
                total_return=total_return,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                expectancy=expectancy,
                profit_factor=profit_factor,
                total_trades=total_trades,
                final_value=final_value,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                avg_trade_return=avg_trade,
            )
        except Exception:
            return None

    def _extract_full(self, portfolio: Any, init_cash: float) -> BacktestMetrics:
        stats = portfolio.stats(silence_warnings=True)
        trades = portfolio.trades

        total_return = float(stats.get("Total Return [%]", 0)) / 100.0
        sharpe = float(stats.get("Sharpe Ratio", 0) or 0)
        max_dd = abs(float(stats.get("Max Drawdown [%]", 0) or 0)) / 100.0

        total_trades = int(stats.get("Total Trades", 0) or 0)
        win_rate = float(stats.get("Win Rate [%]", 0) or 0) / 100.0

        expectancy, profit_factor, gross_profit, gross_loss, avg_trade = self._trade_stats(trades)
        final_value = float(portfolio.value().iloc[-1])

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe if np.isfinite(sharpe) else 0.0,
            max_drawdown=max_dd,
            win_rate=win_rate,
            expectancy=expectancy,
            profit_factor=profit_factor,
            total_trades=total_trades,
            final_value=final_value,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            avg_trade_return=avg_trade,
        )

    def _trade_stats(self, trades: Any) -> tuple[float, float, float, float, float]:
        try:
            records = trades.records_readable
            if records is None or len(records) == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            pnl_col = None
            for col in ("Return", "PnL", "return", "pnl"):
                if col in records.columns:
                    pnl_col = col
                    break
            if pnl_col is None:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            pnls = pd.to_numeric(records[pnl_col], errors="coerce").dropna()
            if len(pnls) == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            rets = pnls.to_numpy(dtype=float)
            win_rate, expectancy, pf, gp, gl = trade_stats_from_returns(rets)
            return expectancy, pf, gp, gl, float(np.mean(rets))
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0
