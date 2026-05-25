"""Walk-forward, Monte Carlo, and parameter sensitivity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.standard.config import InstitutionalBacktestConfig
from fortuna.paper.blackbox import walk_forward_folds
from fortuna.reporting.strategy_tester.trade import TradeRecord


@dataclass
class WalkForwardSummary:
    folds: int
    oos_profit_total: float
    oos_win_rate_avg: float
    fold_profits: list[float] = field(default_factory=list)


@dataclass
class MonteCarloSummary:
    iterations: int
    median_final_equity: float
    p5_final_equity: float
    p95_final_equity: float
    prob_profit: float
    max_dd_median: float


@dataclass
class RobustnessReport:
    walk_forward: Optional[WalkForwardSummary] = None
    monte_carlo: Optional[MonteCarloSummary] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"notes": self.notes}
        if self.walk_forward:
            out["walk_forward"] = {
                "folds": self.walk_forward.folds,
                "oos_profit_total": self.walk_forward.oos_profit_total,
                "oos_win_rate_avg": self.walk_forward.oos_win_rate_avg,
                "fold_profits": self.walk_forward.fold_profits,
            }
        if self.monte_carlo:
            out["monte_carlo"] = {
                "iterations": self.monte_carlo.iterations,
                "median_final_equity": self.monte_carlo.median_final_equity,
                "p5_final_equity": self.monte_carlo.p5_final_equity,
                "p95_final_equity": self.monte_carlo.p95_final_equity,
                "prob_profit": self.monte_carlo.prob_profit,
                "max_dd_median": self.monte_carlo.max_dd_median,
            }
        return out


def run_walk_forward_summary(
    ohlcv: pd.DataFrame,
    fold_profits: list[float],
    fold_win_rates: list[float],
) -> WalkForwardSummary:
    return WalkForwardSummary(
        folds=len(fold_profits),
        oos_profit_total=float(sum(fold_profits)),
        oos_win_rate_avg=float(np.mean(fold_win_rates)) if fold_win_rates else 0.0,
        fold_profits=fold_profits,
    )


def monte_carlo_trade_shuffle(
    trades: list[TradeRecord],
    initial_capital: float,
    *,
    iterations: int = 500,
    seed: int = 42,
) -> MonteCarloSummary:
    """
    Shuffle trade order to test equity-path luck (institutional robustness check).
    """
    if not trades:
        return MonteCarloSummary(
            iterations=0,
            median_final_equity=initial_capital,
            p5_final_equity=initial_capital,
            p95_final_equity=initial_capital,
            prob_profit=0.0,
            max_dd_median=0.0,
        )

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades], dtype=np.float64)
    finals: list[float] = []
    max_dds: list[float] = []

    for _ in range(iterations):
        shuffled = rng.permutation(pnls)
        equity = initial_capital + np.cumsum(shuffled)
        finals.append(float(equity[-1]))
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
        max_dds.append(float(dd.max()) if len(dd) else 0.0)

    arr = np.array(finals)
    return MonteCarloSummary(
        iterations=iterations,
        median_final_equity=float(np.median(arr)),
        p5_final_equity=float(np.percentile(arr, 5)),
        p95_final_equity=float(np.percentile(arr, 95)),
        prob_profit=float((arr > initial_capital).mean()),
        max_dd_median=float(np.median(max_dds)),
    )


def plan_walk_forward_folds(
    ohlcv: pd.DataFrame,
    config: InstitutionalBacktestConfig,
) -> list:
    """Return walk-forward fold plan (for paper league integration)."""
    return walk_forward_folds(
        ohlcv,
        train_bars=config.walk_forward_train_bars,
        test_bars=config.walk_forward_test_bars,
        step_bars=config.walk_forward_step_bars,
        min_folds=2,
    )
