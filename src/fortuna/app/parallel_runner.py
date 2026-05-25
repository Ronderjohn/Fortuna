"""Run multiple strategies in parallel on one shared OHLCV frame."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.app.config import AppConfig
from fortuna.backtesting.engine import BacktestResult
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.config.settings import Settings
from fortuna.reporting.strategy_tester.chart_context import build_report_from_backtest
from fortuna.reporting.strategy_tester.costs import CostConfig
from fortuna.reporting.strategy_tester.report import StrategyTesterReport
from fortuna.strategy.loader import load_strategy
from fortuna.strategies.builtin.dispatch import is_builtin_strategy, run_builtin_backtest
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyRunResult:
    strategy_name: str
    strategy_path: Path
    report: Optional[StrategyTesterReport] = None
    backtest: Optional[BacktestResult] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.report is not None and self.error is None


@dataclass
class BatchRunResult:
    results: dict[str, StrategyRunResult] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def leaderboard_rows(self) -> list[dict]:
        rows = []
        for name, r in sorted(
            self.results.items(),
            key=lambda kv: (
                -(kv[1].report.performance.total_net_profit if kv[1].report else -1e18),
            ),
        ):
            if not r.report:
                continue
            p = r.report.performance
            risk = r.report.risk
            rows.append(
                {
                    "strategy": name,
                    "profit_pct": round(
                        p.total_net_profit / r.report.initial_capital * 100.0, 2
                    ),
                    "net_profit": round(p.total_net_profit, 2),
                    "trades": p.total_trades,
                    "wins": p.winning_trades,
                    "losses": p.losing_trades,
                    "win_rate_pct": round(p.win_rate_pct, 1),
                    "profit_factor": round(p.profit_factor, 2),
                    "max_dd_pct": round(risk.max_drawdown_pct * 100, 2),
                    "sharpe": round(risk.sharpe_ratio, 2),
                }
            )
        return rows

    @property
    def winner(self) -> Optional[str]:
        rows = self.leaderboard_rows()
        return rows[0]["strategy"] if rows else None


def _run_one(
    path: Path,
    ohlcv: pd.DataFrame,
    symbol: str,
    timeframe: str,
    settings: Settings,
    costs: CostConfig,
) -> StrategyRunResult:
    t0 = time.perf_counter()
    name = path.stem
    try:
        strategy = load_strategy(path)
        if is_builtin_strategy(strategy):
            bt = run_builtin_backtest(
                strategy, ohlcv, symbol=symbol, init_cash=settings.init_cash
            )
        else:
            bt = NumPyBacktestRunner(settings).run(strategy, ohlcv, symbol=symbol)
        report = build_report_from_backtest(
            strategy,
            ohlcv,
            bt,
            strategy_name=name,
            symbol=symbol,
            timeframe=timeframe,
            init_cash=settings.init_cash,
            costs=costs,
        )
        return StrategyRunResult(
            strategy_name=name,
            strategy_path=path,
            report=report,
            backtest=bt,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        logger.warning("Strategy %s failed: %s", name, e)
        return StrategyRunResult(
            strategy_name=name,
            strategy_path=path,
            error=str(e),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )


class ParallelStrategyRunner:
    """Execute all strategies on the same OHLCV using a thread pool."""

    def __init__(
        self,
        settings: Settings,
        app_config: Optional[AppConfig] = None,
        costs: Optional[CostConfig] = None,
    ) -> None:
        self.settings = settings
        self.app_config = app_config or AppConfig.from_settings(settings)
        self.costs = costs or CostConfig(
            commission_rate=settings.fees,
            slippage_rate=settings.slippage,
        )

    def run_sequential(
        self,
        ohlcv: pd.DataFrame,
        paths: list[Path],
        *,
        symbol: str,
        timeframe: str,
    ) -> BatchRunResult:
        t0 = time.perf_counter()
        results: dict[str, StrategyRunResult] = {}
        for path in paths:
            r = _run_one(path, ohlcv, symbol, timeframe, self.settings, self.costs)
            results[r.strategy_name] = r
        return BatchRunResult(
            results=results,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def run_parallel(
        self,
        ohlcv: pd.DataFrame,
        paths: list[Path],
        *,
        symbol: str,
        timeframe: str,
        max_workers: Optional[int] = None,
    ) -> BatchRunResult:
        workers = max_workers or self.app_config.parallel_workers
        t0 = time.perf_counter()
        results: dict[str, StrategyRunResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_one, path, ohlcv, symbol, timeframe, self.settings, self.costs
                ): path
                for path in paths
            }
            for fut in as_completed(futures):
                r = fut.result()
                results[r.strategy_name] = r
        return BatchRunResult(
            results=results,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
