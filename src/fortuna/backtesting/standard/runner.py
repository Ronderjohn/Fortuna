"""Run strategies under institutional standard on each data split."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.arena.param_search import ParamSearchEngine, _params_summary
from fortuna.backtesting.standard.config import InstitutionalBacktestConfig
from fortuna.backtesting.standard.splits import DataSplit
from fortuna.config.settings import Settings, get_settings
from fortuna.paper.engine import PaperTradeEngine
from fortuna.reporting.strategy_tester import CostConfig, StrategyTesterReport, build_strategy_report
from fortuna.reporting.strategy_tester.adapters import (
    bar_equity_series,
    trades_from_mmts_dataframe,
)
from fortuna.reporting.strategy_tester.backtest_bridge import trades_from_backtest
from fortuna.reporting.strategy_tester.metrics import compute_all_metrics
from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import StrategyDefinition
from fortuna.strategies.builtin.dispatch import is_builtin_strategy, run_builtin_backtest


@dataclass
class SplitResult:
    phase: str
    report: StrategyTesterReport
    strategy_used: StrategyDefinition
    candidate_id: str


@dataclass
class StrategyBacktestResult:
    strategy_stem: str
    source_path: Path
    train: SplitResult
    validation: SplitResult
    test: SplitResult
    optimized_on: str = "train"


@dataclass
class InstitutionalRunResult:
    symbol: str
    timeframe: str
    calendar_days: int
    trading_days: int
    split: DataSplit
    strategies: list[StrategyBacktestResult] = field(default_factory=list)
    run_dir: Optional[Path] = None


class InstitutionalRunner:
    """Standardized backtest runner with train optimization + OOS evaluation."""

    def __init__(
        self,
        config: InstitutionalBacktestConfig,
        settings: Optional[Settings] = None,
    ) -> None:
        self.config = config
        self.settings = settings or get_settings()
        if config.data_source:
            self.settings = self.settings.model_copy(update={"data_source": config.data_source})
        self.settings = self.settings.model_copy(
            update={
                "init_cash": config.initial_capital,
                "fees": config.costs.brokerage_rate + config.costs.exchange_fee_rate,
                "slippage": config.costs.slippage_rate + config.costs.spread_rate,
            }
        )
        self._paper = PaperTradeEngine(self.settings)
        self._search = ParamSearchEngine(self.settings, max_workers=2)
        self._costs = CostConfig(
            commission_rate=config.costs.brokerage_rate + config.costs.exchange_fee_rate,
            slippage_rate=config.costs.slippage_rate + config.costs.spread_rate,
        )

    def _cost_config(self) -> CostConfig:
        return self._costs

    def _run_phase(
        self,
        strategy: StrategyDefinition,
        ohlcv: pd.DataFrame,
        *,
        phase: str,
        symbol: str,
        candidate_id: str,
    ) -> SplitResult:
        cfg = self.config
        cap = cfg.initial_capital

        if is_builtin_strategy(strategy):
            bt = run_builtin_backtest(strategy, ohlcv, symbol=symbol, init_cash=cap)
            trades_df = bt.enriched_data.attrs.get("trades")
            trades = trades_from_mmts_dataframe(trades_df, ohlcv, initial_capital=cap, costs=self._costs)
            bar_eq = bar_equity_series(ohlcv.index, bt.equity_curve) if bt.equity_curve is not None else None
        else:
            from fortuna.backtesting.numpy_runner import NumPyBacktestRunner

            bt = NumPyBacktestRunner(self.settings).run(strategy, ohlcv, symbol=symbol)
            trades = trades_from_backtest(strategy, bt, ohlcv, initial_capital=cap, costs=self._costs)
            bar_eq = (
                bar_equity_series(ohlcv.index, bt.equity_curve)
                if bt.equity_curve is not None
                else None
            )

        report = build_strategy_report(
            strategy_name=strategy.name,
            trades=trades,
            initial_capital=cap,
            symbol=symbol,
            timeframe=cfg.timeframe,
            costs=self._costs,
            bar_equity=bar_eq,
        )
        report.strategy_name = f"{strategy.name}_{phase}"
        return SplitResult(
            phase=phase,
            report=report,
            strategy_used=strategy,
            candidate_id=candidate_id,
        )

    def run_strategy(
        self,
        path: Path,
        split: DataSplit,
        *,
        symbol: str,
        grid_override: Optional[dict] = None,
    ) -> StrategyBacktestResult:
        cfg = self.config
        search = self._search.search(
            path,
            split.train,
            grid_override=grid_override,
            max_candidates=cfg.max_candidates,
            parallel=False,
            time_budget_sec=cfg.time_budget_sec,
            rank_by="profit_pct",
        )
        best = search.best_strategy
        cid = search.best_candidate_id

        train_r = self._run_phase(best, split.train, phase="train", symbol=symbol, candidate_id=cid)
        val_r = self._run_phase(best, split.validation, phase="validation", symbol=symbol, candidate_id=cid)
        test_r = self._run_phase(best, split.test, phase="test", symbol=symbol, candidate_id=cid)

        return StrategyBacktestResult(
            strategy_stem=path.stem,
            source_path=path,
            train=train_r,
            validation=val_r,
            test=test_r,
        )

    def run_strategy_fixed(
        self,
        path: Path,
        split: DataSplit,
        *,
        symbol: str,
    ) -> StrategyBacktestResult:
        """No param search — evaluate JSON as-is on all splits."""
        base = load_strategy(path)
        cid = path.stem
        return StrategyBacktestResult(
            strategy_stem=path.stem,
            source_path=path,
            train=self._run_phase(base, split.train, phase="train", symbol=symbol, candidate_id=cid),
            validation=self._run_phase(
                base, split.validation, phase="validation", symbol=symbol, candidate_id=cid
            ),
            test=self._run_phase(base, split.test, phase="test", symbol=symbol, candidate_id=cid),
            optimized_on="none",
        )
