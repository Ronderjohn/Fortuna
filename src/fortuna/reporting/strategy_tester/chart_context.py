"""Attach OHLCV + signal overlay to a strategy tester report."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.backtesting.engine import BacktestResult
from fortuna.reporting.strategy_tester.adapters import bar_equity_series, trades_from_mmts_dataframe
from fortuna.reporting.strategy_tester.backtest_bridge import trades_from_backtest
from fortuna.reporting.strategy_tester.costs import CostConfig
from fortuna.reporting.strategy_tester.report import StrategyTesterReport, build_strategy_report
from fortuna.reporting.strategy_tester.signal_overlay import (
    build_signal_frame,
    suggest_overlay_columns,
)
from fortuna.strategy.schema import StrategyDefinition
from fortuna.strategies.builtin.dispatch import is_builtin_strategy


def attach_chart_context(
    report: StrategyTesterReport,
    *,
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    enriched: Optional[pd.DataFrame] = None,
) -> StrategyTesterReport:
    """Populate ``ohlcv``, ``signal_df``, and ``overlay_columns`` on an existing report."""
    report.ohlcv = ohlcv
    report.enriched_data = enriched
    report.signal_df = build_signal_frame(strategy, ohlcv, enriched)
    report.overlay_columns = suggest_overlay_columns(strategy, enriched)
    return report


def chart_context_from_backtest(
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    bt: BacktestResult,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, list[str]]:
    enriched = bt.enriched_data
    signal_df = build_signal_frame(strategy, ohlcv, enriched)
    overlays = suggest_overlay_columns(strategy, enriched)
    return ohlcv, enriched, signal_df, overlays


def build_report_from_backtest(
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    bt: BacktestResult,
    *,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    init_cash: float,
    costs: Optional[CostConfig] = None,
) -> StrategyTesterReport:
    """Assemble a strategy tester report (with chart context) from a backtest result."""
    costs = costs or CostConfig()
    if is_builtin_strategy(strategy):
        trades_df = bt.enriched_data.attrs.get("trades")
        trades = trades_from_mmts_dataframe(
            trades_df, ohlcv, initial_capital=init_cash, costs=costs
        )
    else:
        trades = trades_from_backtest(strategy, bt, ohlcv, initial_capital=init_cash, costs=costs)

    bar_eq = (
        bar_equity_series(ohlcv.index, bt.equity_curve) if bt.equity_curve is not None else None
    )
    _, enriched, signal_df, overlays = chart_context_from_backtest(strategy, ohlcv, bt)
    return build_strategy_report(
        strategy_name=strategy_name,
        trades=trades,
        initial_capital=init_cash,
        symbol=symbol,
        timeframe=timeframe,
        costs=costs,
        candle_timestamps=ohlcv.index,
        bar_equity=bar_eq,
        ohlcv=ohlcv,
        enriched_data=enriched,
        signal_df=signal_df,
        overlay_columns=overlays,
    )


def export_strategy_chart_bundle(
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    bt: BacktestResult,
    output_dir: Path,
    *,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    init_cash: float,
    costs: Optional[CostConfig] = None,
    generate_charts: bool = False,
) -> tuple[Path, StrategyTesterReport]:
    """Build report with TV-style chart and export to ``output_dir``."""
    report = build_report_from_backtest(
        strategy,
        ohlcv,
        bt,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        init_cash=init_cash,
        costs=costs,
    )
    out = report.export(Path(output_dir), generate_charts=generate_charts)
    return out, report
