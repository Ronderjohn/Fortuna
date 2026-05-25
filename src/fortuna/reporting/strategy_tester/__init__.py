"""
TradingView-style Strategy Tester reporting for intraday backtests.

Example::

    from fortuna.reporting.strategy_tester import (
        build_strategy_report,
        compare_reports,
        TradeRecord,
        CostConfig,
    )

    report = build_strategy_report(
        strategy_name="mmts",
        trades=ledger,
        initial_capital=100_000,
        symbol="ICICIBANK.NS",
        timeframe="5m",
    )
    report.export(Path("logs/reports/mmts"))
"""

from fortuna.reporting.strategy_tester.adapters import (
    bar_equity_series,
    trades_from_dataframe,
    trades_from_mmts_dataframe,
    trades_from_return_series,
)
from fortuna.reporting.strategy_tester.backtest_bridge import trades_from_backtest
from fortuna.reporting.strategy_tester.costs import CostConfig
from fortuna.reporting.strategy_tester.chart_context import (
    attach_chart_context,
    export_strategy_chart_bundle,
)
from fortuna.reporting.strategy_tester.report import (
    StrategyTesterReport,
    build_strategy_report,
    compare_reports,
    export_comparison,
)
from fortuna.reporting.strategy_tester.signal_overlay import build_signal_frame
from fortuna.reporting.strategy_tester.trade import TradeRecord, TradeSide, trades_to_dataframe

__all__ = [
    "CostConfig",
    "StrategyTesterReport",
    "TradeRecord",
    "TradeSide",
    "attach_chart_context",
    "bar_equity_series",
    "build_signal_frame",
    "build_strategy_report",
    "compare_reports",
    "export_comparison",
    "export_strategy_chart_bundle",
    "trades_from_dataframe",
    "trades_from_mmts_dataframe",
    "trades_from_backtest",
    "trades_from_return_series",
    "trades_to_dataframe",
]
