"""TradingView-style strategy performance reports."""

from fortuna.reporting.strategy_report import StrategyReport, build_report, reports_to_dataframe
from fortuna.reporting.strategy_tester import (
    CostConfig,
    StrategyTesterReport,
    TradeRecord,
    TradeSide,
    build_strategy_report,
    compare_reports,
    export_comparison,
)

__all__ = [
    "CostConfig",
    "StrategyReport",
    "StrategyTesterReport",
    "TradeRecord",
    "TradeSide",
    "build_report",
    "build_strategy_report",
    "compare_reports",
    "export_comparison",
    "reports_to_dataframe",
]
