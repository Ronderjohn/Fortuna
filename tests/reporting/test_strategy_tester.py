"""Strategy Tester metrics and report tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fortuna.reporting.strategy_tester import (
    CostConfig,
    TradeRecord,
    TradeSide,
    build_strategy_report,
    compare_reports,
    trades_from_dataframe,
)
from fortuna.reporting.strategy_tester.metrics import compute_performance


def _sample_trades() -> list[TradeRecord]:
    base = pd.Timestamp("2026-03-01 09:20:00")
    return [
        TradeRecord(
            entry_time=base,
            exit_time=base + pd.Timedelta(minutes=25),
            entry_price=100.0,
            exit_price=102.0,
            side=TradeSide.LONG,
            qty=100.0,
            pnl=200.0,
            pnl_percent=2.0,
            holding_time=pd.Timedelta(minutes=25),
            commission=10.0,
            slippage=5.0,
        ),
        TradeRecord(
            entry_time=base + pd.Timedelta(hours=1),
            exit_time=base + pd.Timedelta(hours=1, minutes=15),
            entry_price=101.0,
            exit_price=99.0,
            side=TradeSide.LONG,
            qty=100.0,
            pnl=-200.0,
            pnl_percent=-1.98,
            holding_time=pd.Timedelta(minutes=15),
            commission=10.0,
            slippage=5.0,
        ),
        TradeRecord(
            entry_time=base + pd.Timedelta(hours=2),
            exit_time=base + pd.Timedelta(hours=2, minutes=30),
            entry_price=98.0,
            exit_price=99.5,
            side=TradeSide.SHORT,
            qty=100.0,
            pnl=150.0,
            pnl_percent=1.53,
            holding_time=pd.Timedelta(minutes=30),
            commission=10.0,
            slippage=5.0,
        ),
    ]


def test_performance_metrics_tv_formulas() -> None:
    trades = _sample_trades()
    p = compute_performance(trades)
    assert p.total_trades == 3
    assert p.winning_trades == 2
    assert p.losing_trades == 1
    assert p.gross_profit == 350.0
    assert p.gross_loss == 200.0
    assert abs(p.total_net_profit - 150.0) < 1e-6
    assert abs(p.profit_factor - 1.75) < 1e-6
    assert abs(p.win_rate_pct - 66.666) < 0.2


def test_build_report_exports(tmp_path: Path) -> None:
    trades = _sample_trades()
    report = build_strategy_report(
        strategy_name="test_strategy",
        trades=trades,
        initial_capital=100_000.0,
        symbol="TEST.NS",
        timeframe="5m",
    )
    out = report.export(tmp_path / "r", generate_charts=False)
    assert (out / "summary.json").exists()
    assert (out / "trades.csv").exists()
    assert (out / "equity_curve.csv").exists()
    row = report.comparison_row()
    assert row["total_trades"] == 3
    assert "sharpe" in row


def test_compare_multiple_strategies() -> None:
    t1 = build_strategy_report(
        strategy_name="a", trades=_sample_trades(), initial_capital=100_000
    )
    t2 = build_strategy_report(
        strategy_name="b", trades=_sample_trades()[:1], initial_capital=100_000
    )
    df = compare_reports([t1, t2])
    assert len(df) == 2
    assert "profit_factor" in df.columns


def test_trades_from_dataframe() -> None:
    df = pd.DataFrame(
        {
            "entry_time": ["2026-03-01 09:15:00"],
            "exit_time": ["2026-03-01 10:00:00"],
            "entry_price": [100.0],
            "exit_price": [101.0],
            "side": ["LONG"],
            "qty": [50.0],
        }
    )
    trades = trades_from_dataframe(df, costs=CostConfig(0.001, 0.0))
    assert len(trades) == 1
    assert trades[0].pnl > 0
