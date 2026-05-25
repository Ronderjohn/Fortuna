"""TradingView-style strategy performance report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.evaluation.scorer import StrategyScore


@dataclass
class StrategyReport:
    """
    Performance summary comparable to TradingView strategy tester.

    Primary columns: profit %, win ratio, trades, max drawdown, profit factor.
    """

    strategy_name: str
    candidate_id: str
    symbol: str
    timeframe: str
    profit_pct: float
    win_ratio_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown_pct: float
    profit_factor: float
    sharpe_ratio: float
    avg_trade_pct: float
    gross_profit_pct: float
    gross_loss_pct: float
    composite_score: float
    passed_validation: bool
    params_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_row(self) -> str:
        return (
            f"{self.strategy_name:20} | Profit {self.profit_pct:7.2f}% | "
            f"Win {self.win_ratio_pct:5.1f}% | Trades {self.total_trades:4d} | "
            f"DD {self.max_drawdown_pct:5.1f}% | PF {self.profit_factor:4.2f} | "
            f"Score {self.composite_score:5.1f}"
        )


def build_report(
    *,
    strategy_name: str,
    candidate_id: str,
    symbol: str,
    timeframe: str,
    metrics: BacktestMetrics,
    score: Optional[StrategyScore] = None,
    params_summary: str = "",
    winning_trades: Optional[int] = None,
    losing_trades: Optional[int] = None,
) -> StrategyReport:
    if winning_trades is not None and losing_trades is not None:
        wins, losses = winning_trades, losing_trades
    else:
        wins = int(round(metrics.win_rate * metrics.total_trades)) if metrics.total_trades else 0
        losses = max(0, metrics.total_trades - wins)
    composite = score.composite if score else 0.0
    passed = score.passed if score else False

    return StrategyReport(
        strategy_name=strategy_name,
        candidate_id=candidate_id,
        symbol=symbol,
        timeframe=timeframe,
        profit_pct=metrics.profit_pct,
        win_ratio_pct=metrics.win_ratio_pct,
        total_trades=metrics.total_trades,
        winning_trades=wins,
        losing_trades=losses,
        max_drawdown_pct=metrics.max_drawdown * 100.0,
        profit_factor=metrics.profit_factor,
        sharpe_ratio=metrics.sharpe_ratio,
        avg_trade_pct=metrics.avg_trade_return * 100.0,
        gross_profit_pct=metrics.gross_profit * 100.0,
        gross_loss_pct=metrics.gross_loss * 100.0,
        composite_score=composite,
        passed_validation=passed,
        params_summary=params_summary,
    )


def reports_to_dataframe(reports: list[StrategyReport]) -> pd.DataFrame:
    if not reports:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in reports])


def write_report_bundle(
    reports: list[StrategyReport],
    output_dir: Path,
    *,
    basename: str = "arena_comparison",
) -> tuple[Path, Path]:
    """Write CSV + JSON comparison tables (TradingView-style leaderboard)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = reports_to_dataframe(reports)
    if not df.empty:
        df = df.sort_values(["profit_pct", "win_ratio_pct", "composite_score"], ascending=False)

    csv_path = output_dir / f"{basename}.csv"
    json_path = output_dir / f"{basename}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    return csv_path, json_path


def print_leaderboard(reports: list[StrategyReport], top_n: int = 20) -> None:
    ranked = sorted(
        reports,
        key=lambda r: (r.profit_pct, r.win_ratio_pct, r.composite_score),
        reverse=True,
    )
    print("\n" + "=" * 88)
    print("STRATEGY LEADERBOARD (Profit % | Win Ratio | Trades | Max DD | Profit Factor)")
    print("=" * 88)
    for i, r in enumerate(ranked[:top_n], 1):
        print(f"{i:3}. {r.format_row()}")
    print("=" * 88 + "\n")
