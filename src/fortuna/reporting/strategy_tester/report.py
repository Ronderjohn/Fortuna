"""Strategy Tester report builder and export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from fortuna.reporting.strategy_tester.costs import CostConfig
from fortuna.reporting.strategy_tester.metrics import (
    IntradayMetrics,
    PerformanceMetrics,
    RiskMetrics,
    StrategyMetrics,
    compute_all_metrics,
)
from fortuna.reporting.strategy_tester.trade import TradeRecord, trades_to_dataframe


@dataclass
class StrategyTesterReport:
    """
    Full TradingView-style strategy analytics package.

    Plug any strategy's trade ledger + optional bar equity to generate
    summary JSON, DataFrames, comparison rows, and charts.
    """

    strategy_name: str
    symbol: str
    timeframe: str
    initial_capital: float
    costs: CostConfig
    trades: list[TradeRecord]
    metrics: StrategyMetrics
    trades_df: pd.DataFrame
    trade_equity_df: pd.DataFrame
    bar_equity: Optional[pd.Series] = None
    candle_timestamps: Optional[pd.DatetimeIndex] = None
    ohlcv: Optional[pd.DataFrame] = None
    enriched_data: Optional[pd.DataFrame] = None
    signal_df: Optional[pd.DataFrame] = None
    overlay_columns: list[str] = field(default_factory=list)

    @property
    def performance(self) -> PerformanceMetrics:
        return self.metrics.performance

    @property
    def risk(self) -> RiskMetrics:
        return self.metrics.risk

    @property
    def intraday(self) -> IntradayMetrics:
        return self.metrics.intraday

    def comparison_row(self) -> dict[str, Any]:
        row = self.metrics.comparison_row(self.strategy_name)
        row["total_profit_pct"] = round(
            self.performance.total_net_profit / self.initial_capital * 100.0, 4
        )
        row["symbol"] = self.symbol
        row["timeframe"] = self.timeframe
        return row

    def summary_dict(self) -> dict[str, Any]:
        p, r, i = self.performance, self.risk, self.intraday
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "initial_capital": self.initial_capital,
            "final_equity": round(
                self.initial_capital + p.total_net_profit, 2
            ),
            "costs": asdict(self.costs),
            "core_performance": {
                "total_net_profit": round(p.total_net_profit, 2),
                "gross_profit": round(p.gross_profit, 2),
                "gross_loss": round(p.gross_loss, 2),
                "profit_factor": round(p.profit_factor, 4),
                "total_trades": p.total_trades,
                "winning_trades": p.winning_trades,
                "losing_trades": p.losing_trades,
                "breakeven_trades": p.breakeven_trades,
                "win_rate_pct": round(p.win_rate_pct, 2),
                "average_trade": round(p.average_trade, 2),
                "average_winning_trade": round(p.average_winning_trade, 2),
                "average_losing_trade": round(p.average_losing_trade, 2),
                "largest_winning_trade": round(p.largest_winning_trade, 2),
                "largest_losing_trade": round(p.largest_losing_trade, 2),
                "total_commission": round(p.total_commission, 2),
                "total_slippage": round(p.total_slippage, 2),
            },
            "risk": {
                "max_drawdown": round(r.max_drawdown, 2),
                "max_drawdown_pct": round(r.max_drawdown_pct * 100, 4),
                "max_consecutive_wins": r.max_consecutive_wins,
                "max_consecutive_losses": r.max_consecutive_losses,
                "average_hold_time": str(r.average_hold_time),
                "expectancy": round(r.expectancy, 4),
                "risk_reward_ratio": round(r.risk_reward_ratio, 4),
                "sharpe_ratio": round(r.sharpe_ratio, 4),
                "sortino_ratio": round(r.sortino_ratio, 4),
                "calmar_ratio": round(r.calmar_ratio, 4),
            },
            "intraday": {
                "trades_per_day": round(i.trades_per_day, 2),
                "average_pnl_per_day": round(i.average_pnl_per_day, 2),
                "best_trading_hour": i.best_trading_hour,
                "worst_trading_hour": i.worst_trading_hour,
                "best_hour_pnl": round(i.best_hour_pnl, 2),
                "worst_hour_pnl": round(i.worst_hour_pnl, 2),
                "session_opening_pnl": round(i.session_opening_pnl, 2),
                "session_midday_pnl": round(i.session_midday_pnl, 2),
                "session_closing_pnl": round(i.session_closing_pnl, 2),
                "trading_days": i.trading_days,
            },
        }

    def export(self, output_dir: Path, *, generate_charts: bool = False) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(self.summary_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        self.trades_df.to_csv(output_dir / "trades.csv", index=False)
        self.trade_equity_df.to_csv(output_dir / "equity_curve.csv", index=False)

        if self.bar_equity is not None:
            self.bar_equity.to_frame("equity").to_csv(output_dir / "bar_equity.csv")

        if self.signal_df is not None:
            self.signal_df.to_csv(output_dir / "signals.csv")

        if generate_charts:
            from fortuna.reporting.strategy_tester.visualize import generate_all_charts

            generate_all_charts(self, output_dir / "charts")

        return output_dir

    def print_summary(self) -> None:
        s = self.summary_dict()
        cp = s["core_performance"]
        rk = s["risk"]
        print("\n" + "=" * 72)
        print(f"STRATEGY TESTER — {self.strategy_name} ({self.symbol} {self.timeframe})")
        print("=" * 72)
        print(f"  Net Profit:      {cp['total_net_profit']:+,.2f}  ({self.comparison_row()['total_profit_pct']:+.2f}%)")
        print(f"  Gross Profit:    {cp['gross_profit']:,.2f}  |  Gross Loss: {cp['gross_loss']:,.2f}")
        print(f"  Profit Factor:   {cp['profit_factor']:.2f}  |  Trades: {cp['total_trades']}")
        print(
            f"  Win Rate:        {cp['win_rate_pct']:.1f}%  "
            f"({cp['winning_trades']}W / {cp['losing_trades']}L)"
        )
        print(f"  Avg Trade:       {cp['average_trade']:+,.2f}")
        print(f"  Max Drawdown:    {rk['max_drawdown']:,.2f} ({rk['max_drawdown_pct']:.2f}%)")
        print(f"  Sharpe / Sortino:{rk['sharpe_ratio']:.2f} / {rk['sortino_ratio']:.2f}")
        print(f"  Expectancy:      {rk['expectancy']:+,.2f}  |  R:R {rk['risk_reward_ratio']:.2f}")
        iday = s["intraday"]
        print(
            f"  Trades/day:      {iday['trades_per_day']:.1f}  |  "
            f"Best hour: {iday['best_trading_hour']} ({iday['best_hour_pnl']:+,.0f})"
        )
        print("=" * 72 + "\n")


def build_strategy_report(
    *,
    strategy_name: str,
    trades: list[TradeRecord],
    initial_capital: float,
    symbol: str = "",
    timeframe: str = "5m",
    costs: Optional[CostConfig] = None,
    candle_timestamps: Optional[pd.DatetimeIndex] = None,
    bar_equity: Optional[pd.Series] = None,
    ohlcv: Optional[pd.DataFrame] = None,
    enriched_data: Optional[pd.DataFrame] = None,
    signal_df: Optional[pd.DataFrame] = None,
    overlay_columns: Optional[list[str]] = None,
) -> StrategyTesterReport:
    costs = costs or CostConfig()
    # "Pyramid" continuation rows are signal markers, not real round-trips —
    # they would distort win-rate / trade-count metrics. Strip them before
    # computing metrics but keep them in the rendered trades_df so the chart
    # can still draw the continuation markers (matches TV's MMTS pyramiding).
    real_trades = [
        t for t in trades if str(t.metadata.get("exit_reason", "")).lower() != "pyramid"
    ]
    metrics, trade_eq = compute_all_metrics(real_trades, initial_capital, bar_equity)
    return StrategyTesterReport(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=initial_capital,
        costs=costs,
        trades=trades,
        metrics=metrics,
        trades_df=trades_to_dataframe(trades),
        trade_equity_df=trade_eq,
        bar_equity=bar_equity,
        candle_timestamps=candle_timestamps,
        ohlcv=ohlcv,
        enriched_data=enriched_data,
        signal_df=signal_df,
        overlay_columns=overlay_columns or [],
    )


def compare_reports(reports: list[StrategyTesterReport]) -> pd.DataFrame:
    """Comparison-ready metrics for multiple strategies."""
    if not reports:
        return pd.DataFrame()
    rows = [r.comparison_row() for r in reports]
    df = pd.DataFrame(rows)
    return df.sort_values(["total_profit", "win_rate"], ascending=False)


def export_comparison(
    reports: list[StrategyTesterReport],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = compare_reports(reports)
    csv_p = output_dir / "strategy_comparison.csv"
    json_p = output_dir / "strategy_comparison.json"
    df.to_csv(csv_p, index=False)
    json_p.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    return csv_p, json_p
