"""Matplotlib charts for Strategy Tester reports (static PNG exports).

The live dashboard renders TradingView-style charts in-browser via
:mod:`fortuna.app.lightweight_chart` (Lightweight Charts v5 over a CDN-loaded
custom HTML embed). This module is **only** for the CLI/static report bundle
written to ``logs/strategy_tester/.../charts/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fortuna.reporting.strategy_tester.report import StrategyTesterReport


def _import_plt():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib required for charts. Install: uv sync --group charts"
        ) from e


def generate_all_charts(report: StrategyTesterReport, output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(plot_equity_curve(report, output_dir / "equity_curve.png"))
    paths.append(plot_drawdown_curve(report, output_dir / "drawdown_curve.png"))
    paths.append(plot_cumulative_returns(report, output_dir / "cumulative_returns.png"))
    paths.append(plot_win_loss_distribution(report, output_dir / "win_loss_distribution.png"))
    paths.append(plot_monthly_pnl_heatmap(report, output_dir / "monthly_pnl_heatmap.png"))
    return paths


def plot_equity_curve(report: StrategyTesterReport, path: Path) -> Path:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(12, 5))
    eq = report.trade_equity_df
    if eq.empty and report.bar_equity is not None:
        ax.plot(report.bar_equity.index, report.bar_equity.values, color="#2962FF", lw=1.2)
    elif not eq.empty:
        ax.plot(eq["timestamp"], eq["equity"], color="#2962FF", lw=1.2)
        ax.axhline(report.initial_capital, color="gray", ls="--", alpha=0.6)
    ax.set_title(f"{report.strategy_name} — Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_drawdown_curve(report: StrategyTesterReport, path: Path) -> Path:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(12, 4))
    eq = report.trade_equity_df
    if not eq.empty:
        dd_pct = eq["drawdown_pct"] * 100
        ax.fill_between(eq["timestamp"], 0, -dd_pct, color="#E53935", alpha=0.5)
        ax.plot(eq["timestamp"], -dd_pct, color="#B71C1C", lw=1)
    ax.set_title("Drawdown %")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_cumulative_returns(report: StrategyTesterReport, path: Path) -> Path:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(12, 4))
    if not report.trades_df.empty:
        cum = report.trades_df["pnl"].cumsum()
        ax.bar(range(len(cum)), cum.values, color=np.where(cum >= 0, "#26A69A", "#EF5350"), width=1.0)
    ax.set_title("Cumulative P&L by Trade #")
    ax.set_xlabel("Trade")
    ax.set_ylabel("Cumulative P&L")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_win_loss_distribution(report: StrategyTesterReport, path: Path) -> Path:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not report.trades_df.empty:
        pnls = report.trades_df["pnl"]
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        ax.hist(wins, bins=30, alpha=0.7, color="#26A69A", label="Wins")
        ax.hist(losses, bins=30, alpha=0.7, color="#EF5350", label="Losses")
        ax.legend()
    ax.set_title("P&L Distribution")
    ax.set_xlabel("Trade P&L")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_monthly_pnl_heatmap(report: StrategyTesterReport, path: Path) -> Path:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(10, 4))
    if not report.trades_df.empty:
        df = report.trades_df.copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df["year"] = df["exit_time"].dt.year
        df["month"] = df["exit_time"].dt.month
        pivot = df.pivot_table(index="year", columns="month", values="pnl", aggfunc="sum").fillna(0)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Monthly P&L Heatmap")
        fig.colorbar(im, ax=ax, label="P&L")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
