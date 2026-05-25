"""Multi-strategy institutional comparison tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fortuna.backtesting.standard.filters import FilterVerdict, evaluate_strategy
from fortuna.backtesting.standard.runner import StrategyBacktestResult


@dataclass
class ComparisonRow:
    strategy: str
    phase: str
    total_profit: float
    total_profit_pct: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe: float
    sortino: float
    win_rate: float
    expectancy: float
    total_trades: int
    risk_reward: float
    filter_passed: bool
    filter_score: float
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "phase": self.phase,
            "total_profit": round(self.total_profit, 2),
            "total_profit_pct": round(self.total_profit_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "win_rate": round(self.win_rate, 2),
            "expectancy": round(self.expectancy, 4),
            "total_trades": self.total_trades,
            "risk_reward": round(self.risk_reward, 4),
            "filter_passed": self.filter_passed,
            "filter_score": round(self.filter_score, 2),
            "rank": self.rank,
        }


def build_comparison_table(
    results: list[StrategyBacktestResult],
    *,
    phase: str = "test",
    period_days: int,
    thresholds,
    benchmarks,
) -> pd.DataFrame:
    """Standardized comparison on OOS test (or train/validation)."""
    rows: list[ComparisonRow] = []

    for res in results:
        split_r = getattr(res, phase)
        rep = split_r.report
        perf = rep.performance
        risk = rep.risk
        verdict = evaluate_strategy(perf, risk, thresholds=thresholds, benchmarks=benchmarks, period_days=period_days)
        rows.append(
            ComparisonRow(
                strategy=res.strategy_stem,
                phase=phase,
                total_profit=perf.total_net_profit,
                total_profit_pct=rep.comparison_row()["total_profit_pct"],
                max_drawdown_pct=risk.max_drawdown_pct * 100,
                profit_factor=perf.profit_factor,
                sharpe=risk.sharpe_ratio,
                sortino=risk.sortino_ratio,
                win_rate=perf.win_rate_pct,
                expectancy=risk.expectancy,
                total_trades=perf.total_trades,
                risk_reward=risk.risk_reward_ratio,
                filter_passed=verdict.passed,
                filter_score=verdict.score,
            )
        )

    rows.sort(key=lambda r: r.filter_score, reverse=True)
    for i, r in enumerate(rows, 1):
        r.rank = i
    return pd.DataFrame([r.to_dict() for r in rows])


def print_comparison_table(df: pd.DataFrame, title: str = "INSTITUTIONAL COMPARISON (OOS TEST)") -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if df.empty:
        print("  (no results)")
    else:
        cols = [
            "rank",
            "strategy",
            "total_profit",
            "total_profit_pct",
            "max_drawdown_pct",
            "profit_factor",
            "sharpe",
            "win_rate",
            "expectancy",
            "total_trades",
            "filter_passed",
        ]
        show = [c for c in cols if c in df.columns]
        print(df[show].to_string(index=False))
    print("=" * 100 + "\n")


def export_comparison(df: pd.DataFrame, output_dir: Path, basename: str = "institutional_comparison") -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_p = output_dir / f"{basename}.csv"
    json_p = output_dir / f"{basename}.json"
    df.to_csv(csv_p, index=False)
    json_p.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    return csv_p, json_p
