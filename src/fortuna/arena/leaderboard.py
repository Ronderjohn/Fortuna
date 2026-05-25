"""Cross-strategy leaderboard and winner selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fortuna.arena.param_search import ParamSearchResult, _rank_reports
from fortuna.reporting.strategy_report import StrategyReport, print_leaderboard, write_report_bundle


@dataclass
class StrategyLeaderboard:
    """Aggregate best variant per strategy family and pick overall champion."""

    results: list[ParamSearchResult]
    rank_by: str = "profit_pct"

    def best_per_strategy(self) -> list[StrategyReport]:
        return [r.best_report for r in self.results]

    def overall_winner(self) -> Optional[StrategyReport]:
        reports = self.best_per_strategy()
        if not reports:
            return None
        return _rank_reports(reports, self.rank_by)[0]

    def all_variant_reports(self) -> list[StrategyReport]:
        out: list[StrategyReport] = []
        for r in self.results:
            out.extend(r.all_reports)
        return out

    def export(self, output_dir, *, basename: str = "arena_leaderboard") -> tuple:
        from pathlib import Path

        output_dir = Path(output_dir)
        best = self.best_per_strategy()
        return write_report_bundle(best, output_dir, basename=basename)

    def print_summary(self, top_n: int = 20) -> None:
        print_leaderboard(self.best_per_strategy(), top_n=top_n)
        winner = self.overall_winner()
        if winner:
            print(f"CHAMPION: {winner.strategy_name} — Profit {winner.profit_pct:.2f}% | Win {winner.win_ratio_pct:.1f}%")
