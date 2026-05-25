"""Rank strategies and persist to validated/rejected folders."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fortuna.backtesting.engine import BacktestResult
from fortuna.config.settings import Settings, get_settings
from fortuna.evaluation.scorer import StrategyScore, StrategyScorer
from fortuna.strategy.loader import load_strategy
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RankedStrategy:
    """Strategy with backtest and score."""

    strategy_path: Path
    result: BacktestResult
    score: StrategyScore


class StrategyRanker:
    """Rank backtest results and route strategies to storage folders."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        scorer: Optional[StrategyScorer] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.scorer = scorer or StrategyScorer(self.settings)

    def rank_one(self, strategy_path: Path, result: BacktestResult) -> RankedStrategy:
        strategy = load_strategy(strategy_path)
        score = self.scorer.score(result.metrics, strategy.metadata)
        return RankedStrategy(strategy_path=strategy_path, result=result, score=score)

    def persist(self, ranked: RankedStrategy) -> Path:
        """Copy strategy JSON to validated or rejected folder with score sidecar."""
        dest_dir = (
            self.settings.resolve_path(self.settings.strategies_validated)
            if ranked.score.passed
            else self.settings.resolve_path(self.settings.strategies_rejected)
        )
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / ranked.strategy_path.name
        shutil.copy2(ranked.strategy_path, dest_path)

        sidecar = dest_path.with_suffix(".score.json")
        sidecar.write_text(
            json.dumps(
                {
                    "composite_score": ranked.score.composite,
                    "passed": ranked.score.passed,
                    "metrics": ranked.result.metrics.to_dict(),
                    "notes": ranked.score.notes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Strategy %s -> %s (score=%.1f)",
            ranked.strategy_path.name,
            dest_dir.name,
            ranked.score.composite,
        )
        return dest_path

    def rank_many(self, ranked_list: list[RankedStrategy]) -> list[RankedStrategy]:
        """Sort traded strategies above zero-trade ones, then by composite score.

        A strategy that never executed has no evidence to outrank one with
        real (even losing) PnL. The previous sort by raw composite score
        could crown zero-trade strategies — the same bug family that hit
        the dashboard leaderboard. Primary key: ``total_trades > 0``;
        secondary: composite score; tertiary: total_trades for tie-break.
        """
        return sorted(
            ranked_list,
            key=lambda r: (
                r.result.metrics.total_trades > 0,
                r.score.composite,
                r.result.metrics.total_trades,
            ),
            reverse=True,
        )
