"""Adaptive learning: refine param grids from paper-trade out-of-sample results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fortuna.reporting.strategy_report import StrategyReport
from fortuna.strategy.schema import StrategyDefinition


@dataclass
class FoldRecord:
    fold_id: int
    profit_pct: float
    win_ratio_pct: float
    total_trades: int
    candidate_id: str
    params_summary: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization (numpy scalars become Python floats, ints stay ints)."""
        return {
            "fold_id": int(self.fold_id),
            "profit_pct": float(self.profit_pct),
            "win_ratio_pct": float(self.win_ratio_pct),
            "total_trades": int(self.total_trades),
            "candidate_id": str(self.candidate_id),
            "params_summary": str(self.params_summary),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FoldRecord":
        return cls(
            fold_id=int(data["fold_id"]),
            profit_pct=float(data["profit_pct"]),
            win_ratio_pct=float(data["win_ratio_pct"]),
            total_trades=int(data["total_trades"]),
            candidate_id=str(data.get("candidate_id", "")),
            params_summary=str(data.get("params_summary", "")),
        )


@dataclass
class StrategyLearningState:
    """Persistent per-strategy learning state across paper league runs."""

    strategy_stem: str
    best_params: dict[str, Any] = field(default_factory=dict)
    fold_history: list[FoldRecord] = field(default_factory=list)
    cumulative_oos_profit_pct: float = 0.0
    generation: int = 0
    param_grid: dict[str, list[Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_stem": self.strategy_stem,
            "best_params": self.best_params,
            "fold_history": [f.to_dict() for f in self.fold_history],
            "cumulative_oos_profit_pct": self.cumulative_oos_profit_pct,
            "generation": self.generation,
            "param_grid": self.param_grid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyLearningState:
        folds = [FoldRecord.from_dict(f) for f in data.get("fold_history", [])]
        return cls(
            strategy_stem=data["strategy_stem"],
            best_params=data.get("best_params", {}),
            fold_history=folds,
            cumulative_oos_profit_pct=float(data.get("cumulative_oos_profit_pct", 0)),
            generation=int(data.get("generation", 0)),
            param_grid=data.get("param_grid", {}),
        )


class AdaptiveLearner:
    """
    Update strategy knowledge after each OOS paper fold.

    Narrows ``param_grid`` around winning indicator windows (exploit)
    while keeping one neighbor step for exploration — continuous improvement
    without overwriting the base JSON until promoted.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stem: str) -> Path:
        return self.state_dir / f"{stem}_learning.json"

    def load(self, stem: str) -> Optional[StrategyLearningState]:
        path = self._path(stem)
        if not path.exists():
            return None
        return StrategyLearningState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, state: StrategyLearningState) -> None:
        self._path(state.strategy_stem).write_text(
            json.dumps(state.to_dict(), indent=2),
            encoding="utf-8",
        )

    def record_fold(
        self,
        stem: str,
        fold_id: int,
        report: StrategyReport,
        strategy: StrategyDefinition,
        base_grid: dict[str, list[Any]],
    ) -> StrategyLearningState:
        state = self.load(stem) or StrategyLearningState(
            strategy_stem=stem, param_grid=dict(base_grid)
        )
        state.fold_history.append(
            FoldRecord(
                fold_id=fold_id,
                profit_pct=report.profit_pct,
                win_ratio_pct=report.win_ratio_pct,
                total_trades=report.total_trades,
                candidate_id=report.candidate_id,
                params_summary=report.params_summary,
            )
        )
        state.cumulative_oos_profit_pct += report.profit_pct
        state.generation += 1
        state.best_params = _extract_params(strategy)
        if base_grid:
            state.param_grid = self.refine_grid(base_grid, strategy, report.profit_pct)
        self.save(state)
        return state

    def refine_grid(
        self,
        grid: dict[str, list[Any]],
        winner: StrategyDefinition,
        oos_profit_pct: float,
    ) -> dict[str, list[Any]]:
        """Narrow grid toward winner; keep neighbors if OOS profit was weak (explore)."""
        if oos_profit_pct < 0:
            return grid

        refined: dict[str, list[Any]] = {}
        winner_params = _extract_params(winner)

        for path, values in grid.items():
            wval = winner_params.get(path)
            if wval is None:
                refined[path] = values
                continue
            if isinstance(wval, (int, float)) and all(isinstance(v, (int, float)) for v in values):
                nums = sorted(set(int(v) for v in values))
                w = int(wval)
                neighbors = {w}
                if w - 1 in nums or w > min(nums):
                    neighbors.add(max(w - 2, min(nums)))
                if w + 1 in nums or w < max(nums):
                    neighbors.add(min(w + 2, max(nums)))
                neighbors.add(w)
                refined[path] = sorted(neighbors)
            else:
                refined[path] = [wval]

        return refined or grid

    def grid_for_search(
        self, stem: str, base_grid: Optional[dict[str, list[Any]]]
    ) -> Optional[dict[str, list[Any]]]:
        state = self.load(stem)
        if state and state.param_grid:
            return state.param_grid
        return base_grid


def _extract_params(strategy: StrategyDefinition) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ind in strategy.indicators:
        for key, val in ind.params.items():
            out[f"indicators.{ind.id}.params.{key}"] = val
    return out
