"""Per-strategy parameter search with TradingView-style reports."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.config.settings import Settings, get_settings
from fortuna.evaluation.scorer import StrategyScorer
from fortuna.reporting.strategy_report import StrategyReport, build_report
from fortuna.search.batch_evaluator import BatchEvaluator
from fortuna.search.candidate import CandidateStrategy
from fortuna.search.param_expander import ParamExpander, filter_param_grid
from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import StrategyDefinition
from fortuna.utils.timing import timed_step


@dataclass
class ParamSearchResult:
    """Best parameter set for one base strategy."""

    base_name: str
    source_path: Path
    best_candidate_id: str
    best_strategy: StrategyDefinition
    best_report: StrategyReport
    all_reports: list[StrategyReport]
    variants_tested: int


class ParamSearchEngine:
    """Search param_grid for one strategy; rank by profit and win ratio."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        max_workers: int = 2,
        chunk_size: int = 10,
    ) -> None:
        self.settings = settings or get_settings()
        self._expander = ParamExpander()
        self._evaluator = BatchEvaluator(
            self.settings, max_workers=max_workers, chunk_size=chunk_size
        )
        self._scorer = StrategyScorer(self.settings)

    def load_candidates(
        self,
        strategy_path: Path,
        *,
        grid_override: Optional[dict] = None,
        max_candidates: int = 500,
    ) -> list[CandidateStrategy]:
        base = load_strategy(strategy_path)
        if grid_override:
            merged = filter_param_grid(base, grid_override)
            if merged:
                data = base.model_dump()
                data["param_grid"] = merged
                base = StrategyDefinition.model_validate(data)
        variants = self._expander.expand(base, max_candidates=max_candidates)
        out: list[CandidateStrategy] = []
        for i, v in enumerate(variants):
            cid = f"{strategy_path.stem}_{i}_{uuid.uuid4().hex[:6]}"
            out.append(CandidateStrategy(strategy=v, candidate_id=cid, source_path=str(strategy_path)))
        return out

    def search(
        self,
        strategy_path: Path,
        ohlcv: pd.DataFrame,
        *,
        grid_override: Optional[dict] = None,
        max_candidates: int = 500,
        parallel: bool = False,
        time_budget_sec: Optional[float] = None,
        rank_by: str = "profit_pct",
    ) -> ParamSearchResult:
        with timed_step(f"arena.param_search {strategy_path.name}") as d:
            candidates = self.load_candidates(
                strategy_path, grid_override=grid_override, max_candidates=max_candidates
            )
            d["candidates"] = str(len(candidates))
            d["parallel"] = str(parallel)
            if not candidates:
                raise ValueError(f"No candidates for {strategy_path}")

            try:
                eval_out = self._evaluator.evaluate_all(
                    candidates,
                    ohlcv,
                    tier=1,
                    parallel=parallel,
                    time_budget_sec=time_budget_sec,
                )
            finally:
                self._evaluator._cache.clear()
            d["timed_out"] = str(eval_out.timed_out)

            reports: list[StrategyReport] = []
            for cand in candidates:
                tr = next((r for r in eval_out.results if r.candidate_id == cand.candidate_id), None)
                if tr is None or not tr.metrics_dict:
                    continue
                metrics = _metrics_from_dict(tr.metrics_dict)
                score = self._scorer.score(metrics, cand.strategy.metadata)
                reports.append(
                    build_report(
                        strategy_name=cand.strategy.name,
                        candidate_id=cand.candidate_id,
                        symbol=cand.strategy.symbol,
                        timeframe=cand.strategy.timeframe,
                        metrics=metrics,
                        score=score,
                        params_summary=_params_summary(cand.strategy),
                    )
                )

            if not reports:
                raise ValueError(f"All candidates failed for {strategy_path}")

            best = _rank_reports(reports, rank_by)[0]
            best_cand = next(c for c in candidates if c.candidate_id == best.candidate_id)

            return ParamSearchResult(
                base_name=best_cand.strategy.name.split("_")[0],
                source_path=strategy_path,
                best_candidate_id=best.candidate_id,
                best_strategy=best_cand.strategy,
                best_report=best,
                all_reports=reports,
                variants_tested=len(candidates),
            )


def _metrics_from_dict(md: dict) -> BacktestMetrics:
    return BacktestMetrics(
        total_return=float(md.get("total_return", 0)),
        sharpe_ratio=float(md.get("sharpe_ratio", 0)),
        max_drawdown=float(md.get("max_drawdown", 0)),
        win_rate=float(md.get("win_rate", 0)),
        expectancy=float(md.get("expectancy", 0)),
        profit_factor=float(md.get("profit_factor", 0)),
        total_trades=int(md.get("total_trades", 0)),
        final_value=float(md.get("final_value", 0)),
        gross_profit=float(md.get("gross_profit", 0)),
        gross_loss=float(md.get("gross_loss", 0)),
        avg_trade_return=float(md.get("avg_trade_return", 0)),
    )


def _params_summary(strategy: StrategyDefinition) -> str:
    parts = []
    for ind in strategy.indicators:
        w = ind.params.get("window")
        if w is not None:
            parts.append(f"{ind.id}={w}")
    return ",".join(parts) if parts else ""


def _rank_reports(reports: list[StrategyReport], rank_by: str) -> list[StrategyReport]:
    key_map = {
        "profit_pct": lambda r: (
            r.total_trades > 0,
            r.profit_pct,
            r.win_ratio_pct,
            r.composite_score,
        ),
        "win_ratio_pct": lambda r: (
            r.total_trades > 0,
            r.win_ratio_pct,
            r.profit_pct,
            r.composite_score,
        ),
        "composite_score": lambda r: (
            r.total_trades > 0,
            r.composite_score,
            r.profit_pct,
            r.win_ratio_pct,
        ),
    }
    key_fn = key_map.get(rank_by, key_map["profit_pct"])
    return sorted(reports, key=key_fn, reverse=True)


def save_search_result(result: ParamSearchResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.base_name}_param_search.json"
    payload = {
        "base_name": result.base_name,
        "source_path": str(result.source_path),
        "best_candidate_id": result.best_candidate_id,
        "variants_tested": result.variants_tested,
        "best_report": result.best_report.to_dict(),
        "top_5": [r.to_dict() for r in _rank_reports(result.all_reports, "profit_pct")[:5]],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
