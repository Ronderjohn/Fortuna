"""Paper trading league: walk-forward competition + adaptive learning."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.arena.param_search import ParamSearchEngine, _params_summary
from fortuna.config.settings import Settings, get_settings
from fortuna.data.stream import StreamConfig, stream_from_manager
from fortuna.evaluation.scorer import StrategyScorer
from fortuna.paper.blackbox import WalkForwardFold, walk_forward_folds
from fortuna.paper.config import PaperLeagueConfig
from fortuna.paper.engine import PaperTradeEngine
from fortuna.paper.learner import AdaptiveLearner
from fortuna.data.manager import MarketDataManager
from fortuna.reporting.strategy_report import (
    StrategyReport,
    build_report,
    print_leaderboard,
    write_report_bundle,
)
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CompetitorResult:
    strategy_stem: str
    source_path: Path
    oos_reports: list[StrategyReport] = field(default_factory=list)
    cumulative_profit_pct: float = 0.0
    cumulative_win_ratio: float = 0.0
    folds_completed: int = 0
    learning_generation: int = 0


@dataclass
class PaperLeagueResult:
    competitors: list[CompetitorResult]
    champion: Optional[StrategyReport]
    run_dir: Path


class PaperLeague:
    """
    Intensive paper-trading league with black-box walk-forward validation.

    Each fold:
    1. **Train** — search ``param_grid`` (learning).
    2. **Test** — paper trade best variant on unseen bars only.
    3. **Learn** — narrow grid toward winner for next fold.

    Strategies compete on cumulative OOS Profit % and Win Ratio.
    """

    def __init__(self, config: PaperLeagueConfig, settings: Optional[Settings] = None) -> None:
        self.config = config
        self.settings = settings or get_settings()
        if config.data_source:
            self.settings = self.settings.model_copy(update={"data_source": config.data_source})
        if config.init_cash:
            self.settings = self.settings.model_copy(update={"init_cash": config.init_cash})
        self._mdm = MarketDataManager(self.settings, data_source=config.data_source)
        self._paper = PaperTradeEngine(self.settings)
        self._search = ParamSearchEngine(self.settings, max_workers=config.max_workers)
        self._scorer = StrategyScorer(self.settings)

    def _strategy_paths(self) -> list[Path]:
        dirs = [self.config.strategy_dir, Path("strategies/builtin")]
        paths: list[Path] = []
        seen: set[str] = set()
        for rel in dirs:
            d = self.settings.resolve_path(rel)
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                if p.name not in seen:
                    seen.add(p.name)
                    paths.append(p)
        if not paths:
            raise FileNotFoundError(f"No strategies in {dirs}")
        return paths

    def _base_grid(self) -> Optional[dict]:
        p = self.config.param_grid_path
        if p and Path(p).exists():
            return json.loads(Path(p).read_text(encoding="utf-8"))
        return None

    def run(self) -> PaperLeagueResult:
        cfg = self.config
        stream = stream_from_manager(
            self._mdm,
            cfg.symbol,
            cfg.timeframe,
            config=StreamConfig(window_bars=cfg.train_bars + cfg.test_bars),
            days=cfg.days,
        )
        ohlcv = stream.full_series()
        folds = walk_forward_folds(
            ohlcv,
            train_bars=cfg.train_bars,
            test_bars=cfg.test_bars,
            step_bars=cfg.fold_step_bars,
            min_folds=cfg.min_folds,
        )

        run_id = f"{cfg.symbol}_{cfg.timeframe}_{int(time.time())}"
        run_dir = self.settings.resolve_path(cfg.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        learner = AdaptiveLearner(run_dir / "learning") if cfg.enable_learning else None
        base_grid = self._base_grid()

        competitors: dict[str, CompetitorResult] = {}
        all_oos_reports: list[StrategyReport] = []

        for path in self._strategy_paths():
            competitors[path.stem] = CompetitorResult(
                strategy_stem=path.stem, source_path=path
            )

        fold_log: list[dict] = []

        for fold in folds:
            logger.info(
                "Fold %s train [%s .. %s] test [%s .. %s]",
                fold.fold_id,
                fold.train_start,
                fold.train_end,
                fold.test_start,
                fold.test_end,
            )
            fold_entry = {"fold_id": fold.fold_id, "strategies": {}}

            for path in self._strategy_paths():
                stem = path.stem
                comp = competitors[stem]
                grid = base_grid
                if learner:
                    grid = learner.grid_for_search(stem, base_grid) or base_grid

                search = self._search.search(
                    path,
                    fold.train,
                    grid_override=grid,
                    max_candidates=cfg.max_candidates_per_strategy,
                    parallel=False,
                    time_budget_sec=cfg.time_budget_sec,
                    rank_by=cfg.rank_by,
                )
                best = search.best_strategy

                paper = self._paper.run(
                    best,
                    fold.test,
                    symbol=cfg.symbol,
                    init_cash=cfg.init_cash,
                    fold_id=fold.fold_id,
                    phase="oos",
                )
                acc = paper.account
                score = self._scorer.score(paper.metrics, best.metadata)
                report = build_report(
                    strategy_name=best.name,
                    candidate_id=search.best_candidate_id,
                    symbol=cfg.symbol,
                    timeframe=cfg.timeframe,
                    metrics=paper.metrics,
                    score=score,
                    params_summary=_params_summary(best),
                    winning_trades=acc.wins,
                    losing_trades=acc.losses,
                )

                comp.oos_reports.append(report)
                comp.cumulative_profit_pct += report.profit_pct
                comp.folds_completed += 1
                if comp.folds_completed > 0:
                    comp.cumulative_win_ratio += report.win_ratio_pct
                all_oos_reports.append(report)

                if learner:
                    state = learner.record_fold(stem, fold.fold_id, report, best, grid or {})
                    comp.learning_generation = state.generation

                fold_entry["strategies"][stem] = report.to_dict()

            fold_log.append(fold_entry)

        self._write_results(run_dir, competitors, all_oos_reports, fold_log, folds)
        champion = self._champion(competitors, cfg.rank_by)
        return PaperLeagueResult(
            competitors=list(competitors.values()),
            champion=champion,
            run_dir=run_dir,
        )

    def _champion(
        self, competitors: dict[str, CompetitorResult], rank_by: str
    ) -> Optional[StrategyReport]:
        if not competitors:
            return None
        best_comp = max(
            competitors.values(),
            key=lambda c: (
                c.cumulative_profit_pct,
                c.cumulative_win_ratio / max(c.folds_completed, 1),
                c.folds_completed,
            ),
        )
        if not best_comp.oos_reports:
            return None
        if rank_by == "win_ratio_pct":
            return max(best_comp.oos_reports, key=lambda r: r.win_ratio_pct)
        if rank_by == "composite_score":
            return max(best_comp.oos_reports, key=lambda r: r.composite_score)
        return max(best_comp.oos_reports, key=lambda r: r.profit_pct)

    def _write_results(
        self,
        run_dir: Path,
        competitors: dict[str, CompetitorResult],
        all_oos: list[StrategyReport],
        fold_log: list[dict],
        folds: list[WalkForwardFold],
    ) -> None:
        summary_rows = []
        for stem, comp in competitors.items():
            avg_win = (
                comp.cumulative_win_ratio / comp.folds_completed
                if comp.folds_completed
                else 0.0
            )
            summary_rows.append(
                {
                    "strategy": stem,
                    "folds": comp.folds_completed,
                    "cumulative_profit_pct": round(comp.cumulative_profit_pct, 4),
                    "avg_win_ratio_pct": round(avg_win, 4),
                    "learning_generation": comp.learning_generation,
                }
            )

        summary_path = run_dir / "league_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "methodology": "walk_forward_black_box",
                    "folds": len(folds),
                    "competitors": summary_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "fold_log.json").write_text(json.dumps(fold_log, indent=2), encoding="utf-8")

        agg_reports: list[StrategyReport] = []
        for comp in competitors.values():
            if not comp.oos_reports:
                continue
            last = max(comp.oos_reports, key=lambda r: r.profit_pct)
            agg = StrategyReport(
                strategy_name=comp.strategy_stem,
                candidate_id=last.candidate_id,
                symbol=last.symbol,
                timeframe=last.timeframe,
                profit_pct=comp.cumulative_profit_pct,
                win_ratio_pct=comp.cumulative_win_ratio / max(comp.folds_completed, 1),
                total_trades=sum(r.total_trades for r in comp.oos_reports),
                winning_trades=sum(r.winning_trades for r in comp.oos_reports),
                losing_trades=sum(r.losing_trades for r in comp.oos_reports),
                max_drawdown_pct=max(r.max_drawdown_pct for r in comp.oos_reports),
                profit_factor=last.profit_factor,
                sharpe_ratio=last.sharpe_ratio,
                avg_trade_pct=last.avg_trade_pct,
                gross_profit_pct=sum(r.gross_profit_pct for r in comp.oos_reports),
                gross_loss_pct=sum(r.gross_loss_pct for r in comp.oos_reports),
                composite_score=last.composite_score,
                passed_validation=last.passed_validation,
                params_summary=last.params_summary,
            )
            agg_reports.append(agg)

        write_report_bundle(agg_reports, run_dir, basename="oos_leaderboard")
        write_report_bundle(all_oos, run_dir, basename="all_oos_folds")
        print_leaderboard(agg_reports)

        champ = self._champion(competitors, self.config.rank_by)
        if champ:
            comp = next(
                (c for c in competitors.values() if c.strategy_stem in champ.strategy_name),
                None,
            )
            cum = comp.cumulative_profit_pct if comp else champ.profit_pct
            logger.info(
                "Paper league champion: %s (cumulative OOS profit %.2f%%)",
                champ.strategy_name,
                cum,
            )
