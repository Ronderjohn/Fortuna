"""Run many strategies in parallel on streamed OHLCV with parameter search."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fortuna.arena.config import ArenaConfig
from fortuna.arena.leaderboard import StrategyLeaderboard
from fortuna.arena.param_search import ParamSearchEngine, ParamSearchResult, save_search_result
from fortuna.config.settings import Settings, get_settings
from fortuna.data.manager import MarketDataManager
from fortuna.data.stream import OhlcvBarStream, StreamConfig, stream_from_manager
from fortuna.reporting.strategy_report import StrategyReport, build_report, write_report_bundle
from fortuna.search.batch_evaluator import BatchEvaluator
from fortuna.search.candidate import CandidateStrategy
from fortuna.search.param_expander import ParamExpander
from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import StrategyDefinition
from fortuna.strategy.warmup import required_warmup_bars, slice_for_backtest
from fortuna.utils.logging import get_logger
from fortuna.utils.timing import timed_step

logger = get_logger(__name__)


class StrategyArena:
    """
    Compete multiple strategies with parameter search on DuckDB-backed OHLCV.

    Modes:
    - ``run_full_sample``: one backtest window per strategy (all param variants in parallel).
    - ``run_streamed``: rolling windows; per window pick best variant per strategy, aggregate reports.
    """

    def __init__(self, config: ArenaConfig, settings: Optional[Settings] = None) -> None:
        self.config = config
        self.settings = settings or get_settings()
        if config.data_source:
            self.settings = self.settings.model_copy(update={"data_source": config.data_source})
        self._mdm = MarketDataManager(self.settings, data_source=config.data_source)
        self._search = ParamSearchEngine(
            self.settings,
            max_workers=config.max_workers,
            chunk_size=config.chunk_size,
        )

    def _strategy_paths(self) -> list[Path]:
        dirs = [
            self.config.strategy_dir,
            Path("strategies/intraday"),
            Path("strategies/builtin"),
        ]
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

    def _grid_override(self) -> Optional[dict]:
        if self.config.param_grid_path and self.config.param_grid_path.exists():
            return json.loads(self.config.param_grid_path.read_text(encoding="utf-8"))
        return None

    def run_full_sample(self) -> StrategyLeaderboard:
        """Search best parameters per strategy on the full cached series."""
        cfg = self.config
        with timed_step("arena.run_full_sample"):
            stream = stream_from_manager(
                self._mdm,
                cfg.symbol,
                cfg.timeframe,
                config=StreamConfig(window_bars=cfg.window_bars, warmup_bars=0),
                days=cfg.days,
            )
            with timed_step("arena.load_ohlcv_window") as d:
                ohlcv = stream.full_series()
                if len(ohlcv) < cfg.window_bars:
                    raise ValueError(f"Need at least {cfg.window_bars} bars, got {len(ohlcv)}")
                d["bars_total"] = str(len(ohlcv))
            grid = self._grid_override()
            paths = self._strategy_paths()
            results: list[ParamSearchResult] = []

            for path in paths:
                strategy = load_strategy(path)
                if cfg.eval_full_series:
                    ohlcv_window = ohlcv.copy()
                else:
                    ohlcv_window = slice_for_backtest(
                        ohlcv,
                        window_bars=cfg.window_bars,
                        warmup_bars=cfg.warmup_bars,
                        strategy=strategy,
                    )
                logger.info(
                    "Param search: %s (%d bars, warmup>=%d)",
                    path.name,
                    len(ohlcv_window),
                    required_warmup_bars(strategy.indicators),
                )
                results.append(
                    self._search.search(
                        path,
                        ohlcv_window,
                        grid_override=grid,
                        max_candidates=cfg.max_candidates_per_strategy,
                        parallel=cfg.max_workers > 1,
                        time_budget_sec=cfg.time_budget_sec,
                        rank_by=cfg.rank_by,
                    )
                )

            return self._finalize(results)

    def run_streamed(self) -> StrategyLeaderboard:
        """Evaluate strategies on each rolling window from the stream; aggregate champions."""
        cfg = self.config
        stream = stream_from_manager(
            self._mdm,
            cfg.symbol,
            cfg.timeframe,
            config=StreamConfig(
                window_bars=cfg.window_bars,
                warmup_bars=cfg.warmup_bars,
                step_bars=cfg.stream_step,
            ),
            days=cfg.days,
        )

        paths = self._strategy_paths()
        grid = self._grid_override()
        all_candidates = self._load_all_candidates(paths, grid)

        window_reports: dict[str, list[StrategyReport]] = {p.stem: [] for p in paths}
        evaluator = BatchEvaluator(
            self.settings,
            max_workers=cfg.max_workers,
            chunk_size=cfg.chunk_size,
        )

        for bar_idx, window in stream.windows():
            logger.debug("Stream window ending bar %s (%s rows)", bar_idx, len(window))
            eval_out = evaluator.evaluate_all(
                all_candidates,
                window,
                tier=1,
                parallel=len(all_candidates) > cfg.chunk_size,
                time_budget_sec=cfg.time_budget_sec,
            )
            by_source: dict[str, list] = {}
            for tr in eval_out.results:
                cand = next(c for c in all_candidates if c.candidate_id == tr.candidate_id)
                stem = Path(cand.source_path or "").stem
                by_source.setdefault(stem, []).append((cand, tr))

            for stem, rows in by_source.items():
                if not rows:
                    continue
                best_tr = max(rows, key=lambda x: x[1].score)
                cand, tr = best_tr
                md = tr.metrics_dict
                from fortuna.arena.param_search import _metrics_from_dict

                metrics = _metrics_from_dict(md)
                window_reports[stem].append(
                    build_report(
                        strategy_name=cand.strategy.name,
                        candidate_id=cand.candidate_id,
                        symbol=cfg.symbol,
                        timeframe=cfg.timeframe,
                        metrics=metrics,
                        params_summary=_brief_params(cand.strategy),
                    )
                )

        results: list[ParamSearchResult] = []
        full_series = stream.full_series()
        for path in paths:
            strategy = load_strategy(path)
            full_ohlcv = slice_for_backtest(
                full_series,
                window_bars=cfg.window_bars,
                warmup_bars=cfg.warmup_bars,
                strategy=strategy,
            )
            stem = path.stem
            wr = window_reports.get(stem, [])
            if wr:
                from fortuna.arena.param_search import _rank_reports

                best_rep = _rank_reports(wr, cfg.rank_by)[0]
                full = self._search.search(
                    path,
                    full_ohlcv,
                    grid_override=grid,
                    max_candidates=cfg.max_candidates_per_strategy,
                    parallel=False,
                    time_budget_sec=cfg.time_budget_sec,
                    rank_by=cfg.rank_by,
                )
                full.best_report = best_rep
                results.append(full)
            else:
                results.append(
                    self._search.search(
                        path,
                        full_ohlcv,
                        grid_override=grid,
                        max_candidates=cfg.max_candidates_per_strategy,
                        rank_by=cfg.rank_by,
                    )
                )

        return self._finalize(results)

    def run(self) -> StrategyLeaderboard:
        if self.config.use_streaming:
            return self.run_streamed()
        return self.run_full_sample()

    def _load_all_candidates(
        self, paths: list[Path], grid: Optional[dict]
    ) -> list[CandidateStrategy]:
        expander = ParamExpander()
        out: list[CandidateStrategy] = []
        for path in paths:
            base = load_strategy(path)
            if grid:
                data = base.model_dump()
                data["param_grid"] = grid
                base = StrategyDefinition.model_validate(data)
            variants = expander.expand(base, max_candidates=self.config.max_candidates_per_strategy)
            for i, v in enumerate(variants):
                cid = f"{path.stem}_{i}_{uuid.uuid4().hex[:6]}"
                out.append(CandidateStrategy(strategy=v, candidate_id=cid, source_path=str(path)))
        return out

    def _finalize(self, results: list[ParamSearchResult]) -> StrategyLeaderboard:
        cfg = self.config
        out_dir = self.settings.resolve_path(cfg.output_dir)
        run_id = f"{cfg.symbol}_{cfg.timeframe}_{int(time.time())}"
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            save_search_result(r, run_dir / "param_search")

        board = StrategyLeaderboard(results, rank_by=cfg.rank_by)
        board.export(run_dir, basename="leaderboard")
        write_report_bundle(
            board.all_variant_reports(),
            run_dir,
            basename="all_variants",
        )
        board.print_summary()
        meta = {
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "strategies": len(results),
            "rank_by": cfg.rank_by,
            "champion": board.overall_winner().to_dict() if board.overall_winner() else None,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Arena results: %s", run_dir)
        return board


def _brief_params(strategy: StrategyDefinition) -> str:
    return ",".join(
        f"{ind.id}={ind.params.get('window', '')}" for ind in strategy.indicators if ind.params
    )
