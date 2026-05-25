"""Per-bar tournament runner."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.config.settings import Settings, get_settings
from fortuna.data.manager import MarketDataManager
from fortuna.search.batch_evaluator import BatchEvaluator
from fortuna.search.candidate import CandidateStrategy
from fortuna.search.param_expander import ParamExpander, filter_param_grid
from fortuna.strategy.loader import load_strategy
from fortuna.tournament.config import TournamentConfig
from fortuna.tournament.logger import TournamentLogger
from fortuna.tournament.models import TournamentBarResult
from fortuna.tournament.winner import select_winner
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class TournamentRunner:
    """Run per-bar strategy tournament on intraday OHLCV."""

    def __init__(
        self,
        config: TournamentConfig,
        settings: Optional[Settings] = None,
    ) -> None:
        self.config = config
        self.settings = settings or get_settings()
        if config.data_source:
            self.settings = self.settings.model_copy(
                update={"data_source": config.data_source}
            )
        self._mdm = MarketDataManager(self.settings, data_source=config.data_source)
        self._expander = ParamExpander()
        self._evaluator = BatchEvaluator(
            self.settings,
            max_workers=config.max_workers,
            chunk_size=config.chunk_size,
        )

    def _load_candidates(self) -> list[CandidateStrategy]:
        candidates: list[CandidateStrategy] = []
        paths = self.config.strategy_paths
        if not paths:
            gen = self.settings.resolve_path(self.settings.strategies_generated)
            paths = list(gen.glob("*.json"))

        grid_override = None
        if self.config.param_grid_path and self.config.param_grid_path.exists():
            import json

            grid_override = json.loads(
                self.config.param_grid_path.read_text(encoding="utf-8")
            )

        for path in paths:
            base = load_strategy(path)
            if grid_override:
                merged = filter_param_grid(base, grid_override)
                if merged:
                    data = base.model_dump()
                    data["param_grid"] = merged
                    from fortuna.strategy.schema import StrategyDefinition

                    base = StrategyDefinition.model_validate(data)
            variants = self._expander.expand(base, max_candidates=self.config.max_candidates)
            for i, v in enumerate(variants):
                cid = f"{path.stem}_{i}_{uuid.uuid4().hex[:6]}"
                candidates.append(
                    CandidateStrategy(strategy=v, candidate_id=cid, source_path=str(path))
                )
        return candidates

    def run(self) -> list[TournamentBarResult]:
        cfg = self.config
        logger.info("Tournament %s %s (%s candidates max)", cfg.symbol, cfg.timeframe, cfg.max_candidates)

        ohlcv = self._mdm.get_ohlcv(
            cfg.symbol,
            cfg.timeframe,
            force_refresh=False,
            days=cfg.days,
        )
        if len(ohlcv) < cfg.warmup_bars + 10:
            raise ValueError(
                f"Insufficient bars: {len(ohlcv)} < warmup {cfg.warmup_bars}"
            )

        candidates = self._load_candidates()
        if not candidates:
            raise ValueError("No strategy candidates loaded")

        tlog = TournamentLogger(self.settings.resolve_path(cfg.output_dir))
        tlog.write_meta(
            {
                "symbol": cfg.symbol,
                "timeframe": cfg.timeframe,
                "bars": len(ohlcv),
                "candidates": len(candidates),
                "window_bars": cfg.window_bars,
            }
        )

        results: list[TournamentBarResult] = []
        t_run = time.perf_counter()

        step = max(1, cfg.bar_step)
        for t in range(cfg.warmup_bars, len(ohlcv), step):
            window = ohlcv.iloc[t - cfg.window_bars : t].copy()
            ts = str(ohlcv.index[t])

            eval_out = self._evaluator.evaluate_all(
                candidates,
                window,
                tier=cfg.tier,
                time_budget_sec=cfg.time_budget_sec,
                parallel=cfg.parallel,
            )

            winner = select_winner(eval_out.results)
            if winner is None:
                results.append(
                    TournamentBarResult(
                        timestamp=ts,
                        signal="HOLD",
                        winner_strategy="",
                        winner_candidate_id="",
                        score=0.0,
                        candidates_evaluated=0,
                        elapsed_ms=eval_out.elapsed_ms,
                        timed_out=eval_out.timed_out,
                    )
                )
                continue
            win_cand = next(c for c in candidates if c.candidate_id == winner.candidate_id)
            results.append(
                TournamentBarResult(
                    timestamp=ts,
                    signal=winner.signal,
                    winner_strategy=win_cand.strategy.name,
                    winner_candidate_id=winner.candidate_id,
                    score=winner.score,
                    candidates_evaluated=len(eval_out.results),
                    elapsed_ms=eval_out.elapsed_ms,
                    timed_out=eval_out.timed_out,
                )
            )

        tlog.write_signals(results)
        logger.info(
            "Tournament done: %s bars, %.1fs total",
            len(results),
            time.perf_counter() - t_run,
        )
        return results
