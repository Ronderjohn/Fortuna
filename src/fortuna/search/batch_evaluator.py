"""Batch evaluation of strategy candidates with time budget."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from fortuna.config.settings import Settings, get_settings
from fortuna.compute.scheduler import get_scheduler
from fortuna.search.candidate import CandidateStrategy
from fortuna.search.evaluator import evaluate_candidate
from fortuna.search.indicator_cache import IndicatorCache
from fortuna.search.types import EvaluationResult, TierResult
from fortuna.search.worker import eval_chunk
from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.evaluation.scorer import StrategyScorer
from fortuna.utils.timing import timed_step


class BatchEvaluator:
    """Evaluate many candidates in parallel with optional time budget."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        max_workers: int = 2,
        chunk_size: int = 10,
    ) -> None:
        self.settings = settings or get_settings()
        sched = get_scheduler()
        self.max_workers = sched.effective_workers(max_workers)
        self.chunk_size = chunk_size
        self._cache = IndicatorCache()
        self._compiler = StrategyCompiler()
        self._runner = NumPyBacktestRunner(self.settings)
        self._scorer = StrategyScorer(self.settings)

    def _allow_process_pool(self) -> bool:
        """ProcessPool on Windows spawns full Python interpreters (~hundreds of MB each)."""
        if os.environ.get("FORTUNA_ALLOW_PROCESS_POOL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return True
        if os.environ.get("FORTUNA_LOW_MEMORY", "").strip().lower() in ("1", "true", "yes"):
            return False
        # Default off on Windows — common OOM on 8GB RAM with 2+ workers + CuPy imports
        if sys.platform == "win32":
            return False
        return True

    def _should_parallel(self, parallel: bool, n: int) -> bool:
        """Avoid ProcessPool unless enough candidates to amortize spawn + RAM cost."""
        if not parallel or not self._allow_process_pool():
            return False
        min_for_parallel = max(20, self.chunk_size * max(4, self.max_workers))
        return n >= min_for_parallel

    def evaluate_all(
        self,
        candidates: list[CandidateStrategy],
        window: pd.DataFrame,
        *,
        tier: int = 1,
        time_budget_sec: Optional[float] = None,
        parallel: bool = False,
    ) -> EvaluationResult:
        use_parallel = self._should_parallel(parallel, len(candidates))
        with timed_step("eval.evaluate_all") as d:
            d["candidates"] = str(len(candidates))
            d["path"] = "parallel" if use_parallel else "sequential"
            t0 = time.perf_counter()
            results: list[TierResult] = []
            timed_out = False

            spec_lists = [c.strategy.indicators for c in candidates]
            if len(candidates) >= 4:
                with timed_step("eval.enrich_many"):
                    self._cache.enrich_many(window, spec_lists)

            if use_parallel:
                results, timed_out = self._evaluate_parallel(
                    candidates, window, tier=tier, time_budget_sec=time_budget_sec
                )
            else:
                for cand in candidates:
                    if time_budget_sec and (time.perf_counter() - t0) > time_budget_sec:
                        timed_out = True
                        break
                    results.append(
                        evaluate_candidate(
                            cand,
                            window,
                            indicator_cache=self._cache,
                            compiler=self._compiler,
                            runner=self._runner,
                            scorer=self._scorer,
                            tier=tier,
                        )
                    )

            elapsed = (time.perf_counter() - t0) * 1000
            d["timed_out"] = str(timed_out)
            d["results"] = str(len(results))
            results.sort(key=lambda r: r.score, reverse=True)
            return EvaluationResult(results=results, elapsed_ms=elapsed, timed_out=timed_out)

    def _evaluate_parallel(
        self,
        candidates: list[CandidateStrategy],
        window: pd.DataFrame,
        *,
        tier: int,
        time_budget_sec: Optional[float],
    ) -> tuple[list[TierResult], bool]:
        t0 = time.perf_counter()
        timed_out = False
        results: list[TierResult] = []

        window_records = window.to_dict(orient="index")
        chunks: list[list[tuple[str, dict]]] = []
        batch: list[tuple[str, dict]] = []
        for c in candidates:
            batch.append((c.candidate_id, c.strategy.model_dump()))
            if len(batch) >= self.chunk_size:
                chunks.append(batch)
                batch = []
        if batch:
            chunks.append(batch)

        try:
            with timed_step("eval.parallel_pool") as pool_d:
                pool_d["chunks"] = str(len(chunks))
                pool_d["workers"] = str(self.max_workers)
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {
                        pool.submit(eval_chunk, ch, window_records, tier): ch for ch in chunks
                    }
                    for fut in as_completed(futures):
                        if time_budget_sec and (time.perf_counter() - t0) > time_budget_sec:
                            timed_out = True
                            pool.shutdown(wait=False, cancel_futures=True)
                            break
                        ch = futures[fut]
                        with timed_step(f"eval.chunk n={len(ch)}"):
                            for row in fut.result():
                                results.append(
                                    TierResult(
                                        candidate_id=row["candidate_id"],
                                        score=row["score"],
                                        passed=row["passed"],
                                        signal=row["signal"],
                                        entry=row["entry"],
                                        exit=row["exit"],
                                        total_trades=row["total_trades"],
                                        elapsed_ms=row["elapsed_ms"],
                                        metrics_dict=row["metrics_dict"],
                                    )
                                )
        except Exception:
            timed_out = True
            for c in candidates:
                if any(r.candidate_id == c.candidate_id for r in results):
                    continue
                results.append(
                    evaluate_candidate(
                        c,
                        window,
                        indicator_cache=self._cache,
                        compiler=self._compiler,
                        runner=self._runner,
                        scorer=self._scorer,
                        tier=tier,
                    )
                )

        return results, timed_out
