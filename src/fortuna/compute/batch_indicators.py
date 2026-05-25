"""Batch unique indicator jobs for param sweeps (GPU or CPU)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fortuna.compute.device import free_gpu_memory
from fortuna.compute.gpu_kernels import compute_series_cpu, compute_series_gpu
from fortuna.compute.policy import ComputePolicy, get_compute_policy
from fortuna.strategy.schema import IndicatorSpec, IndicatorType
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndicatorJob:
    """Unique indicator computation key."""

    kind: str
    source: str
    window: int
    extra: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_spec(cls, spec: IndicatorSpec) -> IndicatorJob | None:
        ind_type = spec.type.value if hasattr(spec.type, "value") else str(spec.type)
        source = spec.source.value if hasattr(spec.source, "value") else str(spec.source)
        if ind_type in (IndicatorType.EMA.value, IndicatorType.SMA.value, IndicatorType.RSI.value):
            return cls(ind_type, source, int(spec.params.get("window", 14)))
        if ind_type == IndicatorType.ATR.value:
            return cls("atr", "close", int(spec.params.get("window", 14)))
        return None


def collect_jobs(specs: list[IndicatorSpec]) -> list[IndicatorJob]:
    seen: set[IndicatorJob] = set()
    jobs: list[IndicatorJob] = []
    for spec in specs:
        job = IndicatorJob.from_spec(spec)
        if job and job not in seen:
            seen.add(job)
            jobs.append(job)
    return jobs


def collect_jobs_from_many(spec_lists: list[list[IndicatorSpec]]) -> list[IndicatorJob]:
    all_specs: list[IndicatorSpec] = []
    for sl in spec_lists:
        all_specs.extend(sl)
    return collect_jobs(all_specs)


class BatchIndicatorComputer:
    """Compute many indicator series with GPU batching and VRAM limits."""

    def __init__(self, policy: ComputePolicy | None = None) -> None:
        self.policy = policy or get_compute_policy()

    def run(
        self,
        df: pd.DataFrame,
        jobs: list[IndicatorJob],
    ) -> dict[IndicatorJob, np.ndarray]:
        if not jobs:
            return {}

        ohlcv = {
            "open": df["open"].to_numpy(dtype=np.float64, copy=False),
            "high": df["high"].to_numpy(dtype=np.float64, copy=False),
            "low": df["low"].to_numpy(dtype=np.float64, copy=False),
            "close": df["close"].to_numpy(dtype=np.float64, copy=False),
            "volume": df["volume"].to_numpy(dtype=np.float64, copy=False),
        }
        source_cache: dict[str, np.ndarray] = {}

        def _close_for(source: str) -> np.ndarray:
            if source in source_cache:
                return source_cache[source]
            if source in ohlcv:
                arr = ohlcv[source]
            elif source == "hlc3":
                arr = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
            elif source == "ohlc4":
                arr = (ohlcv["open"] + ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 4.0
            else:
                arr = ohlcv["close"]
            source_cache[source] = arr
            return arr

        n_bars = len(df)
        use_gpu = self.policy.should_gpu_indicators(n_bars, len(jobs))
        batch_size = self.policy.series_batch_size(n_bars, len(jobs)) if use_gpu else len(jobs)

        results: dict[IndicatorJob, np.ndarray] = {}
        compute_fn = compute_series_gpu if use_gpu else compute_series_cpu

        if use_gpu:
            logger.debug("GPU indicator batch: %s jobs, chunk=%s", len(jobs), batch_size)

        try:
            for start in range(0, len(jobs), batch_size):
                chunk = jobs[start : start + batch_size]
                for job in chunk:
                    close = _close_for(job.source)
                    results[job] = compute_fn(job.kind, close, job.window, ohlcv)
                if use_gpu:
                    free_gpu_memory()
        except Exception as e:
            logger.warning("GPU batch failed, CPU fallback: %s", e)
            for job in jobs:
                if job not in results:
                    close = _close_for(job.source)
                    results[job] = compute_series_cpu(job.kind, close, job.window, ohlcv)

        return results


def attach_specs(
    df: pd.DataFrame,
    specs: list[IndicatorSpec],
    arrays: dict[IndicatorJob, np.ndarray],
) -> pd.DataFrame:
    """Attach indicator columns from precomputed arrays."""
    if not specs:
        return df
    new_cols: dict[str, pd.Series] = {}
    for spec in specs:
        job = IndicatorJob.from_spec(spec)
        if job and job in arrays:
            new_cols[spec.id] = pd.Series(arrays[job], index=df.index, dtype=float)
    if not new_cols:
        return df
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
