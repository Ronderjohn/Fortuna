"""Central CPU/GPU scheduler for Fortuna compute paths."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd

from fortuna.compute.batch_indicators import (
    BatchIndicatorComputer,
    attach_specs,
    collect_jobs_from_many,
)
from fortuna.compute.device import gpu_device_info, is_gpu_available
from fortuna.compute.policy import ComputePolicy, get_compute_policy
from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.schema import IndicatorSpec
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class ComputeScheduler:
    """
    Routes indicator work to GPU (batched) or CPU (Numba/pandas).

    Design goals (4GB GPU):
    - One OHLCV upload per window; many indicator series on device.
    - Chunk series to stay under VRAM budget.
    - Cap CPU parallelism while GPU is active.
    - Non-GPU indicators (MACD, Bollinger) stay on CPU via IndicatorEngine.
    """

    def __init__(self, policy: Optional[ComputePolicy] = None) -> None:
        self.policy = policy or get_compute_policy()
        self._cpu_engine = IndicatorEngine()
        self._batch = BatchIndicatorComputer(self.policy)

    def status_line(self) -> str:
        if not self.policy.use_gpu:
            return "compute: CPU only (FORTUNA_USE_GPU=0)"
        if not is_gpu_available():
            return "compute: CPU only (install: uv sync --group gpu)"
        info = gpu_device_info(self.policy.gpu_mem_fraction)
        if info is None:
            return "compute: CPU only (GPU probe failed)"
        mb = info.usable_vram_bytes // (1024 * 1024)
        return f"compute: GPU {info.name} (~{mb}MB usable)"

    def effective_workers(self, requested: int) -> int:
        return self.policy.effective_cpu_workers(requested)

    def compute_indicators(self, df: pd.DataFrame, specs: list[IndicatorSpec]) -> pd.DataFrame:
        if not specs:
            return df
        gpu_specs, cpu_specs = _split_specs(specs)
        out = df
        if gpu_specs:
            from fortuna.compute.batch_indicators import collect_jobs

            jobs = collect_jobs(gpu_specs)
            arrays = self._batch.run(out, jobs)
            out = attach_specs(out, gpu_specs, arrays)
        if cpu_specs:
            out = self._cpu_engine.compute(out, cpu_specs)
        return out

    def precompute_for_candidates(
        self,
        df: pd.DataFrame,
        spec_lists: list[list[IndicatorSpec]],
    ) -> dict[str, pd.DataFrame]:
        """One GPU sweep for all unique indicator sets across candidates."""
        from fortuna.search.indicator_cache import IndicatorCache

        unique: dict[str, list[IndicatorSpec]] = {}
        for specs in spec_lists:
            if specs:
                unique[IndicatorCache._key(specs)] = specs

        if not unique:
            return {}

        all_jobs = collect_jobs_from_many(list(unique.values()))
        arrays = self._batch.run(df, all_jobs)

        enriched: dict[str, pd.DataFrame] = {}
        for key, specs in unique.items():
            gpu_specs, cpu_specs = _split_specs(specs)
            frame = df
            if gpu_specs:
                from fortuna.compute.batch_indicators import collect_jobs

                frame = attach_specs(frame, gpu_specs, arrays)
            if cpu_specs:
                frame = self._cpu_engine.compute(frame, cpu_specs)
            enriched[key] = frame
        return enriched


def _split_specs(specs: list[IndicatorSpec]) -> tuple[list[IndicatorSpec], list[IndicatorSpec]]:
    gpu: list[IndicatorSpec] = []
    cpu: list[IndicatorSpec] = []
    gpu_types = {"ema", "sma", "rsi", "atr"}
    for spec in specs:
        t = spec.type.value if hasattr(spec.type, "value") else str(spec.type)
        if t in gpu_types:
            gpu.append(spec)
        else:
            cpu.append(spec)
    return gpu, cpu


@lru_cache(maxsize=1)
def get_scheduler() -> ComputeScheduler:
    return ComputeScheduler()
