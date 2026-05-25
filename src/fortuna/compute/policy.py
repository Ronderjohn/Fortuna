"""Decide when to route work to GPU vs CPU (GTX 1650 4GB tuned)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from fortuna.compute.device import gpu_device_info, is_gpu_available
from fortuna.compute.memory import estimate_indicator_batch_bytes


@dataclass
class ComputePolicy:
    """
    Hybrid CPU/GPU policy.

    Defaults target 4GB GPUs (e.g. GTX 1650): use ~75% VRAM, batch indicators,
    keep CPU workers low while GPU is busy to avoid system RAM pressure.
    """

    use_gpu: bool = True
    gpu_mem_fraction: float = 0.75
    gpu_min_bars: int = 32
    gpu_min_unique_series: int = 2
    gpu_max_series_per_batch: int = 64
    cpu_max_workers_when_gpu: int = 1
    prefer_gpu_for_param_sweep: bool = True

    @classmethod
    def from_env(cls) -> ComputePolicy:
        def _float(k: str, d: float) -> float:
            try:
                return float(os.environ.get(k, d))
            except ValueError:
                return d

        def _int(k: str, d: int) -> int:
            try:
                return int(os.environ.get(k, d))
            except ValueError:
                return d

        use = os.environ.get("FORTUNA_USE_GPU", "1").strip().lower() not in ("0", "false", "no")
        return cls(
            use_gpu=use,
            gpu_mem_fraction=_float("FORTUNA_GPU_MEM_FRACTION", 0.75),
            gpu_min_bars=_int("FORTUNA_GPU_MIN_BARS", 32),
            gpu_min_unique_series=_int("FORTUNA_GPU_MIN_SERIES", 2),
            gpu_max_series_per_batch=_int("FORTUNA_GPU_MAX_SERIES", 64),
            cpu_max_workers_when_gpu=_int("FORTUNA_CPU_WORKERS_GPU", 1),
        )

    def effective_cpu_workers(self, requested: int) -> int:
        if self.use_gpu and is_gpu_available():
            return min(requested, self.cpu_max_workers_when_gpu)
        return requested

    def should_gpu_indicators(
        self,
        n_bars: int,
        n_unique_series: int,
    ) -> bool:
        if not self.use_gpu or not is_gpu_available():
            return False
        min_bars = self.gpu_min_bars
        min_series = self.gpu_min_unique_series
        if self.prefer_gpu_for_param_sweep:
            min_bars = min(min_bars, 20)
            min_series = min(min_series, 1)
        if n_bars < min_bars:
            return False
        if n_unique_series < min_series:
            return False
        info = gpu_device_info(self.gpu_mem_fraction)
        if info is None:
            return False
        need = estimate_indicator_batch_bytes(n_bars, n_unique_series)
        if need > info.usable_vram_bytes:
            return n_unique_series <= 2  # try tiny batches only
        return True

    def series_batch_size(self, n_bars: int, n_series: int) -> int:
        info = gpu_device_info(self.gpu_mem_fraction)
        if info is None:
            return min(n_series, self.gpu_max_series_per_batch)
        per_series = estimate_indicator_batch_bytes(n_bars, 1)
        fit = max(1, info.usable_vram_bytes // max(per_series, 1))
        return min(n_series, fit, self.gpu_max_series_per_batch)


@lru_cache(maxsize=1)
def get_compute_policy() -> ComputePolicy:
    return ComputePolicy.from_env()
