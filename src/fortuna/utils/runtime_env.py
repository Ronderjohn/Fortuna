"""Default runtime env for low-RAM PCs with a 4GB-class GPU (e.g. GTX 1650)."""

from __future__ import annotations

import os


def apply_low_spec_gpu_defaults() -> None:
    """
    Prefer GPU indicator batches on every research run.

    - GPU on (CuPy) for EMA/SMA/RSI/ATR sweeps
    - ``FORTUNA_LOW_MEMORY=1``: keep ProcessPool off on Windows (OOM-safe)
    - Low bar/series thresholds so arena windows (40–80 bars) still use GPU

    Existing env vars are not overwritten (use ``setdefault``).
    """
    os.environ.setdefault("FORTUNA_USE_GPU", "1")
    os.environ.setdefault("FORTUNA_LOW_MEMORY", "1")
    os.environ.setdefault("FORTUNA_GPU_MIN_BARS", "32")
    os.environ.setdefault("FORTUNA_GPU_MIN_SERIES", "2")
    os.environ.setdefault("FORTUNA_GPU_MEM_FRACTION", "0.75")
    os.environ.setdefault("FORTUNA_CPU_WORKERS_GPU", "1")
