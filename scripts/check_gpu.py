#!/usr/bin/env python
"""Report GPU/CuPy status and Fortuna compute policy (GTX 1650 / 4GB tuning)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.compute.device import gpu_device_info, is_gpu_available
from fortuna.compute.policy import get_compute_policy
from fortuna.compute.scheduler import get_scheduler


def main() -> int:
    policy = get_compute_policy()
    sched = get_scheduler()
    print(sched.status_line())
    print(f"  FORTUNA_USE_GPU={policy.use_gpu}")
    print(f"  mem_fraction={policy.gpu_mem_fraction}")
    print(f"  min_bars={policy.gpu_min_bars} min_series={policy.gpu_min_unique_series}")
    print(f"  cpu_workers_when_gpu={policy.cpu_max_workers_when_gpu}")

    if is_gpu_available():
        info = gpu_device_info(policy.gpu_mem_fraction)
        if info:
            print(f"  VRAM total={info.total_vram_bytes // (1024**2)}MB")
            print(f"  VRAM free={info.free_vram_bytes // (1024**2)}MB")
            print(f"  VRAM usable (~{policy.gpu_mem_fraction:.0%})={info.usable_vram_bytes // (1024**2)}MB")
            print(f"  compute_capability={info.compute_capability}")
        return 0
    print("\nInstall GPU support:")
    print("  uv sync --group gpu")
    print("  CUDA 11.x: replace cupy-cuda12x with cupy-cuda11x in pyproject.toml")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
