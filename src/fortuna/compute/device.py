"""GPU device detection and memory management (4GB-class cards)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

_cp: Any = None
_cupy_import_attempted = False


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def is_gpu_enabled() -> bool:
    return _env_flag("FORTUNA_USE_GPU", True)


def get_cupy():
    """Lazy CuPy import; returns None if unavailable."""
    global _cp, _cupy_import_attempted
    if not is_gpu_enabled():
        return None
    if _cupy_import_attempted:
        return _cp
    _cupy_import_attempted = True
    try:
        import cupy as cp  # type: ignore

        _ = cp.cuda.runtime.getDeviceCount()
        _cp = cp
        logger.info("CuPy loaded: CUDA device ready")
    except Exception as e:
        logger.debug("CuPy not available: %s", e)
        _cp = None
    return _cp


def is_gpu_available() -> bool:
    return get_cupy() is not None


@dataclass
class GpuDeviceInfo:
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    compute_capability: tuple[int, int]
    usable_vram_bytes: int


@lru_cache(maxsize=1)
def gpu_device_info(mem_fraction: float = 0.75) -> Optional[GpuDeviceInfo]:
    cp = get_cupy()
    if cp is None:
        return None
    try:
        dev = cp.cuda.Device(0)
        dev.use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        total = int(dev.mem_info[1])
        free = int(dev.mem_info[0])
        cc = dev.compute_capability
        usable = int(total * mem_fraction)
        return GpuDeviceInfo(
            name=name,
            total_vram_bytes=total,
            free_vram_bytes=free,
            compute_capability=cc,
            usable_vram_bytes=usable,
        )
    except Exception as e:
        logger.warning("GPU probe failed: %s", e)
        return None


def free_gpu_memory() -> None:
    cp = get_cupy()
    if cp is None:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def synchronize_gpu() -> None:
    cp = get_cupy()
    if cp is not None:
        cp.cuda.Stream.null.synchronize()
