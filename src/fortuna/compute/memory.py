"""VRAM/RAM estimation for batch sizing."""

from __future__ import annotations

BYTES_PER_FLOAT64 = 8


def estimate_indicator_batch_bytes(n_bars: int, n_series: int) -> int:
    """Rough GPU memory for OHLCV + indicator outputs."""
    ohlcv = n_bars * 5 * BYTES_PER_FLOAT64
    outputs = n_bars * n_series * BYTES_PER_FLOAT64
    overhead = 64 * 1024 * 1024  # kernel workspace
    return ohlcv + outputs + overhead


def chunk_count_for_vram(
    n_items: int,
    nbytes_per_item: int,
    usable_vram: int,
) -> int:
    if usable_vram <= 0 or nbytes_per_item <= 0:
        return max(1, n_items)
    fit = max(1, usable_vram // nbytes_per_item)
    return max(1, (n_items + fit - 1) // fit)
