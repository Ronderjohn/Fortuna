"""Reusable pipeline timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def timed_step(name: str) -> Iterator[dict[str, str]]:
    """Print and log ``[TIMING] name: Nms`` when the block exits.

    Yield a dict; set key/value pairs inside the block to append detail to the line.
    """
    t0 = time.perf_counter()
    details: dict[str, str] = {}
    try:
        yield details
    finally:
        ms = (time.perf_counter() - t0) * 1000
        suffix = ""
        if details:
            suffix = " (" + ", ".join(f"{k}={v}" for k, v in details.items()) + ")"
        msg = f"[TIMING] {name}: {ms:.0f}ms{suffix}"
        print(msg, flush=True)
        logger.info(msg)
