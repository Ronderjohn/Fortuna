"""Simple request throttle for SmartAPI REST (~3 req/s)."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Thread-safe minimum interval between calls."""

    def __init__(self, requests_per_second: float = 3.0) -> None:
        self._min_interval = 1.0 / max(requests_per_second, 0.1)
        self._lock = threading.Lock()
        self._last_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_at
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_at = time.monotonic()
