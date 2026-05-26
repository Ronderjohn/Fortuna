"""Rolling NumPy window buffer for the RL observation builder."""

from __future__ import annotations

import numpy as np


class WindowBuffer:
    """Fixed-size rolling buffer of shape ``(n_bars, n_features)``.

    Push one row per env step; ``push()`` returns the flattened
    ``(n_bars * n_features,)`` vector ready for the policy network.

    Implementation note: ``np.roll`` is allocation-free here because we feed
    its output back into ``self._buf`` via ``out=`` — the buffer is never
    re-allocated after construction.
    """

    def __init__(self, n_bars: int, n_features: int) -> None:
        if n_bars <= 0 or n_features <= 0:
            raise ValueError("n_bars and n_features must be positive")
        self.n_bars = int(n_bars)
        self.n_features = int(n_features)
        self._buf = np.zeros((self.n_bars, self.n_features), dtype=np.float32)

    def push(self, feature_row: np.ndarray) -> np.ndarray:
        if feature_row.shape != (self.n_features,):
            raise ValueError(
                f"feature_row shape mismatch: got {feature_row.shape}, "
                f"expected ({self.n_features},)"
            )
        # In-place shift up one row, then write the new row at the bottom.
        # Avoids a fresh allocation on every step (numpy.roll always copies).
        if self.n_bars > 1:
            self._buf[:-1] = self._buf[1:]
        self._buf[-1] = feature_row.astype(np.float32, copy=False)
        return self._buf.reshape(-1)

    def latest(self) -> np.ndarray:
        return self._buf[-1].copy()

    def snapshot(self) -> np.ndarray:
        return self._buf.copy()

    def reset(self) -> None:
        self._buf.fill(0.0)

    @property
    def shape(self) -> tuple[int, int]:
        return self._buf.shape
