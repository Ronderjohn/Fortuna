"""Online z-score normalization using Welford's algorithm.

Never fits on future data — every ``update_and_normalize`` call updates the
running mean and variance from the single sample being normalized.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class OnlineNormalizer:
    """Per-feature running z-score with warm-up zero-out.

    Returns a numerical zero for any feature whose count is still below
    ``warmup_bars`` — this prevents the very first observations from
    swinging the policy with huge unnormalized values.
    """

    def __init__(self, n_features: int, warmup_bars: int = 30) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = int(n_features)
        self.warmup_bars = int(warmup_bars)
        self._n = np.zeros(self.n_features, dtype=np.float64)
        self._mean = np.zeros(self.n_features, dtype=np.float64)
        self._M2 = np.zeros(self.n_features, dtype=np.float64)

    def update_and_normalize(self, x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        if x64.shape != (self.n_features,):
            raise ValueError(
                f"input shape {x64.shape} != ({self.n_features},)"
            )

        self._n += 1
        delta = x64 - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (x64 - self._mean)

        variance = np.where(self._n > 1, self._M2 / np.maximum(self._n - 1, 1.0), 1.0)
        std = np.maximum(np.sqrt(variance), 1e-8)
        normed = (x64 - self._mean) / std

        # Zero-out features still in warm-up.
        normed = np.where(self._n < self.warmup_bars, 0.0, normed)
        return normed.astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Read-only normalization using current stats (no update)."""
        x64 = np.asarray(x, dtype=np.float64)
        variance = np.where(self._n > 1, self._M2 / np.maximum(self._n - 1, 1.0), 1.0)
        std = np.maximum(np.sqrt(variance), 1e-8)
        normed = (x64 - self._mean) / std
        normed = np.where(self._n < self.warmup_bars, 0.0, normed)
        return normed.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "warmup_bars": self.warmup_bars,
            "n": self._n.tolist(),
            "mean": self._mean.tolist(),
            "M2": self._M2.tolist(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        n = np.array(d["n"], dtype=np.float64)
        mean = np.array(d["mean"], dtype=np.float64)
        m2 = np.array(d["M2"], dtype=np.float64)
        if not (n.shape == mean.shape == m2.shape == (self.n_features,)):
            raise ValueError(
                f"normalizer state shape mismatch for n_features={self.n_features}: "
                f"n={n.shape}, mean={mean.shape}, M2={m2.shape}"
            )
        self._n = n
        self._mean = mean
        self._M2 = m2
        self.warmup_bars = int(d.get("warmup_bars", self.warmup_bars))

    def reset(self) -> None:
        self._n.fill(0.0)
        self._mean.fill(0.0)
        self._M2.fill(0.0)

    @property
    def count(self) -> np.ndarray:
        return self._n.copy()
