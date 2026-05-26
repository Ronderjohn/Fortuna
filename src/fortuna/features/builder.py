"""Top-level observation builder.

Converts an enriched indicator row + position state + timestamp into a flat
float32 numpy vector ready for the RL policy. The vector layout is:

    [ window_buf  | scalar_features ]
       n_bars * n_windowed              n_scalar

Edit the registry (``features/registry.py``) to change the layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from fortuna.features.normalizer import OnlineNormalizer
from fortuna.features.position import PositionState
from fortuna.features.registry import (
    FEATURE_REGISTRY,
    TRANSFORM_FN,
    FeatureSpec,
    feature_registry_hash,
)
from fortuna.features.window import WindowBuffer


class FeatureBuilder:
    """Build one observation per env step.

    Usage:
        builder = FeatureBuilder()
        obs = builder.build(indicator_row, position_state, timestamp)
        assert obs.shape == builder.obs_shape

    Call ``builder.reset()`` at every episode boundary (fold start). Use
    ``save()/load()`` to persist normalizer state alongside the policy.
    """

    DEFAULT_N_BARS = 20

    def __init__(
        self,
        registry: Optional[list[FeatureSpec]] = None,
        n_bars: int = DEFAULT_N_BARS,
        warmup_bars: int = 30,
    ) -> None:
        reg = registry if registry is not None else FEATURE_REGISTRY
        self._registry = reg
        self._windowed = [f for f in reg if f.windowed]
        self._scalars = [f for f in reg if not f.windowed]

        self.n_windowed_features = len(self._windowed)
        self.n_scalar_features = len(self._scalars)
        self.n_bars = int(n_bars)

        self._window = WindowBuffer(self.n_bars, self.n_windowed_features)
        self._normalizer = OnlineNormalizer(
            n_features=self.n_windowed_features + self.n_scalar_features,
            warmup_bars=warmup_bars,
        )
        self._prev_close: Optional[float] = None

    @property
    def obs_shape(self) -> tuple[int]:
        return (self.n_bars * self.n_windowed_features + self.n_scalar_features,)

    @property
    def registry_hash(self) -> str:
        return feature_registry_hash(self._registry)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self._windowed] + [f.name for f in self._scalars]

    def reset(self) -> None:
        """Call at every episode start (fold boundary)."""
        self._window.reset()
        self._normalizer.reset()
        self._prev_close = None

    def build(
        self,
        indicator_row: pd.Series,
        position_state: PositionState,
        timestamp: pd.Timestamp,
    ) -> np.ndarray:
        """Convert a single bar to a normalized observation vector."""
        # Attach prior close so the pct_change transform can use it.
        if hasattr(indicator_row, "attrs"):
            try:
                indicator_row.attrs["_prev_close"] = self._prev_close
            except Exception:  # noqa: BLE001 — some Series subclasses freeze attrs
                pass

        windowed_raw = self._compute_features(self._windowed, indicator_row,
                                              position_state, timestamp)
        scalar_raw = self._compute_features(self._scalars, indicator_row,
                                            position_state, timestamp)

        all_raw = np.concatenate([windowed_raw, scalar_raw]).astype(np.float64)
        all_normed = self._normalizer.update_and_normalize(all_raw)

        windowed_normed = all_normed[: self.n_windowed_features].copy()
        scalar_normed = all_normed[self.n_windowed_features:].copy()

        for i, spec in enumerate(self._windowed):
            if spec.clip_sigma > 0:
                windowed_normed[i] = float(
                    np.clip(windowed_normed[i], -spec.clip_sigma, spec.clip_sigma)
                )

        for i, spec in enumerate(self._scalars):
            if spec.clip_sigma > 0:
                scalar_normed[i] = float(
                    np.clip(scalar_normed[i], -spec.clip_sigma, spec.clip_sigma)
                )

        window_vec = self._window.push(windowed_normed)

        obs = np.concatenate([window_vec, scalar_normed]).astype(np.float32)

        # Update prev_close for next bar's pct_change.
        if "close" in indicator_row.index:
            try:
                self._prev_close = float(indicator_row["close"])
            except (TypeError, ValueError):
                self._prev_close = None
        return obs

    def _compute_features(
        self,
        specs: list[FeatureSpec],
        row: pd.Series,
        position: PositionState,
        timestamp: pd.Timestamp,
    ) -> np.ndarray:
        out = np.zeros(len(specs), dtype=np.float64)
        for i, spec in enumerate(specs):
            fn = TRANSFORM_FN.get(spec.transform)
            if fn is None:
                raise KeyError(f"Unknown transform '{spec.transform}' for feature {spec.name}")
            try:
                value = fn(spec.sources, row, position, timestamp)
            except Exception:  # noqa: BLE001 — bad indicator data should not kill the env
                value = 0.0
            if value is None or np.isnan(value) or np.isinf(value):
                value = 0.0
            out[i] = float(value)
        return out

    def save(self, path: str | Path) -> None:
        state: dict[str, Any] = {
            "registry_hash": self.registry_hash,
            "n_bars": self.n_bars,
            "n_windowed_features": self.n_windowed_features,
            "n_scalar_features": self.n_scalar_features,
            "feature_names": self.feature_names,
            "obs_shape": list(self.obs_shape),
            "normalizer": self._normalizer.state_dict(),
        }
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        saved_hash = state.get("registry_hash")
        if saved_hash and saved_hash != self.registry_hash:
            raise ValueError(
                f"FeatureRegistry hash mismatch: checkpoint={saved_hash}, "
                f"current={self.registry_hash}. Refusing to load to avoid silent obs drift."
            )
        if int(state.get("n_bars", self.n_bars)) != self.n_bars:
            raise ValueError(
                f"n_bars mismatch: checkpoint={state['n_bars']}, current={self.n_bars}"
            )
        self._normalizer.load_state_dict(state["normalizer"])
