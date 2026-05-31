"""Phase 2 feature engineering: observation space builder for RL policies.

Public surface:
    - ``FeatureSpec`` and ``FEATURE_REGISTRY``: declarative feature catalog.
    - ``WindowBuffer``: rolling NumPy buffer for windowed features.
    - ``OnlineNormalizer``: Welford z-score (no future leakage).
    - ``PositionState``: lightweight position tracker for RL environment state.
    - ``FeatureBuilder``: top-level entry point that turns an enriched
      indicator row + position state + timestamp into a flat observation vector.
    - ``RL_INDICATORS``: the canonical IndicatorSpec list the RL env always
      computes — it covers every source column referenced by FEATURE_REGISTRY.
"""

from __future__ import annotations

from fortuna.features.builder import FeatureBuilder
from fortuna.features.normalizer import OnlineNormalizer
from fortuna.features.position import PositionState
from fortuna.features.registry import (
    FEATURE_REGISTRY,
    TRANSFORM_FN,
    FeatureSpec,
    feature_registry_hash,
)
from fortuna.features.rl_indicators import RL_INDICATORS
from fortuna.features.window import WindowBuffer

__all__ = [
    "FeatureBuilder",
    "OnlineNormalizer",
    "PositionState",
    "FEATURE_REGISTRY",
    "TRANSFORM_FN",
    "FeatureSpec",
    "feature_registry_hash",
    "RL_INDICATORS",
    "WindowBuffer",
]
