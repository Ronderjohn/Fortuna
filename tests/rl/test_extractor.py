"""Tests for ``rl/models/extractor.py``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from fortuna.features.registry import FEATURE_REGISTRY
from fortuna.rl.models.extractor import TemporalCNNExtractor


def _counts() -> tuple[int, int]:
    n_w = sum(1 for f in FEATURE_REGISTRY if f.windowed)
    n_s = sum(1 for f in FEATURE_REGISTRY if not f.windowed)
    return n_w, n_s


def test_extractor_infers_n_bars_from_obs_shape():
    n_w, n_s = _counts()
    n_bars = 20
    flat = n_bars * n_w + n_s
    obs_space = Box(low=-np.inf, high=np.inf, shape=(flat,), dtype=np.float32)

    ext = TemporalCNNExtractor(obs_space, features_dim=128)
    assert ext.n_bars == n_bars
    assert ext.n_windowed == n_w
    assert ext.n_scalar == n_s


def test_extractor_forward_shape():
    n_w, n_s = _counts()
    n_bars = 20
    flat = n_bars * n_w + n_s
    obs_space = Box(low=-np.inf, high=np.inf, shape=(flat,), dtype=np.float32)
    ext = TemporalCNNExtractor(obs_space, features_dim=128)

    batch = torch.randn(8, flat)
    out = ext(batch)
    assert out.shape == (8, 128)


def test_extractor_rejects_obs_size_mismatch():
    obs_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
    with pytest.raises(ValueError):
        TemporalCNNExtractor(obs_space, features_dim=128, n_bars=20)


def test_ppo_can_load_cnn_extractor():
    """Sanity: SB3 PPO with the CNN extractor should construct without error."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    # Pre-warm import cycle.
    from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401

    import pandas as pd

    from fortuna.paper.blackbox import walk_forward_folds
    from fortuna.rl.env.trading_env import FortunaTradingEnv

    rng = np.random.default_rng(0)
    n = 400
    bars_per_day = 75
    sessions = (n + bars_per_day - 1) // bars_per_day
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2026-04-01 09:15")
    for s in range(sessions):
        d0 = day + pd.Timedelta(days=s)
        stamps.extend(d0 + pd.Timedelta(minutes=5 * b) for b in range(bars_per_day))
    stamps = stamps[:n]
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.1, n),
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
            "close": close,
            "volume": rng.integers(1_000, 5_000, n).astype(float),
        },
        index=pd.DatetimeIndex(stamps),
    )
    folds = walk_forward_folds(df, train_bars=200, test_bars=80, step_bars=80)
    env = FortunaTradingEnv(folds=folds, n_bars=10, warmup_bars=5)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": TemporalCNNExtractor,
            "features_extractor_kwargs": {"features_dim": 64, "n_bars": 10},
        },
        n_steps=64,
        batch_size=16,
        n_epochs=1,
        verbose=0,
    )
    assert model is not None
