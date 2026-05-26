"""Tests for ``features/builder.py`` + position/registry integration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

# Warm the data.manager import cycle (see tests/data/test_manager_multisymbol.py).
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401

from fortuna.features import (
    FEATURE_REGISTRY,
    FeatureBuilder,
    PositionState,
    feature_registry_hash,
)
from fortuna.features.rl_indicators import enrich_for_rl


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-04-01 09:15", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_obs_shape_matches_registry():
    builder = FeatureBuilder(n_bars=20)
    n_windowed = sum(1 for f in FEATURE_REGISTRY if f.windowed)
    n_scalar = sum(1 for f in FEATURE_REGISTRY if not f.windowed)
    assert builder.n_windowed_features == n_windowed
    assert builder.n_scalar_features == n_scalar
    assert builder.obs_shape == (20 * n_windowed + n_scalar,)


def test_build_returns_finite_float32():
    builder = FeatureBuilder(n_bars=10, warmup_bars=5)
    enriched = enrich_for_rl(_make_ohlcv(40))
    pos = PositionState()
    for i in range(len(enriched)):
        obs = builder.build(enriched.iloc[i], pos, enriched.index[i])
    assert obs.dtype == np.float32
    assert obs.shape == builder.obs_shape
    assert np.all(np.isfinite(obs))


def test_build_during_warmup_returns_zeros():
    builder = FeatureBuilder(n_bars=10, warmup_bars=50)
    enriched = enrich_for_rl(_make_ohlcv(40))
    pos = PositionState()
    obs = builder.build(enriched.iloc[10], pos, enriched.index[10])
    # During warm-up everything is zeroed by the normalizer.
    np.testing.assert_allclose(obs, np.zeros_like(obs))


def test_reset_clears_window_and_normalizer():
    builder = FeatureBuilder(n_bars=10, warmup_bars=5)
    enriched = enrich_for_rl(_make_ohlcv(30))
    pos = PositionState()
    for i in range(20):
        builder.build(enriched.iloc[i], pos, enriched.index[i])

    builder.reset()
    obs = builder.build(enriched.iloc[20], pos, enriched.index[20])
    # First call after reset is the first sample of the fresh normalizer -> zeros.
    np.testing.assert_allclose(obs, np.zeros_like(obs))


def test_save_load_round_trip(tmp_path):
    builder = FeatureBuilder(n_bars=10, warmup_bars=5)
    enriched = enrich_for_rl(_make_ohlcv(40))
    pos = PositionState()
    for i in range(30):
        builder.build(enriched.iloc[i], pos, enriched.index[i])

    out = tmp_path / "normalizer.json"
    builder.save(out)
    saved = json.loads(out.read_text())
    assert saved["registry_hash"] == feature_registry_hash()

    fresh = FeatureBuilder(n_bars=10, warmup_bars=5)
    fresh.load(out)
    # Normalizer stats should match.
    np.testing.assert_allclose(
        np.array(fresh._normalizer.state_dict()["mean"]),
        np.array(builder._normalizer.state_dict()["mean"]),
    )


def test_save_load_rejects_shape_drift(tmp_path):
    builder = FeatureBuilder(n_bars=10, warmup_bars=5)
    out = tmp_path / "normalizer.json"
    builder.save(out)

    other = FeatureBuilder(n_bars=20, warmup_bars=5)  # different n_bars
    with pytest.raises(ValueError):
        other.load(out)


def test_position_unrealized_pct_long():
    pos = PositionState()
    pos.open_long(price=100.0, timestamp=pd.Timestamp("2026-04-01 10:00"))
    assert pos.unrealized_pct(110.0) == pytest.approx(10.0)


def test_position_unrealized_pct_short():
    pos = PositionState()
    pos.open_short(price=100.0, timestamp=pd.Timestamp("2026-04-01 10:00"))
    assert pos.unrealized_pct(90.0) == pytest.approx(10.0)


def test_position_close_resets_state():
    pos = PositionState()
    pos.open_long(price=100.0, timestamp=pd.Timestamp("2026-04-01 10:00"))
    realized = pos.close(price=105.0, timestamp=pd.Timestamp("2026-04-01 10:30"))
    assert realized == pytest.approx(5.0)
    assert pos.is_flat
    assert pos.realized_pnl_pct == pytest.approx(5.0)


def test_feature_registry_hash_stable():
    h1 = feature_registry_hash()
    h2 = feature_registry_hash()
    assert h1 == h2
