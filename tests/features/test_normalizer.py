"""Tests for ``features/normalizer.py``."""

from __future__ import annotations

import numpy as np

from fortuna.features.normalizer import OnlineNormalizer


def test_warmup_returns_zeros():
    nz = OnlineNormalizer(n_features=2, warmup_bars=10)
    rng = np.random.default_rng(0)
    for _ in range(5):
        out = nz.update_and_normalize(rng.normal(size=2))
        np.testing.assert_allclose(out, np.zeros(2))


def test_post_warmup_z_score_converges():
    """After warm-up the running stats should approximate true mean/std."""
    nz = OnlineNormalizer(n_features=1, warmup_bars=10)
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=5.0, scale=2.0, size=2000)
    last = None
    for x in samples:
        last = nz.update_and_normalize(np.array([x]))

    # Stats after 2000 samples should be very close to mean=5, std=2.
    state = nz.state_dict()
    assert abs(state["mean"][0] - 5.0) < 0.2
    variance = state["M2"][0] / (state["n"][0] - 1)
    assert abs(np.sqrt(variance) - 2.0) < 0.2

    # The last normalized value should be a valid z-score (finite and small).
    assert np.isfinite(last[0])
    assert abs(last[0]) < 10


def test_state_dict_round_trip():
    nz = OnlineNormalizer(n_features=3, warmup_bars=5)
    rng = np.random.default_rng(1)
    for _ in range(30):
        nz.update_and_normalize(rng.normal(size=3))

    state = nz.state_dict()
    nz2 = OnlineNormalizer(n_features=3, warmup_bars=5)
    nz2.load_state_dict(state)

    np.testing.assert_allclose(nz2._n, nz._n)
    np.testing.assert_allclose(nz2._mean, nz._mean)
    np.testing.assert_allclose(nz2._M2, nz._M2)


def test_reset_clears_stats():
    nz = OnlineNormalizer(n_features=2)
    for _ in range(20):
        nz.update_and_normalize(np.array([1.0, 2.0]))
    nz.reset()
    state = nz.state_dict()
    assert all(c == 0 for c in state["n"])
    assert all(m == 0 for m in state["mean"])
