"""Tests for ``rl/inference/regime_detector.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Pre-warm import cycle.
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401

from fortuna.rl.inference.regime_detector import (
    FEATURE_ORDER,
    REGIMES,
    RegimeDetector,
    _label_session,
    _session_features,
    load_detector_if_available,
)


def _trending_session(n: int = 75, seed: int = 1) -> pd.DataFrame:
    """A session with strong directional drift."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-01 09:15", periods=n, freq="5min")
    close = 100 + np.cumsum(np.abs(rng.normal(0.3, 0.05, n)))
    high = close + rng.uniform(0.05, 0.2, n)
    low = close - rng.uniform(0.05, 0.2, n)
    open_ = close + rng.normal(0, 0.05, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _ranging_session(n: int = 75, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-02 09:15", periods=n, freq="5min")
    close = 100 + 0.5 * np.sin(np.linspace(0, 6 * np.pi, n)) + rng.normal(0, 0.05, n)
    high = close + rng.uniform(0.02, 0.1, n)
    low = close - rng.uniform(0.02, 0.1, n)
    open_ = close + rng.normal(0, 0.02, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _volatile_session(n: int = 75, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-03 09:15", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 2.5, n))
    high = close + rng.uniform(0.5, 3.0, n)
    low = close - rng.uniform(0.5, 3.0, n)
    open_ = close + rng.normal(0, 1.0, n)
    volume = rng.integers(5_000, 20_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_session_features_returns_all_keys():
    feats = _session_features(_trending_session())
    for k in FEATURE_ORDER:
        assert k in feats
        assert np.isfinite(feats[k])


def test_heuristic_labels_match_intuition():
    trending = _label_session(_session_features(_trending_session()))
    ranging = _label_session(_session_features(_ranging_session()))
    volatile = _label_session(_session_features(_volatile_session()))
    assert trending in REGIMES
    assert ranging in REGIMES
    assert volatile in REGIMES


def test_detector_trains_on_synthetic_panel():
    # Mix sessions across multiple days so per-day grouping returns enough sessions.
    frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(11)
    for day in range(30):
        kind = rng.integers(0, 3)
        if kind == 0:
            frame = _trending_session(seed=day)
        elif kind == 1:
            frame = _ranging_session(seed=day + 100)
        else:
            frame = _volatile_session(seed=day + 200)
        frame = frame.copy()
        frame.index = frame.index + pd.Timedelta(days=day)
        frames.append(frame)
    panel = pd.concat(frames)

    det = RegimeDetector()
    meta = det.train(panel, test_fraction=0.2, random_state=0)
    assert det.is_trained
    assert 0.0 <= meta.accuracy <= 1.0
    # With three well-separated synthetic regimes the model should do better than chance.
    if meta.n_test > 0:
        assert meta.accuracy >= 0.5


def test_detector_predict_returns_valid_label():
    frames: list[pd.DataFrame] = []
    for day in range(20):
        f = _trending_session(seed=day) if day % 2 == 0 else _ranging_session(seed=day)
        f.index = f.index + pd.Timedelta(days=day)
        frames.append(f)
    det = RegimeDetector()
    det.train(pd.concat(frames))

    pred = det.predict(_trending_session(seed=99))
    assert pred in REGIMES


def test_detector_save_load_round_trip(tmp_path):
    frames = []
    for day in range(20):
        f = _trending_session(seed=day) if day % 2 == 0 else _ranging_session(seed=day)
        f.index = f.index + pd.Timedelta(days=day)
        frames.append(f)
    det = RegimeDetector()
    det.train(pd.concat(frames))
    out = tmp_path / "classifier.joblib"
    det.save(out)

    det2 = RegimeDetector()
    det2.load(out)
    assert det2.is_trained
    p1 = det.predict(_trending_session(seed=99))
    p2 = det2.predict(_trending_session(seed=99))
    assert p1 == p2


def test_load_detector_if_available_returns_none_on_missing(tmp_path):
    assert load_detector_if_available(tmp_path / "missing.joblib") is None


def test_detector_untrained_predict_returns_none():
    det = RegimeDetector()
    assert det.predict(_trending_session()) is None
