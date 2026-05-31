"""Tests for SignalScorer train/save/load/predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fortuna.ml.features import ML_FEATURE_NAMES
from fortuna.ml.signal_scorer import SignalScorer, score_or_neutral
from fortuna.ml.types import LabelConfig, ScorerMetadata, SplitRange


def _synthetic_xy(n: int = 30, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(ML_FEATURE_NAMES)))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_fit_predict_proba_in_unit_interval():
    X, y = _synthetic_xy()
    scorer = SignalScorer(random_state=42)
    scorer.fit(X, y)
    proba = scorer.predict_proba(X)
    assert proba.shape == (len(y),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_fit_raises_on_single_class():
    X, _ = _synthetic_xy()
    y = np.zeros(len(X), dtype=int)
    scorer = SignalScorer()
    with pytest.raises(ValueError, match="both classes"):
        scorer.fit(X, y)


def test_save_load_roundtrip(tmp_path: Path):
    X, y = _synthetic_xy()
    scorer = SignalScorer(random_state=42)
    meta = ScorerMetadata(
        run_id="test_run",
        symbol="RELIANCE.NS",
        timeframe="5m",
        strategy_name="orb",
        label_config=LabelConfig(),
        split_ranges=[
            SplitRange(name="train", start="2026-01-01", end="2026-01-15", bar_count=100),
            SplitRange(name="test", start="2026-01-16", end="2026-01-31", bar_count=50),
        ],
    )
    scorer.fit(X, y, metadata=meta)
    out = tmp_path / "artifact"
    scorer.save(out)

    loaded = SignalScorer.load(out)
    assert loaded is not None
    assert loaded.metadata is not None
    assert loaded.metadata.run_id == "test_run"
    assert len(loaded.metadata.split_ranges) == 2
    np.testing.assert_allclose(scorer.predict_proba(X), loaded.predict_proba(X))


def test_save_with_fresh_metadata_preserves_trained_artifact_fields(tmp_path: Path):
    X, y = _synthetic_xy()
    scorer = SignalScorer(random_state=42)
    scorer.fit(X, y)
    meta = ScorerMetadata(
        run_id="fresh_meta",
        symbol="RELIANCE.NS",
        timeframe="5m",
        strategy_name="orb",
        label_config=LabelConfig(),
    )

    out = tmp_path / "artifact"
    scorer.save(out, metadata=meta)
    loaded = SignalScorer.load(out)

    assert loaded is not None
    assert loaded.metadata is not None
    assert loaded.metadata.run_id == "fresh_meta"
    assert loaded.metadata.feature_schema_hash
    assert loaded.metadata.train_metrics.n_samples == len(y)


def test_score_signal_on_enriched_frame():
    idx = pd.date_range("2026-01-06 09:15", periods=5, freq="5min")
    enriched = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000] * 5,
            "atr": [1.0] * 5,
        },
        index=idx,
    )
    X, y = _synthetic_xy(n=40)
    scorer = SignalScorer(random_state=42)
    scorer.fit(X, y)
    result = scorer.score_signal(enriched, 3, action="BUY", bar_time=idx[3])
    assert result.available is True
    assert 0.0 <= result.probability <= 1.0
    assert result.model_id is None or isinstance(result.model_id, str)


def test_unfitted_score_returns_unavailable():
    scorer = SignalScorer()
    idx = pd.date_range("2026-01-06 09:15", periods=3, freq="5min")
    enriched = pd.DataFrame({"close": [100, 101, 102]}, index=idx)
    result = score_or_neutral(scorer, enriched, 1, action="BUY")
    assert result.available is False
    assert result.probability == 0.5


def test_score_or_neutral_catches_prediction_errors():
    class BrokenScorer:
        is_available = True

        def score_signal(self, *args, **kwargs):
            raise RuntimeError("boom")

    idx = pd.date_range("2026-01-06 09:15", periods=3, freq="5min")
    enriched = pd.DataFrame({"close": [100, 101, 102]}, index=idx)
    result = score_or_neutral(BrokenScorer(), enriched, 1, action="BUY")
    assert result.available is False
    assert "boom" in result.reason
