"""Tests for artifact load fail-soft behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fortuna.ml.artifacts import load_artifact, load_metadata, load_pipeline
from fortuna.ml.features import ML_FEATURE_NAMES
from fortuna.ml.signal_scorer import SignalScorer, score_or_neutral
from fortuna.ml.types import LabelConfig, ScorerMetadata, SignalScoreResult


def test_load_missing_dir_returns_none(tmp_path: Path):
    assert SignalScorer.load(tmp_path / "missing") is None
    assert load_pipeline(tmp_path / "missing") is None
    assert load_metadata(tmp_path / "missing") is None
    pipe, meta = load_artifact(tmp_path / "missing")
    assert pipe is None and meta is None


def test_score_or_neutral_none_scorer():
    import pandas as pd

    idx = pd.date_range("2026-01-06 09:15", periods=3, freq="5min")
    enriched = pd.DataFrame({"close": [100, 101, 102]}, index=idx)
    result = score_or_neutral(None, enriched, 1, action="BUY")
    assert isinstance(result, SignalScoreResult)
    assert result.available is False
    assert result.predicted_class == -1


def test_corrupt_metadata_returns_none(tmp_path: Path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "metadata.json").write_text("{not valid json", encoding="utf-8")
    assert load_metadata(bad) is None


def test_schema_hash_mismatch_blocks_load(tmp_path: Path):
    X = np.random.default_rng(0).normal(size=(20, len(ML_FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)
    scorer = SignalScorer(random_state=42)
    meta = ScorerMetadata(
        run_id="hash_test",
        symbol="X",
        timeframe="5m",
        label_config=LabelConfig(),
    )
    scorer.fit(X, y, metadata=meta)
    out = tmp_path / "artifact"
    scorer.save(out)
    # Corrupt stored hash after save to simulate schema drift.
    import json

    meta_path = out / "metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["feature_schema_hash"] = "deadbeef"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    assert SignalScorer.load(out) is None
