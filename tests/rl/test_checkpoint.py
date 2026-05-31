"""Tests for PolicyCheckpoint metadata and advisory readiness."""

from __future__ import annotations

from pathlib import Path

from fortuna.rl.training.checkpoint import (
    OOSMetricsSummary,
    PolicyCheckpoint,
    build_failure_modes,
    compute_advisory_ready,
)


def _sample_checkpoint(**overrides) -> PolicyCheckpoint:
    base = PolicyCheckpoint(
        run_id="test_run",
        symbol="RELIANCE.NS",
        timeframe="5m",
        obs_shape=[20, 32],
        policy_type="MlpPolicy",
        total_timesteps=1000,
        n_folds=3,
        oos_metrics=OOSMetricsSummary(total_trades=5, sharpe_ratio=1.2),
        verdict_passed=True,
    )
    for key, val in overrides.items():
        setattr(base, key, val)
    return base


def test_checkpoint_write_read_roundtrip(tmp_path: Path):
    cp = _sample_checkpoint(
        baseline_sharpe=0.8,
        beats_baseline=True,
        oos_fold_metrics=[{"sharpe_ratio": 1.0, "total_trades": 2}],
    )
    cp.failure_modes = build_failure_modes(cp)
    cp.advisory_ready, _ = compute_advisory_ready(cp)
    path = tmp_path / "metadata.json"
    cp.write(path)
    loaded = PolicyCheckpoint.read(path)
    assert loaded.run_id == "test_run"
    assert loaded.baseline_sharpe == 0.8
    assert loaded.beats_baseline is True
    assert len(loaded.oos_fold_metrics) == 1


def test_compute_advisory_ready_passes_when_checks_ok():
    cp = _sample_checkpoint(beats_baseline=True)
    ready, failures = compute_advisory_ready(cp)
    assert ready is True
    assert failures == []


def test_compute_advisory_ready_fails_on_hold_lock():
    cp = _sample_checkpoint(
        oos_metrics=OOSMetricsSummary(total_trades=0, sharpe_ratio=0.0),
        verdict_passed=False,
    )
    ready, failures = compute_advisory_ready(cp)
    assert ready is False
    assert "hold_locked" in failures
    assert "verdict_failed" in failures


def test_compute_advisory_ready_fails_below_baseline():
    cp = _sample_checkpoint(beats_baseline=False, baseline_sharpe=1.5)
    ready, failures = compute_advisory_ready(cp)
    assert ready is False
    assert "below_baseline" in failures


def test_from_dict_legacy_advisory_ready_defaults_to_verdict():
    payload = {
        "run_id": "legacy",
        "symbol": "X.NS",
        "timeframe": "5m",
        "obs_shape": [1, 2],
        "policy_type": "MlpPolicy",
        "total_timesteps": 1,
        "n_folds": 1,
        "oos_metrics": {"total_trades": 3, "sharpe_ratio": 0.5},
        "verdict_passed": True,
    }
    cp = PolicyCheckpoint.from_dict(payload)
    assert cp.advisory_ready is True
