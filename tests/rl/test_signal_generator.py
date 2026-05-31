"""Tests for ``rl/inference/signal_generator.py``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fortuna.app.live_signals import (
    SignalType,
    compute_live_signals_with_rl,
)

# Pre-warm import cycle.
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401
from fortuna.features.builder import FeatureBuilder
from fortuna.features.position import PositionState
from fortuna.features.registry import feature_registry_hash
from fortuna.features.rl_indicators import enrich_for_rl_last_bar
from fortuna.rl.inference import RLSignalGenerator, resolve_live_checkpoint_dir
from fortuna.rl.training.checkpoint import OOSMetricsSummary, PolicyCheckpoint


def _make_ohlcv(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-01 09:15", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_generator_missing_checkpoint_is_unavailable(tmp_path):
    gen = RLSignalGenerator(tmp_path / "does_not_exist")
    assert gen.is_available is False
    assert gen.predict(pd.Series(dtype=float), PositionState(), pd.Timestamp.now()) is None


def test_generator_with_no_path_is_unavailable():
    gen = RLSignalGenerator(None)
    assert gen.is_available is False


def test_compute_live_signals_with_rl_falls_back_when_unavailable(project_root):
    gen = RLSignalGenerator(None)
    df = _make_ohlcv()
    paths = list((project_root / "strategies" / "generated").glob("*.json"))[:2]
    signals = compute_live_signals_with_rl(paths, df, rl_generator=gen)
    # Without an RL policy the output looks like compute_live_signals output.
    rl_keys = [k for k in signals if k.startswith("RL:")]
    assert not rl_keys


def test_resolve_live_checkpoint_dir_returns_none_when_empty(tmp_path):
    assert resolve_live_checkpoint_dir(tmp_path) is None


def test_resolve_live_checkpoint_dir_with_pointer_file(tmp_path):
    # Build a fake validated checkpoint dir with a sentinel policy.zip.
    run_id = "20260601_120000"
    cp_dir = tmp_path / "validated" / run_id
    cp_dir.mkdir(parents=True)
    (cp_dir / "policy.zip").write_bytes(b"\x00")
    (cp_dir / "metadata.json").write_text("{}")

    (tmp_path / "live.json").write_text(json.dumps({"run_id": run_id}))

    resolved = resolve_live_checkpoint_dir(tmp_path)
    assert resolved == cp_dir


def test_resolve_live_checkpoint_dir_falls_back_to_latest_validated(tmp_path):
    for run_id in ("20260601_120000", "20260602_120000"):
        d = tmp_path / "validated" / run_id
        d.mkdir(parents=True)
        (d / "policy.zip").write_bytes(b"\x00")

    resolved = resolve_live_checkpoint_dir(tmp_path)
    assert resolved is not None
    assert resolved.name == "20260602_120000"


def test_resolve_live_checkpoint_dir_can_disable_latest_validated_fallback(tmp_path):
    d = tmp_path / "validated" / "20260602_120000"
    d.mkdir(parents=True)
    (d / "policy.zip").write_bytes(b"\x00")

    resolved = resolve_live_checkpoint_dir(tmp_path, allow_latest_validated=False)

    assert resolved is None


def test_action_signal_mapping_complete():
    """All five discrete actions must be reachable through ACTION_TO_SIGNAL."""
    from fortuna.rl.env.actions import (
        ACTION_ENTER_LONG,
        ACTION_ENTER_SHORT,
        ACTION_EXIT_LONG,
        ACTION_EXIT_SHORT,
        ACTION_HOLD,
        ACTION_TO_SIGNAL,
    )

    assert ACTION_TO_SIGNAL[ACTION_HOLD] == SignalType.HOLD
    assert ACTION_TO_SIGNAL[ACTION_ENTER_LONG] == SignalType.BUY
    assert ACTION_TO_SIGNAL[ACTION_EXIT_LONG] == SignalType.EXIT
    assert ACTION_TO_SIGNAL[ACTION_ENTER_SHORT] == SignalType.SELL
    assert ACTION_TO_SIGNAL[ACTION_EXIT_SHORT] == SignalType.EXIT


def test_generator_loads_real_checkpoint(tmp_path):
    """End-to-end: train a tiny PPO, save, reload via RLSignalGenerator."""
    from fortuna.rl.env.reward import RewardConfig
    from fortuna.rl.training.trainer import FortunaRLTrainer, TrainerConfig

    df = _make_ohlcv(400, seed=7)

    class _FakeMDM:
        def get_ohlcv(self, *a, **k):
            return df

    import fortuna.rl.training.trainer as trainer_mod

    orig = trainer_mod.MarketDataManager
    trainer_mod.MarketDataManager = lambda *a, **k: _FakeMDM()
    try:
        cfg = TrainerConfig(
            symbol="SYN",
            timeframe="5m",
            days=30,
            train_bars=150,
            test_bars=60,
            step_bars=60,
            min_folds=2,
            total_timesteps=200,
            n_envs=1,
            n_bars=8,
            warmup_bars=4,
            ppo_n_steps=64,
            batch_size=16,
            n_epochs=1,
            checkpoint_dir=tmp_path,
            eval_freq=99_999,
            early_stop_patience=99,
            reward_config=RewardConfig(),
        )
        meta = FortunaRLTrainer(cfg).train()
    finally:
        trainer_mod.MarketDataManager = orig

    candidates = list(tmp_path.glob("validated/*/")) + list(tmp_path.glob("rejected/*/"))
    cp_dir = candidates[0]

    gen = RLSignalGenerator(cp_dir)
    assert gen.is_available

    if not gen.is_advisory_ready:
        meta = PolicyCheckpoint.read(cp_dir / "metadata.json")
        meta.advisory_ready = True
        meta.failure_modes = []
        meta.write(cp_dir / "metadata.json")
        gen = RLSignalGenerator(cp_dir)

    sig = gen.predict_from_ohlcv(df, PositionState())
    assert sig is not None
    assert sig.signal in set(SignalType)
    assert sig.metadata["run_id"] == meta.run_id


def _write_minimal_checkpoint(tmp_path: Path, *, hash_value: str | None = None) -> Path:
    cp_dir = tmp_path / "cp"
    cp_dir.mkdir()
    (cp_dir / "policy.zip").write_bytes(b"\x00")
    meta = PolicyCheckpoint(
        run_id="unit",
        symbol="TEST.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=1),
        verdict_passed=True,
        feature_registry_hash=hash_value or feature_registry_hash(),
        advisory_ready=True,
    )
    meta.write(cp_dir / "metadata.json")
    builder = FeatureBuilder(n_bars=8)
    builder.save(cp_dir / "normalizer.json")
    return cp_dir


def test_generator_rejects_feature_hash_mismatch(tmp_path):
    cp_dir = _write_minimal_checkpoint(tmp_path, hash_value="deadbeef")
    gen = RLSignalGenerator(cp_dir)
    assert gen.is_available is False
    assert gen.load_error is not None


def test_generator_rejects_missing_normalizer(tmp_path):
    cp_dir = _write_minimal_checkpoint(tmp_path)
    (cp_dir / "normalizer.json").unlink()
    gen = RLSignalGenerator(cp_dir)
    assert gen.is_available is False
    assert "normalizer" in (gen.load_error or "").lower()


def test_generator_predict_blocked_when_not_advisory_ready(tmp_path):
    cp_dir = _write_minimal_checkpoint(tmp_path)
    meta = PolicyCheckpoint.read(cp_dir / "metadata.json")
    meta.advisory_ready = False
    meta.failure_modes = ["verdict_failed"]
    meta.write(cp_dir / "metadata.json")
    # Load will succeed only if PPO.load works — skip PPO by testing property path
    gen = RLSignalGenerator(None)
    gen._model = object()
    gen._builder = FeatureBuilder(n_bars=8)
    gen._metadata = meta
    assert gen.is_available is True
    assert gen.is_advisory_ready is False
    row = enrich_for_rl_last_bar(_make_ohlcv())
    assert gen.predict(row, PositionState(), row.name) is None


def test_generator_blocks_non_advisory_active_regime_policy(tmp_path):
    base_meta = PolicyCheckpoint(
        run_id="base",
        symbol="TEST.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=1),
        verdict_passed=True,
        advisory_ready=True,
    )
    regime_meta = PolicyCheckpoint(
        run_id="trend",
        symbol="TEST.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=0),
        verdict_passed=False,
        advisory_ready=False,
        failure_modes=["verdict_failed"],
    )
    gen = RLSignalGenerator(None)
    gen._model = object()
    gen._builder = FeatureBuilder(n_bars=8)
    gen._metadata = base_meta
    gen._regime_policies["TRENDING"] = (object(), FeatureBuilder(n_bars=8), regime_meta)
    gen.set_active_regime("TRENDING")

    row = enrich_for_rl_last_bar(_make_ohlcv())

    assert gen.is_advisory_ready is False
    assert gen.predict(row, PositionState(), row.name) is None
