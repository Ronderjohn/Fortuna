"""End-to-end PPO smoke test — verifies the trainer can run a few hundred steps.

Marked ``@pytest.mark.slow`` so it stays out of the default ``pytest -q`` sweep.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

# Pre-warm import cycle.
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401


pytestmark = pytest.mark.slow


def _make_synthetic_ohlcv(n: int = 600, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bars_per_day = 75
    sessions = (n + bars_per_day - 1) // bars_per_day
    timestamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2026-04-01 09:15:00")
    for s in range(sessions):
        session_start = day + pd.Timedelta(days=s)
        timestamps.extend(
            session_start + pd.Timedelta(minutes=5 * b) for b in range(bars_per_day)
        )
    timestamps = timestamps[:n]
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(timestamps),
    )


def test_trainer_runs_500_timesteps(tmp_path, monkeypatch):
    from fortuna.rl.env.reward import RewardConfig
    from fortuna.rl.training.trainer import FortunaRLTrainer, TrainerConfig

    df = _make_synthetic_ohlcv(600)

    # Patch the data manager so the trainer does not hit the network.
    monkeypatch.setattr(
        "fortuna.rl.training.trainer.MarketDataManager",
        lambda *a, **kw: type("FakeMDM", (), {
            "get_ohlcv": lambda self, *a, **k: df,
        })(),
    )

    cfg = TrainerConfig(
        symbol="SYN",
        timeframe="5m",
        days=30,
        train_bars=200,
        test_bars=80,
        step_bars=80,
        min_folds=2,
        total_timesteps=500,
        n_envs=1,
        n_bars=10,
        warmup_bars=5,
        ppo_n_steps=128,
        batch_size=32,
        n_epochs=2,
        use_subproc=False,
        seed=0,
        checkpoint_dir=tmp_path,
        eval_freq=10_000,  # do not trigger eval mid-run for a 500-step smoke
        early_stop_patience=99,
        reward_config=RewardConfig(),
    )

    trainer = FortunaRLTrainer(cfg)
    meta = trainer.train()

    # Find checkpoint dir
    candidates = list(tmp_path.glob("validated/*/")) + list(tmp_path.glob("rejected/*/"))
    assert candidates, "trainer did not produce a checkpoint directory"
    final_dir = candidates[0]

    assert (final_dir / "policy.zip").exists()
    assert (final_dir / "normalizer.json").exists()
    assert (final_dir / "metadata.json").exists()

    payload = json.loads((final_dir / "metadata.json").read_text())
    assert payload["symbol"] == "SYN"
    assert payload["total_timesteps"] == 500
    assert "feature_registry_hash" in payload
    assert "oos_metrics" in payload

    assert meta.run_id in str(final_dir)
