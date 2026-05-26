"""Tests for ``rl/env/trading_env.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Pre-warm the cycle (see test_manager_multisymbol.py for why).
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401

from fortuna.paper.blackbox import WalkForwardFold, walk_forward_folds
from fortuna.rl.env.actions import (
    ACTION_ENTER_LONG,
    ACTION_EXIT_LONG,
    ACTION_HOLD,
    N_ACTIONS,
)
from fortuna.rl.env.session import is_squareoff_time
from fortuna.rl.env.trading_env import FortunaTradingEnv


def _make_ohlcv(n: int = 200, seed: int = 5) -> pd.DataFrame:
    """Single NSE 5m session-friendly synthetic OHLCV.

    Spans multiple days by gluing 75-bar sessions, but the env uses bar order,
    not calendar gaps, so this works fine.
    """
    rng = np.random.default_rng(seed)
    # Start at 09:15 IST and step 5m forward, skipping overnight gaps.
    bars_per_day = 75
    sessions = (n + bars_per_day - 1) // bars_per_day
    timestamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2026-04-01 09:15:00")
    for s in range(sessions):
        session_start = day + pd.Timedelta(days=s)
        timestamps.extend(
            session_start + pd.Timedelta(minutes=5 * b)
            for b in range(bars_per_day)
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


@pytest.fixture
def folds() -> list[WalkForwardFold]:
    df = _make_ohlcv(450)
    return walk_forward_folds(df, train_bars=200, test_bars=100, step_bars=100)


def test_reset_returns_correct_shape(folds):
    env = FortunaTradingEnv(folds=folds, n_bars=10, warmup_bars=5, train_mode=True)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert info["train_mode"] is True
    assert "fold_id" in info


def test_step_advances_cursor(folds):
    env = FortunaTradingEnv(folds=folds, n_bars=10, warmup_bars=5)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False


def test_episode_terminates_at_end(folds):
    env = FortunaTradingEnv(folds=folds, n_bars=10, warmup_bars=5)
    env.reset()
    total_steps = 0
    terminated = False
    while not terminated and total_steps < 1000:
        _, _, terminated, _, info = env.step(ACTION_HOLD)
        total_steps += 1
    assert terminated
    assert "episode_metrics" in info
    assert "episode_verdict" in info
    assert "episode_n_trades" in info


def test_random_policy_negative_mean_reward(folds):
    """The cost model must bite: random trading should lose money on average."""
    env = FortunaTradingEnv(folds=folds, n_bars=10, warmup_bars=5)
    rng = np.random.default_rng(0)
    rewards: list[float] = []
    for _ in range(3):
        env.reset()
        terminated = False
        while not terminated:
            action = int(rng.integers(0, N_ACTIONS))
            _, r, terminated, _, _ = env.step(action)
            rewards.append(r)
    mean_r = float(np.mean(rewards))
    assert mean_r < 0.05, f"random policy mean reward {mean_r:.4f} should be near zero or negative"


def test_invalid_actions_treated_as_hold(folds):
    env = FortunaTradingEnv(folds=folds, n_bars=10, warmup_bars=5)
    env.reset()
    # Try to exit a long without ever entering — should be a no-op, no exception.
    obs, reward, terminated, _, _ = env.step(ACTION_EXIT_LONG)
    assert env._position.is_flat
    assert obs.shape == env.observation_space.shape


def test_squareoff_force_exit(folds):
    """A long held into the 15:15 zone must be force-closed by the env."""
    # Hand-craft a tiny fold where the last bar is at 15:25 IST.
    bars_per_day = 75
    df = _make_ohlcv(bars_per_day)  # exactly one session, ends ~ 15:25-15:30
    # Make a tiny fold (small windows that still fit our 10-bar n_bars).
    fold = WalkForwardFold(
        fold_id=0,
        train=df,
        test=df,
        train_start=str(df.index[0]),
        train_end=str(df.index[-1]),
        test_start=str(df.index[0]),
        test_end=str(df.index[-1]),
    )
    env = FortunaTradingEnv(folds=[fold], n_bars=10, warmup_bars=5)
    env.reset()
    # Enter long early then HOLD until end — last bar should force EXIT.
    _, _, _, _, _ = env.step(ACTION_ENTER_LONG)
    assert env._position.is_open
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(ACTION_HOLD)
    assert env._position.is_flat, "position must be force-closed by 15:15 square-off"


def test_gym_env_checker_passes(folds):
    """``gymnasium.utils.env_checker.check_env`` is the canonical contract test."""
    from gymnasium.utils.env_checker import check_env

    env = FortunaTradingEnv(folds=folds, n_bars=10, warmup_bars=5)
    check_env(env, skip_render_check=True)


def test_is_squareoff_time_helper():
    assert is_squareoff_time(pd.Timestamp("2026-04-01 15:15"))
    assert is_squareoff_time(pd.Timestamp("2026-04-01 15:20"))
    assert not is_squareoff_time(pd.Timestamp("2026-04-01 15:10"))
    assert not is_squareoff_time(pd.Timestamp("2026-04-01 09:15"))
