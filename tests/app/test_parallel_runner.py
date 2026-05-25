"""Parallel strategy runner tests (synthetic data, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fortuna.app.parallel_runner import ParallelStrategyRunner
from fortuna.app.strategy_paths import list_strategy_paths
from fortuna.config.settings import load_settings


def _synthetic_ohlcv(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2026-04-01 09:15", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.15, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
        },
        index=idx,
    )


def test_parallel_runner_on_synthetic() -> None:
    settings = load_settings()
    paths = list_strategy_paths(settings)
    if not paths:
        return
    paths = paths[:3]
    ohlcv = _synthetic_ohlcv()
    runner = ParallelStrategyRunner(settings)
    batch = runner.run_parallel(ohlcv, paths, symbol="TEST.NS", timeframe="5m")
    assert len(batch.results) == len(paths)
    ok = sum(1 for r in batch.results.values() if r.ok)
    assert ok >= 1


def test_batch_leaderboard_winner() -> None:
    settings = load_settings()
    paths = list_strategy_paths(settings)
    if len(paths) < 2:
        return
    ohlcv = _synthetic_ohlcv(250)
    runner = ParallelStrategyRunner(settings)
    batch = runner.run_parallel(ohlcv, paths[:2], symbol="TEST.NS", timeframe="5m")
    rows = batch.leaderboard_rows()
    if rows:
        assert batch.winner == rows[0]["strategy"]
