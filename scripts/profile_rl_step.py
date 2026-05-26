#!/usr/bin/env python
"""Micro-profile the RL env hot path so we know what to optimize.

Measures the per-call cost of:
    - ``enrich_for_rl(fold.train)``         (called on every ``reset``)
    - ``FeatureBuilder.build(row, ...)``    (called on every ``step``)
    - one ``env.reset()`` + 200 ``env.step()`` round-trip
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401

import numpy as np

from fortuna.config.settings import get_settings
from fortuna.data.manager import MarketDataManager
from fortuna.features.builder import FeatureBuilder
from fortuna.features.position import PositionState
from fortuna.features.rl_indicators import enrich_for_rl
from fortuna.paper.blackbox import walk_forward_folds
from fortuna.rl.env.trading_env import FortunaTradingEnv


def main() -> int:
    s = get_settings()
    mdm = MarketDataManager(s, data_source="smartapi")
    df = mdm.get_ohlcv("ICICIBANK.NS", "5m", days=180)
    print(f"OHLCV rows: {len(df)}")

    folds = walk_forward_folds(df, train_bars=750, test_bars=375, step_bars=375)
    print(f"folds: {len(folds)}")

    # --- enrich_for_rl latency
    t0 = time.perf_counter()
    enriched = enrich_for_rl(folds[0].train)
    t1 = time.perf_counter()
    print(f"enrich_for_rl(fold.train: {len(folds[0].train)} rows): {(t1 - t0) * 1000:.1f} ms")

    # --- FeatureBuilder.build latency (warm)
    builder = FeatureBuilder(n_bars=20, warmup_bars=30)
    pos = PositionState()
    # Warm-up so the build path is in steady state.
    for i in range(50):
        builder.build(enriched.iloc[i], pos, enriched.index[i])

    n_iters = min(500, len(enriched) - 60)
    t0 = time.perf_counter()
    for i in range(50, 50 + n_iters):
        builder.build(enriched.iloc[i], pos, enriched.index[i])
    t1 = time.perf_counter()
    print(f"FeatureBuilder.build (avg of {n_iters}): {(t1 - t0) / n_iters * 1000:.3f} ms")

    # --- env.reset + 200 steps round-trip (no policy, just stepping)
    env = FortunaTradingEnv(folds=folds, n_bars=20, warmup_bars=30)

    t0 = time.perf_counter()
    env.reset()
    reset_ms = (time.perf_counter() - t0) * 1000

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(200):
        env.step(int(rng.integers(0, 5)))
    step_ms_each = (time.perf_counter() - t0) / 200 * 1000
    print(f"env.reset(): {reset_ms:.1f} ms")
    print(f"env.step() avg of 200: {step_ms_each:.3f} ms  =>  ~{1000 / step_ms_each:.0f} FPS per env")

    # --- cProfile a longer slice
    print("\n=== cProfile top time consumers (200 steps after fresh reset) ===")
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()
    env.reset()
    for _ in range(200):
        env.step(int(rng.integers(0, 5)))
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(25)
    print(buf.getvalue())

    print("\n=== cProfile top time consumers (just enrich_for_rl) ===")
    pr2 = cProfile.Profile()
    pr2.enable()
    for _ in range(3):
        enrich_for_rl(folds[0].train)
    pr2.disable()
    buf2 = io.StringIO()
    pstats.Stats(pr2, stream=buf2).sort_stats("cumulative").print_stats(20)
    print(buf2.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
