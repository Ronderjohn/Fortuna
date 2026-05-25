"""Pytest fixtures — session-scoped where safe to avoid repeated heavy setup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

OHLCV_BARS = 60


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ema_crossover_path(project_root: Path) -> Path:
    return project_root / "strategies" / "generated" / "ema_crossover.json"


@pytest.fixture(scope="session")
def ema_crossover_strategy(ema_crossover_path: Path):
    from fortuna.strategy.loader import load_strategy

    return load_strategy(ema_crossover_path)


def _make_ohlcv(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture(scope="session")
def sample_ohlcv() -> pd.DataFrame:
    return _make_ohlcv(OHLCV_BARS)


@pytest.fixture(scope="session")
def sample_ohlcv_backtest(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    return sample_ohlcv


@pytest.fixture(scope="session")
def backtest_engine():
    pytest.importorskip("vectorbt")
    from fortuna.backtesting.engine import BacktestEngine

    return BacktestEngine()


@pytest.fixture(scope="session")
def numpy_backtest_runner():
    from fortuna.backtesting.numpy_runner import NumPyBacktestRunner

    return NumPyBacktestRunner()


@pytest.fixture(scope="session")
def indicator_engine():
    from fortuna.indicators.engine import IndicatorEngine

    return IndicatorEngine()


@pytest.fixture(scope="session")
def strategy_compiler():
    from fortuna.backtesting.compiler import StrategyCompiler

    return StrategyCompiler()
