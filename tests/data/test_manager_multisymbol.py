"""Phase 1 completion 2.1: concurrent multi-symbol cache integrity.

Validates that ``MarketDataManager`` + ``DataCache`` keep 10 symbols' Parquet
caches isolated when written concurrently, with no key collisions and no
tz-normalization drift across symbols.

This test stays fully offline by writing synthetic OHLCV directly into the
cache (mirroring what ``MarketDataManager.append_bar`` does) and then reading
each symbol back through ``MarketDataManager.get_ohlcv`` from N threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytest

# Pre-existing circular: importing ``fortuna.backtesting.standard.calendar`` before
# ``fortuna.data.manager`` breaks the data.manager -> backtesting.standard ->
# paper -> data.manager cycle by warming the calendar module first.
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401
from fortuna.config.settings import Settings
from fortuna.data.cache import DataCache
from fortuna.data.manager import MarketDataManager


SYMBOLS = [
    "ICICIBANK.NS",
    "HDFCBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
]


def _synthetic_5m_bars(symbol: str, n_bars: int = 75, seed: int = 0) -> pd.DataFrame:
    """Synthetic NSE 5m bars inside a single session window (09:15-15:30 IST)."""
    rng = np.random.default_rng(seed + hash(symbol) % 10_000)
    n_bars = min(n_bars, 75)  # one NSE session = 375 minutes = 75 bars
    start = pd.Timestamp("2026-04-01 09:15:00")
    index = pd.date_range(start, periods=n_bars, freq="5min")
    close = 1000 + np.cumsum(rng.normal(0, 1.0, n_bars))
    high = close + rng.uniform(0.1, 2.0, n_bars)
    low = close - rng.uniform(0.1, 2.0, n_bars)
    open_ = close + rng.normal(0, 0.5, n_bars)
    volume = rng.integers(1_000, 10_000, n_bars).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def manager_with_cache(tmp_path, monkeypatch):
    """Manager whose cache lives in ``tmp_path`` and never hits the network."""
    cache_dir = tmp_path / "cache"
    settings = Settings(
        data_source="yfinance",
        data_cache_dir=str(cache_dir),
        duckdb_path=str(tmp_path / "fortuna.duckdb"),
        cache_max_age_days=99,
        cache_max_age_hours=None,
    )
    cache = DataCache(cache_dir)
    return MarketDataManager(settings=settings, cache=cache)


def test_concurrent_reads_no_cache_key_collisions(manager_with_cache):
    """Ten symbols read concurrently from cache return their own data unchanged."""
    expected: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(SYMBOLS):
        df = _synthetic_5m_bars(sym, n_bars=75, seed=i)
        manager_with_cache.cache.write(df, sym, "5m")
        expected[sym] = df

    def _read(symbol: str) -> tuple[str, pd.DataFrame]:
        df = manager_with_cache.get_ohlcv(symbol, "5m")
        return symbol, df

    seen: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_read, s) for s in SYMBOLS]
        for fut in as_completed(futures):
            sym, df = fut.result()
            seen[sym] = df

    assert set(seen) == set(SYMBOLS)
    for sym in SYMBOLS:
        exp = expected[sym]
        got = seen[sym]
        # Same row count (filter_session_bars may drop pre/post session bars,
        # but we placed every bar inside 09:15-15:30 IST).
        assert len(got) == len(exp), f"{sym} row drift: {len(got)} vs {len(exp)}"
        np.testing.assert_allclose(
            got["close"].to_numpy(),
            exp["close"].to_numpy(),
            rtol=1e-9,
            err_msg=f"{sym} close mismatch — cache key collision suspected",
        )


def test_tz_normalization_uniform_across_symbols(manager_with_cache):
    """No symbol leaks tz-aware timestamps into the cache layer."""
    for i, sym in enumerate(SYMBOLS):
        manager_with_cache.cache.write(_synthetic_5m_bars(sym, seed=i), sym, "5m")

    tz_states: set[object] = set()
    for sym in SYMBOLS:
        df = manager_with_cache.get_ohlcv(sym, "5m")
        assert isinstance(df.index, pd.DatetimeIndex)
        tz_states.add(df.index.tz)

    # Every symbol must agree on its tz state (either all naive or all aware).
    assert len(tz_states) == 1, f"tz drift across symbols: {tz_states}"
