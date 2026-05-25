"""Live signal computation: BUY / SELL / EXIT / IN-POSITION on the latest bar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fortuna.app.live_signals import (
    LiveSignal,
    _trade_state,
    compute_live_signal,
    compute_live_signals,
)
from fortuna.app.strategy_paths import list_strategy_paths
from fortuna.config.settings import load_settings
from fortuna.strategy.loader import load_strategy


def _synthetic_ohlcv(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2026-05-01 09:15", periods=n, freq="5min")
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
        },
        index=idx,
    )


def test_trade_state_open_position() -> None:
    n = 10
    entries = pd.Series([False] * n)
    exits = pd.Series([False] * n)
    entries.iloc[3] = True
    in_pos, idx, bars = _trade_state(entries, exits)
    assert in_pos is True
    assert idx == 3
    assert bars == n - 1 - 3


def test_trade_state_closed_position() -> None:
    n = 10
    entries = pd.Series([False] * n)
    exits = pd.Series([False] * n)
    entries.iloc[2] = True
    exits.iloc[5] = True
    in_pos, idx, bars = _trade_state(entries, exits)
    assert in_pos is False
    assert idx == -1
    assert bars == 0


def test_compute_live_signal_returns_label() -> None:
    path = Path("strategies/intraday/rsi_scalp_5m.json")
    if not path.exists():
        return
    strategy = load_strategy(path)
    sig = compute_live_signal(strategy, _synthetic_ohlcv(300), strategy_name="rsi_scalp_5m")
    assert sig is not None
    assert sig.action in {"BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "IN_LONG", "IN_SHORT", "HOLD"}
    assert sig.label in {"BUY", "SELL", "EXIT", "LONG", "SHORT", "HOLD"}
    assert isinstance(sig.bar_close, float)


def test_compute_live_signals_for_all_strategies() -> None:
    settings = load_settings()
    paths = list_strategy_paths(settings)
    if not paths:
        return
    ohlcv = _synthetic_ohlcv(400)
    sigs = compute_live_signals(paths[:3], ohlcv)
    assert len(sigs) >= 1
    for name, sig in sigs.items():
        assert isinstance(sig, LiveSignal)
        assert sig.bar_time is not None
        assert sig.color.startswith("#")


def test_signal_bar_time_matches_last_ohlcv_index() -> None:
    path = Path("strategies/intraday/rsi_scalp_5m.json")
    if not path.exists():
        return
    strategy = load_strategy(path)
    ohlcv = _synthetic_ohlcv(200)
    sig = compute_live_signal(strategy, ohlcv)
    assert sig is not None
    assert pd.Timestamp(sig.bar_time) == pd.Timestamp(ohlcv.index[-1])
