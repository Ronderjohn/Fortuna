"""Warmup bar estimation for backtests."""

from pathlib import Path

from fortuna.strategy.loader import load_strategy
from fortuna.strategy.warmup import required_warmup_bars, slice_for_backtest


def test_rsi_warmup_from_indicator_windows():
    path = Path("strategies/generated/rsi_mean_reversion.json")
    strategy = load_strategy(path)
    # RSI window 14 + SMA 15 → max window 15 + buffer 5
    assert required_warmup_bars(strategy.indicators) >= 20


def test_slice_includes_warmup_plus_window():
    import pandas as pd

    path = Path("strategies/generated/ema_crossover.json")
    strategy = load_strategy(path)
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    ohlcv = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1000},
        index=idx,
    )
    sliced = slice_for_backtest(ohlcv, window_bars=40, warmup_bars=10, strategy=strategy)
    assert len(sliced) == 66  # max(10, 21+5)+40 for EMA slow window 21
