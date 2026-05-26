"""Phase 1 completion 2.4: ``compute_live_signal`` emits a [TIMING] line."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from fortuna.app.live_signals import SignalType, compute_live_signal
from fortuna.strategy.loader import load_strategy


def _make_ohlcv(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2026-01-01 09:15", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_compute_live_signal_emits_timing_line(project_root, caplog):
    strategy = load_strategy(project_root / "strategies" / "generated" / "ema_crossover.json")
    df = _make_ohlcv(80)

    buf = io.StringIO()
    with redirect_stdout(buf), caplog.at_level(logging.INFO):
        compute_live_signal(strategy, df, strategy_name="ema_crossover")

    stdout_text = buf.getvalue()
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    combined = stdout_text + "\n" + log_text

    assert "[TIMING]" in combined
    assert "live_signal" in combined


def test_signal_type_enum_maps_actions():
    assert SignalType.from_action("BUY") == SignalType.BUY
    assert SignalType.from_action("SELL") == SignalType.SELL
    assert SignalType.from_action("EXIT_LONG") == SignalType.EXIT
    assert SignalType.from_action("EXIT_SHORT") == SignalType.EXIT
    assert SignalType.from_action("HOLD") == SignalType.HOLD
    assert SignalType.from_action("IN_LONG") == SignalType.IN_LONG
