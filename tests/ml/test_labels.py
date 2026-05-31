"""Tests for ML label generation."""

from __future__ import annotations

import pandas as pd
import pytest

from fortuna.backtesting.standard.config import MarketCostModel
from fortuna.ml.labels import (
    build_signal_examples,
    directional_multiplier,
    forward_return_at_bar,
    label_from_forward_return,
)
from fortuna.ml.types import LabelConfig


def _synthetic_ohlcv(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 09:15", periods=len(closes), freq="5min")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )


def test_buy_rising_close_labels_positive():
    ohlcv = _synthetic_ohlcv([100.0, 100.5, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    events = pd.DataFrame(
        [{"bar_idx": 2, "strategy_name": "orb", "action": "BUY"}],
    )
    cfg = LabelConfig(horizon_bars=3, cost_model=MarketCostModel())
    examples = build_signal_examples(ohlcv, events, label_config=cfg)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.label == 1
    assert ex.forward_return_pct > 0


def test_buy_falling_close_labels_negative():
    ohlcv = _synthetic_ohlcv([100.0, 99.5, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0])
    events = pd.DataFrame(
        [{"bar_idx": 2, "strategy_name": "orb", "action": "BUY"}],
    )
    examples = build_signal_examples(ohlcv, events, label_config=LabelConfig(horizon_bars=3))
    assert len(examples) == 1
    assert examples[0].label == 0


def test_sell_short_entry_direction():
    ohlcv = _synthetic_ohlcv([100.0, 99.5, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0])
    fwd = forward_return_at_bar(ohlcv, 2, "SELL", 3)
    assert fwd is not None
    assert fwd > 0


def test_exit_long_direction():
    ohlcv = _synthetic_ohlcv([100.0, 99.5, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0])
    fwd = forward_return_at_bar(ohlcv, 2, "EXIT_LONG", 3)
    assert fwd is not None
    assert fwd > 0


def test_exit_short_direction():
    ohlcv = _synthetic_ohlcv([100.0, 100.5, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    fwd = forward_return_at_bar(ohlcv, 2, "EXIT_SHORT", 3)
    assert fwd is not None
    assert fwd > 0


def test_cost_subtraction_on_entry():
    cfg = LabelConfig(cost_model=MarketCostModel())
    cost = cfg.cost_model.total_cost_pct()
    label, adjusted = label_from_forward_return(cost + 0.01, "BUY", label_config=cfg)
    assert label == 1
    label2, _ = label_from_forward_return(cost - 0.01, "BUY", label_config=cfg)
    assert label2 == 0
    _, adjusted_val = label_from_forward_return(1.0, "BUY", label_config=cfg)
    assert adjusted_val == pytest.approx(1.0 - cost)


def test_directional_multipliers():
    assert directional_multiplier("BUY") == 1.0
    assert directional_multiplier("SELL") == -1.0
    assert directional_multiplier("EXIT_LONG") == -1.0
    assert directional_multiplier("EXIT_SHORT") == 1.0


def test_label_uses_future_close():
    """Labels depend on future bars; verify forward return changes with future close."""
    ohlcv = _synthetic_ohlcv([100.0, 100.0, 100.0, 110.0, 100.0, 100.0, 100.0, 100.0])
    fwd_before = forward_return_at_bar(ohlcv, 2, "BUY", 3)
    ohlcv2 = ohlcv.copy()
    ohlcv2.iloc[5, ohlcv2.columns.get_loc("close")] = 80.0
    fwd_after = forward_return_at_bar(ohlcv2, 2, "BUY", 3)
    assert fwd_before != fwd_after
