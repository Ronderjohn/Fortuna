"""Tests for ML feature builder — no leakage."""

from __future__ import annotations

import pandas as pd

from fortuna.ml.features import ML_FEATURE_NAMES, build_feature_row


def _enriched_frame(n: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 09:15", periods=n, freq="5min")
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000 + i * 10 for i in range(n)],
            "atr": [1.0] * n,
            "vwap": closes,
            "volume_sma": [1000.0] * n,
            "ema_fast": closes,
            "ema_slow": closes,
            "rsi": [50.0] * n,
            "macd_hist": [0.1] * n,
            "bb_pct_b": [0.5] * n,
        },
        index=idx,
    )


def test_close_ret_uses_previous_bar_only():
    enriched = _enriched_frame()
    row = build_feature_row(enriched, 3, action="BUY", bar_time=enriched.index[3])
    expected = (103.0 - 102.0) / 102.0
    assert row["close_ret"] == expected


def test_features_unchanged_when_future_bars_mutated():
    enriched = _enriched_frame()
    before = build_feature_row(enriched, 2, action="BUY", bar_time=enriched.index[2])
    mutated = enriched.copy()
    mutated.iloc[4:, mutated.columns.get_loc("close")] = 999.0
    after = build_feature_row(mutated, 2, action="BUY", bar_time=mutated.index[2])
    assert before == after


def test_missing_indicator_columns_default_safely():
    idx = pd.date_range("2026-01-06 09:15", periods=3, freq="5min")
    enriched = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
    row = build_feature_row(enriched, 2, action="SELL", bar_time=idx[2])
    assert set(row.keys()) == set(ML_FEATURE_NAMES)
    assert all(isinstance(v, float) for v in row.values())


def test_action_one_hot_encoding():
    enriched = _enriched_frame()
    buy = build_feature_row(enriched, 2, action="BUY", bar_time=enriched.index[2])
    sell = build_feature_row(enriched, 2, action="SELL", bar_time=enriched.index[2])
    assert buy["action_buy"] == 1.0 and buy["action_sell"] == 0.0
    assert sell["action_sell"] == 1.0 and sell["action_buy"] == 0.0
