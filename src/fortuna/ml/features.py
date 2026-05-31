"""Tabular ML features at signal bars — past-only, no leakage."""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from fortuna.features.position import PositionState
from fortuna.features.registry import TRANSFORM_FN, FeatureSpec
from fortuna.ml.types import SignalExample

_ML_INDICATOR_SPECS: list[FeatureSpec] = [
    FeatureSpec("hl_range_atr", ["high", "low", "atr"], "hl_over_atr"),
    FeatureSpec("close_vwap_dist", ["close", "vwap"], "pct_diff"),
    FeatureSpec("vol_ratio", ["volume", "volume_sma"], "divide"),
    FeatureSpec("ema_fast_dist", ["close", "ema_fast"], "pct_diff"),
    FeatureSpec("ema_slow_dist", ["close", "ema_slow"], "pct_diff"),
    FeatureSpec("ema_cross", ["ema_fast", "ema_slow"], "pct_diff"),
    FeatureSpec("rsi_norm", ["rsi"], "divide_100"),
    FeatureSpec("macd_hist_atr", ["macd_hist", "atr"], "divide"),
    FeatureSpec("bb_pct_b", ["bb_pct_b"], "identity"),
    FeatureSpec("atr_pct", ["atr", "close"], "divide"),
    FeatureSpec("session_progress", ["timestamp"], "session_progress_norm", windowed=False),
    FeatureSpec("is_opening_range", ["timestamp"], "is_first_15min", windowed=False),
    FeatureSpec("is_squareoff_zone", ["timestamp"], "is_last_30min", windowed=False),
    FeatureSpec("day_of_week", ["timestamp"], "day_norm", windowed=False),
]

_ACTION_FEATURES = (
    "action_buy",
    "action_sell",
    "action_exit_long",
    "action_exit_short",
)

_SIDE_FEATURES = (
    "side_long",
    "side_short",
    "side_flat",
)

ML_FEATURE_NAMES: tuple[str, ...] = (
    ("close_ret",)
    + tuple(s.name for s in _ML_INDICATOR_SPECS)
    + _ACTION_FEATURES
    + _SIDE_FEATURES
)


def ml_feature_schema_hash() -> str:
    payload = "|".join(ML_FEATURE_NAMES)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _close_ret_pct(enriched: pd.DataFrame, bar_idx: int) -> float:
    """Pct change using only bar_idx and bar_idx-1 closes."""
    if bar_idx <= 0 or bar_idx >= len(enriched):
        return 0.0
    if "close" not in enriched.columns:
        return 0.0
    prev = float(enriched.iloc[bar_idx - 1]["close"])
    cur = float(enriched.iloc[bar_idx]["close"])
    if prev == 0:
        return 0.0
    return (cur - prev) / prev


def _position_from_side(side: str) -> PositionState:
    pos = PositionState()
    side_u = str(side or "FLAT").upper()
    if side_u == "LONG":
        pos.side = "LONG"
        pos.entry_price = 1.0
        pos.bars_held = 1
    elif side_u == "SHORT":
        pos.side = "SHORT"
        pos.entry_price = 1.0
        pos.bars_held = 1
    return pos


def _encode_action(action: str) -> dict[str, float]:
    act = str(action).upper()
    return {
        "action_buy": 1.0 if act == "BUY" else 0.0,
        "action_sell": 1.0 if act == "SELL" else 0.0,
        "action_exit_long": 1.0 if act == "EXIT_LONG" else 0.0,
        "action_exit_short": 1.0 if act == "EXIT_SHORT" else 0.0,
    }


def _encode_side(position_side: str) -> dict[str, float]:
    side = str(position_side or "FLAT").upper()
    return {
        "side_long": 1.0 if side == "LONG" else 0.0,
        "side_short": 1.0 if side == "SHORT" else 0.0,
        "side_flat": 1.0 if side not in {"LONG", "SHORT"} else 0.0,
    }


def build_feature_row(
    enriched: pd.DataFrame,
    bar_idx: int,
    *,
    action: str,
    position_side: str = "FLAT",
    bar_time: Optional[pd.Timestamp] = None,
) -> dict[str, float]:
    """Build one feature row using data at bar_idx and bar_idx-1 only."""
    if bar_idx < 0 or bar_idx >= len(enriched):
        return {name: 0.0 for name in ML_FEATURE_NAMES}

    row = enriched.iloc[bar_idx]
    ts = bar_time if bar_time is not None else pd.Timestamp(enriched.index[bar_idx])
    position = _position_from_side(position_side)

    out: dict[str, float] = {"close_ret": _close_ret_pct(enriched, bar_idx)}

    for spec in _ML_INDICATOR_SPECS:
        fn = TRANSFORM_FN[spec.transform]
        out[spec.name] = float(fn(spec.sources, row, position, ts))

    out.update(_encode_action(action))
    out.update(_encode_side(position_side))
    return out


def build_feature_matrix(
    examples: list[SignalExample],
    enriched: pd.DataFrame,
    *,
    position_side: str = "FLAT",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Stack feature rows for labeled examples."""
    if not examples:
        return np.zeros((0, len(ML_FEATURE_NAMES)), dtype=np.float64), ML_FEATURE_NAMES

    rows: list[list[float]] = []
    for ex in examples:
        feat = build_feature_row(
            enriched,
            ex.bar_idx,
            action=ex.action,
            position_side=position_side,
            bar_time=pd.Timestamp(ex.bar_time),
        )
        rows.append([feat[name] for name in ML_FEATURE_NAMES])

    return np.array(rows, dtype=np.float64), ML_FEATURE_NAMES


def feature_vector_from_row(feat: dict[str, float]) -> np.ndarray:
    """Ordered feature vector matching ML_FEATURE_NAMES."""
    return np.array([feat.get(name, 0.0) for name in ML_FEATURE_NAMES], dtype=np.float64)
