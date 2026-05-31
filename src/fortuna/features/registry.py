"""Declarative feature catalog for the RL observation space.

Editing ``FEATURE_REGISTRY`` is the only knob that changes ``obs_shape``,
normalization, and the feature index map. The ``FeatureBuilder`` consumes
this registry directly — no hard-coded feature names elsewhere.

Each ``FeatureSpec`` declares:
    - ``name``     : unique identifier (used in debug logs).
    - ``sources``  : column names this feature reads from the enriched indicator row.
    - ``transform``: key into ``TRANSFORM_FN`` selecting the pure function that
                     maps (row, position, timestamp) -> float.
    - ``windowed`` : True = part of the rolling (N_bars x N_feat) buffer,
                     False = appended as scalar (position / time context).
    - ``clip_sigma``: post-normalization clipping (0.0 = no clip).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fortuna.features.position import PositionState


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    sources: list[str]
    transform: str
    windowed: bool = True
    clip_sigma: float = 3.0


# ──────────────────────────── pure transform helpers ────────────────────────────


def _safe(value: float, default: float = 0.0) -> float:
    if value is None:
        return default
    f = float(value)
    if np.isnan(f) or np.isinf(f):
        return default
    return f


def _row_value(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col not in row.index:
        return default
    return _safe(row[col], default)


# Each transform takes (sources_list, row, position, timestamp) -> float.
TransformFn = Callable[[list[str], pd.Series, "PositionState", pd.Timestamp], float]


def _pct_change(sources: list[str], row: pd.Series, pos, ts) -> float:
    prev = row.attrs.get("_prev_close") if hasattr(row, "attrs") else None
    cur = _row_value(row, sources[0])
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev


def _hl_over_atr(sources: list[str], row: pd.Series, pos, ts) -> float:
    h = _row_value(row, sources[0])
    l = _row_value(row, sources[1])
    a = _row_value(row, sources[2])
    if a <= 0:
        return 0.0
    return (h - l) / a


def _pct_diff(sources: list[str], row: pd.Series, pos, ts) -> float:
    a = _row_value(row, sources[0])
    b = _row_value(row, sources[1])
    if b == 0:
        return 0.0
    return (a - b) / b


def _divide(sources: list[str], row: pd.Series, pos, ts) -> float:
    a = _row_value(row, sources[0])
    b = _row_value(row, sources[1])
    if b == 0:
        return 0.0
    return a / b


def _divide_100(sources: list[str], row: pd.Series, pos, ts) -> float:
    return _row_value(row, sources[0]) / 100.0


def _identity(sources: list[str], row: pd.Series, pos, ts) -> float:
    return _row_value(row, sources[0])


# Position-aware transforms — the position argument is mandatory here.


def _is_long_flag(sources, row, position, ts) -> float:
    return 1.0 if (position is not None and position.is_long) else 0.0


def _is_short_flag(sources, row, position, ts) -> float:
    return 1.0 if (position is not None and position.is_short) else 0.0


def _unrealized_pct(sources, row, position, ts) -> float:
    if position is None or position.is_flat:
        return 0.0
    cur = _row_value(row, "close")
    return position.unrealized_pct(cur)


def _bars_norm(sources, row, position, ts) -> float:
    if position is None or position.is_flat:
        return 0.0
    # 75 = one NSE 5m session; cap at 1.0
    return min(position.bars_held / 75.0, 1.0)


def _size_fraction(sources, row, position, ts) -> float:
    if position is None:
        return 0.0
    return float(position.size_fraction)


# Time-aware transforms — derived from the timestamp.

_OPEN_MIN = 9 * 60 + 15      # 09:15
_CLOSE_MIN = 15 * 60 + 30    # 15:30
_SESSION_LEN = _CLOSE_MIN - _OPEN_MIN  # 375 minutes


def _timestamp_minutes(ts: pd.Timestamp) -> int:
    if ts is None:
        return _OPEN_MIN
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Kolkata").tz_localize(None)
    return t.hour * 60 + t.minute


def _session_progress_norm(sources, row, position, ts) -> float:
    minutes = _timestamp_minutes(ts) - _OPEN_MIN
    return max(0.0, min(1.0, minutes / _SESSION_LEN))


def _is_first_15min(sources, row, position, ts) -> float:
    minutes = _timestamp_minutes(ts)
    return 1.0 if _OPEN_MIN <= minutes < _OPEN_MIN + 15 else 0.0


def _is_last_30min(sources, row, position, ts) -> float:
    minutes = _timestamp_minutes(ts)
    return 1.0 if _CLOSE_MIN - 30 <= minutes <= _CLOSE_MIN else 0.0


def _day_norm(sources, row, position, ts) -> float:
    if ts is None:
        return 0.0
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Kolkata").tz_localize(None)
    return t.weekday() / 4.0  # Mon=0..Fri=1.0


TRANSFORM_FN: dict[str, TransformFn] = {
    "pct_change":              _pct_change,
    "hl_over_atr":             _hl_over_atr,
    "pct_diff":                _pct_diff,
    "divide":                  _divide,
    "divide_100":              _divide_100,
    "identity":                _identity,
    "is_long_flag":            _is_long_flag,
    "is_short_flag":           _is_short_flag,
    "unrealized_pct":          _unrealized_pct,
    "bars_norm":               _bars_norm,
    "size_fraction":           _size_fraction,
    "session_progress_norm":   _session_progress_norm,
    "is_first_15min":          _is_first_15min,
    "is_last_30min":           _is_last_30min,
    "day_norm":                _day_norm,
}


FEATURE_REGISTRY: list[FeatureSpec] = [
    # Price structure (per bar)
    FeatureSpec("close_ret",        ["close"],                  "pct_change"),
    FeatureSpec("hl_range_atr",     ["high", "low", "atr"],     "hl_over_atr"),
    FeatureSpec("close_vwap_dist",  ["close", "vwap"],          "pct_diff"),
    FeatureSpec("vol_ratio",        ["volume", "volume_sma"],   "divide"),

    # Trend
    FeatureSpec("ema_fast_dist",    ["close", "ema_fast"],      "pct_diff"),
    FeatureSpec("ema_slow_dist",    ["close", "ema_slow"],      "pct_diff"),
    FeatureSpec("ema_cross",        ["ema_fast", "ema_slow"],   "pct_diff"),

    # Oscillators
    FeatureSpec("rsi_norm",         ["rsi"],                    "divide_100"),
    FeatureSpec("macd_hist_atr",    ["macd_hist", "atr"],       "divide"),
    FeatureSpec("bb_pct_b",         ["bb_pct_b"],               "identity"),

    # Volatility
    FeatureSpec("atr_pct",          ["atr", "close"],           "divide"),

    # Position context (scalar)
    FeatureSpec("is_long",          ["position"], "is_long_flag",    windowed=False, clip_sigma=0.0),
    FeatureSpec("is_short",         ["position"], "is_short_flag",   windowed=False, clip_sigma=0.0),
    FeatureSpec("unrealized_pnl",   ["position"], "unrealized_pct",  windowed=False),
    FeatureSpec("bars_in_trade",    ["position"], "bars_norm",       windowed=False, clip_sigma=0.0),
    FeatureSpec("position_size",    ["position"], "size_fraction",   windowed=False, clip_sigma=0.0),

    # NSE time context (scalar)
    FeatureSpec("session_progress", ["timestamp"], "session_progress_norm", windowed=False, clip_sigma=0.0),
    FeatureSpec("is_opening_range", ["timestamp"], "is_first_15min",        windowed=False, clip_sigma=0.0),
    FeatureSpec("is_squareoff_zone", ["timestamp"], "is_last_30min",        windowed=False, clip_sigma=0.0),
    FeatureSpec("day_of_week",      ["timestamp"], "day_norm",              windowed=False, clip_sigma=0.0),
]


def feature_registry_hash(registry: Optional[list[FeatureSpec]] = None) -> str:
    """MD5 hash of the feature catalog (used for checkpoint reproducibility)."""
    specs = registry if registry is not None else FEATURE_REGISTRY
    payload = "|".join(
        f"{s.name}::{','.join(s.sources)}::{s.transform}::{s.windowed}::{s.clip_sigma:.4f}"
        for s in specs
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
