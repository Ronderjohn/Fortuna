"""Canonical indicator basis the RL environment always computes.

The Phase 2 observation space (see ``features/registry.py``) is independent of
any per-strategy indicator list — the RL policy needs every source column to
exist regardless of what the deterministic strategy JSON declares. This module
declares the fixed indicator stack and a thin ``enrich_for_rl()`` helper that
attaches the post-processed aliases the registry references
(``macd_hist`` and ``bb_pct_b``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.schema import IndicatorSpec, IndicatorType, PriceSource

EMA_FAST_WINDOW = 9
EMA_SLOW_WINDOW = 21
ATR_WINDOW = 14
RSI_WINDOW = 14
VOLUME_SMA_WINDOW = 20
BB_WINDOW = 20
BB_DEV = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


RL_INDICATORS: list[IndicatorSpec] = [
    IndicatorSpec(
        id="ema_fast",
        type=IndicatorType.EMA,
        source=PriceSource.CLOSE,
        params={"window": EMA_FAST_WINDOW},
    ),
    IndicatorSpec(
        id="ema_slow",
        type=IndicatorType.EMA,
        source=PriceSource.CLOSE,
        params={"window": EMA_SLOW_WINDOW},
    ),
    IndicatorSpec(
        id="atr",
        type=IndicatorType.ATR,
        source=PriceSource.CLOSE,
        params={"window": ATR_WINDOW},
    ),
    IndicatorSpec(
        id="rsi",
        type=IndicatorType.RSI,
        source=PriceSource.CLOSE,
        params={"window": RSI_WINDOW},
    ),
    IndicatorSpec(
        id="vwap",
        type=IndicatorType.VWAP,
        source=PriceSource.HLC3,
    ),
    IndicatorSpec(
        id="volume_sma",
        type=IndicatorType.VOLUME_SMA,
        source=PriceSource.VOLUME,
        params={"window": VOLUME_SMA_WINDOW},
    ),
    IndicatorSpec(
        id="bb",
        type=IndicatorType.BOLLINGER,
        source=PriceSource.CLOSE,
        params={"window": BB_WINDOW, "window_dev": BB_DEV},
    ),
    IndicatorSpec(
        id="macd",
        type=IndicatorType.MACD,
        source=PriceSource.CLOSE,
        params={
            "window_slow": MACD_SLOW,
            "window_fast": MACD_FAST,
            "window_sign": MACD_SIGNAL,
        },
    ),
]


def _attach_aliases(enriched: pd.DataFrame) -> pd.DataFrame:
    """Add ``macd_hist`` and ``bb_pct_b`` columns the FeatureBuilder reads.

    Operates on a copy so the caller's DataFrame is not mutated.
    """
    df = enriched
    new_cols: dict[str, pd.Series] = {}

    if "macd_hist" not in df.columns and "macd_macd_hist" in df.columns:
        new_cols["macd_hist"] = df["macd_macd_hist"]

    if "bb_pct_b" not in df.columns and {"bb_upper", "bb_lower", "close"}.issubset(df.columns):
        denom = df["bb_upper"] - df["bb_lower"]
        denom = denom.replace(0, np.nan)
        new_cols["bb_pct_b"] = ((df["close"] - df["bb_lower"]) / denom).fillna(0.5)

    if not new_cols:
        return df
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def enrich_for_rl(
    df: pd.DataFrame,
    *,
    engine: IndicatorEngine | None = None,
) -> pd.DataFrame:
    """Compute ``RL_INDICATORS`` on ``df`` and attach derived aliases."""
    eng = engine or IndicatorEngine()
    enriched = eng.compute(df, RL_INDICATORS)
    return _attach_aliases(enriched)


def enrich_for_rl_last_bar(
    df: pd.DataFrame,
    *,
    engine: IndicatorEngine | None = None,
    lookback: int = 300,
) -> pd.Series:
    """Hot-path equivalent of ``enrich_for_rl`` returning only the last row."""
    eng = engine or IndicatorEngine()
    if df is None or len(df) == 0:
        raise ValueError("enrich_for_rl_last_bar requires non-empty df")
    tail = df.iloc[-lookback:] if lookback and len(df) > lookback else df
    enriched = enrich_for_rl(tail, engine=eng)
    return enriched.iloc[-1]
