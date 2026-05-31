"""Apply indicator specs to OHLCV DataFrames."""

from __future__ import annotations

import pandas as pd

from fortuna.indicators.registry import INDICATOR_REGISTRY, compute_atr
from fortuna.strategy.schema import IndicatorSpec, IndicatorType
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class IndicatorEngine:
    """Compute and attach indicators defined in a strategy."""

    def compute(self, df: pd.DataFrame, indicators: list[IndicatorSpec]) -> pd.DataFrame:
        """Return df with indicator columns added."""
        if not indicators:
            return df
        new_cols: dict[str, pd.Series] = {}
        base = df
        for spec in indicators:
            self._apply_one(base, spec, new_cols)
        if not new_cols:
            return df
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def compute_for_bar(
        self,
        raw_df: pd.DataFrame,
        indicators: list[IndicatorSpec],
        *,
        lookback: int = 300,
    ) -> pd.Series:
        """Compute only the latest row of indicators without reprocessing full history.

        Hot path for live inference (called every 5m bar). Recomputes the indicator
        stack over a fixed tail window (``lookback`` bars, default 300 = ~25 NSE
        5m sessions) and returns only the last row. For pandas-based rolling
        windows (EMA/SMA/ATR/MACD/Bollinger/RSI) this is numerically identical
        to a full-history pass once ``lookback`` exceeds the largest window.

        Target: < 10 ms on CPU for the Phase 2 RL feature stack.
        """
        if raw_df is None or len(raw_df) == 0:
            raise ValueError("compute_for_bar() requires a non-empty OHLCV frame")
        tail = raw_df.iloc[-lookback:] if lookback and len(raw_df) > lookback else raw_df
        enriched = self.compute(tail, indicators)
        return enriched.iloc[-1]

    def _apply_one(
        self,
        df: pd.DataFrame,
        spec: IndicatorSpec,
        new_cols: dict[str, pd.Series],
    ) -> None:
        ind_type = spec.type.value if hasattr(spec.type, "value") else str(spec.type)
        params = dict(spec.params)
        source = spec.source.value if hasattr(spec.source, "value") else str(spec.source)

        if ind_type == IndicatorType.ATR.value:
            window = int(params.get("window", 14))
            new_cols[spec.id] = compute_atr(df, window=window)
            return

        if ind_type == IndicatorType.VWAP.value:
            new_cols[spec.id] = INDICATOR_REGISTRY["vwap"](df, source=source)
            return

        if ind_type == IndicatorType.MACD.value:
            macd_df = INDICATOR_REGISTRY["macd"](
                df,
                source,
                window_slow=int(params.get("window_slow", 26)),
                window_fast=int(params.get("window_fast", 12)),
                window_sign=int(params.get("window_sign", 9)),
            )
            suffix = params.get("suffix", "")
            for col in macd_df.columns:
                col_id = f"{spec.id}_{col}" if suffix == "" else f"{spec.id}_{col}"
                new_cols[col_id] = macd_df[col]
            if params.get("attach_main", True):
                new_cols[spec.id] = macd_df["macd"]
                new_cols[f"{spec.id}_signal"] = macd_df["macd_signal"]
            return

        if ind_type == IndicatorType.BOLLINGER.value:
            bb_df = INDICATOR_REGISTRY["bollinger"](
                df,
                source,
                window=int(params.get("window", 20)),
                window_dev=float(params.get("window_dev", 2.0)),
            )
            new_cols[f"{spec.id}_upper"] = bb_df["bb_upper"]
            new_cols[f"{spec.id}_middle"] = bb_df["bb_middle"]
            new_cols[f"{spec.id}_lower"] = bb_df["bb_lower"]
            new_cols[spec.id] = bb_df["bb_middle"]
            return

        if ind_type == IndicatorType.VOLUME_SMA.value:
            window = int(params.get("window", 20))
            new_cols[spec.id] = INDICATOR_REGISTRY["volume_sma"](df, window=window)
            return

        if ind_type == IndicatorType.ROLLING_HIGH.value:
            window = int(params.get("window", 20))
            shift = int(params.get("shift", 1))
            new_cols[spec.id] = INDICATOR_REGISTRY["rolling_high"](
                df, source, window=window, shift=shift
            )
            return

        if ind_type == IndicatorType.ROLLING_LOW.value:
            window = int(params.get("window", 20))
            shift = int(params.get("shift", 1))
            new_cols[spec.id] = INDICATOR_REGISTRY["rolling_low"](
                df, source, window=window, shift=shift
            )
            return

        fn = INDICATOR_REGISTRY.get(ind_type)
        if fn is None:
            raise ValueError(f"Unsupported indicator type: {ind_type}")

        window = int(params.get("window", 14))
        new_cols[spec.id] = fn(df, source, window=window)
        logger.debug("Computed indicator %s (%s)", spec.id, ind_type)
