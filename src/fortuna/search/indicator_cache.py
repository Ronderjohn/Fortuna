"""Cache indicator arrays by type and parameters."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.schema import IndicatorSpec


class IndicatorCache:
    """Reuse enriched DataFrames for identical indicator sets."""

    def __init__(self, engine: IndicatorEngine | None = None) -> None:
        self._engine = engine or IndicatorEngine()
        self._frame_cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _key(indicators: list[IndicatorSpec]) -> str:
        payload = [
            {
                "id": s.id,
                "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                "params": s.params,
                "source": s.source.value if hasattr(s.source, "value") else str(s.source),
            }
            for s in indicators
        ]
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def enrich(self, df: pd.DataFrame, indicators: list[IndicatorSpec]) -> pd.DataFrame:
        if not indicators:
            return df
        key = self._key(indicators)
        if key in self._frame_cache and len(self._frame_cache[key]) == len(df):
            cached = self._frame_cache[key]
            if cached.index.equals(df.index):
                return cached
        try:
            from fortuna.compute.scheduler import get_scheduler

            enriched = get_scheduler().compute_indicators(df, indicators)
        except Exception:
            enriched = self._engine.compute(df, indicators)
        self._frame_cache[key] = enriched
        return enriched

    def clear(self) -> None:
        """Drop cached frames (call between strategy searches to limit RAM)."""
        self._frame_cache.clear()

    def enrich_many(
        self,
        df: pd.DataFrame,
        spec_lists: list[list[IndicatorSpec]],
    ) -> dict[str, pd.DataFrame]:
        """GPU-accelerated batch enrich for many candidates sharing one OHLCV window."""
        try:
            from fortuna.compute.scheduler import get_scheduler

            pre = get_scheduler().precompute_for_candidates(df, spec_lists)
            if pre:
                self._frame_cache.update(pre)
                return pre
        except Exception:
            pass
        out: dict[str, pd.DataFrame] = {}
        for specs in spec_lists:
            if specs:
                out[self._key(specs)] = self.enrich(df, specs)
        return out
