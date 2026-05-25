"""Indicator cache tests."""

import pandas as pd

from fortuna.search.indicator_cache import IndicatorCache
from fortuna.strategy.schema import IndicatorSpec, IndicatorType


def test_indicator_cache_reuses_frame(sample_ohlcv: pd.DataFrame) -> None:
    cache = IndicatorCache()
    specs = [
        IndicatorSpec(id="ema_12", type=IndicatorType.EMA, params={"window": 12}),
    ]
    first = cache.enrich(sample_ohlcv, specs)
    second = cache.enrich(sample_ohlcv, specs)
    assert "ema_12" in first.columns
    assert "ema_12" in second.columns
    assert len(first) == len(sample_ohlcv)
