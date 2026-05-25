"""Indicator engine tests."""

import pandas as pd

from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.schema import IndicatorSpec, IndicatorType


def test_indicator_engine_adds_columns(
    indicator_engine: IndicatorEngine,
    sample_ohlcv: pd.DataFrame,
) -> None:
    specs = [
        IndicatorSpec(id="ema_12", type=IndicatorType.EMA, params={"window": 12}),
        IndicatorSpec(id="rsi", type=IndicatorType.RSI, params={"window": 14}),
    ]
    result = indicator_engine.compute(sample_ohlcv, specs)
    assert "ema_12" in result.columns
    assert "rsi" in result.columns
    assert result["ema_12"].notna().sum() > 0
