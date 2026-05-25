"""Minimum OHLCV bars needed before indicators and signals are valid."""

from __future__ import annotations

from fortuna.strategy.schema import IndicatorSpec, IndicatorType, StrategyDefinition


def required_warmup_bars(indicators: list[IndicatorSpec], *, buffer: int = 5) -> int:
    """
    Estimate warmup from indicator windows (EMA/RSI/SMA/VWAP, etc.).

    VWAP is cumulative intraday — still needs a few bars to stabilize crossovers.
    """
    max_window = 0
    for spec in indicators:
        params = spec.params or {}
        ind_type = spec.type.value if hasattr(spec.type, "value") else str(spec.type)

        for key in ("window", "window_slow", "window_fast", "window_sign"):
            if key in params:
                max_window = max(max_window, int(params[key]))

        if ind_type == IndicatorType.MACD.value:
            max_window = max(
                max_window,
                int(params.get("window_slow", 26)),
                int(params.get("window_fast", 12)),
            )
        if ind_type == IndicatorType.BOLLINGER.value:
            max_window = max(max_window, int(params.get("window", 20)))

    return max(max_window + buffer, buffer + 2)


def slice_for_backtest(
    ohlcv,
    *,
    window_bars: int,
    warmup_bars: int,
    strategy: StrategyDefinition,
):
    """Return OHLCV slice with enough history for indicators + evaluation window."""
    import pandas as pd

    need_warmup = max(warmup_bars, required_warmup_bars(strategy.indicators))
    total = need_warmup + window_bars
    if len(ohlcv) <= total:
        return ohlcv.copy()
    return ohlcv.iloc[-total:].copy()
