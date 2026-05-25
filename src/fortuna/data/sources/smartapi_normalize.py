"""Normalize SmartAPI candle payloads to Fortuna OHLCV DataFrames."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def candles_to_dataframe(
    candles: Iterable[list[Any]],
    symbol: str,
) -> pd.DataFrame:
    """Convert getCandleData rows to datetime-indexed OHLCV."""
    rows = list(candles)
    if not rows:
        raise ValueError(f"No candle data for {symbol}")

    df = pd.DataFrame(
        rows,
        columns=["datetime", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "datetime"

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["symbol"] = symbol
    return df.sort_index()
