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
    has_oi = any(len(row) >= 7 for row in rows)
    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if has_oi:
        columns.append("open_interest")
    normalized = [list(row[: len(columns)]) for row in rows]
    df = pd.DataFrame(normalized, columns=columns)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "datetime"

    numeric_cols = ["open", "high", "low", "close", "volume"]
    if "open_interest" in df.columns:
        numeric_cols.append("open_interest")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["symbol"] = symbol
    return df.sort_index()
