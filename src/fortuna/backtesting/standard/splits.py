"""Chronological train / validation / test splits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from fortuna.backtesting.standard.config import SplitRatios


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_range: tuple[str, str]
    val_range: tuple[str, str]
    test_range: tuple[str, str]


def validate_history_length(ohlcv: pd.DataFrame, min_days: int) -> None:
    if ohlcv.empty:
        raise ValueError("OHLCV is empty")
    span = (ohlcv.index.max() - ohlcv.index.min()).days
    if span < min_days:
        raise ValueError(
            f"Insufficient history: {span} calendar days < minimum {min_days}. "
            "5m strategies require 6+ months (trending, sideways, volatile regimes)."
        )


def chronological_split(ohlcv: pd.DataFrame, ratios: SplitRatios) -> DataSplit:
    """
    Split by time (no shuffle) — 70% train, 15% validation, 15% OOS test.

    OOS test is used for final institutional evaluation only.
    """
    n = len(ohlcv)
    if n < 100:
        raise ValueError(f"Need at least 100 bars for split, got {n}")

    train_end = int(n * ratios.train)
    val_end = train_end + int(n * ratios.validation)

    train = ohlcv.iloc[:train_end].copy()
    validation = ohlcv.iloc[train_end:val_end].copy()
    test = ohlcv.iloc[val_end:].copy()

    if len(test) < 50:
        raise ValueError(f"Test set too small: {len(test)} bars")

    def _rng(df: pd.DataFrame) -> tuple[str, str]:
        return str(df.index[0]), str(df.index[-1])

    return DataSplit(
        train=train,
        validation=validation,
        test=test,
        train_range=_rng(train),
        val_range=_rng(validation),
        test_range=_rng(test),
    )


def estimate_calendar_days(ohlcv: pd.DataFrame) -> int:
    if ohlcv.empty:
        return 0
    return max(1, (ohlcv.index.max() - ohlcv.index.min()).days)
