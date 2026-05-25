"""Walk-forward black-box splits — strict train / test isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class WalkForwardFold:
    """One out-of-sample evaluation fold."""

    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def walk_forward_folds(
    ohlcv: pd.DataFrame,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    min_folds: int = 1,
) -> list[WalkForwardFold]:
    """
    Sliding walk-forward folds.

    - **Train**: parameter search / learning only.
    - **Test**: paper trading only (black-box OOS).

    Same principle as calibrating a pricing model (e.g. Black–Scholes vol) on
    in-sample dates and measuring hedge PnL on unseen dates.
    """
    n = len(ohlcv)
    need = train_bars + test_bars
    if n < need:
        raise ValueError(f"Need at least {need} bars, got {n}")

    folds: list[WalkForwardFold] = []
    offset = 0
    fold_id = 0

    while offset + need <= n:
        train = ohlcv.iloc[offset : offset + train_bars].copy()
        test = ohlcv.iloc[offset + train_bars : offset + need].copy()
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train=train,
                test=test,
                train_start=str(train.index[0]),
                train_end=str(train.index[-1]),
                test_start=str(test.index[0]),
                test_end=str(test.index[-1]),
            )
        )
        fold_id += 1
        offset += step_bars

    if len(folds) < min_folds:
        raise ValueError(f"Only {len(folds)} folds (min {min_folds})")
    return folds


def iter_oos_windows(
    ohlcv: pd.DataFrame,
    window_bars: int,
    step_bars: int,
) -> Iterator[pd.DataFrame]:
    """Sequential OOS windows for intensive paper-trading competition."""
    for start in range(0, len(ohlcv) - window_bars + 1, step_bars):
        yield ohlcv.iloc[start : start + window_bars].copy()
