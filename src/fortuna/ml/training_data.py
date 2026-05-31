"""Helpers for building ML scorer datasets from agentic learning rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fortuna.agentic.learning import (
    LearningExample,
    build_learning_rows,
    to_signal_examples,
)
from fortuna.app.session_engine import _enrich_ohlcv_for_ml
from fortuna.ml.features import build_feature_matrix
from fortuna.ml.types import SplitRange


@dataclass(frozen=True)
class TrainingDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    split_ranges: list[SplitRange]
    example_count: int


def load_learning_rows(
    learning_rows_path: Path | str,
    *,
    symbol: str,
    timeframe: str,
) -> list[LearningExample]:
    path = Path(learning_rows_path)
    if not path.is_file():
        return []
    rows: list[LearningExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(LearningExample.from_dict(json.loads(line)))
    deduped = build_learning_rows(rows, resolved_only=True)
    sym = symbol.upper().strip()
    return [r for r in deduped if r.symbol.upper().strip() == sym and r.timeframe == timeframe]


def build_dataset_from_learning_rows(
    rows: list[LearningExample],
    ohlcv: pd.DataFrame,
    *,
    eval_fraction: float = 0.2,
    position_side: str = "FLAT",
) -> TrainingDataset:
    if not rows:
        raise ValueError("No resolved learning rows available for training")
    enriched = _enrich_ohlcv_for_ml(ohlcv)
    examples = to_signal_examples(rows)
    if len(examples) < 2:
        raise ValueError("Need at least two resolved signal examples for training")

    X, _ = build_feature_matrix(examples, enriched, position_side=position_side)
    y = np.array([int(ex.label) for ex in examples], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise ValueError("Training labels must contain both classes")

    eval_count = max(1, int(round(len(examples) * eval_fraction)))
    if eval_count >= len(examples):
        eval_count = max(1, len(examples) // 3)
    split_idx = max(1, len(examples) - eval_count)
    if split_idx >= len(examples):
        split_idx = len(examples) - 1

    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    if not eval_examples:
        eval_examples = examples[-1:]
        train_examples = examples[:-1]

    X_train = X[: len(train_examples)]
    y_train = y[: len(train_examples)]
    X_eval = X[len(train_examples) :]
    y_eval = y[len(train_examples) :]

    split_ranges = [
        _split_range("train", train_examples),
        _split_range("eval", eval_examples),
    ]
    return TrainingDataset(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        split_ranges=split_ranges,
        example_count=len(examples),
    )


def default_run_id(symbol: str, timeframe: str) -> str:
    sym = symbol.upper().replace(".", "_")
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f"ml_{sym}_{timeframe}_{ts}"


def _split_range(name: str, examples) -> SplitRange:
    start = pd.Timestamp(examples[0].bar_time).isoformat()
    end = pd.Timestamp(examples[-1].bar_time).isoformat()
    return SplitRange(name=name, start=start, end=end, bar_count=len(examples))
