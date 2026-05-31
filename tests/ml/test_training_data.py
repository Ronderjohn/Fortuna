from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fortuna.agentic import LearningExample, LearningOutcome
from fortuna.ml.training_data import build_dataset_from_learning_rows, load_learning_rows


def _ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 09:15", periods=12, freq="5min")
    close = [100.0, 100.4, 100.9, 101.2, 101.8, 102.1, 101.7, 101.1, 100.7, 101.4, 101.9, 102.3]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [1000] * len(close),
        },
        index=idx,
    )


def test_load_learning_rows_filters_to_symbol_and_resolved(tmp_path: Path):
    rows_path = tmp_path / "learning_rows.jsonl"
    rows_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "decision_hash": "a",
                        "bar_time": "2026-01-06T09:25:00",
                        "bar_idx": 2,
                        "symbol": "RELIANCE.NS",
                        "timeframe": "5m",
                        "action": "BUY",
                        "confidence": 0.8,
                        "current_side": None,
                        "bar_close": 100.0,
                        "outcome": {"status": "resolved", "directionally_correct": True},
                    }
                ),
                json.dumps(
                    {
                        "decision_hash": "b",
                        "bar_time": "2026-01-06T09:30:00",
                        "bar_idx": 3,
                        "symbol": "ICICIBANK.NS",
                        "timeframe": "5m",
                        "action": "BUY",
                        "confidence": 0.8,
                        "current_side": None,
                        "bar_close": 100.0,
                        "outcome": {"status": "resolved", "directionally_correct": True},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    rows = load_learning_rows(rows_path, symbol="RELIANCE.NS", timeframe="5m")
    assert len(rows) == 1
    assert rows[0].decision_hash == "a"


def test_build_dataset_from_learning_rows_builds_train_and_eval():
    ohlcv = _ohlcv()
    rows = [
        LearningExample(
            decision_hash=f"d{i}",
            bar_time=ohlcv.index[i].isoformat(),
            bar_idx=i,
            symbol="RELIANCE.NS",
            timeframe="5m",
            action="BUY" if i % 2 == 0 else "SELL",
            confidence=0.7,
            current_side=None,
            bar_close=float(ohlcv.iloc[i]["close"]),
            outcome=LearningOutcome(
                status="resolved",
                forward_return_pct=1.0 if i % 2 == 0 else -1.0,
                directionally_correct=bool(i % 2 == 0),
            ),
        )
        for i in range(2, 10)
    ]
    dataset = build_dataset_from_learning_rows(rows, ohlcv, eval_fraction=0.25)
    assert dataset.example_count == len(rows)
    assert len(dataset.y_train) > 0
    assert len(dataset.y_eval) > 0
    assert len(dataset.split_ranges) == 2
