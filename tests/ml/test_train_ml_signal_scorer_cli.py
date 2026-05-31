from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fortuna.data.cache import DataCache
from fortuna.ml.signal_scorer import SignalScorer


def test_train_ml_signal_scorer_cli_writes_loadable_artifact(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    prices = [100.0, 100.4, 100.9, 101.2, 101.8, 102.1, 101.7, 101.1, 100.7, 101.4, 101.9, 102.3]
    DataCache(cache_dir).write(
        pd.DataFrame(
            {
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000] * 12,
            },
            index=pd.date_range("2026-01-06 09:15", periods=12, freq="5min"),
        ),
        "RELIANCE.NS",
        "5m",
    )
    learning_dir = tmp_path / "agentic"
    learning_dir.mkdir()
    rows = []
    for i in range(2, 10):
        rows.append(
            {
                "decision_hash": f"d{i}",
                "bar_time": pd.Timestamp("2026-01-06 09:15") + pd.Timedelta(minutes=5 * i),
                "bar_idx": i,
                "symbol": "RELIANCE.NS",
                "timeframe": "5m",
                "action": "BUY" if i % 2 == 0 else "SELL",
                "confidence": 0.7,
                "current_side": None,
                "bar_close": 100.0,
                "outcome": {
                    "status": "resolved",
                    "forward_return_pct": 1.0 if i % 2 == 0 else -1.0,
                    "directionally_correct": bool(i % 2 == 0),
                },
            }
        )
    with (learning_dir / "learning_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            payload = dict(row)
            payload["bar_time"] = pd.Timestamp(payload["bar_time"]).isoformat()
            f.write(json.dumps(payload, default=str) + "\n")

    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        "data:\n"
        f"  cache_dir: {cache_dir.as_posix()}\n"
        f"  duckdb_path: {(tmp_path / 'fortuna.duckdb').as_posix()}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "ml_out"
    cmd = [
        sys.executable,
        "scripts/train_ml_signal_scorer.py",
        "--config",
        str(cfg),
        "--symbol",
        "RELIANCE.NS",
        "--timeframe",
        "5m",
        "--learning-dir",
        str(learning_dir),
        "--output-dir",
        str(output_dir),
        "--run-id",
        "unit_ml_run",
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    loaded = SignalScorer.load(output_dir / "unit_ml_run")
    assert loaded is not None
    assert loaded.metadata is not None
    assert loaded.metadata.run_id == "unit_ml_run"
