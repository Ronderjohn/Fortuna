"""Tests for deterministic baseline Sharpe lookup."""

from __future__ import annotations

import json

from fortuna.config.settings import Settings
from fortuna.rl.evaluation.baseline import baseline_sharpe_for_symbol


def test_baseline_returns_none_without_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=cache,
        duckdb_path=cache / "fortuna.duckdb",
        logs_dir=tmp_path / "logs",
    )
    assert baseline_sharpe_for_symbol("RELIANCE.NS", "5m", settings=settings) is None


def test_baseline_reads_rl_learner_state(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    learner_dir = tmp_path / "logs" / "learner"
    learner_dir.mkdir(parents=True)
    (learner_dir / "rl_RELIANCE_NS_5m_learning.json").write_text(
        json.dumps(
            {
                "strategy_stem": "rl_RELIANCE_NS_5m",
                "fold_history": [
                    {"profit_pct": 3.0, "total_trades": 2, "fold_id": 0, "win_ratio_pct": 50.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=cache,
        duckdb_path=cache / "fortuna.duckdb",
        logs_dir=tmp_path / "logs",
    )

    sharpe = baseline_sharpe_for_symbol("RELIANCE.NS", "5m", settings=settings)

    assert sharpe == 1.5
