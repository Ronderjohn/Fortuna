from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fortuna.agentic import AgenticLearningStore, LearningExample, LearningOutcome
from fortuna.app.model_status import build_learning_summary, build_model_status
from fortuna.config.settings import Settings
from fortuna.models.metadata import ModelKind
from fortuna.models.registry import append_audit, live_pointer_path


def test_build_learning_summary_handles_empty_store(tmp_path):
    store = AgenticLearningStore(tmp_path / "agentic")
    summary = build_learning_summary(store)
    assert summary.total_rows == 0
    assert summary.recent_rows == ()


def test_build_learning_summary_filters_by_symbol(tmp_path):
    store = AgenticLearningStore(tmp_path / "agentic")
    store.upsert(
        LearningExample(
            decision_hash="d1",
            bar_time="2026-01-06T09:30:00",
            bar_idx=3,
            symbol="RELIANCE.NS",
            timeframe="5m",
            action="BUY",
            confidence=0.8,
            current_side=None,
            bar_close=100.0,
            outcome=LearningOutcome(status="resolved", paper_closed=True),
        )
    )
    store.upsert(
        LearningExample(
            decision_hash="d2",
            bar_time="2026-01-06T10:30:00",
            bar_idx=4,
            symbol="TCS.NS",
            timeframe="5m",
            action="SELL",
            confidence=0.6,
            current_side=None,
            bar_close=200.0,
            outcome=LearningOutcome(status="resolved", paper_closed=False),
        )
    )
    summary = build_learning_summary(store, symbol="RELIANCE.NS")
    assert summary.total_rows == 1
    assert summary.recent_rows[0].symbol == "RELIANCE.NS"


def test_build_model_status_disabled_ml_rl(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        agentic_enabled=False,
        agentic_ml_scorer_enabled=False,
    )
    engine = SimpleNamespace(
        settings=settings,
        _agentic_store=None,
        _agentic_learning_store=None,
        rl_generator=None,
        state=SimpleNamespace(symbol=""),
        _agentic_orchestrator=None,
    )
    status = build_model_status(engine)
    assert status.ml.available is False
    assert status.ml.load_error == "not_loaded"
    assert status.rl.available is False
    assert status.rl.load_error == "no_generator"
    payload = status.to_dict()
    assert payload["ml"]["enabled"] is False


def test_build_model_status_includes_promotions_and_learning_summary(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        default_timeframe="5m",
        default_days=10,
        agentic_enabled=True,
        agentic_ml_scorer_enabled=True,
        model_registry_enabled=True,
        model_promotion_required=True,
        agentic_log_dir=tmp_path / "agentic",
        ml_scorer_artifact_dir=Path("models/ml_signal_scorer/validated"),
    )
    learning_store = AgenticLearningStore(tmp_path / "agentic")
    learning_store.upsert(
        LearningExample(
            decision_hash="d1",
            bar_time="2026-01-06T09:30:00",
            bar_idx=3,
            symbol="RELIANCE.NS",
            timeframe="5m",
            action="BUY",
            confidence=0.8,
            current_side=None,
            bar_close=100.0,
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=1.5,
            ),
        )
    )

    models_root = tmp_path / "models"
    models_root.mkdir()
    rl_ptr = live_pointer_path(ModelKind.RL_POLICY, models_root, symbol="RELIANCE.NS")
    rl_ptr.parent.mkdir(parents=True, exist_ok=True)
    rl_ptr.write_text(
        json.dumps(
            {
                "model_kind": "rl_policy",
                "run_id": "rl_1",
                "symbol": "RELIANCE.NS",
                "timeframe": "5m",
                "status": "live",
                "artifact_dir": str(tmp_path / "models" / "validated" / "rl_1"),
            }
        ),
        encoding="utf-8",
    )
    append_audit(
        models_root,
        {
            "model_kind": "rl_policy",
            "run_id": "rl_1",
            "symbol": "RELIANCE.NS",
            "promoted_by": "test",
        },
    )

    engine = SimpleNamespace(
        settings=settings,
        _agentic_store=None,
        _agentic_learning_store=learning_store,
        rl_generator=None,
        state=SimpleNamespace(symbol="RELIANCE.NS"),
        _agentic_orchestrator=None,
    )
    status = build_model_status(engine)
    assert status.registry_enabled is True
    assert status.rl.live_pointer is not None
    assert status.rl.live_pointer.endswith("live.json")
    assert status.rl.last_promotion is not None
    assert status.rl.last_promotion.run_id == "rl_1"
    assert status.agentic.learning_summary.paper_closed_rows == 1

    payload = status.to_dict()
    assert payload["rl"]["last_promotion"]["run_id"] == "rl_1"
