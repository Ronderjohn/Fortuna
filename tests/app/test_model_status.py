from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import fortuna.app.model_status as model_status
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
    assert status.runtime_readiness is not None
    assert status.runtime_readiness.overall_posture == "deterministic_only"
    assert status.runtime_readiness.deterministic_fallback_ok is True
    assert status.runtime_readiness.nightly_posture == "manual_only"
    payload = status.to_dict()
    assert payload["ml"]["enabled"] is False
    assert payload["runtime_readiness"]["overall_posture"] == "deterministic_only"


def test_build_model_status_runtime_readiness_advisory_partial(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        agentic_enabled=True,
        agentic_ml_scorer_enabled=True,
        model_registry_enabled=True,
        model_promotion_required=True,
        ml_scorer_artifact_dir=Path("models/ml_signal_scorer/validated"),
    )
    engine = SimpleNamespace(
        settings=settings,
        _agentic_store=None,
        _agentic_learning_store=None,
        rl_generator=None,
        state=SimpleNamespace(symbol="RELIANCE.NS"),
        _agentic_orchestrator=None,
    )
    status = build_model_status(engine)
    readiness = status.runtime_readiness
    assert readiness is not None
    assert readiness.overall_posture == "advisory_partial"
    warning_names = {row.name for row in readiness.warnings}
    assert "ml_scorer" in warning_names
    assert "rl_policy" in warning_names


def test_build_model_status_includes_scaling_in_payload(tmp_path):
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
    assert status.scaling is not None
    payload = status.to_dict()
    assert payload["scaling"] is not None
    assert payload["scaling"]["basket_posture"] in {
        "small_only",
        "medium_ok",
        "large_not_recommended",
    }


def test_scaling_warnings_do_not_change_overall_posture(tmp_path):
    from fortuna.app.runtime_readiness import build_runtime_readiness_summary

    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        agentic_enabled=True,
        agentic_ml_scorer_enabled=True,
        model_registry_enabled=True,
        model_promotion_required=True,
        ml_scorer_artifact_dir=Path("models/ml_signal_scorer/validated"),
    )
    baseline = build_runtime_readiness_summary(settings, basket_count=0)
    stressed = build_runtime_readiness_summary(
        settings,
        basket_count=12,
        physical_ram_gb=8.0,
    )
    assert baseline.overall_posture == stressed.overall_posture
    scaling_warnings = [row for row in stressed.warnings if row.category == "scaling"]
    assert scaling_warnings


def test_build_model_status_includes_promotions_and_learning_summary(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        model_status_nightly_alignment_recommendation_ratio=0.75,
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
    workflow_snapshot = tmp_path / "reports" / "workflow_snapshot.json"
    nightly_dir = tmp_path / "reports" / "nightly"
    workflow_snapshot.parent.mkdir(parents=True, exist_ok=True)
    workflow_snapshot.write_text(
        json.dumps(
            {
                "source": "auto",
                "timeframe": "5m",
                "lookback_days": 10,
                "universe": {
                    "metrics": {"count": 12, "adaptive_count": 1},
                    "notes": ["adaptive_boost=+0.08 avg_pnl=+0.90% rows=2"],
                },
                "shortlist": {"metrics": {"count": 6}},
                "briefing": {"metrics": {"candidates": 3}},
                "allocation": {
                    "metrics": {"selected_count": 2, "skipped_count": 1, "max_positions": 3},
                    "notes": [
                        "portfolio_research_alignment=enabled",
                        "allocation_research_targets: both=1, rl=1",
                        "allocation_refreshed_research_targets: both=1",
                    ],
                },
                "candidates": {"metrics": {"ml_count": 4, "rl_count": 2}},
            }
        ),
        encoding="utf-8",
    )
    nightly_dir.mkdir(parents=True, exist_ok=True)
    for name, detail in (
        (
            "20260603_0100.json",
            {
                "overall_status": "ok",
                "basket": ["RELIANCE.NS", "SBIN.NS"],
                "steps": [
                    {
                        "name": "training_research",
                        "detail": {
                            "discovery_preferred_count": 2,
                            "discovery_regime_mix": "trending=2",
                            "discovery_summary": (
                                "preferred=2 | liquidity=RELIANCE.NS,SBIN.NS | "
                                "activity=SBIN.NS,RELIANCE.NS | regimes=trending=2"
                            ),
                        },
                    },
                    {
                        "name": "training_execution_target",
                        "detail": {
                            "target": "rl",
                            "run_ml": False,
                            "run_rl": True,
                            "selection_source": "training_research",
                        },
                    },
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                            "allocation_research_target_mix": "both=1, rl=1",
                            "allocation_refreshed_research_target_mix": "both=1",
                            "team_research_alignment_summary": "basket/research aligned 3/3",
                            "team_research_alignment_target_mix": "both=1, rl=2",
                            "team_research_refresh_urgency_summary": (
                                "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
                            ),
                            "team_research_refresh_urgency_target": "rl",
                            "team_research_follow_up_action": (
                                "Prioritize RL refresh follow-up from remediation pressure "
                                "across 2 selected row(s)."
                            ),
                            "team_research_discovery_follow_up_target": "ml",
                            "team_research_discovery_follow_up_summary": (
                                "Discovery scouts currently lean ML-focused "
                                "from ml universe posture."
                            ),
                            "team_research_discovery_follow_up_action": (
                                "Refresh market universe and shortlist review with ML-focused "
                                "discovery focus before the next training cycle."
                            ),
                            "team_research_research_follow_up_target": "rl",
                            "team_research_research_follow_up_summary": (
                                "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
                            ),
                            "team_research_research_follow_up_action": (
                                "Prioritize RL refresh follow-up from remediation pressure "
                                "across 2 selected row(s)."
                            ),
                            "team_research_execution_follow_up_target": "rl",
                            "team_research_execution_follow_up_summary": (
                                "Nightly execution drift currently leans RL-focused from rl "
                                "execution posture."
                            ),
                            "team_research_execution_follow_up_action": (
                                "Review nightly execution path and retarget execution "
                                "candidate selection toward RL-focused before the next "
                                "promotion or nightly cycle."
                            ),
                            "team_discovery_alignment_summary": "discovery aligned 2/2",
                            "team_discovery_overlap_symbols": "RELIANCE.NS, SBIN.NS",
                            "team_discovery_alignment_overlap_count": 2,
                            "team_discovery_alignment_compare_count": 2,
                        },
                    }
                ],
            },
        ),
        (
            "20260602_0100.json",
            {
                "overall_status": "ok",
                "basket": ["TCS.NS"],
                "steps": [
                    {
                        "name": "training_execution_target",
                        "detail": {
                            "target": "ml",
                            "run_ml": True,
                            "run_rl": False,
                            "selection_source": "training_candidates",
                        },
                    },
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                            "allocation_research_target_mix": "ml=1",
                            "team_research_alignment_summary": "basket/research aligned 1/1",
                            "team_research_alignment_target_mix": "ml=1",
                            "team_discovery_alignment_summary": "discovery aligned 1/1",
                            "team_discovery_overlap_symbols": "TCS.NS",
                            "team_discovery_alignment_overlap_count": 1,
                            "team_discovery_alignment_compare_count": 1,
                        },
                    }
                    ,
                    {
                        "name": "promotion_reviews",
                        "detail": {
                            "model_kind": "mixed",
                            "model_kinds": ["ml_scorer", "rl_policy"],
                        },
                    }
                ],
            },
        ),
    ):
        (nightly_dir / name).write_text(json.dumps(detail), encoding="utf-8")

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
            "workflow_snapshot_path": str(workflow_snapshot),
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
    assert status.rl.last_promotion.promoted_at is not None
    assert status.rl.last_promotion.workflow_snapshot_path is not None
    assert status.rl.last_promotion.workflow_snapshot is not None
    assert status.rl.last_promotion.workflow_snapshot.universe_count == 12
    assert status.rl.last_promotion.workflow_snapshot.universe_adaptive_count == 1
    assert status.rl.last_promotion.workflow_snapshot.briefing_candidates == 3
    assert status.rl.last_promotion.workflow_snapshot.allocation_selected == 2
    assert status.agentic.learning_summary.paper_closed_rows == 1
    assert status.agentic.nightly_alignment.report_count == 2
    assert status.agentic.nightly_alignment.enabled_reports == 2
    assert status.agentic.nightly_alignment.aligned_reports == 2
    assert status.agentic.nightly_alignment.latest_execution_target == "rl"
    assert status.agentic.nightly_alignment.latest_execution_model_family == "rl"
    assert (
        status.agentic.nightly_alignment.latest_execution_selection_source
        == "training_research"
    )
    assert status.agentic.nightly_alignment.latest_target_mix == "both=1, rl=1"
    assert status.agentic.nightly_alignment.refreshed_aligned_reports == 1
    assert status.agentic.nightly_alignment.latest_refreshed_target_mix == "both=1"
    assert status.agentic.nightly_alignment.recommended_action is None
    assert status.agentic.nightly_alignment.recommended_refresh_target is None
    assert status.agentic.nightly_alignment.effective_refresh_target is None
    assert status.agentic.nightly_alignment.recommended_force_refresh is False
    assert status.agentic.nightly_alignment.workflow_target_mismatch is False
    assert status.agentic.nightly_alignment.latest_training_research_discovery_preferred_count == 2
    assert (
        status.agentic.nightly_alignment.latest_training_research_discovery_regime_mix
        == "trending=2"
    )
    assert "preferred=2" in str(
        status.agentic.nightly_alignment.latest_training_research_discovery_summary or ""
    )
    assert status.agentic.nightly_alignment.recent_trend_window == 2
    assert status.agentic.nightly_alignment.recent_trend_enabled == 2
    assert status.agentic.nightly_alignment.recent_trend_aligned == 2
    assert status.agentic.nightly_alignment.recent_trend_latest_status == "ok"
    assert status.agentic.nightly_alignment.recent_trend_latest_basket_size == 2
    assert status.agentic.nightly_alignment.recent_rows[0].target_mix == "both=1, rl=1"
    assert status.agentic.nightly_alignment.recent_rows[0].refreshed_target_mix == "both=1"
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_alignment_summary
        == "basket/research aligned 3/3"
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_alignment_target_mix
        == "both=1, rl=2"
    )
    assert status.agentic.nightly_alignment.latest_workflow_research_recommended_target == "all"
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_refresh_urgency_summary
        == "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
    )
    assert status.agentic.nightly_alignment.latest_workflow_research_refresh_urgency_target == "rl"
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_follow_up_action
        == "Prioritize RL refresh follow-up from remediation pressure across 2 selected row(s)."
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_discovery_follow_up_target
        == "ml"
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_discovery_follow_up_summary
        == "Discovery scouts currently lean ML-focused from ml universe posture."
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_discovery_follow_up_action
        == "Refresh market universe and shortlist review with ML-focused discovery focus "
        "before the next training cycle."
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_research_follow_up_target
        == "rl"
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_execution_follow_up_target
        == "rl"
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_research_execution_follow_up_action
        == "Review nightly execution path and retarget execution candidate selection "
        "toward RL-focused before the next promotion or nightly cycle."
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_discovery_alignment_summary
        == "discovery aligned 2/2"
    )
    assert (
        status.agentic.nightly_alignment.latest_workflow_discovery_overlap_symbols
        == "RELIANCE.NS, SBIN.NS"
    )
    assert status.agentic.nightly_alignment.latest_workflow_discovery_warning is False
    assert status.agentic.nightly_alignment.latest_promotion_review_model_kind == "mixed"
    assert status.agentic.nightly_alignment.latest_promotion_review_model_kinds == (
        "ml_scorer",
        "rl_policy",
    )

    payload = status.to_dict()
    assert payload["rl"]["last_promotion"]["run_id"] == "rl_1"
    assert payload["rl"]["last_promotion"]["workflow_snapshot_path"].endswith(
        "workflow_snapshot.json"
    )
    assert payload["rl"]["last_promotion"]["workflow_snapshot"]["ml_count"] == 4
    assert (
        payload["rl"]["last_promotion"]["workflow_snapshot"][
            "allocation_refreshed_research_target_mix"
        ]
        == "both=1"
    )
    assert (
        payload["rl"]["last_promotion"]["workflow_snapshot"]["universe_top_adaptive_note"]
        == "adaptive_boost=+0.08 avg_pnl=+0.90% rows=2"
    )
    assert payload["rl"]["last_promotion"]["workflow_snapshot"]["allocation_skipped"] == 1
    assert payload["agentic"]["nightly_alignment"]["report_count"] == 2
    assert payload["agentic"]["nightly_alignment"]["latest_execution_target"] == "rl"
    assert payload["agentic"]["nightly_alignment"]["latest_execution_model_family"] == "rl"
    assert (
        payload["agentic"]["nightly_alignment"]["latest_execution_selection_source"]
        == "training_research"
    )
    assert payload["agentic"]["nightly_alignment"]["latest_target_mix"] == "both=1, rl=1"
    assert payload["agentic"]["nightly_alignment"]["latest_refreshed_target_mix"] == "both=1"
    assert (
        payload["agentic"]["nightly_alignment"]["latest_workflow_research_refresh_urgency_summary"]
        == "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
    )
    assert (
        payload["agentic"]["nightly_alignment"][
            "latest_workflow_research_refresh_urgency_target"
        ]
        == "rl"
    )
    assert (
        payload["agentic"]["nightly_alignment"]["latest_workflow_research_follow_up_action"]
        == "Prioritize RL refresh follow-up from remediation pressure across 2 selected row(s)."
    )
    assert (
        payload["agentic"]["nightly_alignment"][
            "latest_workflow_research_discovery_follow_up_target"
        ]
        == "ml"
    )
    assert (
        payload["agentic"]["nightly_alignment"][
            "latest_workflow_research_execution_follow_up_target"
        ]
        == "rl"
    )
    assert (
        payload["agentic"]["nightly_alignment"]["latest_workflow_discovery_alignment_summary"]
        == "discovery aligned 2/2"
    )
    assert (
        payload["agentic"]["nightly_alignment"]["latest_promotion_review_model_kind"]
        == "mixed"
    )
    assert payload["agentic"]["nightly_alignment"]["latest_promotion_review_model_kinds"] == [
        "ml_scorer",
        "rl_policy",
    ]


def test_build_model_status_sets_nightly_alignment_recommendation_on_drift(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        model_status_nightly_alignment_recommendation_ratio=0.75,
    )
    nightly_dir = tmp_path / "reports" / "nightly"
    nightly_dir.mkdir(parents=True, exist_ok=True)
    for name, detail in (
        (
            "20260603_0100.json",
            {
                "overall_status": "ok",
                "basket": ["RELIANCE.NS"],
                "steps": [
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                            "allocation_research_target_mix": "both=1, rl=1",
                            "team_research_alignment_summary": "basket/research drifting 1/3",
                            "team_research_alignment_target_mix": "ml=1",
                            "team_discovery_alignment_summary": "discovery drifting 1/3",
                            "team_discovery_overlap_symbols": "RELIANCE.NS",
                            "team_discovery_alignment_overlap_count": 1,
                            "team_discovery_alignment_compare_count": 3,
                        },
                    }
                ],
            },
        ),
        (
            "20260602_0100.json",
            {
                "overall_status": "ok",
                "basket": ["TCS.NS"],
                "steps": [
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                        },
                    }
                ],
            },
        ),
    ):
        (nightly_dir / name).write_text(json.dumps(detail), encoding="utf-8")

    engine = SimpleNamespace(
        settings=settings,
        _agentic_store=None,
        _agentic_learning_store=None,
        rl_generator=None,
        state=SimpleNamespace(symbol=""),
        _agentic_orchestrator=None,
    )
    status = build_model_status(engine)

    assert status.agentic.nightly_alignment.report_count == 2
    assert status.agentic.nightly_alignment.aligned_reports == 1
    assert status.agentic.nightly_alignment.recommended_action is not None
    assert status.agentic.nightly_alignment.recommended_cli_command is not None
    assert status.agentic.nightly_alignment.recommended_refresh_target == "all"
    assert status.agentic.nightly_alignment.effective_refresh_target == "rl"
    assert status.agentic.nightly_alignment.recommended_force_refresh is True
    assert status.agentic.nightly_alignment.workflow_target_mismatch is True
    assert status.agentic.nightly_alignment.latest_workflow_research_recommended_target == "rl"
    assert (
        status.agentic.nightly_alignment.latest_workflow_discovery_alignment_summary
        == "discovery drifting 1/3"
    )
    assert status.agentic.nightly_alignment.latest_workflow_discovery_warning is True
    assert (
        "RL-oriented training research plan"
        in status.agentic.nightly_alignment.recommended_action
    )
    assert (
        "build_training_research_plan.py"
        in status.agentic.nightly_alignment.recommended_cli_command
    )
    assert "--refresh-target rl" in status.agentic.nightly_alignment.recommended_cli_command
    assert status.agentic.nightly_alignment.recommended_discovery_action is not None
    assert (
        "Rebuild the market universe"
        in status.agentic.nightly_alignment.recommended_discovery_action
    )
    assert status.agentic.nightly_alignment.recommended_discovery_cli_command is not None
    assert (
        "build_market_universe.py"
        in status.agentic.nightly_alignment.recommended_discovery_cli_command
    )


def test_build_model_status_biases_recommendation_target_from_refreshed_mix(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        model_status_nightly_alignment_recommendation_ratio=0.75,
    )
    nightly_dir = tmp_path / "reports" / "nightly"
    nightly_dir.mkdir(parents=True, exist_ok=True)
    for name, detail in (
        (
            "20260603_0100.json",
            {
                "overall_status": "ok",
                "basket": ["INFY.NS"],
                "steps": [
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                            "allocation_research_target_mix": "ml=1",
                            "allocation_refreshed_research_target_mix": "ml=1",
                            "team_research_alignment_summary": "basket/research aligned 1/1",
                            "team_research_alignment_target_mix": "rl=1",
                        },
                    }
                ],
            },
        ),
        (
            "20260602_0100.json",
            {
                "overall_status": "ok",
                "basket": ["TCS.NS"],
                "steps": [
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                        },
                    }
                ],
            },
        ),
    ):
        (nightly_dir / name).write_text(json.dumps(detail), encoding="utf-8")

    engine = SimpleNamespace(
        settings=settings,
        _agentic_store=None,
        _agentic_learning_store=None,
        rl_generator=None,
        state=SimpleNamespace(symbol=""),
        _agentic_orchestrator=None,
    )
    status = build_model_status(engine)

    assert status.agentic.nightly_alignment.recommended_action is not None
    assert status.agentic.nightly_alignment.recommended_refresh_target == "ml"
    assert status.agentic.nightly_alignment.effective_refresh_target == "ml"
    assert status.agentic.nightly_alignment.recommended_force_refresh is False
    assert status.agentic.nightly_alignment.workflow_target_mismatch is False
    assert (
        "ML-oriented training research plan"
        in status.agentic.nightly_alignment.recommended_action
    )
    assert status.agentic.nightly_alignment.recommended_cli_command is not None
    assert "--refresh-target ml" in status.agentic.nightly_alignment.recommended_cli_command


def test_nightly_alignment_posture_prefers_refreshed_recent_signal():
    posture = model_status._nightly_alignment_posture(
        recent_rows=(
            model_status.NightlyAlignmentRow(
                path="reports/nightly/latest.json",
                target_mix="both=1, rl=1",
                refreshed_target_mix="rl=2",
            ),
            model_status.NightlyAlignmentRow(
                path="reports/nightly/prior.json",
                target_mix="ml=1",
                refreshed_target_mix="ml=1",
            ),
        ),
        latest_refreshed_target_mix="rl=2",
        latest_target_mix="both=1, rl=1",
    )

    assert posture == "rl"


def test_build_model_status_uses_discovery_context_to_narrow_refresh_target(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        model_status_nightly_alignment_recommendation_ratio=0.75,
        default_timeframe="5m",
        default_days=10,
        market_universe_screener_csv=Path("data/universe/screener_export.csv"),
    )
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    screener_path.parent.mkdir(parents=True, exist_ok=True)
    screener_path.write_text(
        "Symbol,Company Name,Current Price,Volume\nRELIANCE,Rel,100,1000\n",
        encoding="utf-8",
    )

    nightly_dir = tmp_path / "reports" / "nightly"
    nightly_dir.mkdir(parents=True, exist_ok=True)
    (nightly_dir / "20260603_0100.json").write_text(
        json.dumps(
            {
                "overall_status": "ok",
                "basket": ["RELIANCE.NS"],
                "steps": [
                    {
                        "name": "training_research",
                        "detail": {
                            "path": "reports/nightly/training_research_plan.json",
                            "count": 3,
                            "ml_count": 2,
                            "rl_count": 1,
                            "selection_policy": "diversified",
                            "refresh_target": "all",
                            "discovery_preferred_count": 2,
                            "discovery_regime_mix": "ranging=3",
                            "discovery_summary": "preferred=2 | regimes=ranging=3",
                        },
                    },
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                            "universe_count": 10,
                            "shortlist_count": 5,
                            "briefing_candidates": 2,
                            "ml_count": 3,
                            "rl_count": 1,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (nightly_dir / "20260602_0100.json").write_text(
        json.dumps(
            {
                "overall_status": "ok",
                "basket": ["TCS.NS"],
                "steps": [
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                            "allocation_research_target_mix": "both=1",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
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

    assert status.agentic.nightly_alignment.recommended_refresh_target == "all"
    assert (
        status.agentic.nightly_alignment.latest_training_research_discovery_recommended_target
        == "ml"
    )
    assert status.agentic.nightly_alignment.effective_refresh_target == "ml"
    assert status.agentic.nightly_alignment.recommended_action is not None
    assert (
        "ML-oriented training research plan"
        in status.agentic.nightly_alignment.recommended_action
    )
    assert (
        "Discovery context currently leans ML."
        in status.agentic.nightly_alignment.recommended_action
    )
    assert status.agentic.nightly_alignment.recommended_cli_command is not None
    assert "--refresh-target ml" in status.agentic.nightly_alignment.recommended_cli_command


def test_build_model_status_uses_scout_support_to_break_refresh_target_tie(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        model_status_nightly_alignment_recommendation_ratio=0.75,
        default_timeframe="5m",
        default_days=10,
        market_universe_screener_csv=Path("data/universe/screener_export.csv"),
    )
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    screener_path.parent.mkdir(parents=True, exist_ok=True)
    screener_path.write_text(
        "Symbol,Company Name,Current Price,Volume\nRELIANCE,Rel,100,1000\n",
        encoding="utf-8",
    )

    nightly_dir = tmp_path / "reports" / "nightly"
    nightly_dir.mkdir(parents=True, exist_ok=True)
    (nightly_dir / "20260603_0100.json").write_text(
        json.dumps(
            {
                "overall_status": "warn",
                "basket": ["RELIANCE.NS"],
                "steps": [
                    {
                        "name": "training_research",
                        "detail": {
                            "path": "reports/nightly/training_research_plan.json",
                            "count": 2,
                            "ml_count": 1,
                            "rl_count": 1,
                            "selection_policy": "diversified",
                            "refresh_target": "all",
                            "discovery_preferred_count": 2,
                            "discovery_regime_mix": "ranging=2",
                            "discovery_summary": "preferred=2 | regimes=ranging=2",
                            "scout_support_target": "ml",
                        },
                    },
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                            "allocation_research_target_mix": "both=1",
                            "team_research_alignment_target_mix": "ml=1",
                            "team_research_scout_support_target": "ml",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (nightly_dir / "20260602_0100.json").write_text(
        json.dumps(
            {
                "overall_status": "ok",
                "basket": ["TCS.NS"],
                "steps": [
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
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

    assert status.agentic.nightly_alignment.recommended_refresh_target == "all"
    assert status.agentic.nightly_alignment.latest_workflow_research_recommended_target == "rl"
    assert (
        status.agentic.nightly_alignment.latest_training_research_discovery_recommended_target
        == "ml"
    )
    assert status.agentic.nightly_alignment.latest_workflow_research_scout_support_target == "ml"
    assert status.agentic.nightly_alignment.latest_training_research_scout_support_target == "ml"
    assert status.agentic.nightly_alignment.effective_refresh_target == "ml"
    assert status.agentic.nightly_alignment.recommended_action is not None
    assert "--refresh-target ml" in (
        status.agentic.nightly_alignment.recommended_cli_command or ""
    )


def test_build_model_status_activation_disabled_lanes(tmp_path):
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
    assert status.activation is not None
    assert status.activation.ml.stage == "disabled"
    assert status.activation.rl.stage == "disabled"
    assert status.to_dict()["activation"]["ml"]["stage"] == "disabled"


def test_build_lane_activation_unpromoted_rl(tmp_path):
    from fortuna.app.model_activation import build_lane_activation_summary
    from fortuna.rl.training.checkpoint import OOSMetricsSummary, PolicyCheckpoint

    models_root = tmp_path / "models"
    run_dir = models_root / "validated" / "run_unpromoted"
    run_dir.mkdir(parents=True)
    (run_dir / "policy.zip").write_bytes(b"\x00")
    (run_dir / "normalizer.json").write_text("{}", encoding="utf-8")
    PolicyCheckpoint(
        run_id="run_unpromoted",
        symbol="RELIANCE.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=10, sharpe_ratio=1.2),
        verdict_passed=True,
        verdict_score=0.8,
        advisory_ready=True,
    ).write(run_dir / "metadata.json")

    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        agentic_enabled=True,
        model_registry_enabled=True,
        model_promotion_required=True,
    )
    activation = build_lane_activation_summary(
        settings,
        lane="rl",
        symbol="RELIANCE.NS",
        models_root=models_root,
    )
    assert activation.stage == "unpromoted"
    assert activation.recommended_command is not None
    assert "run_unpromoted" in activation.recommended_command


def test_build_lane_activation_not_loaded_with_pointer(tmp_path):
    from fortuna.app.model_activation import build_lane_activation_summary
    from fortuna.models import PromotionRecord, PromotionStatus, promote
    from fortuna.rl.training.checkpoint import OOSMetricsSummary, PolicyCheckpoint

    models_root = tmp_path / "models"
    run_dir = models_root / "validated" / "run_ready"
    run_dir.mkdir(parents=True)
    (run_dir / "policy.zip").write_bytes(b"\x00")
    (run_dir / "normalizer.json").write_text("{}", encoding="utf-8")
    cp = PolicyCheckpoint(
        run_id="run_ready",
        symbol="RELIANCE.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=10, sharpe_ratio=1.2),
        verdict_passed=True,
        verdict_score=0.8,
        advisory_ready=True,
    )
    cp.write(run_dir / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, run_dir, status=PromotionStatus.VALIDATED)
    promote(record, models_root=models_root, promoted_by="test")

    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        agentic_enabled=True,
        model_registry_enabled=True,
        model_promotion_required=True,
    )
    activation = build_lane_activation_summary(
        settings,
        lane="rl",
        symbol="RELIANCE.NS",
        models_root=models_root,
        rl_status=SimpleNamespace(
            available=False,
            advisory_ready=True,
            load_error="no_generator",
            run_id=None,
            checkpoint_dir="",
        ),
    )
    assert activation.stage == "not_loaded"


def test_build_lane_activation_active_when_loaded(tmp_path):
    from fortuna.agentic.contracts import RlModelStatus
    from fortuna.app.model_activation import build_lane_activation_summary

    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        agentic_enabled=True,
    )
    activation = build_lane_activation_summary(
        settings,
        lane="rl",
        symbol="RELIANCE.NS",
        models_root=tmp_path / "models",
        rl_status=RlModelStatus(
            symbol="RELIANCE.NS",
            available=True,
            advisory_ready=True,
            load_error=None,
            run_id="run_live",
            checkpoint_dir=str(tmp_path / "models" / "validated" / "run_live"),
        ),
    )
    assert activation.stage == "active"
    assert activation.active is True
