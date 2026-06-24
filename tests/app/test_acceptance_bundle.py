from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fortuna.agentic.contracts import NightlyArtifactLinkageSummary
from fortuna.app.acceptance_bundle import (
    AcceptanceBundle,
    build_and_export_acceptance_bundle,
    export_acceptance_bundle,
    format_acceptance_bundle,
    gather_acceptance_bundle,
)
from fortuna.config.settings import Settings
from fortuna.models import PromotionRecord, PromotionStatus, promote
from fortuna.rl.training.checkpoint import OOSMetricsSummary, PolicyCheckpoint


def _write_rl_artifact(root: Path, run_id: str = "run_acceptance") -> Path:
    cp_dir = root / "validated" / run_id
    cp_dir.mkdir(parents=True)
    (cp_dir / "policy.zip").write_bytes(b"\x00")
    (cp_dir / "normalizer.json").write_text("{}", encoding="utf-8")
    PolicyCheckpoint(
        run_id=run_id,
        symbol="RELIANCE.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=11, sharpe_ratio=1.35, profit_factor=1.2),
        verdict_passed=True,
        verdict_score=0.85,
        verdict_reasons=["strong_oos"],
        advisory_ready=True,
    ).write(cp_dir / "metadata.json")
    return cp_dir


def _write_workflow_snapshot(
    path: Path,
    *,
    review_kind: str = "rl_policy",
    review_kinds: str | None = None,
    execution_target: str = "rl",
    execution_family: str = "rl",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": "auto",
                "timeframe": "5m",
                "lookback_days": 20,
                "team": {
                    "metrics": {
                        "count": 6,
                        "ok_count": 5,
                        "nightly_report_count": 2,
                        "nightly_enabled_reports": 2,
                        "nightly_aligned_reports": 1,
                        "nightly_latest_target_mix": "both=1, rl=1",
                        "nightly_latest_execution_target": execution_target,
                        "nightly_latest_execution_model_family": execution_family,
                        "nightly_latest_execution_selection_source": "training_research",
                        "nightly_latest_promotion_review_model_kind": review_kind,
                        "nightly_latest_promotion_review_model_kinds": review_kinds,
                        "nightly_latest_refreshed_target_mix": "both=1",
                        "nightly_recommended_target": "all",
                        "nightly_recommended_force_refresh": True,
                        "nightly_recent_window": 2,
                        "nightly_recent_enabled": 2,
                        "nightly_recent_aligned": 1,
                        "nightly_recent_latest_status": "ok",
                        "nightly_recent_latest_basket_size": 1,
                        "research_alignment_summary": "basket/research aligned 3/3",
                        "research_alignment_target_mix": "both=1, rl=2",
                        "research_alignment_overlap_count": 3,
                        "research_alignment_selected_count": 3,
                        "research_discovery_follow_up_target": "ml",
                        "research_discovery_follow_up_summary": (
                            "Discovery scouts currently lean ML-focused from ml universe posture."
                        ),
                        "research_discovery_follow_up_action": (
                            "Refresh market universe and shortlist review with ML-focused "
                            "discovery focus before the next training cycle."
                        ),
                        "research_research_follow_up_target": "rl",
                        "research_research_follow_up_summary": (
                            "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
                        ),
                        "research_research_follow_up_action": (
                            "Prioritize RL refresh follow-up from remediation pressure "
                            "across 2 selected row(s)."
                        ),
                        "research_execution_follow_up_target": execution_target,
                        "research_execution_follow_up_summary": (
                            "Nightly execution drift currently leans RL-focused from rl "
                            "execution posture."
                            if execution_target == "rl"
                            else "Nightly execution drift currently leans ML-focused from ml "
                            "execution posture."
                        ),
                        "research_execution_follow_up_action": (
                            "Review nightly execution path and retarget execution "
                            f"candidate selection toward {execution_target.upper()}-focused "
                            "before the next promotion or nightly cycle."
                        ),
                        "research_scout_support_summary": (
                            "supported=2, overlap=1, volume_dense=2, "
                            "symbols=RELIANCE.NS,TCS.NS"
                        ),
                        "research_scout_supported_symbols": "RELIANCE.NS,TCS.NS",
                        "research_scout_overlap_selected_count": 1,
                        "research_scout_volume_dense_selected_count": 2,
                        "discovery_refresh_source": "auto",
                        "discovery_refresh_timeframe": "1d",
                        "discovery_refresh_days": 20,
                        "discovery_refresh_requested": True,
                        "discovery_refreshed_count": 15,
                        "discovery_alignment_summary": "discovery partial 2/3",
                        "discovery_overlap_symbols": "RELIANCE.NS, TCS.NS",
                        "discovery_alignment_overlap_count": 2,
                        "discovery_alignment_compare_count": 3,
                    }
                },
                "universe": {
                    "metrics": {"count": 15, "adaptive_count": 1},
                    "notes": ["adaptive_boost=+0.10 avg_pnl=+1.25% rows=2"],
                },
                "shortlist": {"metrics": {"count": 6}},
                "briefing": {"metrics": {"candidates": 4}},
                "allocation": {
                    "metrics": {"selected_count": 3, "skipped_count": 1, "max_positions": 3},
                    "notes": [
                        "portfolio_research_alignment=enabled",
                        "allocation_research_targets: both=1, rl=2",
                    ],
                },
                "candidates": {"metrics": {"ml_count": 5, "rl_count": 3}},
                "research": {
                    "metrics": {
                        "count": 4,
                        "ml_count": 3,
                        "rl_count": 2,
                        "selection_policy": "diversified",
                        "refresh_target": "all",
                        "refresh_requested": True,
                        "refreshed_count": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_nightly_report(
    path: Path,
    workflow_path: Path,
    promotion_paths: list[str] | None = None,
    *,
    basket_mode: str = "derived_training_candidates",
    selection_source: str = "training_research",
    execution_target: str = "rl",
    execution_family: str = "rl",
    promotion_model_kind: str = "rl_policy",
    promotion_model_kinds: list[str] | None = None,
    include_promote_step: bool = True,
    training_candidate_path: str = "reports/nightly/training_candidates.json",
    training_research_path: str = "reports/nightly/training_research_plan.json",
) -> None:
    steps = [
        {
            "name": "basket_resolution",
            "status": "ok",
            "detail": {"mode": basket_mode},
        },
        {
            "name": "training_candidates",
            "status": "ok",
            "detail": {
                "path": training_candidate_path,
                "count": 5,
                "ml_count": 4,
                "rl_count": 2,
                "selection_policy": "diversified",
            },
        },
        {
            "name": "training_research",
            "status": "ok",
            "detail": {
                "path": training_research_path,
                "count": 4,
                "ml_count": 3,
                "rl_count": 2,
                "selection_policy": "diversified",
                "refresh_target": "all",
                "discovery_preferred_count": 2,
                "discovery_regime_mix": "trending=3, volatile=1",
                "discovery_summary": (
                    "preferred=2 | liquidity=RELIANCE.NS,TCS.NS | "
                    "activity=TCS.NS,SBIN.NS | regimes=trending=3, volatile=1"
                ),
                "discovery_recommended_refresh_target": "rl",
                "discovery_follow_up_target": "rl",
                "discovery_follow_up_summary": (
                    "Discovery scouts currently lean RL-focused from rl universe posture."
                ),
                "discovery_follow_up_action": (
                    "Refresh market universe and shortlist review with RL-focused "
                    "discovery focus before the next training cycle."
                ),
                "research_follow_up_target": "rl",
                "research_follow_up_summary": (
                    "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
                ),
                "research_follow_up_action": (
                    "Prioritize RL refresh follow-up from remediation pressure "
                    "across 2 selected row(s)."
                ),
                "execution_follow_up_target": execution_target,
                "execution_follow_up_summary": (
                    "Nightly execution drift currently leans RL-focused from rl execution posture."
                    if execution_target == "rl"
                    else (
                        "Nightly execution drift currently leans ML-focused from ml "
                        "execution posture."
                    )
                ),
                "execution_follow_up_action": (
                    "Review nightly execution path and retarget execution "
                    f"candidate selection toward {execution_target.upper()}-focused "
                    "before the next promotion or nightly cycle."
                ),
                "effective_refresh_target": "rl",
                "scout_support_target": "rl",
            },
        },
        {
            "name": "training_execution_target",
            "status": "ok",
            "detail": {
                "target": execution_target,
                "run_ml": execution_family in {"ml", "hybrid"},
                "run_rl": execution_family in {"rl", "hybrid"},
                "selection_source": selection_source,
            },
        },
        {
            "name": "workflow_snapshot",
            "status": "ok",
            "detail": {
                "path": str(workflow_path),
                "universe_count": 15,
                "shortlist_count": 6,
                "briefing_candidates": 4,
                "ml_count": 5,
                "rl_count": 3,
            },
        },
    ]
    if include_promote_step:
        steps.append(
            {
                "name": "promote_best",
                "status": "ok",
                "detail": {
                    "count": len(promotion_paths or []),
                    "model_kind": promotion_model_kind,
                },
            }
        )
    steps.append(
        {
            "name": "promotion_reviews",
            "status": "ok",
            "detail": {
                "count": len(promotion_paths or []),
                "format": "md",
                "paths": promotion_paths or [],
                "model_kind": promotion_model_kind,
                "model_kinds": promotion_model_kinds or [],
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "started_at": "2026-06-03T01:00:00",
                "ended_at": "2026-06-03T01:10:00",
                "total_duration_s": 600.0,
                "basket": ["RELIANCE.NS"],
                "overall_status": "ok",
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )


def _write_promotion_review_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Promotion Review\n", encoding="utf-8")


def test_gather_acceptance_bundle_links_workflow_review_and_model_health(tmp_path: Path):
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
        agentic_ml_scorer_enabled=False,
        model_registry_enabled=True,
        model_promotion_required=True,
        agentic_log_dir=tmp_path / "agentic",
        ml_scorer_artifact_dir=Path("models/ml_signal_scorer/validated"),
    )
    models_root = tmp_path / "models"
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    nightly_report = tmp_path / "reports" / "nightly" / "20260603_0100.json"
    training_candidate_path = tmp_path / "reports" / "nightly" / "training_candidates.json"
    training_research_path = tmp_path / "reports" / "nightly" / "training_research_plan.json"
    _write_workflow_snapshot(workflow_path)
    training_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    training_candidate_path.write_text("{}", encoding="utf-8")
    training_research_path.write_text("{}", encoding="utf-8")
    _write_nightly_report(nightly_report, workflow_path)
    art = _write_rl_artifact(models_root)
    cp = PolicyCheckpoint.read(art / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, art, status=PromotionStatus.VALIDATED)
    record.workflow_snapshot_path = str(workflow_path)
    promote(record, models_root=models_root, promoted_by="test")

    engine = SimpleNamespace(
        settings=settings,
        _agentic_store=None,
        _agentic_learning_store=None,
        rl_generator=None,
        state=SimpleNamespace(symbol="RELIANCE.NS"),
        _agentic_orchestrator=None,
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        engine=engine,
        models_root=models_root,
        ml_base=models_root / "ml_signal_scorer",
        symbol="RELIANCE.NS",
        nightly_report_path=nightly_report,
        workflow_snapshot_path=workflow_path,
        include_model_health=True,
    )

    assert bundle.overall_status == "pass"
    assert bundle.workflow_snapshot is not None
    assert bundle.nightly_report is not None
    assert bundle.nightly_report.training_research_discovery_preferred_count == 2
    assert bundle.nightly_report.training_research_discovery_regime_mix == "trending=3, volatile=1"
    assert "preferred=2" in str(bundle.nightly_report.training_research_discovery_summary or "")
    assert bundle.nightly_report.training_research_discovery_recommended_refresh_target == "rl"
    assert bundle.nightly_report.training_research_effective_refresh_target == "rl"
    assert bundle.effective_refresh_target is not None
    assert bundle.cross_artifact_alignment is not None
    assert bundle.nightly_report.training_execution_target == "rl"
    assert bundle.nightly_report.training_execution_model_family == "rl"
    assert bundle.nightly_report.promoted_model_kind == "rl_policy"
    assert bundle.nightly_report.promotion_review_model_kind == "rl_policy"
    assert bundle.workflow_snapshot.universe_count == 15
    assert bundle.workflow_snapshot.team_role_count == 6
    assert bundle.workflow_snapshot.team_nightly_recommended_target == "all"
    assert bundle.workflow_snapshot.team_research_discovery_follow_up_target == "ml"
    assert bundle.workflow_snapshot.team_research_research_follow_up_target == "rl"
    assert bundle.workflow_snapshot.team_research_execution_follow_up_target == "rl"
    assert bundle.rl_review is not None
    assert bundle.rl_review.record.run_id == "run_acceptance"
    assert bundle.model_health is not None
    assert bundle.model_health.rl.last_promotion is not None
    assert bundle.model_health.rl.last_promotion.workflow_snapshot is not None
    text = format_acceptance_bundle(bundle)
    assert "acceptance bundle: pass" in text
    assert "target=rl family=rl source=training_research" in text
    assert "team_posture=roles=5/6 nightly=1/2 target=all force_refresh=1" in text
    assert "team_research_alignment=basket/research aligned 3/3 mix=both=1, rl=2" in text
    assert (
        "team_research_scouts=supported=2, overlap=1, volume_dense=2, symbols=RELIANCE.NS,TCS.NS"
        in text
    )
    assert "team_follow_up=discovery=ml research=rl execution=rl" in text
    assert "refresh_context=effective_target=" in text
    assert "cross_artifact_alignment=status=" in text
    assert "team_discovery_refresh=15 source=auto timeframe=1d days=20" in text
    assert "team_discovery_alignment=discovery partial 2/3 overlap=RELIANCE.NS, TCS.NS" in text
    assert "team_recent_trend=1/2 over 2 run(s) latest=ok basket=1" in text
    assert "[pass] promotion_workflow_linkage" in text


def test_gather_acceptance_bundle_uses_nightly_report_artifacts(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    models_root = tmp_path / "models"
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    nightly_report = tmp_path / "reports" / "nightly" / "20260603_0100.json"
    training_candidate_path = tmp_path / "reports" / "nightly" / "training_candidates.json"
    training_research_path = tmp_path / "reports" / "nightly" / "training_research_plan.json"
    review_path = tmp_path / "reports" / "nightly" / "promotions" / "RELIANCE_NS__run_ready.md"
    _write_workflow_snapshot(workflow_path)
    training_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    training_candidate_path.write_text("{}", encoding="utf-8")
    training_research_path.write_text("{}", encoding="utf-8")
    _write_promotion_review_artifact(review_path)
    _write_nightly_report(
        nightly_report,
        workflow_path,
        promotion_paths=["reports/nightly/promotions/RELIANCE_NS__run_ready.md"],
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=models_root,
        ml_base=models_root / "ml_signal_scorer",
        nightly_report_path=nightly_report,
        include_model_health=False,
    )

    assert bundle.nightly_report is not None
    assert bundle.nightly_report.path.endswith("20260603_0100.json")
    assert (
        bundle.nightly_report.training_candidate_path
        == "reports/nightly/training_candidates.json"
    )
    assert bundle.nightly_report.training_candidate_exists is True
    assert bundle.nightly_report.training_candidate_count == 5
    assert bundle.nightly_report.training_research_exists is True
    assert bundle.nightly_report.training_research_count == 4
    assert bundle.nightly_report.training_execution_target == "rl"
    assert bundle.nightly_report.training_execution_model_family == "rl"
    assert bundle.nightly_report.workflow_snapshot_path == str(workflow_path)
    assert bundle.nightly_report.workflow_snapshot_exists is True
    assert bundle.nightly_report.basket_mode == "derived_training_candidates"
    assert bundle.nightly_report.promotion_review_model_kind == "rl_policy"
    assert bundle.workflow_snapshot is not None
    assert bundle.workflow_snapshot.universe_count == 15
    assert any(row.name == "nightly_report" and row.status == "pass" for row in bundle.checks)
    assert any(
        row.name == "candidate_nightly_shape" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "nightly_workflow_execution_consistency" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "nightly_training_candidates_reference" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "nightly_training_research_reference" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "nightly_review_artifacts" and row.status == "pass" for row in bundle.checks
    )
    assert any(
        row.name == "allocation_research_alignment" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "training_research_refresh" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "workflow_discovery_refresh" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "multi_agent_team_posture" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "workflow_basket_research_alignment" and row.status == "pass"
        for row in bundle.checks
    )
    assert any(
        row.name == "multi_agent_team_posture"
        and "recent=1/2 over 2 run(s) latest=ok:1." in row.detail
        and "basket_research=basket/research aligned 3/3 mix=both=1, rl=2." in row.detail
        and "discovery=discovery partial 2/3 overlap=RELIANCE.NS, TCS.NS." in row.detail
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_tracks_ml_nightly_execution_and_review_kind(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    models_root = tmp_path / "models"
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    nightly_report = tmp_path / "reports" / "nightly" / "20260603_0100.json"
    review_path = tmp_path / "reports" / "nightly" / "promotions" / "ml_rel.md"
    _write_workflow_snapshot(workflow_path)
    _write_promotion_review_artifact(review_path)
    candidate_path = tmp_path / "reports" / "nightly" / "training_candidates.json"
    research_path = tmp_path / "reports" / "nightly" / "training_research_plan.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text("{}", encoding="utf-8")
    research_path.write_text("{}", encoding="utf-8")
    _write_nightly_report(
        nightly_report,
        workflow_path,
        promotion_paths=["reports/nightly/promotions/ml_rel.md"],
        execution_target="ml",
        execution_family="ml",
        promotion_model_kind="ml_scorer",
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=models_root,
        ml_base=models_root / "ml_signal_scorer",
        nightly_report_path=nightly_report,
        include_model_health=False,
    )

    assert bundle.nightly_report is not None
    assert bundle.nightly_report.training_execution_target == "ml"
    assert bundle.nightly_report.training_execution_model_family == "ml"
    assert bundle.nightly_report.training_execution_selection_source == "training_research"
    assert bundle.nightly_report.promoted_model_kind == "ml_scorer"
    assert bundle.nightly_report.promotion_review_model_kind == "ml_scorer"
    text = format_acceptance_bundle(bundle)
    assert "target=ml family=ml source=training_research" in text


def test_gather_acceptance_bundle_tracks_mixed_nightly_review_kinds(tmp_path: Path):
    settings = Settings(agentic_enabled=True)
    workflow_path = tmp_path / "reports" / "nightly" / "workflow_snapshot.json"
    _write_workflow_snapshot(
        workflow_path,
        review_kind="mixed",
        review_kinds="ml_scorer, rl_policy",
        execution_target="all",
        execution_family="hybrid",
    )
    nightly_path = tmp_path / "reports" / "nightly" / "nightly_report.json"
    training_candidate_path = tmp_path / "reports" / "nightly" / "training_candidates.json"
    training_candidate_path.write_text("{}", encoding="utf-8")
    training_research_path = tmp_path / "reports" / "nightly" / "training_research_plan.json"
    training_research_path.write_text("{}", encoding="utf-8")
    _write_nightly_report(
        nightly_path,
        workflow_path,
        promotion_paths=[
            "reports/nightly/promotions/ml_rel.md",
            "reports/nightly/promotions/rl_rel.md",
        ],
        execution_target="all",
        execution_family="hybrid",
        promotion_model_kind="mixed",
        promotion_model_kinds=["ml_scorer", "rl_policy"],
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        include_model_health=False,
        nightly_report_path=nightly_path,
        workflow_snapshot_path=workflow_path,
    )

    assert bundle.nightly_report is not None
    assert bundle.nightly_report.training_execution_target == "all"
    assert bundle.nightly_report.training_execution_model_family == "hybrid"
    assert bundle.nightly_report.promotion_review_model_kind == "mixed"
    assert bundle.nightly_report.promotion_review_model_kinds == (
        "ml_scorer",
        "rl_policy",
    )
    text = format_acceptance_bundle(bundle)
    assert "target=all family=hybrid source=training_research" in text
    assert "review_kind=ml_scorer, rl_policy" in text
    assert any(
        row.name == "nightly_workflow_execution_consistency"
        and row.status == "pass"
        and "review_kinds=ml_scorer, rl_policy" in row.detail
        for row in bundle.checks
    )
    assert bundle.dual_promotion_review is not None
    assert bundle.dual_promotion_review.both_present is True
    assert bundle.dual_promotion_review.posture in {"expected_dual", "coherent_mixed"}
    assert bundle.dual_promotion_review.warning is False
    text = format_acceptance_bundle(bundle)
    assert "dual_promotion=posture=" in text
    assert any(
        row.name == "dual_promotion_posture" and row.status == "pass"
        for row in bundle.checks
    )


def test_format_acceptance_bundle_includes_replay_linkage_diagnostics():
    bundle = AcceptanceBundle(
        created_at="2026-06-13T09:15:00",
        overall_status="warn",
        replay_linkage=NightlyArtifactLinkageSummary(
            overall_status="partial",
            replay_status="partial",
            run_identifier="dry_run_emit_replay",
            missing_artifacts=("promotion_review",),
            linkage_warnings=("promotion review was not emitted for a promoted replay artifact",),
            summary="optional_missing=promotion_review",
        ),
    )

    text = format_acceptance_bundle(bundle)

    assert "replay_linkage=status=partial" in text
    assert "run=dry_run_emit_replay" in text
    assert "missing=promotion_review" in text


def test_gather_acceptance_bundle_flags_conflicting_dual_promotion_posture(tmp_path: Path):
    settings = Settings(agentic_enabled=True)
    workflow_path = tmp_path / "reports" / "nightly" / "workflow_snapshot.json"
    _write_workflow_snapshot(
        workflow_path,
        review_kind="mixed",
        review_kinds="ml_scorer, rl_policy",
        execution_target="rl",
        execution_family="rl",
    )
    nightly_path = tmp_path / "reports" / "nightly" / "nightly_report.json"
    training_candidate_path = tmp_path / "reports" / "nightly" / "training_candidates.json"
    training_candidate_path.write_text("{}", encoding="utf-8")
    training_research_path = tmp_path / "reports" / "nightly" / "training_research_plan.json"
    training_research_path.write_text("{}", encoding="utf-8")
    _write_nightly_report(
        nightly_path,
        workflow_path,
        promotion_paths=[
            "reports/nightly/promotions/ml_rel.md",
            "reports/nightly/promotions/rl_rel.md",
        ],
        execution_target="rl",
        execution_family="rl",
        promotion_model_kind="mixed",
        promotion_model_kinds=["ml_scorer", "rl_policy"],
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        include_model_health=False,
        nightly_report_path=nightly_path,
        workflow_snapshot_path=workflow_path,
    )

    assert bundle.dual_promotion_review is not None
    assert bundle.dual_promotion_review.posture == "conflicting"
    assert bundle.dual_promotion_review.warning is True
    assert any(
        row.name == "dual_promotion_posture"
        and row.status == "warn"
        and "conflicts" in row.detail.lower()
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_single_lane_dual_promotion_posture(tmp_path: Path):
    settings = Settings(agentic_enabled=True)
    workflow_path = tmp_path / "reports" / "nightly" / "workflow_snapshot.json"
    _write_workflow_snapshot(workflow_path)
    nightly_path = tmp_path / "reports" / "nightly" / "nightly_report.json"
    training_candidate_path = tmp_path / "reports" / "nightly" / "training_candidates.json"
    training_candidate_path.write_text("{}", encoding="utf-8")
    training_research_path = tmp_path / "reports" / "nightly" / "training_research_plan.json"
    training_research_path.write_text("{}", encoding="utf-8")
    _write_nightly_report(nightly_path, workflow_path)

    bundle = gather_acceptance_bundle(
        settings=settings,
        include_model_health=False,
        nightly_report_path=nightly_path,
        workflow_snapshot_path=workflow_path,
    )

    assert bundle.dual_promotion_review is not None
    assert bundle.dual_promotion_review.posture == "single_lane"
    assert bundle.dual_promotion_review.warning is False
    assert any(
        row.name == "dual_promotion_posture" and row.status == "pass"
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_warns_on_weak_basket_research_alignment(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        acceptance_bundle_team_research_alignment_warn_ratio=0.75,
    )
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    _write_workflow_snapshot(workflow_path)
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    payload["team"]["metrics"]["research_alignment_summary"] = "basket/research drifting 1/3"
    payload["team"]["metrics"]["research_alignment_target_mix"] = "none=2, rl=1"
    payload["team"]["metrics"]["research_alignment_overlap_count"] = 1
    payload["team"]["metrics"]["research_alignment_selected_count"] = 3
    workflow_path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = gather_acceptance_bundle(
        settings=settings,
        workflow_snapshot_path=workflow_path,
        include_model_health=False,
    )

    assert any(
        row.name == "workflow_basket_research_alignment"
        and row.status == "warn"
        and "ratio=0.33 (1/3)" in row.detail
        and "recommended_target=all." in row.detail
        for row in bundle.checks
    )
    assert bundle.recommended_refresh_target == "all"


def test_gather_acceptance_bundle_warns_on_weak_discovery_alignment(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        acceptance_bundle_team_discovery_alignment_warn_ratio=0.75,
    )
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    _write_workflow_snapshot(workflow_path)
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    payload["team"]["metrics"]["discovery_alignment_summary"] = "discovery drifting 1/3"
    payload["team"]["metrics"]["discovery_overlap_symbols"] = "RELIANCE.NS"
    payload["team"]["metrics"]["discovery_alignment_overlap_count"] = 1
    payload["team"]["metrics"]["discovery_alignment_compare_count"] = 3
    workflow_path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = gather_acceptance_bundle(
        settings=settings,
        workflow_snapshot_path=workflow_path,
        include_model_health=False,
    )

    assert any(
        row.name == "workflow_discovery_alignment"
        and row.status == "warn"
        and "ratio=0.33 (1/3)" in row.detail
        and "discovery drifting 1/3" in row.detail
        and "overlap=RELIANCE.NS." in row.detail
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_surfaces_recent_nightly_alignment_warning(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        acceptance_bundle_nightly_alignment_min_enabled_reports=2,
        acceptance_bundle_nightly_alignment_warn_ratio=0.75,
    )
    models_root = tmp_path / "models"
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    nightly_dir = tmp_path / "reports" / "nightly"
    nightly_report = nightly_dir / "20260603_0100.json"
    training_candidate_path = nightly_dir / "training_candidates.json"
    training_research_path = nightly_dir / "training_research_plan.json"
    _write_workflow_snapshot(workflow_path)
    training_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    training_candidate_path.write_text("{}", encoding="utf-8")
    training_research_path.write_text("{}", encoding="utf-8")
    _write_nightly_report(nightly_report, workflow_path, promotion_paths=[])
    (
        nightly_dir / "20260602_0100.json"
    ).write_text(
        json.dumps(
            {
                "overall_status": "ok",
                "basket": ["SBIN.NS"],
                "steps": [
                    {
                        "name": "workflow_snapshot",
                        "detail": {
                            "allocation_research_alignment_enabled": True,
                            "allocation_research_target_mix": "both=1, rl=1",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (
        nightly_dir / "20260601_0100.json"
    ).write_text(
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

    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=models_root,
        ml_base=models_root / "ml_signal_scorer",
        nightly_report_path=nightly_report,
        include_model_health=True,
    )

    assert bundle.model_health is not None
    assert bundle.recommended_refresh_target == "all"
    assert bundle.recommended_force_refresh is True
    assert bundle.recommended_cli_command is not None
    assert bundle.recommended_discovery_action is not None
    assert "Rebuild the market universe" in bundle.recommended_discovery_action
    assert bundle.recommended_discovery_cli_command is not None
    assert "build_market_universe.py" in bundle.recommended_discovery_cli_command
    assert any(
        row.name == "recent_nightly_alignment" and row.status == "warn"
        for row in bundle.checks
    )
    assert any(
        row.name == "recent_refreshed_nightly_alignment" and row.status == "warn"
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_warns_when_research_refresh_requested_but_empty(
    tmp_path: Path,
):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(
        json.dumps(
            {
                "source": "auto",
                "timeframe": "5m",
                "lookback_days": 20,
                "universe": {"metrics": {"count": 15, "adaptive_count": 1}},
                "shortlist": {"metrics": {"count": 6}},
                "briefing": {"metrics": {"candidates": 4}},
                "allocation": {
                    "metrics": {"selected_count": 3, "skipped_count": 1, "max_positions": 3},
                },
                "candidates": {"metrics": {"ml_count": 5, "rl_count": 3}},
                "research": {
                    "metrics": {
                        "count": 4,
                        "ml_count": 3,
                        "rl_count": 2,
                        "refresh_target": "rl",
                        "refresh_requested": True,
                        "refreshed_count": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=tmp_path / "models",
        ml_base=tmp_path / "models" / "ml_signal_scorer",
        workflow_snapshot_path=workflow_path,
        include_model_health=False,
    )

    assert any(
        row.name == "training_research_refresh" and row.status == "warn"
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_warns_when_discovery_refresh_requested_but_empty(
    tmp_path: Path,
):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    _write_workflow_snapshot(workflow_path)
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    payload["team"]["metrics"]["discovery_refreshed_count"] = 0
    workflow_path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = gather_acceptance_bundle(
        settings=settings,
        workflow_snapshot_path=workflow_path,
        include_model_health=False,
    )

    assert any(
        row.name == "workflow_discovery_refresh"
        and row.status == "warn"
        and "source=auto" in row.detail
        and "timeframe=1d" in row.detail
        and "days=20" in row.detail
        and "refreshed=0" in row.detail
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_flags_missing_nightly_artifacts(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    nightly_report = tmp_path / "reports" / "nightly" / "20260603_0100.json"
    missing_workflow = tmp_path / "reports" / "workflow_snapshot.json"
    _write_nightly_report(
        nightly_report,
        missing_workflow,
        promotion_paths=["reports/nightly/promotions/missing_review.md"],
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=tmp_path / "models",
        ml_base=tmp_path / "models" / "ml_signal_scorer",
        nightly_report_path=nightly_report,
        include_model_health=False,
    )

    assert bundle.overall_status == "fail"
    assert any(
        row.name == "nightly_training_candidates_reference" and row.status == "fail"
        for row in bundle.checks
    )
    assert any(
        row.name == "nightly_training_research_reference" and row.status == "fail"
        for row in bundle.checks
    )
    assert any(
        row.name == "nightly_workflow_reference" and row.status == "fail"
        for row in bundle.checks
    )
    assert any(
        row.name == "nightly_review_artifacts" and row.status == "fail"
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_flags_missing_candidate_nightly_steps(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    nightly_report = tmp_path / "reports" / "nightly" / "20260603_0100.json"
    _write_workflow_snapshot(workflow_path)
    _write_nightly_report(
        nightly_report,
        workflow_path,
        promotion_paths=[],
        include_promote_step=False,
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=tmp_path / "models",
        ml_base=tmp_path / "models" / "ml_signal_scorer",
        nightly_report_path=nightly_report,
        include_model_health=False,
    )

    assert bundle.overall_status == "fail"
    assert any(
        row.name == "candidate_nightly_shape" and row.status == "fail"
        for row in bundle.checks
    )


def test_gather_acceptance_bundle_fails_on_nightly_workflow_execution_mismatch(
    tmp_path: Path,
):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    nightly_report = tmp_path / "reports" / "nightly" / "20260603_0100.json"
    _write_workflow_snapshot(workflow_path)
    _write_nightly_report(
        nightly_report,
        workflow_path,
        promotion_paths=["reports/nightly/promotions/ml_rel.md"],
        execution_target="ml",
        execution_family="ml",
        selection_source="training_candidates",
        promotion_model_kind="ml_scorer",
    )

    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=tmp_path / "models",
        ml_base=tmp_path / "models" / "ml_signal_scorer",
        nightly_report_path=nightly_report,
        workflow_snapshot_path=workflow_path,
        include_model_health=False,
    )

    assert bundle.overall_status == "fail"
    mismatch = next(
        row
        for row in bundle.checks
        if row.name == "nightly_workflow_execution_consistency"
    )
    assert mismatch.status == "fail"
    assert "target nightly=ml workflow=rl" in mismatch.detail
    assert "family nightly=ml workflow=rl" in mismatch.detail
    assert "source nightly=training_candidates workflow=training_research" in mismatch.detail
    assert "review_kind nightly=ml_scorer workflow=rl_policy" in mismatch.detail


def test_export_acceptance_bundle_writes_markdown_and_json(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    _write_workflow_snapshot(workflow_path)

    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=tmp_path / "models",
        ml_base=tmp_path / "models" / "ml_signal_scorer",
        workflow_snapshot_path=workflow_path,
        include_model_health=False,
    )

    md_path = export_acceptance_bundle(bundle, tmp_path / "acceptance.md", fmt="md")
    json_path = export_acceptance_bundle(bundle, tmp_path / "acceptance.json", fmt="json")

    md_text = md_path.read_text(encoding="utf-8")
    assert "# Multi-Agent Acceptance Bundle" in md_text
    assert "- Refresh context: `effective_target=" in md_text
    assert "- Recommended follow-up lane: `" in md_text
    assert "- Cross-artifact alignment: `status=" in md_text
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["workflow_snapshot"]["universe_count"] == 15
    assert payload["workflow_snapshot"]["team_role_count"] == 6
    assert payload["workflow_snapshot"]["team_nightly_recommended_target"] == "all"
    assert payload["workflow_snapshot"]["team_research_execution_follow_up_target"] == "rl"


def test_build_and_export_acceptance_bundle_from_workflow_snapshot(tmp_path: Path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    _write_workflow_snapshot(workflow_path)

    bundle, md_path, json_path = build_and_export_acceptance_bundle(
        settings=settings,
        models_root=tmp_path / "models",
        ml_base=tmp_path / "models" / "ml_signal_scorer",
        workflow_snapshot_path=workflow_path,
        out_dir=tmp_path / "reports" / "dashboard",
        basename="acceptance_bundle_dashboard",
        include_model_health=False,
    )

    assert bundle.workflow_snapshot is not None
    assert md_path.name == "acceptance_bundle_dashboard.md"
    assert json_path.name == "acceptance_bundle_dashboard.json"
    assert md_path.is_file()
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == bundle.overall_status
    assert payload["workflow_snapshot"]["team_discovery_refresh_requested"] is True
    assert payload["workflow_snapshot"]["team_discovery_refreshed_count"] == 15
    assert payload["workflow_snapshot"]["team_nightly_recommended_force_refresh"] is True
    assert payload["workflow_snapshot"]["team_nightly_recent_window"] == 2
    assert payload["workflow_snapshot"]["team_nightly_recent_aligned"] == 1
    assert payload["workflow_snapshot"]["team_research_discovery_follow_up_target"] == "ml"
    assert payload["recommended_refresh_target"] == "all"
    assert payload["recommended_force_refresh"] is False
    assert payload["recommended_cli_command"] is None
    assert payload["recommended_discovery_action"] is not None
    assert "Rebuild the market universe" in payload["recommended_discovery_action"]
    assert payload["recommended_discovery_cli_command"] is not None
    assert "build_market_universe.py" in payload["recommended_discovery_cli_command"]
    assert payload["nightly_report"] is None
    assert payload["checks"][0]["name"] == "nightly_report"


def test_load_nightly_report_evidence_parses_lane_readiness(tmp_path: Path):
    from fortuna.app.acceptance_bundle import load_nightly_report_evidence

    report_path = tmp_path / "reports" / "nightly" / "20260613_0100.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "started_at": "2026-06-13T01:00:00",
                "ended_at": "2026-06-13T01:05:00",
                "total_duration_s": 300.0,
                "basket": ["RELIANCE.NS"],
                "overall_status": "blocked",
                "steps": [
                    {
                        "name": "lane_readiness",
                        "status": "ok",
                        "detail": {
                            "global_blockers": ["smartapi_credentials_missing"],
                            "should_abort": True,
                            "lane_decisions": [
                                {
                                    "lane": "backfill",
                                    "disposition": "block",
                                    "skip_class": "missing_prerequisites",
                                    "detail": "credentials missing",
                                }
                            ],
                        },
                    },
                    {
                        "name": "backfill",
                        "status": "skip",
                        "detail": {"skip_class": "missing_prerequisites"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = load_nightly_report_evidence(nightly_report_path=report_path)
    assert evidence is not None
    assert evidence.lane_readiness_global_blockers == ("smartapi_credentials_missing",)
    assert evidence.lane_readiness_should_abort is True
    assert evidence.step_statuses is not None
    assert evidence.step_statuses["lane_readiness"] == "ok"


def test_legacy_nightly_report_without_lane_readiness_still_loads(tmp_path: Path):
    from fortuna.app.acceptance_bundle import load_nightly_report_evidence

    report_path = tmp_path / "reports" / "nightly" / "legacy.json"
    workflow_path = tmp_path / "reports" / "workflow_snapshot.json"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text("{}", encoding="utf-8")
    _write_nightly_report(
        report_path,
        workflow_path=workflow_path,
        include_promote_step=True,
        promotion_paths=[],
    )
    evidence = load_nightly_report_evidence(nightly_report_path=report_path)
    assert evidence is not None
    assert evidence.lane_readiness_global_blockers == ()
    assert evidence.lane_readiness_should_abort is False
