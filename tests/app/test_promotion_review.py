from __future__ import annotations

import json
from pathlib import Path

from fortuna.app.model_activation import build_lane_activation_summary
from fortuna.app.promotion_review import (
    build_and_export_promotion_review,
    build_promotion_review,
    export_promotion_review,
    format_promotion_review,
)
from fortuna.config.settings import Settings
from fortuna.models import PromotionRecord, PromotionStatus, promote
from fortuna.models.metadata import ModelKind
from fortuna.rl.training.checkpoint import OOSMetricsSummary, PolicyCheckpoint


def _write_rl_artifact(root: Path, run_id: str = "run_review") -> Path:
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
        oos_metrics=OOSMetricsSummary(total_trades=11, sharpe_ratio=1.4, profit_factor=1.2),
        verdict_passed=True,
        verdict_score=0.85,
        verdict_reasons=["strong_oos"],
        advisory_ready=True,
    ).write(cp_dir / "metadata.json")
    return cp_dir


def _write_workflow_snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "reports" / "workflow_snapshot.json"
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
                        "discovery_alignment_summary": "discovery partial 2/3",
                        "discovery_overlap_symbols": "RELIANCE.NS, TCS.NS",
                        "discovery_alignment_overlap_count": 2,
                        "discovery_alignment_compare_count": 3,
                        "research_headline": "4 ML/RL research rows prepared",
                        "research_selection_policy": "diversified",
                        "research_refresh_target": "rl",
                        "research_refresh_requested": True,
                        "research_refreshed_count": 2,
                        "research_ml_count": 3,
                        "research_rl_count": 2,
                    }
                },
                "universe": {
                    "metrics": {"count": 15, "adaptive_count": 1},
                    "notes": ["adaptive_penalty=-0.12 avg_pnl=-1.40% rows=3"],
                },
                "shortlist": {"metrics": {"count": 6}},
                "briefing": {"metrics": {"candidates": 4}},
                "allocation": {
                    "metrics": {"selected_count": 3, "skipped_count": 1, "max_positions": 3},
                    "notes": [
                        "portfolio_research_alignment=enabled",
                        "allocation_research_targets: both=1, rl=1",
                        "allocation_refreshed_research_targets: both=1",
                    ],
                },
                "candidates": {"metrics": {"ml_count": 5, "rl_count": 3}},
                "research": {
                    "metrics": {
                        "count": 4,
                        "ml_count": 3,
                        "rl_count": 2,
                        "refresh_target": "rl",
                        "refresh_requested": True,
                        "refreshed_count": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_promotion_review_reads_pointer_and_workflow(tmp_path: Path):
    art = _write_rl_artifact(tmp_path)
    workflow = _write_workflow_snapshot(tmp_path)
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
                            "allocation_refreshed_research_target_mix": "both=1",
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
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        acceptance_bundle_nightly_alignment_min_enabled_reports=2,
        acceptance_bundle_nightly_alignment_warn_ratio=0.75,
    )
    cp = PolicyCheckpoint.read(art / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, art, status=PromotionStatus.VALIDATED)
    record.workflow_snapshot_path = str(workflow)
    promote(record, models_root=tmp_path, promoted_by="test")

    review = build_promotion_review(
        kind=ModelKind.RL_POLICY,
        models_root=tmp_path,
        symbol="RELIANCE.NS",
        settings=settings,
    )

    assert review is not None
    assert review.record.run_id == "run_review"
    assert review.record.metrics["sharpe_ratio"] == 1.4
    assert review.workflow_summary_text is not None
    assert "universe=15" in review.workflow_summary_text
    assert "allocation=3/3" in review.workflow_summary_text
    assert "adaptive=adaptive_penalty=-0.12 avg_pnl=-1.40% rows=3" in review.workflow_summary_text
    assert "training_research=4(3/2) refreshed=2" in review.workflow_summary_text
    assert (
        review.nightly_alignment_summary_text
        == "ratio=0.50 (1/2) latest=both=1, rl=1 refreshed=both=1 target=all refreshed_ratio=0.50"
    )
    assert review.nightly_alignment_warning is True
    assert review.nightly_alignment_recommended_target == "all"
    assert review.nightly_alignment_recommended_force_refresh is False
    assert review.nightly_alignment_recommended_command is not None
    assert review.nightly_alignment_recommended_discovery_action is not None
    assert "Rebuild the market universe" in review.nightly_alignment_recommended_discovery_action
    assert review.nightly_alignment_recommended_discovery_command is not None
    assert "build_market_universe.py" in review.nightly_alignment_recommended_discovery_command
    text = format_promotion_review(review)
    assert "promotion review: rl_policy" in text
    assert "reasons: strong_oos" in text
    assert "training research: count=4 ml=3 rl=2" in text
    assert "training research refresh: target=rl refreshed=2" in text
    assert (
        "team research posture: 4 ML/RL research rows prepared "
        "ml=3 rl=2 policy=diversified target=rl refreshed=2"
    ) in text
    assert "team posture: roles=5/6 nightly=1/2 target=all force_refresh=1" in text
    assert "team recent trend: 1/2 over 2 run(s) latest=ok basket=1" in text
    assert "team discovery posture: discovery partial 2/3 overlap=RELIANCE.NS, TCS.NS" in text
    assert (
        "workflow basket/research alignment: ratio=1.00 (3/3) "
        "basket/research aligned 3/3 mix=both=1, rl=2"
    ) in text
    assert (
        "warning: ratio=0.50 (1/2) latest=both=1, rl=1 "
        "refreshed=both=1 target=all refreshed_ratio=0.50"
    ) in text
    assert "recommended discovery follow-up:" in text
    assert "recommended discovery command:" in text
    assert "recommended remediation: target=all force_refresh=0" in text
    assert "recommended command: uv run python scripts/build_training_research_plan.py" in text

    md_path = export_promotion_review(review, tmp_path / "promotion_review.md", fmt="md")
    json_path = export_promotion_review(review, tmp_path / "promotion_review.json", fmt="json")
    md_text = md_path.read_text(encoding="utf-8")
    assert "# Promotion Review" in md_text
    assert "- Cross-artifact alignment: `status=" in md_text
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["workflow_snapshot"]["universe_count"] == 15
    assert payload["workflow_snapshot"]["allocation_selected"] == 3
    assert payload["workflow_snapshot"]["research_refreshed_count"] == 2
    assert payload["workflow_snapshot"]["allocation_refreshed_research_target_mix"] == "both=1"
    assert payload["workflow_snapshot"]["team_role_count"] == 6
    assert payload["workflow_snapshot"]["team_nightly_recommended_target"] == "all"
    assert payload["workflow_snapshot"]["team_nightly_recommended_force_refresh"] is True
    assert payload["workflow_snapshot"]["team_nightly_recent_window"] == 2
    assert payload["workflow_snapshot"]["team_nightly_recent_aligned"] == 1
    assert (
        payload["workflow_snapshot"]["team_research_headline"]
        == "4 ML/RL research rows prepared"
    )
    assert payload["workflow_snapshot"]["team_research_selection_policy"] == "diversified"
    assert payload["workflow_snapshot"]["team_research_refresh_target"] == "rl"
    assert payload["workflow_snapshot"]["team_research_refresh_requested"] is True
    assert payload["workflow_snapshot"]["team_research_refreshed_count"] == 2
    assert payload["workflow_research_alignment_summary_text"] is not None
    assert payload["workflow_research_alignment_warning"] is False
    assert payload["nightly_alignment_warning"] is True
    assert payload["nightly_alignment_recommended_target"] == "all"
    assert payload["nightly_alignment_recommended_force_refresh"] is False
    assert payload["nightly_alignment_recommended_command"] is not None
    assert payload["nightly_alignment_recommended_discovery_action"] is not None
    assert payload["nightly_alignment_recommended_discovery_command"] is not None


def test_build_promotion_review_warns_on_weak_basket_research_alignment(tmp_path: Path):
    art = _write_rl_artifact(tmp_path, run_id="run_review_warn")
    workflow = _write_workflow_snapshot(tmp_path)
    payload = json.loads(workflow.read_text(encoding="utf-8"))
    payload["team"]["metrics"]["research_alignment_summary"] = "basket/research drifting 1/3"
    payload["team"]["metrics"]["research_alignment_target_mix"] = "none=2, rl=1"
    payload["team"]["metrics"]["research_alignment_overlap_count"] = 1
    payload["team"]["metrics"]["research_alignment_selected_count"] = 3
    workflow.write_text(json.dumps(payload), encoding="utf-8")
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        acceptance_bundle_team_research_alignment_warn_ratio=0.75,
    )
    cp = PolicyCheckpoint.read(art / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, art, status=PromotionStatus.VALIDATED)
    record.workflow_snapshot_path = str(workflow)
    promote(record, models_root=tmp_path, promoted_by="test")

    review = build_promotion_review(
        kind=ModelKind.RL_POLICY,
        models_root=tmp_path,
        symbol="RELIANCE.NS",
        settings=settings,
    )

    assert review is not None
    assert review.workflow_research_alignment_warning is True
    assert "ratio=0.33 (1/3)" in (review.workflow_research_alignment_summary_text or "")
    assert review.workflow_research_alignment_recommended_target == "all"
    text = format_promotion_review(review)
    assert (
        "warning: ratio=0.33 (1/3) basket/research drifting 1/3 mix=none=2, rl=1 "
        "recommended_target=all"
    ) in text


def test_build_promotion_review_warns_on_weak_discovery_alignment(tmp_path: Path):
    art = _write_rl_artifact(tmp_path, run_id="run_review_discovery_warn")
    workflow = _write_workflow_snapshot(tmp_path)
    payload = json.loads(workflow.read_text(encoding="utf-8"))
    payload["team"]["metrics"]["discovery_alignment_summary"] = "discovery drifting 1/3"
    payload["team"]["metrics"]["discovery_overlap_symbols"] = "RELIANCE.NS"
    payload["team"]["metrics"]["discovery_alignment_overlap_count"] = 1
    payload["team"]["metrics"]["discovery_alignment_compare_count"] = 3
    workflow.write_text(json.dumps(payload), encoding="utf-8")
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        acceptance_bundle_team_discovery_alignment_warn_ratio=0.75,
    )
    cp = PolicyCheckpoint.read(art / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, art, status=PromotionStatus.VALIDATED)
    record.workflow_snapshot_path = str(workflow)
    promote(record, models_root=tmp_path, promoted_by="test")

    review = build_promotion_review(
        kind=ModelKind.RL_POLICY,
        models_root=tmp_path,
        symbol="RELIANCE.NS",
        settings=settings,
    )

    assert review is not None
    assert review.workflow_discovery_alignment_warning is True
    assert "ratio=0.33 (1/3)" in (review.workflow_discovery_alignment_summary_text or "")
    assert review.workflow_discovery_alignment_recommended_target == "all"
    text = format_promotion_review(review)
    assert (
        "warning: ratio=0.33 (1/3) discovery drifting 1/3 overlap=RELIANCE.NS "
        "recommended_target=all"
    ) in text


def test_build_and_export_promotion_review_overrides_workflow_snapshot(tmp_path: Path):
    art = _write_rl_artifact(tmp_path, run_id="run_review_override")
    workflow = _write_workflow_snapshot(tmp_path)
    payload = json.loads(workflow.read_text(encoding="utf-8"))
    payload["team"]["metrics"]["discovery_refresh_source"] = "screener"
    payload["team"]["metrics"]["discovery_refresh_timeframe"] = "1d"
    payload["team"]["metrics"]["discovery_refresh_days"] = 30
    payload["team"]["metrics"]["discovery_refresh_requested"] = True
    payload["team"]["metrics"]["discovery_refreshed_count"] = 15
    workflow.write_text(json.dumps(payload), encoding="utf-8")

    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )
    cp = PolicyCheckpoint.read(art / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, art, status=PromotionStatus.VALIDATED)
    record.workflow_snapshot_path = str(tmp_path / "reports" / "other_snapshot.json")
    promote(record, models_root=tmp_path, promoted_by="test")

    built = build_and_export_promotion_review(
        kind=ModelKind.RL_POLICY,
        models_root=tmp_path,
        symbol="RELIANCE.NS",
        workflow_snapshot_path=workflow,
        out_dir=tmp_path / "reports" / "dashboard",
        basename="promotion_review_dashboard",
        settings=settings,
    )

    assert built is not None
    review, md_path, json_path = built
    assert review.record.workflow_snapshot_path == str(workflow)
    assert review.workflow_snapshot is not None
    assert review.workflow_snapshot.team_discovery_refresh_requested is True
    assert review.workflow_snapshot.team_discovery_refreshed_count == 15
    assert md_path.name == "promotion_review_dashboard.md"
    assert json_path.name == "promotion_review_dashboard.json"
    exported = json.loads(json_path.read_text(encoding="utf-8"))
    assert exported["workflow_snapshot"]["team_discovery_refresh_source"] == "screener"
    assert exported["workflow_snapshot"]["team_discovery_refreshed_count"] == 15
    assert exported["workflow_discovery_alignment_summary_text"] is not None
    assert exported["cross_artifact_alignment"] is not None


def test_build_promotion_review_includes_activation(tmp_path: Path):
    art = _write_rl_artifact(tmp_path)
    cp = PolicyCheckpoint.read(art / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, art, status=PromotionStatus.VALIDATED)
    promote(record, models_root=tmp_path, promoted_by="test")

    review = build_promotion_review(
        kind=ModelKind.RL_POLICY,
        models_root=tmp_path,
        symbol="RELIANCE.NS",
    )
    assert review is not None
    assert review.activation is not None
    assert review.activation.lane == "rl"
    assert review.activation.stage in {"active", "not_loaded", "promoted_not_advisory_ready"}


def test_format_promotion_review_renders_activation_block(tmp_path: Path):
    art = _write_rl_artifact(tmp_path)
    cp = PolicyCheckpoint.read(art / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, art, status=PromotionStatus.VALIDATED)
    promote(record, models_root=tmp_path, promoted_by="test")

    review = build_promotion_review(
        kind=ModelKind.RL_POLICY,
        models_root=tmp_path,
        symbol="RELIANCE.NS",
    )
    text = format_promotion_review(review)
    assert "activation: lane=rl" in text
    assert "stage=" in text


def test_build_lane_activation_without_promotion_audit(tmp_path: Path):
    _write_rl_artifact(tmp_path, run_id="run_only")
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
        models_root=tmp_path,
    )
    assert activation.stage == "unpromoted"
    assert "run_only" in (activation.recommended_command or "")
