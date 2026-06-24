from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from fortuna.app.nightly_acceptance import (
    build_candidate_nightly_dry_run,
    build_nightly_emit_replay,
    verify_nightly_artifact_linkage,
)
from fortuna.config.settings import Settings


def _acceptance_settings(tmp_path: Path) -> Settings:
    return Settings(
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


def _acceptance_engine(settings: Settings) -> SimpleNamespace:
    return SimpleNamespace(
        settings=settings,
        _agentic_store=None,
        _agentic_learning_store=None,
        rl_generator=None,
        state=SimpleNamespace(symbol="RELIANCE.NS"),
        _agentic_orchestrator=None,
    )


def test_build_candidate_nightly_dry_run_creates_local_evidence_stack(tmp_path: Path):
    settings = _acceptance_settings(tmp_path)
    engine = _acceptance_engine(settings)

    bundle, artifacts = build_candidate_nightly_dry_run(
        settings=settings,
        out_dir=tmp_path / "dry_run",
        engine=engine,
        export_bundle_path=tmp_path / "dry_run" / "acceptance_bundle.md",
    )

    assert bundle.overall_status == "pass"
    assert Path(artifacts.training_candidate_manifest_path).is_file()
    assert Path(artifacts.training_research_plan_path).is_file()
    assert Path(artifacts.workflow_snapshot_path).is_file()
    assert Path(artifacts.nightly_report_path).is_file()
    assert Path(artifacts.nightly_report_markdown_path or "").is_file()
    assert Path(artifacts.promotion_review_path).is_file()
    assert Path(artifacts.live_pointer_path).is_file()
    assert Path(artifacts.acceptance_bundle_path or "").is_file()
    assert artifacts.training_execution_target == "rl"
    assert artifacts.training_execution_model_family == "rl"
    assert artifacts.training_execution_selection_source == "training_research"
    assert bundle.workflow_snapshot is not None
    assert bundle.workflow_snapshot.team_research_discovery_follow_up_target == "rl"
    assert bundle.workflow_snapshot.team_research_research_follow_up_target == "rl"
    assert bundle.workflow_snapshot.team_research_execution_follow_up_target == "rl"

    nightly_payload = json.loads(Path(artifacts.nightly_report_path).read_text(encoding="utf-8"))
    step_names = [row["name"] for row in nightly_payload["steps"]]
    assert "basket_resolution" in step_names
    assert "training_candidates" in step_names
    assert "training_research" in step_names
    assert "training_execution_target" in step_names
    assert "workflow_snapshot" in step_names
    assert "promote_best" in step_names
    assert "promotion_reviews" in step_names
    research_step = next(
        row for row in nightly_payload["steps"] if row["name"] == "training_research"
    )
    assert research_step["detail"]["discovery_follow_up_target"] == "rl"
    assert research_step["detail"]["research_follow_up_target"] == "rl"
    assert research_step["detail"]["execution_follow_up_target"] == "rl"
    execution_detail = next(
        row["detail"]
        for row in nightly_payload["steps"]
        if row["name"] == "training_execution_target"
    )
    assert execution_detail["target"] == "rl"
    assert execution_detail["selection_source"] == "training_research"
    assert execution_detail["discovery_recommended_refresh_target"] == "rl"
    assert "setup_posture=rl" in execution_detail["discovery_summary"]
    research_detail = next(
        row["detail"] for row in nightly_payload["steps"] if row["name"] == "training_research"
    )
    assert research_detail["discovery_recommended_refresh_target"] == "rl"
    assert "setup_posture=rl" in research_detail["discovery_summary"]
    promote_detail = next(
        row["detail"] for row in nightly_payload["steps"] if row["name"] == "promote_best"
    )
    assert promote_detail["model_kind"] == "rl_policy"
    review_detail = next(
        row["detail"] for row in nightly_payload["steps"] if row["name"] == "promotion_reviews"
    )
    assert review_detail["model_kind"] == "rl_policy"
    md_text = Path(artifacts.nightly_report_markdown_path or "").read_text(encoding="utf-8")
    assert "## Training Candidates" in md_text
    assert "## Training Research Plan" in md_text
    assert "## Workflow Snapshot" in md_text
    assert "Allocation research alignment: `both=1, ml=1, rl=1`" in md_text
    assert "setup_posture=rl" in md_text


def test_build_nightly_emit_replay_creates_linked_chain(tmp_path: Path):
    settings = _acceptance_settings(tmp_path)
    engine = _acceptance_engine(settings)

    bundle, artifacts = build_nightly_emit_replay(
        settings=settings,
        out_dir=tmp_path / "emit_replay",
        engine=engine,
        export_bundle_path=tmp_path / "emit_replay" / "acceptance_bundle.md",
    )

    assert artifacts.replay_mode == "emit"
    assert artifacts.linkage is not None
    assert artifacts.linkage.overall_status == "linked"
    assert Path(artifacts.training_candidate_manifest_path).is_file()
    assert Path(artifacts.training_research_plan_path).is_file()
    assert Path(artifacts.workflow_snapshot_path).is_file()
    assert Path(artifacts.nightly_report_path).is_file()
    assert Path(artifacts.promotion_review_path).is_file()
    assert bundle.replay_linkage is not None
    assert bundle.replay_linkage.overall_status == "linked"
    assert any(
        row.name == "nightly_replay_linkage" and row.status == "pass"
        for row in bundle.checks
    )

    nightly_payload = json.loads(Path(artifacts.nightly_report_path).read_text(encoding="utf-8"))
    candidate_step = next(
        row for row in nightly_payload["steps"] if row["name"] == "training_candidates"
    )
    workflow_step = next(
        row for row in nightly_payload["steps"] if row["name"] == "workflow_snapshot"
    )
    assert Path(candidate_step["detail"]["path"]).is_file()
    assert Path(workflow_step["detail"]["path"]).is_file()


def test_verify_nightly_artifact_linkage_flags_missing_workflow(tmp_path: Path):
    settings = _acceptance_settings(tmp_path)
    engine = _acceptance_engine(settings)
    _, artifacts = build_candidate_nightly_dry_run(
        settings=settings,
        out_dir=tmp_path / "dry_run",
        engine=engine,
    )
    workflow_path = Path(artifacts.workflow_snapshot_path)
    workflow_path.unlink()

    linkage = verify_nightly_artifact_linkage(
        report_dir=Path(artifacts.nightly_report_path).parent,
        project_root=tmp_path / "dry_run",
        nightly_report_path=artifacts.nightly_report_path,
    )

    assert linkage.overall_status == "broken"
    assert "workflow_snapshot" in linkage.missing_artifacts


def test_verify_nightly_artifact_linkage_flags_promote_workflow_mismatch(tmp_path: Path):
    settings = _acceptance_settings(tmp_path)
    engine = _acceptance_engine(settings)
    _, artifacts = build_candidate_nightly_dry_run(
        settings=settings,
        out_dir=tmp_path / "dry_run",
        engine=engine,
    )
    report_path = Path(artifacts.nightly_report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for row in payload["steps"]:
        if row["name"] == "promote_best":
            row["detail"]["workflow_snapshot_path"] = "reports/nightly/missing_workflow.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    linkage = verify_nightly_artifact_linkage(
        report_dir=report_path.parent,
        project_root=tmp_path / "dry_run",
        nightly_report_path=report_path,
    )

    assert linkage.overall_status == "broken"
    assert any("mismatch" in item for item in linkage.broken_links)


def test_verify_nightly_artifact_linkage_treats_missing_promotion_review_as_degraded(
    tmp_path: Path,
):
    settings = _acceptance_settings(tmp_path)
    engine = _acceptance_engine(settings)
    _, artifacts = build_candidate_nightly_dry_run(
        settings=settings,
        out_dir=tmp_path / "dry_run",
        engine=engine,
    )
    review_path = Path(artifacts.promotion_review_path)
    review_path.unlink()

    linkage = verify_nightly_artifact_linkage(
        report_dir=Path(artifacts.nightly_report_path).parent,
        project_root=tmp_path / "dry_run",
        nightly_report_path=artifacts.nightly_report_path,
    )

    assert linkage.overall_status == "partial"
    assert linkage.run_identifier is not None
    assert linkage.artifact_presence["promotion_review"] is False
    assert "promotion_review" in linkage.missing_artifacts
    assert "optional_missing=promotion_review" in linkage.summary


def test_build_nightly_emit_replay_returns_typed_failure_when_emit_chain_is_incomplete(
    tmp_path: Path,
    monkeypatch,
):
    settings = _acceptance_settings(tmp_path)
    engine = _acceptance_engine(settings)

    @contextmanager
    def _noop_patches(**_kwargs):
        yield

    monkeypatch.setattr("fortuna.app.nightly_acceptance._replay_emit_patches", _noop_patches)

    nightly_dir = tmp_path / "emit_replay" / "reports" / "nightly"
    candidate_path = nightly_dir / "training_candidates.json"
    research_path = nightly_dir / "training_research_plan.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text("{}", encoding="utf-8")
    research_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "fortuna.app.nightly_acceptance.nightly_basket.emit_training_candidate_manifest",
        lambda *args, **kwargs: {"path": str(candidate_path)},
    )
    monkeypatch.setattr(
        "fortuna.app.nightly_acceptance.nightly_basket.emit_training_research_plan",
        lambda *args, **kwargs: {"path": str(research_path)},
    )
    monkeypatch.setattr(
        "fortuna.app.nightly_acceptance.nightly_basket.emit_workflow_snapshot",
        lambda *args, **kwargs: None,
    )

    bundle, artifacts = build_nightly_emit_replay(
        settings=settings,
        out_dir=tmp_path / "emit_replay",
        engine=engine,
    )

    assert bundle.overall_status == "fail"
    assert artifacts.linkage is not None
    assert artifacts.linkage.overall_status == "broken"
    assert "workflow_snapshot" in artifacts.linkage.missing_artifacts
    assert artifacts.nightly_report_path == ""
