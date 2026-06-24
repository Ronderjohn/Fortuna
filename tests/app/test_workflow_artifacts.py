from __future__ import annotations

import json
from pathlib import Path

from fortuna.app.workflow_artifacts import (
    artifact_follow_up_prefers_discovery,
    artifact_follow_up_refresh_target,
    artifact_follow_up_team_posture,
    build_workflow_artifact_follow_up,
    build_workflow_artifact_rows,
)


def test_build_workflow_artifact_rows_empty_when_no_workflow():
    assert build_workflow_artifact_rows(None) == ()
    assert build_workflow_artifact_rows({}) == ()
    assert build_workflow_artifact_follow_up(None) is None


def test_build_workflow_artifact_rows_includes_snapshot_acceptance_and_promotion(tmp_path: Path):
    snapshot = tmp_path / "reports" / "dashboard" / "workflow_snapshot.json"
    acceptance_md = tmp_path / "reports" / "dashboard" / "acceptance_bundle.md"
    acceptance_json = tmp_path / "reports" / "dashboard" / "acceptance_bundle.json"
    promotion_md = tmp_path / "reports" / "dashboard" / "promotion_review.md"
    promotion_json = tmp_path / "reports" / "dashboard" / "promotion_review.json"
    for path in (snapshot, acceptance_md, promotion_md):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    acceptance_json.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "nightly_report": {
                    "training_execution_target": "all",
                    "training_execution_model_family": "hybrid",
                    "training_execution_selection_source": "training_research",
                    "promotion_review_model_kind": "mixed",
                    "promotion_review_model_kinds": ["ml_scorer", "rl_policy"],
                },
                "checks": [
                    {"name": "workflow_snapshot", "status": "pass", "detail": "ok"},
                    {
                        "name": "workflow_discovery_alignment",
                        "status": "warn",
                        "detail": "discovery drifting",
                    },
                ],
                "dual_promotion_review": {
                    "both_present": True,
                    "ml_present": True,
                    "rl_present": True,
                    "posture": "expected_dual",
                    "warning": False,
                },
            }
        ),
        encoding="utf-8",
    )
    promotion_json.write_text(
        json.dumps(
            {
                "run_id": "run_123",
                "workflow_discovery_alignment_warning": True,
                "workflow_research_alignment_warning": False,
                "nightly_alignment_warning": True,
            }
        ),
        encoding="utf-8",
    )

    rows = build_workflow_artifact_rows(
        {
            "saved_workflow_snapshot_path": str(snapshot),
            "acceptance_bundle_status": "pass",
            "acceptance_bundle_markdown_path": str(acceptance_md),
            "acceptance_bundle_json_path": str(acceptance_json),
            "promotion_review_kind": "rl",
            "promotion_review_run_id": "run_123",
            "promotion_review_markdown_path": str(promotion_md),
            "promotion_review_json_path": str(promotion_json),
        }
    )

    assert len(rows) == 3
    assert rows[0].kind == "workflow_snapshot"
    assert rows[0].status == "saved"
    assert rows[1].kind == "acceptance_bundle"
    assert rows[1].status == "pass"
    assert rows[1].json_path == str(acceptance_json)
    assert rows[1].detail == (
        "warn: workflow_discovery_alignment "
        "(target=all, family=hybrid, source=training_research, "
        "reviews=ml_scorer, rl_policy, dual=expected_dual)"
    )
    assert rows[2].kind == "promotion_review"
    assert rows[2].label == "RL promotion review"
    assert rows[2].detail == "Run run_123: discovery drift, nightly drift"

    follow_up = build_workflow_artifact_follow_up(
        {
            "acceptance_bundle_json_path": str(acceptance_json),
            "promotion_review_json_path": str(promotion_json),
        }
    )
    assert follow_up is not None
    assert follow_up.action_label == "Refresh market universe"
    assert follow_up.target == "discovery"
    assert artifact_follow_up_prefers_discovery(follow_up) is True
    assert artifact_follow_up_refresh_target(follow_up) is None
    assert (
        artifact_follow_up_team_posture(follow_up)
        == "Review drift currently favors discovery recovery before another research pass."
    )


def test_build_workflow_artifact_rows_includes_replay_linkage_in_acceptance_detail(
    tmp_path: Path,
):
    acceptance_json = tmp_path / "reports" / "dashboard" / "acceptance_bundle.json"
    acceptance_json.parent.mkdir(parents=True, exist_ok=True)
    acceptance_json.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "checks": [{"name": "nightly_replay_linkage", "status": "pass", "detail": "ok"}],
                "replay_linkage": {
                    "overall_status": "linked",
                    "summary": "nightly artifact chain linked",
                    "missing_artifacts": [],
                    "broken_links": [],
                },
            }
        ),
        encoding="utf-8",
    )
    rows = build_workflow_artifact_rows(
        {
            "acceptance_bundle_status": "pass",
            "acceptance_bundle_json_path": str(acceptance_json),
        }
    )
    assert len(rows) == 1
    assert rows[0].kind == "acceptance_bundle"
    assert "replay_linkage=linked" in (rows[0].detail or "")


def test_build_workflow_artifact_follow_up_prefers_replay_verification_when_linkage_is_degraded(
    tmp_path: Path,
):
    acceptance_json = tmp_path / "reports" / "dashboard" / "acceptance_bundle.json"
    acceptance_json.parent.mkdir(parents=True, exist_ok=True)
    acceptance_json.write_text(
        json.dumps(
            {
                "overall_status": "warn",
                "checks": [
                    {
                        "name": "nightly_replay_linkage",
                        "status": "warn",
                        "detail": "optional_missing=promotion_review",
                    }
                ],
                "replay_linkage": {
                    "overall_status": "partial",
                    "run_identifier": "dry_run_emit_replay",
                    "summary": "optional_missing=promotion_review",
                    "missing_artifacts": ["promotion_review"],
                    "broken_links": [],
                },
            }
        ),
        encoding="utf-8",
    )

    rows = build_workflow_artifact_rows(
        {
            "acceptance_bundle_status": "warn",
            "acceptance_bundle_json_path": str(acceptance_json),
        }
    )
    assert "replay_linkage=partial run=dry_run_emit_replay missing=promotion_review" in (
        rows[0].detail or ""
    )

    follow_up = build_workflow_artifact_follow_up(
        {"acceptance_bundle_json_path": str(acceptance_json)}
    )
    assert follow_up is not None
    assert follow_up.action_label == "Verify nightly replay"
    assert follow_up.target == "replay"
    assert artifact_follow_up_team_posture(follow_up) == (
        "Review artifact linkage currently favors replay verification before another nightly cycle."
    )


def test_build_workflow_artifact_follow_up_prefers_research_alignment_signal(tmp_path: Path):
    acceptance_json = tmp_path / "reports" / "dashboard" / "acceptance_bundle.json"
    acceptance_json.parent.mkdir(parents=True, exist_ok=True)
    acceptance_json.write_text(
        json.dumps(
            {
                "recommended_refresh_target": "rl",
                "checks": [
                    {
                        "name": "workflow_basket_research_alignment",
                        "status": "warn",
                        "detail": "basket/research drifting",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    follow_up = build_workflow_artifact_follow_up(
        {"acceptance_bundle_json_path": str(acceptance_json)}
    )

    assert follow_up is not None
    assert follow_up.action_label == "Refresh training research"
    assert follow_up.target == "rl"
    assert artifact_follow_up_prefers_discovery(follow_up) is False
    assert artifact_follow_up_refresh_target(follow_up) == "rl"
    assert artifact_follow_up_team_posture(follow_up) == (
        "Review drift currently favors research recovery target=rl."
    )


def test_build_workflow_artifact_follow_up_reads_explicit_execution_guidance(tmp_path: Path):
    acceptance_json = tmp_path / "reports" / "dashboard" / "acceptance_bundle.json"
    acceptance_json.parent.mkdir(parents=True, exist_ok=True)
    acceptance_json.write_text(
        json.dumps(
            {
                "recommended_refresh_target": "rl",
                "workflow_snapshot": {
                    "team_research_execution_follow_up_target": "rl",
                    "team_research_execution_follow_up_action": (
                        "Review nightly execution path and retarget execution candidate "
                        "selection toward RL-focused before the next promotion or nightly cycle."
                    ),
                },
                "checks": [],
            }
        ),
        encoding="utf-8",
    )

    follow_up = build_workflow_artifact_follow_up(
        {"acceptance_bundle_json_path": str(acceptance_json)}
    )

    assert follow_up is not None
    assert follow_up.action_label == "Review nightly execution path"
    assert follow_up.target == "rl"
    assert artifact_follow_up_prefers_discovery(follow_up) is False
    assert artifact_follow_up_refresh_target(follow_up) == "rl"
    assert artifact_follow_up_team_posture(follow_up) == (
        "Review drift currently favors execution-path recovery target=rl."
    )
