"""Pure helpers for dashboard workflow artifact summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class WorkflowArtifactRow:
    kind: str
    label: str
    status: str
    path: str
    json_path: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        payload = {
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "path": self.path,
            "json_path": self.json_path or "",
            "detail": self.detail or "",
        }
        return payload


@dataclass(frozen=True)
class WorkflowArtifactFollowUp:
    action_label: str
    rationale: str
    target: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        return {
            "action_label": self.action_label,
            "rationale": self.rationale,
            "target": self.target or "",
        }


def artifact_follow_up_refresh_target(
    follow_up: Optional[WorkflowArtifactFollowUp],
) -> Optional[str]:
    if follow_up is None:
        return None
    if follow_up.action_label not in {
        "Refresh training research",
        "Review nightly execution path",
    }:
        return None
    target = str(follow_up.target or "").strip().lower()
    return target if target in {"rl", "ml", "all"} else None


def artifact_follow_up_prefers_discovery(
    follow_up: Optional[WorkflowArtifactFollowUp],
) -> bool:
    return bool(follow_up is not None and follow_up.action_label == "Refresh market universe")


def artifact_follow_up_team_posture(
    follow_up: Optional[WorkflowArtifactFollowUp],
) -> Optional[str]:
    if follow_up is None:
        return None
    target = str(follow_up.target or "").strip()
    if follow_up.action_label == "Refresh market universe":
        return "Review drift currently favors discovery recovery before another research pass."
    if follow_up.action_label == "Refresh training research":
        suffix = f" target={target}" if target else ""
        return f"Review drift currently favors research recovery{suffix}."
    if follow_up.action_label == "Review nightly execution path":
        suffix = f" target={target}" if target else ""
        return f"Review drift currently favors execution-path recovery{suffix}."
    if follow_up.action_label == "Verify nightly replay":
        return (
            "Review artifact linkage currently favors replay verification "
            "before another nightly cycle."
        )
    return None


def build_workflow_artifact_rows(
    workflow: dict[str, Any] | None,
) -> tuple[WorkflowArtifactRow, ...]:
    if not isinstance(workflow, dict):
        return ()
    rows: list[WorkflowArtifactRow] = []

    snapshot_path = _clean_path(workflow.get("saved_workflow_snapshot_path"))
    if snapshot_path:
        rows.append(
            WorkflowArtifactRow(
                kind="workflow_snapshot",
                label="Workflow snapshot",
                status=_file_status(snapshot_path),
                path=snapshot_path,
                detail="Current saved multi-agent workflow snapshot.",
            )
        )

    acceptance_md = _clean_path(workflow.get("acceptance_bundle_markdown_path"))
    acceptance_json = _clean_path(workflow.get("acceptance_bundle_json_path"))
    acceptance_status = str(workflow.get("acceptance_bundle_status") or "").strip() or "unknown"
    if acceptance_md or acceptance_json:
        acceptance_detail = _acceptance_detail(acceptance_json)
        rows.append(
            WorkflowArtifactRow(
                kind="acceptance_bundle",
                label="Acceptance review",
                status=acceptance_status,
                path=acceptance_md or acceptance_json or "",
                json_path=acceptance_json,
                detail=acceptance_detail or "Snapshot-linked acceptance bundle.",
            )
        )

    promotion_md = _clean_path(workflow.get("promotion_review_markdown_path"))
    promotion_json = _clean_path(workflow.get("promotion_review_json_path"))
    promotion_kind = str(workflow.get("promotion_review_kind") or "").strip() or "promotion"
    promotion_run_id = str(workflow.get("promotion_review_run_id") or "").strip()
    if promotion_md or promotion_json:
        promotion_detail = _promotion_detail(
            promotion_json,
            promotion_run_id=promotion_run_id,
        )
        rows.append(
            WorkflowArtifactRow(
                kind="promotion_review",
                label=f"{promotion_kind.upper()} promotion review",
                status="available",
                path=promotion_md or promotion_json or "",
                json_path=promotion_json,
                detail=promotion_detail or "Snapshot-linked promotion review.",
            )
        )

    return tuple(rows)


def build_workflow_artifact_follow_up(
    workflow: dict[str, Any] | None,
) -> Optional[WorkflowArtifactFollowUp]:
    if not isinstance(workflow, dict):
        return None
    acceptance_payload = _load_json(workflow.get("acceptance_bundle_json_path"))
    promotion_payload = _load_json(workflow.get("promotion_review_json_path"))

    acceptance_signal = _acceptance_follow_up_signal(acceptance_payload)
    if acceptance_signal is not None:
        return acceptance_signal

    promotion_signal = _promotion_follow_up_signal(promotion_payload)
    if promotion_signal is not None:
        return promotion_signal

    return None


def _clean_path(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _file_status(path_text: str) -> str:
    try:
        return "saved" if Path(path_text).is_file() else "missing"
    except OSError:
        return "missing"


def _acceptance_detail(path_text: Optional[str]) -> Optional[str]:
    payload = _load_json(path_text)
    if not isinstance(payload, dict):
        return None
    nightly_payload = payload.get("nightly_report")
    nightly_target = ""
    nightly_family = ""
    nightly_source = ""
    nightly_review_kind = ""
    nightly_review_kinds = ""
    if isinstance(nightly_payload, dict):
        nightly_target = str(nightly_payload.get("training_execution_target") or "").strip()
        nightly_family = str(
            nightly_payload.get("training_execution_model_family") or ""
        ).strip()
        nightly_source = str(
            nightly_payload.get("training_execution_selection_source") or ""
        ).strip()
        nightly_review_kind = str(
            nightly_payload.get("promotion_review_model_kind") or ""
        ).strip()
        raw_review_kinds = nightly_payload.get("promotion_review_model_kinds")
        if isinstance(raw_review_kinds, list):
            nightly_review_kinds = ", ".join(
                str(kind or "").strip() for kind in raw_review_kinds if str(kind or "").strip()
            )
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return None
    warn_names: list[str] = []
    fail_names: list[str] = []
    for row in checks:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().lower()
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        if status == "warn":
            warn_names.append(name)
        elif status == "fail":
            fail_names.append(name)
    if fail_names:
        detail = "fail: " + ", ".join(fail_names[:2])
        return _append_nightly_posture(
            detail,
            target=nightly_target,
            family=nightly_family,
            source=nightly_source,
            review_kind=nightly_review_kind,
            review_kinds=nightly_review_kinds,
            dual_posture=_dual_posture_from_payload(payload),
            replay_linkage=_replay_linkage_from_payload(payload),
        )
    if warn_names:
        detail = "warn: " + ", ".join(warn_names[:2])
        return _append_nightly_posture(
            detail,
            target=nightly_target,
            family=nightly_family,
            source=nightly_source,
            review_kind=nightly_review_kind,
            review_kinds=nightly_review_kinds,
            dual_posture=_dual_posture_from_payload(payload),
            replay_linkage=_replay_linkage_from_payload(payload),
        )
    overall = str(payload.get("overall_status") or "").strip()
    if not overall:
        return None
    return _append_nightly_posture(
        f"{overall} with no active warnings",
        target=nightly_target,
        family=nightly_family,
        source=nightly_source,
        review_kind=nightly_review_kind,
        review_kinds=nightly_review_kinds,
        dual_posture=_dual_posture_from_payload(payload),
        replay_linkage=_replay_linkage_from_payload(payload),
    )


def _replay_linkage_from_payload(payload: dict) -> str:
    linkage_payload = payload.get("replay_linkage")
    if not isinstance(linkage_payload, dict):
        return ""
    status = str(linkage_payload.get("overall_status") or "").strip()
    if not status or status == "unknown":
        return ""
    run_id = str(linkage_payload.get("run_identifier") or "").strip()
    missing = linkage_payload.get("missing_artifacts")
    broken = linkage_payload.get("broken_links")
    missing_text = ""
    if isinstance(missing, list) and missing:
        missing_text = f" missing={','.join(str(item) for item in missing[:3])}"
    broken_text = ""
    if isinstance(broken, list) and broken:
        broken_text = f" broken={','.join(str(item) for item in broken[:2])}"
    run_text = f" run={run_id}" if run_id else ""
    return f"replay_linkage={status}{run_text}{missing_text}{broken_text}"


def _dual_posture_from_payload(payload: dict) -> str:
    dual_payload = payload.get("dual_promotion_review")
    if not isinstance(dual_payload, dict):
        return ""
    posture = str(dual_payload.get("posture") or "").strip()
    if not posture or posture == "unknown":
        return ""
    warning = bool(dual_payload.get("warning"))
    suffix = " (warn)" if warning else ""
    return f"dual={posture}{suffix}"


def _append_nightly_posture(
    detail: str,
    *,
    target: str,
    family: str,
    source: str,
    review_kind: str,
    review_kinds: str,
    dual_posture: str = "",
    replay_linkage: str = "",
) -> str:
    parts: list[str] = []
    if target:
        parts.append(f"target={target}")
    if family:
        parts.append(f"family={family}")
    if source:
        parts.append(f"source={source}")
    if review_kinds:
        parts.append(f"reviews={review_kinds}")
    elif review_kind:
        parts.append(f"review={review_kind}")
    if dual_posture:
        parts.append(dual_posture)
    if replay_linkage:
        parts.append(replay_linkage)
    if not parts:
        return detail
    return f"{detail} ({', '.join(parts)})"


def _promotion_detail(
    path_text: Optional[str],
    *,
    promotion_run_id: str,
) -> Optional[str]:
    payload = _load_json(path_text)
    if not isinstance(payload, dict):
        return f"Run {promotion_run_id}" if promotion_run_id else None
    warn_parts: list[str] = []
    if bool(payload.get("workflow_research_alignment_warning")):
        warn_parts.append("basket/research drift")
    if bool(payload.get("workflow_discovery_alignment_warning")):
        warn_parts.append("discovery drift")
    if bool(payload.get("nightly_alignment_warning")):
        warn_parts.append("nightly drift")
    run_id = str(payload.get("run_id") or promotion_run_id or "").strip()
    if warn_parts:
        prefix = f"Run {run_id}: " if run_id else ""
        return prefix + ", ".join(warn_parts)
    return f"Run {run_id}" if run_id else "Snapshot-linked promotion review."


def _load_json(path_text: Optional[str]) -> Optional[dict[str, Any]]:
    path = _clean_path(path_text)
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _acceptance_follow_up_signal(
    payload: Optional[dict[str, Any]],
) -> Optional[WorkflowArtifactFollowUp]:
    if not isinstance(payload, dict):
        return None
    workflow = payload.get("workflow_snapshot")
    if isinstance(workflow, dict):
        discovery_action = str(
            workflow.get("team_research_discovery_follow_up_action", "") or ""
        ).strip()
        if discovery_action:
            return WorkflowArtifactFollowUp(
                action_label="Refresh market universe",
                rationale=discovery_action,
                target=(
                    str(workflow.get("team_research_discovery_follow_up_target", "") or "").strip()
                    or "discovery"
                ),
            )
        research_action = str(
            workflow.get("team_research_research_follow_up_action", "") or ""
        ).strip()
        if research_action:
            return WorkflowArtifactFollowUp(
                action_label="Refresh training research",
                rationale=research_action,
                target=(
                    str(workflow.get("team_research_research_follow_up_target", "") or "").strip()
                    or (str(payload.get("recommended_refresh_target") or "").strip() or None)
                ),
            )
        execution_action = str(
            workflow.get("team_research_execution_follow_up_action", "") or ""
        ).strip()
        if execution_action:
            return WorkflowArtifactFollowUp(
                action_label="Review nightly execution path",
                rationale=execution_action,
                target=(
                    str(workflow.get("team_research_execution_follow_up_target", "") or "").strip()
                    or (str(payload.get("recommended_refresh_target") or "").strip() or None)
                ),
            )
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return None
    replay_payload = payload.get("replay_linkage")
    if isinstance(replay_payload, dict):
        replay_status = str(replay_payload.get("overall_status") or "").strip().lower()
        if replay_status in {"partial", "broken"}:
            summary = (
                str(replay_payload.get("summary") or "").strip()
                or "Replay linkage needs verification."
            )
            return WorkflowArtifactFollowUp(
                action_label="Verify nightly replay",
                rationale=summary,
                target="replay",
            )
    warn_names = {
        str(row.get("name") or "").strip(): str(row.get("status") or "").strip().lower()
        for row in checks
        if isinstance(row, dict)
    }
    if warn_names.get("workflow_discovery_alignment") == "warn":
        return WorkflowArtifactFollowUp(
            action_label="Refresh market universe",
            rationale="Latest acceptance review is warning on workflow discovery alignment.",
            target="discovery",
        )
    if warn_names.get("workflow_discovery_refresh") == "warn":
        return WorkflowArtifactFollowUp(
            action_label="Refresh market universe",
            rationale=(
                "Latest acceptance review says discovery refresh was requested "
                "but not completed."
            ),
            target="discovery",
        )
    research_warns = {
        "workflow_basket_research_alignment",
        "training_research_refresh",
        "recent_nightly_alignment",
        "recent_refreshed_nightly_alignment",
    }
    if any(warn_names.get(name) == "warn" for name in research_warns):
        return WorkflowArtifactFollowUp(
            action_label="Refresh training research",
            rationale=(
                "Latest acceptance review is warning on basket/research "
                "or nightly alignment."
            ),
            target=str(payload.get("recommended_refresh_target") or "").strip() or None,
        )
    return None


def _promotion_follow_up_signal(
    payload: Optional[dict[str, Any]],
) -> Optional[WorkflowArtifactFollowUp]:
    if not isinstance(payload, dict):
        return None
    if bool(payload.get("workflow_discovery_alignment_warning")):
        return WorkflowArtifactFollowUp(
            action_label="Refresh market universe",
            rationale="Latest promotion review is warning on workflow discovery alignment.",
            target=(
                str(payload.get("workflow_discovery_alignment_recommended_target") or "").strip()
                or "discovery"
            ),
        )
    if bool(payload.get("workflow_research_alignment_warning")) or bool(
        payload.get("nightly_alignment_warning")
    ):
        return WorkflowArtifactFollowUp(
            action_label="Refresh training research",
            rationale=(
                "Latest promotion review is warning on workflow research "
                "or nightly alignment."
            ),
            target=str(payload.get("nightly_alignment_recommended_target") or "").strip() or None,
        )
    return None


__all__ = [
    "artifact_follow_up_prefers_discovery",
    "artifact_follow_up_refresh_target",
    "artifact_follow_up_team_posture",
    "WorkflowArtifactFollowUp",
    "WorkflowArtifactRow",
    "build_workflow_artifact_follow_up",
    "build_workflow_artifact_rows",
]
