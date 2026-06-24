"""Helpers for reviewing promoted model artifacts and their workflow context."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from fortuna.agentic.contracts import (
    CrossArtifactAlignmentSummary,
    DualPromotionReviewSummary,
    LaneActivationSummary,
    WorkflowSnapshotSummary,
)
from fortuna.app.acceptance_alignment import (
    build_cross_artifact_alignment,
    build_dual_promotion_review_summary,
)
from fortuna.app.model_activation import build_lane_activation_summary, format_lane_activation
from fortuna.app.model_status import build_nightly_alignment_status
from fortuna.app.workflow_snapshot import (
    format_workflow_snapshot_summary,
    load_workflow_snapshot_summary,
)
from fortuna.config.settings import get_settings
from fortuna.models.metadata import ModelKind, PromotionRecord
from fortuna.models.registry import live_pointer_path, read_audit
from fortuna.observability.recorder import workflow_boundary


@dataclass(frozen=True)
class PromotionReview:
    record: PromotionRecord
    pointer_path: str
    audit_ts: Optional[str] = None
    workflow_snapshot: Optional[WorkflowSnapshotSummary] = None
    workflow_summary_text: Optional[str] = None
    nightly_alignment_summary_text: Optional[str] = None
    nightly_alignment_warning: bool = False
    nightly_alignment_recommended_target: Optional[str] = None
    nightly_alignment_recommended_force_refresh: bool = False
    nightly_alignment_recommended_command: Optional[str] = None
    nightly_alignment_recommended_discovery_action: Optional[str] = None
    nightly_alignment_recommended_discovery_command: Optional[str] = None
    workflow_research_alignment_summary_text: Optional[str] = None
    workflow_research_alignment_warning: bool = False
    workflow_research_alignment_recommended_target: Optional[str] = None
    workflow_discovery_alignment_summary_text: Optional[str] = None
    workflow_discovery_alignment_warning: bool = False
    workflow_discovery_alignment_recommended_target: Optional[str] = None
    cross_artifact_alignment: Optional[CrossArtifactAlignmentSummary] = None
    dual_promotion_review: Optional[DualPromotionReviewSummary] = None
    activation: Optional[LaneActivationSummary] = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.record.to_dict()
        payload["pointer_path"] = self.pointer_path
        payload["audit_ts"] = self.audit_ts
        payload["workflow_snapshot"] = (
            self.workflow_snapshot.to_dict() if self.workflow_snapshot is not None else None
        )
        payload["workflow_summary_text"] = self.workflow_summary_text
        payload["nightly_alignment_summary_text"] = self.nightly_alignment_summary_text
        payload["nightly_alignment_warning"] = self.nightly_alignment_warning
        payload["nightly_alignment_recommended_target"] = (
            self.nightly_alignment_recommended_target
        )
        payload["nightly_alignment_recommended_force_refresh"] = (
            self.nightly_alignment_recommended_force_refresh
        )
        payload["nightly_alignment_recommended_command"] = (
            self.nightly_alignment_recommended_command
        )
        payload["nightly_alignment_recommended_discovery_action"] = (
            self.nightly_alignment_recommended_discovery_action
        )
        payload["nightly_alignment_recommended_discovery_command"] = (
            self.nightly_alignment_recommended_discovery_command
        )
        payload["workflow_research_alignment_summary_text"] = (
            self.workflow_research_alignment_summary_text
        )
        payload["workflow_research_alignment_warning"] = self.workflow_research_alignment_warning
        payload["workflow_research_alignment_recommended_target"] = (
            self.workflow_research_alignment_recommended_target
        )
        payload["workflow_discovery_alignment_summary_text"] = (
            self.workflow_discovery_alignment_summary_text
        )
        payload["workflow_discovery_alignment_warning"] = self.workflow_discovery_alignment_warning
        payload["workflow_discovery_alignment_recommended_target"] = (
            self.workflow_discovery_alignment_recommended_target
        )
        payload["cross_artifact_alignment"] = (
            self.cross_artifact_alignment.to_dict()
            if self.cross_artifact_alignment is not None
            else None
        )
        payload["dual_promotion_review"] = (
            self.dual_promotion_review.to_dict()
            if self.dual_promotion_review is not None
            else None
        )
        payload["activation"] = (
            self.activation.to_dict() if self.activation is not None else None
        )
        return payload


def _dual_promotion_for_review(
    *,
    review_kind: ModelKind,
    workflow_summary: Optional[WorkflowSnapshotSummary],
    _workflow_research_target: Optional[str],
    nightly_alignment,
    effective_refresh_target: Optional[str],
) -> DualPromotionReviewSummary:
    extra_kinds: tuple[str, ...] = ()
    execution_target_override = None
    execution_family_override = None
    if nightly_alignment is not None:
        extra_kinds = tuple(
            getattr(nightly_alignment, "latest_promotion_review_model_kinds", ()) or ()
        )
        execution_target_override = getattr(
            nightly_alignment, "latest_execution_target", None
        )
        execution_family_override = getattr(
            nightly_alignment, "latest_execution_model_family", None
        )
    if not extra_kinds:
        extra_kinds = (review_kind.value,)
    elif review_kind.value not in extra_kinds:
        extra_kinds = (*extra_kinds, review_kind.value)

    return build_dual_promotion_review_summary(
        nightly_report=None,
        workflow_summary=workflow_summary,
        rl_review=None,
        ml_review=None,
        effective_refresh_target=effective_refresh_target,
        extra_review_kinds=extra_kinds,
        execution_target_override=execution_target_override,
        execution_family_override=execution_family_override,
    )


def build_promotion_review(
    *,
    kind: ModelKind,
    models_root: Path | str = "models",
    ml_base: Optional[Path | str] = None,
    symbol: Optional[str] = None,
    run_id: Optional[str] = None,
    settings=None,
) -> Optional[PromotionReview]:
    runtime_settings = settings
    if runtime_settings is None:
        try:
            runtime_settings = get_settings()
        except Exception:
            runtime_settings = None
    if runtime_settings is None:
        return _build_promotion_review_impl(
            kind=kind,
            models_root=models_root,
            ml_base=ml_base,
            symbol=symbol,
            run_id=run_id,
            settings=settings,
        )
    with workflow_boundary(
        runtime_settings,
        event_name="promotion_review",
        module="fortuna.app.promotion_review",
        workflow_id="promotion_review",
        symbol=(symbol or "").upper().strip() or None,
        context={"kind": kind.value},
    ) as span:
        review = _build_promotion_review_impl(
            kind=kind,
            models_root=models_root,
            ml_base=ml_base,
            symbol=symbol,
            run_id=run_id,
            settings=settings,
        )
        if review is None:
            span.set_status("warn")
        elif review.activation is not None:
            span.set_context(
                stage=review.activation.stage,
                blocker_count=len(review.activation.blockers),
            )
        return review


def _build_promotion_review_impl(
    *,
    kind: ModelKind,
    models_root: Path | str = "models",
    ml_base: Optional[Path | str] = None,
    symbol: Optional[str] = None,
    run_id: Optional[str] = None,
    settings=None,
) -> Optional[PromotionReview]:
    event = _find_promotion_event(
        kind=kind,
        models_root=models_root,
        symbol=symbol,
        run_id=run_id,
    )
    if event is None:
        return None

    pointer_path = _pointer_path_for_event(
        kind=kind,
        models_root=models_root,
        ml_base=ml_base,
        symbol=event.get("symbol") or symbol,
        explicit=event.get("pointer"),
    )
    record = _record_from_pointer(pointer_path)
    if record is None:
        return None

    workflow_summary = load_workflow_snapshot_summary(record.workflow_snapshot_path)
    workflow_text = (
        format_workflow_snapshot_summary(workflow_summary)
        if workflow_summary is not None
        else None
    )
    runtime_settings = settings
    if runtime_settings is None:
        try:
            runtime_settings = get_settings()
        except Exception:
            runtime_settings = None
    nightly_alignment_text = None
    nightly_alignment_warning = False
    nightly_alignment_target = None
    nightly_alignment_force_refresh = False
    nightly_alignment_command = None
    nightly_alignment_discovery_action = None
    nightly_alignment_discovery_command = None
    workflow_research_alignment_text = None
    workflow_research_alignment_warning = False
    workflow_research_alignment_target = None
    workflow_discovery_alignment_text = None
    workflow_discovery_alignment_warning = False
    workflow_discovery_alignment_target = None
    workflow_discovery_alignment_action = None
    workflow_discovery_alignment_command = None
    nightly_alignment_obj = None
    if runtime_settings is not None:
        nightly_alignment = build_nightly_alignment_status(runtime_settings)
        nightly_alignment_obj = nightly_alignment
        nightly_alignment_text, nightly_alignment_warning = _nightly_alignment_review_text(
            nightly_alignment,
            settings=runtime_settings,
        )
        nightly_alignment_target = nightly_alignment.recommended_refresh_target
        nightly_alignment_force_refresh = nightly_alignment.recommended_force_refresh
        nightly_alignment_command = nightly_alignment.recommended_cli_command
        nightly_alignment_discovery_action = nightly_alignment.recommended_discovery_action
        nightly_alignment_discovery_command = nightly_alignment.recommended_discovery_cli_command
        workflow_research_alignment_text, workflow_research_alignment_warning = (
            _workflow_research_alignment_review_text(
                workflow_summary,
                settings=runtime_settings,
            )
        )
        workflow_research_alignment_target = _workflow_research_alignment_target(
            workflow_summary
        )
        workflow_discovery_alignment_text, workflow_discovery_alignment_warning = (
            _workflow_discovery_alignment_review_text(
                workflow_summary,
                settings=runtime_settings,
            )
        )
        workflow_discovery_alignment_target = _workflow_discovery_alignment_target(
            workflow_summary
        )
        workflow_discovery_alignment_action = _workflow_discovery_recommendation_action(
            workflow_summary
        )
        workflow_discovery_alignment_command = _workflow_discovery_recommendation_command(
            workflow_summary=workflow_summary,
            settings=runtime_settings,
        )
    cross_artifact_alignment = None
    if runtime_settings is not None and nightly_alignment_obj is not None:
        cross_artifact_alignment = build_cross_artifact_alignment(
            nightly_report=None,
            workflow_summary=workflow_summary,
            rl_review=None,
            ml_review=None,
            model_health=None,
            effective_refresh_target=getattr(
                nightly_alignment_obj, "effective_refresh_target", None
            ),
            nightly_alignment=nightly_alignment_obj,
        )
    lane = "ml" if kind == ModelKind.ML_SCORER else "rl"
    activation = None
    if runtime_settings is not None:
        activation = build_lane_activation_summary(
            runtime_settings,
            lane=lane,
            symbol=record.symbol or symbol,
            models_root=models_root,
            ml_base=ml_base,
        )
    return PromotionReview(
        record=record,
        pointer_path=str(pointer_path),
        audit_ts=event.get("ts"),
        workflow_snapshot=workflow_summary,
        workflow_summary_text=workflow_text,
        nightly_alignment_summary_text=nightly_alignment_text,
        nightly_alignment_warning=nightly_alignment_warning,
        nightly_alignment_recommended_target=nightly_alignment_target,
        nightly_alignment_recommended_force_refresh=nightly_alignment_force_refresh,
        nightly_alignment_recommended_command=nightly_alignment_command,
        nightly_alignment_recommended_discovery_action=(
            nightly_alignment_discovery_action or workflow_discovery_alignment_action
        ),
        nightly_alignment_recommended_discovery_command=(
            nightly_alignment_discovery_command or workflow_discovery_alignment_command
        ),
        workflow_research_alignment_summary_text=workflow_research_alignment_text,
        workflow_research_alignment_warning=workflow_research_alignment_warning,
        workflow_research_alignment_recommended_target=workflow_research_alignment_target,
        workflow_discovery_alignment_summary_text=workflow_discovery_alignment_text,
        workflow_discovery_alignment_warning=workflow_discovery_alignment_warning,
        workflow_discovery_alignment_recommended_target=workflow_discovery_alignment_target,
        cross_artifact_alignment=cross_artifact_alignment,
        dual_promotion_review=_dual_promotion_for_review(
            review_kind=kind,
            workflow_summary=workflow_summary,
            _workflow_research_target=workflow_research_alignment_target,
            nightly_alignment=nightly_alignment_obj,
            effective_refresh_target=getattr(
                nightly_alignment_obj, "effective_refresh_target", None
            )
            if nightly_alignment_obj is not None
            else None,
        ),
        activation=activation,
    )


def format_promotion_review(review: PromotionReview) -> str:
    record = review.record
    lines: list[str] = []
    if review.activation is not None:
        lines.extend(format_lane_activation(review.activation))
        lines.append("")
    lines.extend(
        [
            f"promotion review: {record.model_kind.value}",
            f"run_id={record.run_id}",
            f"symbol={record.symbol or '—'} timeframe={record.timeframe or '—'}",
            f"status={record.status.value} advisory_ready={record.advisory_ready} "
            f"verdict_passed={record.verdict_passed}",
            f"pointer={review.pointer_path}",
            f"artifact_dir={record.artifact_dir}",
        ]
    )
    if review.audit_ts:
        lines.append(f"audit_ts={review.audit_ts}")
    if record.promoted_by:
        lines.append(f"promoted_by={record.promoted_by}")
    if record.metrics:
        metrics = " ".join(f"{k}={v}" for k, v in sorted(record.metrics.items()))
        lines.append(f"metrics: {metrics}")
    if record.reasons:
        lines.append("reasons: " + " | ".join(record.reasons))
    if review.workflow_summary_text:
        lines.append(review.workflow_summary_text)
    if review.workflow_snapshot is not None and review.workflow_snapshot.research_count > 0:
        lines.append(
            "training research: "
            f"count={review.workflow_snapshot.research_count} "
            f"ml={review.workflow_snapshot.research_ml_count} "
            f"rl={review.workflow_snapshot.research_rl_count}"
        )
    if (
        review.workflow_snapshot is not None
        and review.workflow_snapshot.research_refresh_requested
    ):
        lines.append(
            "training research refresh: "
            f"target={review.workflow_snapshot.research_refresh_target or '—'} "
            f"refreshed={review.workflow_snapshot.research_refreshed_count}"
        )
    if (
        review.workflow_snapshot is not None
        and review.workflow_snapshot.team_research_headline
    ):
        lines.append(
            "team research posture: "
            f"{review.workflow_snapshot.team_research_headline} "
            f"ml={review.workflow_snapshot.team_research_ml_count} "
            f"rl={review.workflow_snapshot.team_research_rl_count} "
            f"policy={review.workflow_snapshot.team_research_selection_policy or '—'} "
            f"target={review.workflow_snapshot.team_research_refresh_target or '—'}"
            + (
                f" refreshed={review.workflow_snapshot.team_research_refreshed_count}"
                if review.workflow_snapshot.team_research_refresh_requested
                else ""
            )
        )
    if review.workflow_snapshot is not None and review.workflow_snapshot.team_role_count > 0:
        lines.append(
            "team posture: "
            f"roles={review.workflow_snapshot.team_ok_role_count}/"
            f"{review.workflow_snapshot.team_role_count} "
            f"nightly={review.workflow_snapshot.team_nightly_aligned_reports}/"
            f"{review.workflow_snapshot.team_nightly_enabled_reports} "
            f"target={review.workflow_snapshot.team_nightly_recommended_target or '—'} "
            "force_refresh="
            f"{1 if review.workflow_snapshot.team_nightly_recommended_force_refresh else 0}"
        )
        if review.workflow_snapshot.team_nightly_recent_window > 0:
            lines.append(
                "team recent trend: "
                f"{review.workflow_snapshot.team_nightly_recent_aligned}/"
                f"{review.workflow_snapshot.team_nightly_recent_enabled} "
                f"over {review.workflow_snapshot.team_nightly_recent_window} run(s) "
                f"latest={review.workflow_snapshot.team_nightly_recent_latest_status or '—'} "
                "basket="
                f"{review.workflow_snapshot.team_nightly_recent_latest_basket_size}"
            )
        if review.workflow_snapshot.team_discovery_alignment_summary:
            lines.append(
                "team discovery posture: "
                f"{review.workflow_snapshot.team_discovery_alignment_summary}"
                + (
                    f" overlap={review.workflow_snapshot.team_discovery_overlap_symbols}"
                    if review.workflow_snapshot.team_discovery_overlap_symbols
                    else ""
                )
            )
    if review.workflow_research_alignment_summary_text:
        prefix = (
            "warning: "
            if review.workflow_research_alignment_warning
            else "workflow basket/research alignment: "
        )
        line = prefix + review.workflow_research_alignment_summary_text
        if review.workflow_research_alignment_recommended_target:
            line += (
                " recommended_target="
                f"{review.workflow_research_alignment_recommended_target}"
            )
        lines.append(line)
    if review.workflow_discovery_alignment_summary_text:
        prefix = (
            "warning: "
            if review.workflow_discovery_alignment_warning
            else "workflow discovery alignment: "
        )
        line = prefix + review.workflow_discovery_alignment_summary_text
        if review.workflow_discovery_alignment_recommended_target:
            line += (
                " recommended_target="
                f"{review.workflow_discovery_alignment_recommended_target}"
            )
        lines.append(line)
    if review.nightly_alignment_summary_text:
        prefix = "warning: " if review.nightly_alignment_warning else "recent nightly alignment: "
        lines.append(prefix + review.nightly_alignment_summary_text)
    if review.nightly_alignment_recommended_target:
        lines.append(
            "recommended remediation: "
            f"target={review.nightly_alignment_recommended_target} "
            f"force_refresh={1 if review.nightly_alignment_recommended_force_refresh else 0}"
        )
    if review.cross_artifact_alignment is not None:
        alignment = review.cross_artifact_alignment
        lines.append(
            "cross_artifact_alignment="
            f"status={alignment.overall_status} summary={alignment.summary or '—'}"
        )
    if review.dual_promotion_review is not None:
        dual = review.dual_promotion_review
        lines.append(
            "dual_promotion="
            f"posture={dual.posture} warning={1 if dual.warning else 0} "
            f"summary={dual.summary or '—'}"
        )
    if review.nightly_alignment_recommended_command:
        lines.append(f"recommended command: {review.nightly_alignment_recommended_command}")
    if review.nightly_alignment_recommended_discovery_action:
        lines.append(
            "recommended discovery follow-up: "
            f"{review.nightly_alignment_recommended_discovery_action}"
        )
    if review.nightly_alignment_recommended_discovery_command:
        lines.append(
            "recommended discovery command: "
            f"{review.nightly_alignment_recommended_discovery_command}"
        )
    return "\n".join(lines)


def _find_promotion_event(
    *,
    kind: ModelKind,
    models_root: Path | str,
    symbol: Optional[str],
    run_id: Optional[str],
) -> Optional[dict[str, Any]]:
    events = read_audit(models_root, limit=500)
    symbol_norm = str(symbol or "").upper().strip()
    run_norm = str(run_id or "").strip()
    for row in events:
        if row.get("action") != "promote":
            continue
        if row.get("model_kind") != kind.value:
            continue
        if run_norm and str(row.get("run_id", "")).strip() != run_norm:
            continue
        if symbol_norm and str(row.get("symbol", "")).upper().strip() != symbol_norm:
            continue
        return row
    return None


def _pointer_path_for_event(
    *,
    kind: ModelKind,
    models_root: Path | str,
    ml_base: Optional[Path | str],
    symbol: Optional[str],
    explicit: Optional[str],
) -> Path:
    if explicit:
        return Path(str(explicit))
    return live_pointer_path(
        kind,
        models_root,
        symbol=symbol if kind == ModelKind.RL_POLICY else None,
        ml_base=ml_base,
    )


def _record_from_pointer(path: Path) -> Optional[PromotionRecord]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return PromotionRecord.from_dict(payload)
    except (KeyError, ValueError):
        return None


__all__ = [
    "PromotionReview",
    "build_and_export_promotion_review",
    "build_promotion_review",
    "export_promotion_review",
    "format_promotion_review",
    "format_promotion_review_markdown",
]


def format_promotion_review_markdown(review: PromotionReview) -> str:
    record = review.record
    lines: list[str] = []
    if review.activation is not None:
        act = review.activation
        lines.extend(
            [
                f"# Activation — {act.lane.upper()}",
                "",
                f"- Stage: `{act.stage}`",
                f"- Active: `{act.active}`",
                f"- Summary: {act.summary}",
            ]
        )
        if act.recommended_command:
            lines.append(f"- Recommended command: `{act.recommended_command}`")
        if act.candidate_run_id:
            lines.append(f"- Candidate run ID: `{act.candidate_run_id}`")
        if act.blockers:
            lines.append("")
            lines.append("## Blockers")
            lines.append("")
            for row in act.blockers:
                lines.append(f"- **{row.severity}** `{row.code}`: {row.detail}")
        lines.extend(["", "---", ""])
    lines.extend(
        [
            f"# Promotion Review — {record.model_kind.value}",
            "",
            f"- Run ID: `{record.run_id}`",
            f"- Symbol: `{record.symbol or '—'}`",
            f"- Timeframe: `{record.timeframe or '—'}`",
            f"- Status: `{record.status.value}`",
            f"- Advisory ready: `{record.advisory_ready}`",
            f"- Verdict passed: `{record.verdict_passed}`",
            f"- Pointer: `{review.pointer_path}`",
            f"- Artifact dir: `{record.artifact_dir}`",
        ]
    )
    if review.audit_ts:
        lines.append(f"- Audit timestamp: `{review.audit_ts}`")
    if record.promoted_by:
        lines.append(f"- Promoted by: `{record.promoted_by}`")
    if record.metrics:
        lines.extend(
            [
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for key, value in sorted(record.metrics.items()):
            lines.append(f"| `{key}` | {value} |")
    if record.reasons:
        lines.extend(
            [
                "",
                "## Reasons",
                "",
            ]
        )
        lines.extend(f"- {reason}" for reason in record.reasons)
    if record.workflow_snapshot_path:
        lines.extend(
            [
                "",
                "## Workflow Snapshot",
                "",
                f"- Path: `{record.workflow_snapshot_path}`",
            ]
        )
    if review.workflow_summary_text:
        lines.append(f"- Summary: `{review.workflow_summary_text}`")
    if review.workflow_snapshot is not None and review.workflow_snapshot.research_count > 0:
        lines.append(
            "- Training research: "
            f"`count={review.workflow_snapshot.research_count} "
            f"ml={review.workflow_snapshot.research_ml_count} "
            f"rl={review.workflow_snapshot.research_rl_count}`"
        )
    if (
        review.workflow_snapshot is not None
        and review.workflow_snapshot.research_refresh_requested
    ):
        lines.append(
            "- Training research refresh: "
            f"`target={review.workflow_snapshot.research_refresh_target or '—'} "
            f"refreshed={review.workflow_snapshot.research_refreshed_count}`"
        )
    if (
        review.workflow_snapshot is not None
        and review.workflow_snapshot.team_research_headline
    ):
        lines.append(
            "- Team research posture: "
            f"`{review.workflow_snapshot.team_research_headline} "
            f"ml={review.workflow_snapshot.team_research_ml_count} "
            f"rl={review.workflow_snapshot.team_research_rl_count} "
            f"policy={review.workflow_snapshot.team_research_selection_policy or '—'} "
            f"target={review.workflow_snapshot.team_research_refresh_target or '—'}"
            + (
                f" refreshed={review.workflow_snapshot.team_research_refreshed_count}"
                if review.workflow_snapshot.team_research_refresh_requested
                else ""
            )
            + "`"
        )
    if review.workflow_snapshot is not None and review.workflow_snapshot.team_role_count > 0:
        lines.append(
            "- Team posture: "
            f"`roles={review.workflow_snapshot.team_ok_role_count}/"
            f"{review.workflow_snapshot.team_role_count} "
            f"nightly={review.workflow_snapshot.team_nightly_aligned_reports}/"
            f"{review.workflow_snapshot.team_nightly_enabled_reports} "
            f"target={review.workflow_snapshot.team_nightly_recommended_target or '—'} "
            "force_refresh="
            f"{1 if review.workflow_snapshot.team_nightly_recommended_force_refresh else 0}`"
        )
        if review.workflow_snapshot.team_nightly_recent_window > 0:
            lines.append(
                "- Team recent trend: "
                f"`{review.workflow_snapshot.team_nightly_recent_aligned}/"
                f"{review.workflow_snapshot.team_nightly_recent_enabled} "
                f"over {review.workflow_snapshot.team_nightly_recent_window} run(s) "
                f"latest={review.workflow_snapshot.team_nightly_recent_latest_status or '—'} "
                f"basket={review.workflow_snapshot.team_nightly_recent_latest_basket_size}`"
            )
        if review.workflow_snapshot.team_discovery_alignment_summary:
            lines.append(
                "- Team discovery posture: "
                f"`{review.workflow_snapshot.team_discovery_alignment_summary}"
                + (
                    f" overlap={review.workflow_snapshot.team_discovery_overlap_symbols}"
                    if review.workflow_snapshot.team_discovery_overlap_symbols
                    else ""
                )
                + "`"
            )
    if review.workflow_research_alignment_summary_text:
        label = (
            "Warning"
            if review.workflow_research_alignment_warning
            else "Workflow basket/research alignment"
        )
        detail = review.workflow_research_alignment_summary_text
        if review.workflow_research_alignment_recommended_target:
            detail += (
                " recommended_target="
                f"{review.workflow_research_alignment_recommended_target}"
            )
        lines.append(f"- {label}: `{detail}`")
    if review.workflow_discovery_alignment_summary_text:
        label = (
            "Warning"
            if review.workflow_discovery_alignment_warning
            else "Workflow discovery alignment"
        )
        detail = review.workflow_discovery_alignment_summary_text
        if review.workflow_discovery_alignment_recommended_target:
            detail += (
                " recommended_target="
                f"{review.workflow_discovery_alignment_recommended_target}"
            )
        lines.append(f"- {label}: `{detail}`")
    if review.nightly_alignment_summary_text:
        label = "Warning" if review.nightly_alignment_warning else "Recent nightly alignment"
        lines.append(f"- {label}: `{review.nightly_alignment_summary_text}`")
    if review.nightly_alignment_recommended_target:
        lines.append(
            "- Recommended remediation: "
            f"`target={review.nightly_alignment_recommended_target} "
            f"force_refresh={1 if review.nightly_alignment_recommended_force_refresh else 0}`"
        )
    if review.cross_artifact_alignment is not None:
        alignment = review.cross_artifact_alignment
        lines.append(
            "- Cross-artifact alignment: "
            f"`status={alignment.overall_status} summary={alignment.summary or '—'}`"
        )
    if review.dual_promotion_review is not None:
        dual = review.dual_promotion_review
        lines.append(
            "- Dual promotion: "
            f"`posture={dual.posture} warning={1 if dual.warning else 0} "
            f"summary={dual.summary or '—'}`"
        )
    if review.nightly_alignment_recommended_command:
        lines.append(f"- Recommended command: `{review.nightly_alignment_recommended_command}`")
    if review.nightly_alignment_recommended_discovery_action:
        lines.append(
            "- Recommended discovery follow-up: "
            f"`{review.nightly_alignment_recommended_discovery_action}`"
        )
    if review.nightly_alignment_recommended_discovery_command:
        lines.append(
            "- Recommended discovery command: "
            f"`{review.nightly_alignment_recommended_discovery_command}`"
        )
    return "\n".join(lines) + "\n"


def _nightly_alignment_review_text(nightly_alignment, *, settings) -> tuple[Optional[str], bool]:
    min_enabled_reports = max(
        1,
        int(getattr(settings, "acceptance_bundle_nightly_alignment_min_enabled_reports", 2)),
    )
    warn_ratio = max(
        0.0,
        min(1.0, float(getattr(settings, "acceptance_bundle_nightly_alignment_warn_ratio", 0.6))),
    )
    refreshed_warn_ratio = max(
        0.0,
        min(
            1.0,
            float(
                getattr(
                    settings,
                    "acceptance_bundle_nightly_refreshed_alignment_warn_ratio",
                    0.35,
                )
            ),
        ),
    )
    if nightly_alignment.enabled_reports < min_enabled_reports:
        return None, False
    ratio = (
        float(nightly_alignment.aligned_reports) / float(nightly_alignment.enabled_reports)
        if nightly_alignment.enabled_reports > 0
        else 0.0
    )
    refreshed_ratio = (
        float(nightly_alignment.refreshed_aligned_reports)
        / float(nightly_alignment.enabled_reports)
        if nightly_alignment.enabled_reports > 0
        else 0.0
    )
    text = (
        f"ratio={ratio:.2f} ({nightly_alignment.aligned_reports}/"
        f"{nightly_alignment.enabled_reports})"
        + (
            f" latest={nightly_alignment.latest_target_mix}"
            if nightly_alignment.latest_target_mix
            else ""
        )
        + (
            f" refreshed={nightly_alignment.latest_refreshed_target_mix}"
            if nightly_alignment.latest_refreshed_target_mix
            else ""
        )
        + (
            f" target={nightly_alignment.recommended_refresh_target}"
            if nightly_alignment.recommended_refresh_target
            else ""
        )
        + (" force_refresh=1" if nightly_alignment.recommended_force_refresh else "")
        + (
            f" refreshed_ratio={refreshed_ratio:.2f}"
            if nightly_alignment.enabled_reports > 0
            else ""
        )
    )
    return text, (ratio < warn_ratio) or (refreshed_ratio < refreshed_warn_ratio)


def _workflow_research_alignment_review_text(
    workflow_summary: Optional[WorkflowSnapshotSummary],
    *,
    settings,
) -> tuple[Optional[str], bool]:
    if workflow_summary is None:
        return None, False
    selected = int(getattr(workflow_summary, "team_research_alignment_selected_count", 0) or 0)
    overlap = int(getattr(workflow_summary, "team_research_alignment_overlap_count", 0) or 0)
    summary = str(getattr(workflow_summary, "team_research_alignment_summary", "") or "").strip()
    mix = str(getattr(workflow_summary, "team_research_alignment_target_mix", "") or "").strip()
    if not summary or selected <= 0:
        return None, False
    warn_ratio = max(
        0.0,
        min(
            1.0,
            float(
                getattr(
                    settings,
                    "acceptance_bundle_team_research_alignment_warn_ratio",
                    0.6,
                )
            ),
        ),
    )
    ratio = float(overlap) / float(selected) if selected > 0 else 0.0
    text = f"ratio={ratio:.2f} ({overlap}/{selected}) {summary}"
    if mix:
        text += f" mix={mix}"
    return text, ratio < warn_ratio


def _workflow_research_alignment_target(
    workflow_summary: Optional[WorkflowSnapshotSummary],
) -> Optional[str]:
    if workflow_summary is None:
        return None
    return _target_from_alignment_mix(workflow_summary.team_research_alignment_target_mix)


def _workflow_discovery_alignment_review_text(
    workflow_summary: Optional[WorkflowSnapshotSummary],
    *,
    settings,
) -> tuple[Optional[str], bool]:
    if workflow_summary is None:
        return None, False
    compare_count = int(
        getattr(workflow_summary, "team_discovery_alignment_compare_count", 0) or 0
    )
    overlap = int(getattr(workflow_summary, "team_discovery_alignment_overlap_count", 0) or 0)
    summary = str(getattr(workflow_summary, "team_discovery_alignment_summary", "") or "").strip()
    overlap_symbols = str(
        getattr(workflow_summary, "team_discovery_overlap_symbols", "") or ""
    ).strip()
    if not summary or compare_count <= 0:
        return None, False
    warn_ratio = max(
        0.0,
        min(
            1.0,
            float(
                getattr(
                    settings,
                    "acceptance_bundle_team_discovery_alignment_warn_ratio",
                    0.6,
                )
            ),
        ),
    )
    ratio = float(overlap) / float(compare_count) if compare_count > 0 else 0.0
    text = f"ratio={ratio:.2f} ({overlap}/{compare_count}) {summary}"
    if overlap_symbols:
        text += f" overlap={overlap_symbols}"
    return text, ratio < warn_ratio


def _workflow_discovery_alignment_target(
    workflow_summary: Optional[WorkflowSnapshotSummary],
) -> Optional[str]:
    if workflow_summary is None:
        return None
    return _target_from_alignment_mix(workflow_summary.team_research_alignment_target_mix)


def _workflow_discovery_recommendation_action(
    workflow_summary: Optional[WorkflowSnapshotSummary],
) -> Optional[str]:
    if workflow_summary is None:
        return None
    compare_count = int(workflow_summary.team_discovery_alignment_compare_count or 0)
    overlap_count = int(workflow_summary.team_discovery_alignment_overlap_count or 0)
    summary = str(workflow_summary.team_discovery_alignment_summary or "").strip()
    if not summary or compare_count <= 0 or overlap_count >= compare_count:
        return None
    action = (
        "Rebuild the market universe from Screener liquidity and recent activity inputs "
        "before the next promotion or nightly cycle."
    )
    overlap_symbols = str(workflow_summary.team_discovery_overlap_symbols or "").strip()
    if overlap_symbols:
        action += f" Latest overlap: {overlap_symbols}."
    return action


def _workflow_discovery_recommendation_command(
    *,
    workflow_summary: Optional[WorkflowSnapshotSummary],
    settings,
) -> Optional[str]:
    if _workflow_discovery_recommendation_action(workflow_summary) is None:
        return None
    from fortuna.app.market_universe import resolve_market_universe_cli_source

    source = resolve_market_universe_cli_source(settings)
    timeframe = str(getattr(settings, "default_timeframe", "5m") or "5m")
    days = int(getattr(settings, "default_days", 30) or 30)
    limit = int(getattr(settings, "market_universe_default_limit", 15) or 15)
    return (
        "uv run python scripts/build_market_universe.py "
        f"--source {source} --timeframe {timeframe} --days {days} --limit {limit}"
    )


def _target_from_alignment_mix(mix: Optional[str]) -> Optional[str]:
    text = str(mix or "").strip().lower()
    if not text:
        return None
    keys: set[str] = set()
    for part in text.split(","):
        chunk = part.strip()
        if not chunk or "=" not in chunk:
            continue
        key = chunk.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    if not keys:
        return None
    if "none" in keys or "both" in keys:
        return "all"
    if "ml" in keys and "rl" not in keys:
        return "rl"
    if "rl" in keys and "ml" not in keys:
        return "ml"
    if "ml" in keys and "rl" in keys:
        return "all"
    return None


def export_promotion_review(
    review: PromotionReview,
    out_path: Path | str,
    *,
    fmt: str,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(review.to_dict(), indent=2), encoding="utf-8")
    elif fmt == "md":
        path.write_text(format_promotion_review_markdown(review), encoding="utf-8")
    else:
        raise ValueError(f"unsupported review format: {fmt}")
    return path


def build_and_export_promotion_review(
    *,
    kind: ModelKind,
    out_dir: Path | str,
    basename: str = "promotion_review",
    models_root: Path | str = "models",
    ml_base: Optional[Path | str] = None,
    symbol: Optional[str] = None,
    run_id: Optional[str] = None,
    workflow_snapshot_path: Optional[Path | str] = None,
    settings=None,
) -> tuple[PromotionReview, Path, Path] | None:
    review = build_promotion_review(
        kind=kind,
        models_root=models_root,
        ml_base=ml_base,
        symbol=symbol,
        run_id=run_id,
        settings=settings,
    )
    if review is None:
        return None
    override_path = str(workflow_snapshot_path or "").strip()
    if override_path:
        workflow_summary = load_workflow_snapshot_summary(override_path)
        workflow_text = (
            format_workflow_snapshot_summary(workflow_summary)
            if workflow_summary is not None
            else None
        )
        workflow_research_text = review.workflow_research_alignment_summary_text
        workflow_research_warning = review.workflow_research_alignment_warning
        workflow_research_target = review.workflow_research_alignment_recommended_target
        workflow_discovery_text = review.workflow_discovery_alignment_summary_text
        workflow_discovery_warning = review.workflow_discovery_alignment_warning
        workflow_discovery_target = review.workflow_discovery_alignment_recommended_target
        runtime_settings = settings
        if runtime_settings is None:
            try:
                runtime_settings = get_settings()
            except Exception:
                runtime_settings = None
        if runtime_settings is not None:
            workflow_research_text, workflow_research_warning = (
                _workflow_research_alignment_review_text(
                    workflow_summary,
                    settings=runtime_settings,
                )
            )
            workflow_research_target = _workflow_research_alignment_target(
                workflow_summary
            )
            workflow_discovery_text, workflow_discovery_warning = (
                _workflow_discovery_alignment_review_text(
                    workflow_summary,
                    settings=runtime_settings,
                )
            )
            workflow_discovery_target = _workflow_discovery_alignment_target(
                workflow_summary
            )
        cross_artifact_alignment = build_cross_artifact_alignment(
            nightly_report=None,
            workflow_summary=workflow_summary,
            rl_review=None,
            ml_review=None,
            model_health=None,
            effective_refresh_target=review.nightly_alignment_recommended_target,
        )
        review = replace(
            review,
            record=replace(review.record, workflow_snapshot_path=override_path),
            workflow_snapshot=workflow_summary,
            workflow_summary_text=workflow_text,
            workflow_research_alignment_summary_text=workflow_research_text,
            workflow_research_alignment_warning=workflow_research_warning,
            workflow_research_alignment_recommended_target=workflow_research_target,
            workflow_discovery_alignment_summary_text=workflow_discovery_text,
            workflow_discovery_alignment_warning=workflow_discovery_warning,
            workflow_discovery_alignment_recommended_target=workflow_discovery_target,
            cross_artifact_alignment=cross_artifact_alignment,
            dual_promotion_review=_dual_promotion_for_review(
                review_kind=review.record.model_kind,
                workflow_summary=workflow_summary,
                _workflow_research_target=workflow_research_target,
                nightly_alignment=None,
                effective_refresh_target=review.nightly_alignment_recommended_target,
            ),
        )
    target_dir = Path(out_dir)
    md_path = export_promotion_review(review, target_dir / f"{basename}.md", fmt="md")
    json_path = export_promotion_review(review, target_dir / f"{basename}.json", fmt="json")
    return review, md_path, json_path
