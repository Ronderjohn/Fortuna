"""Build and export a compact multi-agent acceptance bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fortuna.agentic.contracts import (
    CrossArtifactAlignmentSummary,
    DualPromotionReviewSummary,
    ModelHealthResponse,
    NightlyArtifactLinkageSummary,
    WorkflowSnapshotSummary,
)
from fortuna.app.acceptance_alignment import (
    build_cross_artifact_alignment,
    build_dual_promotion_review_summary,
    follow_up_lane_target,
    resolve_acceptance_refresh_context,
)
from fortuna.app.model_status import build_model_status
from fortuna.app.operator_workflow import build_workflow_snapshot, export_workflow_snapshot
from fortuna.app.promotion_review import PromotionReview, build_promotion_review
from fortuna.models.metadata import ModelKind
from fortuna.observability.recorder import workflow_boundary


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class NightlyReportEvidence:
    path: str
    overall_status: str = ""
    basket: tuple[str, ...] = ()
    basket_mode: Optional[str] = None
    training_execution_target: Optional[str] = None
    training_execution_model_family: Optional[str] = None
    training_execution_selection_source: Optional[str] = None
    training_candidate_path: Optional[str] = None
    training_candidate_exists: bool = False
    training_candidate_count: int = 0
    training_candidate_ml_count: int = 0
    training_candidate_rl_count: int = 0
    training_candidate_selection_policy: Optional[str] = None
    training_research_path: Optional[str] = None
    training_research_exists: bool = False
    training_research_count: int = 0
    training_research_ml_count: int = 0
    training_research_rl_count: int = 0
    training_research_discovery_preferred_count: int = 0
    training_research_discovery_regime_mix: Optional[str] = None
    training_research_discovery_summary: Optional[str] = None
    training_research_discovery_recommended_refresh_target: Optional[str] = None
    training_research_effective_refresh_target: Optional[str] = None
    training_research_scout_support_target: Optional[str] = None
    training_research_discovery_follow_up_target: Optional[str] = None
    training_research_discovery_follow_up_summary: Optional[str] = None
    training_research_discovery_follow_up_action: Optional[str] = None
    training_research_research_follow_up_target: Optional[str] = None
    training_research_research_follow_up_summary: Optional[str] = None
    training_research_research_follow_up_action: Optional[str] = None
    training_research_execution_follow_up_target: Optional[str] = None
    training_research_execution_follow_up_summary: Optional[str] = None
    training_research_execution_follow_up_action: Optional[str] = None
    workflow_snapshot_path: Optional[str] = None
    workflow_snapshot_exists: bool = False
    promoted_model_kind: Optional[str] = None
    promotion_review_paths: tuple[str, ...] = ()
    promotion_review_exists: tuple[bool, ...] = ()
    promotion_review_model_kind: Optional[str] = None
    promotion_review_model_kinds: tuple[str, ...] = ()
    step_statuses: dict[str, str] | None = None
    lane_readiness_global_blockers: tuple[str, ...] = ()
    lane_readiness_should_abort: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "overall_status": self.overall_status,
            "basket": list(self.basket),
            "basket_mode": self.basket_mode,
            "training_execution_target": self.training_execution_target,
            "training_execution_model_family": self.training_execution_model_family,
            "training_execution_selection_source": self.training_execution_selection_source,
            "training_candidate_path": self.training_candidate_path,
            "training_candidate_exists": self.training_candidate_exists,
            "training_candidate_count": self.training_candidate_count,
            "training_candidate_ml_count": self.training_candidate_ml_count,
            "training_candidate_rl_count": self.training_candidate_rl_count,
            "training_candidate_selection_policy": self.training_candidate_selection_policy,
            "training_research_path": self.training_research_path,
            "training_research_exists": self.training_research_exists,
            "training_research_count": self.training_research_count,
            "training_research_ml_count": self.training_research_ml_count,
            "training_research_rl_count": self.training_research_rl_count,
            "training_research_discovery_preferred_count": (
                self.training_research_discovery_preferred_count
            ),
            "training_research_discovery_regime_mix": (
                self.training_research_discovery_regime_mix
            ),
            "training_research_discovery_summary": self.training_research_discovery_summary,
            "training_research_discovery_recommended_refresh_target": (
                self.training_research_discovery_recommended_refresh_target
            ),
            "training_research_effective_refresh_target": (
                self.training_research_effective_refresh_target
            ),
            "training_research_scout_support_target": (
                self.training_research_scout_support_target
            ),
            "training_research_discovery_follow_up_target": (
                self.training_research_discovery_follow_up_target
            ),
            "training_research_discovery_follow_up_summary": (
                self.training_research_discovery_follow_up_summary
            ),
            "training_research_discovery_follow_up_action": (
                self.training_research_discovery_follow_up_action
            ),
            "training_research_research_follow_up_target": (
                self.training_research_research_follow_up_target
            ),
            "training_research_research_follow_up_summary": (
                self.training_research_research_follow_up_summary
            ),
            "training_research_research_follow_up_action": (
                self.training_research_research_follow_up_action
            ),
            "training_research_execution_follow_up_target": (
                self.training_research_execution_follow_up_target
            ),
            "training_research_execution_follow_up_summary": (
                self.training_research_execution_follow_up_summary
            ),
            "training_research_execution_follow_up_action": (
                self.training_research_execution_follow_up_action
            ),
            "workflow_snapshot_path": self.workflow_snapshot_path,
            "workflow_snapshot_exists": self.workflow_snapshot_exists,
            "promoted_model_kind": self.promoted_model_kind,
            "promotion_review_paths": list(self.promotion_review_paths),
            "promotion_review_exists": list(self.promotion_review_exists),
            "promotion_review_model_kind": self.promotion_review_model_kind,
            "promotion_review_model_kinds": list(self.promotion_review_model_kinds),
            "step_statuses": dict(self.step_statuses or {}),
            "lane_readiness_global_blockers": list(self.lane_readiness_global_blockers),
            "lane_readiness_should_abort": self.lane_readiness_should_abort,
        }


@dataclass(frozen=True)
class AcceptanceBundle:
    created_at: str
    overall_status: str
    nightly_report: Optional[NightlyReportEvidence] = None
    workflow_snapshot: Optional[WorkflowSnapshotSummary] = None
    rl_review: Optional[PromotionReview] = None
    ml_review: Optional[PromotionReview] = None
    model_health: Optional[ModelHealthResponse] = None
    recommended_refresh_target: Optional[str] = None
    effective_refresh_target: Optional[str] = None
    refresh_target_source: Optional[str] = None
    recommended_follow_up_lane: Optional[str] = None
    scout_support_target: Optional[str] = None
    scout_support_summary: Optional[str] = None
    recommended_force_refresh: bool = False
    recommended_cli_command: Optional[str] = None
    recommended_discovery_action: Optional[str] = None
    recommended_discovery_cli_command: Optional[str] = None
    cross_artifact_alignment: Optional[CrossArtifactAlignmentSummary] = None
    dual_promotion_review: Optional[DualPromotionReviewSummary] = None
    replay_linkage: Optional[NightlyArtifactLinkageSummary] = None
    checks: tuple[AcceptanceCheck, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "overall_status": self.overall_status,
            "nightly_report": (
                self.nightly_report.to_dict() if self.nightly_report is not None else None
            ),
            "workflow_snapshot": (
                self.workflow_snapshot.to_dict() if self.workflow_snapshot is not None else None
            ),
            "rl_review": self.rl_review.to_dict() if self.rl_review is not None else None,
            "ml_review": self.ml_review.to_dict() if self.ml_review is not None else None,
            "model_health": self.model_health.to_dict() if self.model_health is not None else None,
            "recommended_refresh_target": self.recommended_refresh_target,
            "effective_refresh_target": self.effective_refresh_target,
            "refresh_target_source": self.refresh_target_source,
            "recommended_follow_up_lane": self.recommended_follow_up_lane,
            "scout_support_target": self.scout_support_target,
            "scout_support_summary": self.scout_support_summary,
            "recommended_force_refresh": self.recommended_force_refresh,
            "recommended_cli_command": self.recommended_cli_command,
            "recommended_discovery_action": self.recommended_discovery_action,
            "recommended_discovery_cli_command": self.recommended_discovery_cli_command,
            "cross_artifact_alignment": (
                self.cross_artifact_alignment.to_dict()
                if self.cross_artifact_alignment is not None
                else None
            ),
            "dual_promotion_review": (
                self.dual_promotion_review.to_dict()
                if self.dual_promotion_review is not None
                else None
            ),
            "replay_linkage": (
                self.replay_linkage.to_dict() if self.replay_linkage is not None else None
            ),
            "checks": [row.to_dict() for row in self.checks],
        }


def _format_replay_linkage(linkage: NightlyArtifactLinkageSummary) -> str:
    parts = [f"status={linkage.overall_status}"]
    if linkage.run_identifier:
        parts.append(f"run={linkage.run_identifier}")
    if linkage.missing_artifacts:
        parts.append(f"missing={','.join(linkage.missing_artifacts)}")
    if linkage.broken_links:
        parts.append(f"broken={','.join(linkage.broken_links)}")
    if linkage.linkage_warnings:
        parts.append(f"warnings={' | '.join(linkage.linkage_warnings)}")
    parts.append(f"summary={linkage.summary or '—'}")
    return " ".join(parts)


def gather_acceptance_bundle(
    *,
    settings,
    engine: Any | None = None,
    models_root: Path | str = "models",
    ml_base: Optional[Path | str] = None,
    symbol: Optional[str] = None,
    nightly_report_path: Optional[Path | str] = None,
    nightly_report_dir: Optional[Path | str] = None,
    workflow_snapshot_path: Optional[Path | str] = None,
    build_workflow: bool = False,
    workflow_out: Optional[Path | str] = None,
    universe_limit: int = 10,
    analysis_limit: int = 5,
    candidate_limit: int = 8,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
    include_model_health: bool = True,
    linkage_summary: Optional[NightlyArtifactLinkageSummary] = None,
) -> AcceptanceBundle:
    nightly_report = load_nightly_report_evidence(
        nightly_report_path=nightly_report_path,
        nightly_report_dir=nightly_report_dir,
    )
    workflow_summary: Optional[WorkflowSnapshotSummary] = None
    workflow_path = str(workflow_snapshot_path or "").strip()
    if not workflow_path and nightly_report is not None:
        workflow_path = _resolve_nightly_artifact_path(
            nightly_report,
            nightly_report.workflow_snapshot_path,
        )
    if build_workflow:
        snapshot = build_workflow_snapshot(
            settings=settings,
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            candidate_limit=candidate_limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )
        out_path = (
            Path(workflow_out)
            if workflow_out
            else Path("reports/acceptance/workflow_snapshot.json")
        )
        written = export_workflow_snapshot(snapshot, settings.resolve_path(out_path))
        from fortuna.app.workflow_snapshot import load_workflow_snapshot_summary

        workflow_summary = load_workflow_snapshot_summary(str(written))
    elif workflow_path:
        from fortuna.app.workflow_snapshot import load_workflow_snapshot_summary

        workflow_summary = load_workflow_snapshot_summary(workflow_path)

    rl_review = build_promotion_review(
        kind=ModelKind.RL_POLICY,
        models_root=models_root,
        ml_base=ml_base,
        symbol=symbol,
        settings=settings,
    )
    ml_review = build_promotion_review(
        kind=ModelKind.ML_SCORER,
        models_root=models_root,
        ml_base=ml_base,
        settings=settings,
    )

    model_health = None
    if include_model_health:
        if engine is None:
            from fortuna.app.session_engine import FortunaSessionEngine

            engine = FortunaSessionEngine(settings)
        model_health = build_model_status(engine)

    refresh_context = resolve_acceptance_refresh_context(
        model_health=model_health,
        nightly_report=nightly_report,
        workflow_summary=workflow_summary,
        workflow_research_alignment_target=_workflow_research_alignment_target,
    )
    dual_promotion_review = build_dual_promotion_review_summary(
        nightly_report=nightly_report,
        workflow_summary=workflow_summary,
        rl_review=rl_review,
        ml_review=ml_review,
        effective_refresh_target=refresh_context.get("effective_refresh_target"),
    )
    checks = _build_checks(
        settings=settings,
        nightly_report=nightly_report,
        workflow_summary=workflow_summary,
        rl_review=rl_review,
        ml_review=ml_review,
        model_health=model_health,
        dual_promotion_review=dual_promotion_review,
        linkage_summary=linkage_summary,
    )
    cross_artifact_alignment = build_cross_artifact_alignment(
        nightly_report=nightly_report,
        workflow_summary=workflow_summary,
        rl_review=rl_review,
        ml_review=ml_review,
        model_health=model_health,
        effective_refresh_target=refresh_context.get("effective_refresh_target"),
    )
    return AcceptanceBundle(
        created_at=datetime.now().isoformat(timespec="seconds"),
        overall_status=_overall_status(checks),
        nightly_report=nightly_report,
        workflow_snapshot=workflow_summary,
        rl_review=rl_review,
        ml_review=ml_review,
        model_health=model_health,
        recommended_refresh_target=(
            model_health.agentic.nightly_alignment.recommended_refresh_target
            if model_health is not None
            else _workflow_research_alignment_target(workflow_summary)
        ),
        effective_refresh_target=refresh_context.get("effective_refresh_target"),
        refresh_target_source=refresh_context.get("refresh_target_source"),
        recommended_follow_up_lane=refresh_context.get("recommended_follow_up_lane"),
        scout_support_target=refresh_context.get("scout_support_target"),
        scout_support_summary=refresh_context.get("scout_support_summary"),
        recommended_force_refresh=(
            bool(model_health.agentic.nightly_alignment.recommended_force_refresh)
            if model_health is not None
            else False
        ),
        recommended_cli_command=(
            model_health.agentic.nightly_alignment.recommended_cli_command
            if model_health is not None
            else None
        ),
        recommended_discovery_action=(
            (
                model_health.agentic.nightly_alignment.recommended_discovery_action
                if model_health is not None
                else None
            )
            or _workflow_discovery_recommendation_action(workflow_summary)
        ),
        recommended_discovery_cli_command=(
            (
                model_health.agentic.nightly_alignment.recommended_discovery_cli_command
                if model_health is not None
                else None
            )
            or _workflow_discovery_recommendation_command(
                workflow_summary=workflow_summary,
                settings=settings,
            )
        ),
        cross_artifact_alignment=cross_artifact_alignment,
        dual_promotion_review=dual_promotion_review,
        replay_linkage=linkage_summary,
        checks=checks,
    )


def format_acceptance_bundle(bundle: AcceptanceBundle) -> str:
    lines = [
        f"acceptance bundle: {bundle.overall_status}",
        f"created_at={bundle.created_at}",
    ]
    if bundle.nightly_report is not None:
        review_label = _review_label(
            bundle.nightly_report.promotion_review_model_kind,
            bundle.nightly_report.promotion_review_model_kinds,
        )
        lines.append(
            "nightly_report="
            f"{bundle.nightly_report.path} "
            f"status={bundle.nightly_report.overall_status or '—'} "
            f"target={bundle.nightly_report.training_execution_target or '—'} "
            f"family={bundle.nightly_report.training_execution_model_family or '—'} "
            f"source={bundle.nightly_report.training_execution_selection_source or '—'} "
            f"candidates={bundle.nightly_report.training_candidate_count} "
            f"reviews={len(bundle.nightly_report.promotion_review_paths)} "
            f"review_kind={review_label}"
        )
    if bundle.workflow_snapshot is not None:
        wf = bundle.workflow_snapshot
        lines.append(
            "workflow="
            f"{wf.path} universe={wf.universe_count} shortlist={wf.shortlist_count} "
            "briefing="
            f"{wf.briefing_candidates} "
            "allocation="
            f"{wf.allocation_selected}/{wf.allocation_max_positions} "
            f"ml={wf.ml_count} rl={wf.rl_count}"
        )
        if wf.team_role_count > 0:
            lines.append(
                "team_posture="
                f"roles={wf.team_ok_role_count}/{wf.team_role_count} "
                f"nightly={wf.team_nightly_aligned_reports}/{wf.team_nightly_enabled_reports} "
                f"target={wf.team_nightly_recommended_target or '—'} "
                f"force_refresh={1 if wf.team_nightly_recommended_force_refresh else 0}"
            )
            if wf.team_research_alignment_summary:
                lines.append(
                    "team_research_alignment="
                    f"{wf.team_research_alignment_summary}"
                    + (
                        f" mix={wf.team_research_alignment_target_mix}"
                        if wf.team_research_alignment_target_mix
                        else ""
                    )
                )
            if wf.team_research_scout_support_summary:
                lines.append(
                    "team_research_scouts="
                    f"{wf.team_research_scout_support_summary}"
                )
            if wf.team_discovery_alignment_summary:
                lines.append(
                    "team_discovery_alignment="
                    f"{wf.team_discovery_alignment_summary}"
                    + (
                        f" overlap={wf.team_discovery_overlap_symbols}"
                        if wf.team_discovery_overlap_symbols
                        else ""
                    )
                )
            if wf.team_discovery_refresh_requested:
                lines.append(
                    "team_discovery_refresh="
                    f"{wf.team_discovery_refreshed_count}"
                    + (
                        f" source={wf.team_discovery_refresh_source}"
                        if wf.team_discovery_refresh_source
                        else ""
                    )
                    + (
                        f" timeframe={wf.team_discovery_refresh_timeframe}"
                        if wf.team_discovery_refresh_timeframe
                        else ""
                    )
                    + (
                        f" days={wf.team_discovery_refresh_days}"
                        if wf.team_discovery_refresh_days > 0
                        else ""
                    )
                )
            if (
                wf.team_research_discovery_follow_up_target
                or wf.team_research_research_follow_up_target
                or wf.team_research_execution_follow_up_target
            ):
                lines.append(
                    "team_follow_up="
                    + " ".join(
                        part
                        for part in (
                            (
                                f"discovery={wf.team_research_discovery_follow_up_target}"
                                if wf.team_research_discovery_follow_up_target
                                else ""
                            ),
                            (
                                f"research={wf.team_research_research_follow_up_target}"
                                if wf.team_research_research_follow_up_target
                                else ""
                            ),
                            (
                                f"execution={wf.team_research_execution_follow_up_target}"
                                if wf.team_research_execution_follow_up_target
                                else ""
                            ),
                        )
                        if part
                    )
                )
            if wf.team_nightly_recent_window > 0:
                lines.append(
                    "team_recent_trend="
                    f"{wf.team_nightly_recent_aligned}/{wf.team_nightly_recent_enabled} "
                    f"over {wf.team_nightly_recent_window} run(s) "
                    f"latest={wf.team_nightly_recent_latest_status or '—'} "
                    f"basket={wf.team_nightly_recent_latest_basket_size}"
                )
    if bundle.rl_review is not None:
        lines.append(f"rl_review={bundle.rl_review.record.run_id}")
    if bundle.ml_review is not None:
        lines.append(f"ml_review={bundle.ml_review.record.run_id}")
    if bundle.recommended_refresh_target:
        lines.append(
            "recommended_remediation="
            f"target={bundle.recommended_refresh_target} "
            f"force_refresh={1 if bundle.recommended_force_refresh else 0}"
        )
    if bundle.effective_refresh_target or bundle.refresh_target_source:
        lane_kwargs = {
            "nightly_report": bundle.nightly_report,
            "workflow_summary": bundle.workflow_snapshot,
        }
        lines.append(
            "refresh_context="
            f"effective_target={bundle.effective_refresh_target or '—'} "
            f"source={bundle.refresh_target_source or '—'} "
            f"discovery={follow_up_lane_target(**lane_kwargs, lane='discovery') or '—'} "
            f"research={follow_up_lane_target(**lane_kwargs, lane='research') or '—'} "
            f"execution={follow_up_lane_target(**lane_kwargs, lane='execution') or '—'} "
            f"scouts={bundle.scout_support_target or '—'}"
            + (
                f" ({bundle.scout_support_summary})"
                if bundle.scout_support_summary
                else ""
            )
        )
    if bundle.recommended_follow_up_lane:
        lines.append(f"recommended_follow_up_lane={bundle.recommended_follow_up_lane}")
    if bundle.cross_artifact_alignment is not None:
        alignment = bundle.cross_artifact_alignment
        lines.append(
            "cross_artifact_alignment="
            f"status={alignment.overall_status} "
            f"summary={alignment.summary or '—'}"
        )
    if bundle.dual_promotion_review is not None:
        dual = bundle.dual_promotion_review
        lines.append(
            "dual_promotion="
            f"posture={dual.posture} "
            f"warning={1 if dual.warning else 0} "
            f"kinds={','.join(dual.nightly_review_kinds) or '—'} "
            f"summary={dual.summary or '—'}"
        )
    if bundle.replay_linkage is not None:
        linkage = bundle.replay_linkage
        lines.append("replay_linkage=" + _format_replay_linkage(linkage))
    if bundle.recommended_cli_command:
        lines.append(f"recommended_command={bundle.recommended_cli_command}")
    if bundle.recommended_discovery_action:
        lines.append(f"recommended_discovery={bundle.recommended_discovery_action}")
    if bundle.recommended_discovery_cli_command:
        lines.append(
            f"recommended_discovery_command={bundle.recommended_discovery_cli_command}"
        )
    for check in bundle.checks:
        lines.append(f"[{check.status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def format_acceptance_bundle_markdown(bundle: AcceptanceBundle) -> str:
    lines = [
        "# Multi-Agent Acceptance Bundle",
        "",
        f"- Created at: `{bundle.created_at}`",
        f"- Overall status: `{bundle.overall_status}`",
    ]
    if bundle.nightly_report is not None:
        nightly = bundle.nightly_report
        review_label = _review_label(
            nightly.promotion_review_model_kind,
            nightly.promotion_review_model_kinds,
        )
        lines.extend(
            [
                f"- Nightly report: `{nightly.path}`",
                f"- Nightly status: `{nightly.overall_status or '—'}`",
                f"- Nightly basket size: `{len(nightly.basket)}`",
                (
                    "- Nightly execution posture: "
                    f"`target={nightly.training_execution_target or '—'}` "
                    f"`family={nightly.training_execution_model_family or '—'}` "
                    f"`source={nightly.training_execution_selection_source or '—'}`"
                ),
                (
                    "- Nightly training candidates: "
                    f"`{nightly.training_candidate_count}` "
                    f"(ml=`{nightly.training_candidate_ml_count}` "
                    f"rl=`{nightly.training_candidate_rl_count}`)"
                ),
                (
                    "- Nightly training research rows: "
                    f"`{nightly.training_research_count}` "
                    f"(ml=`{nightly.training_research_ml_count}` "
                    f"rl=`{nightly.training_research_rl_count}`)"
                ),
                (
                    "- Nightly discovery context: "
                    f"`preferred={nightly.training_research_discovery_preferred_count}` "
                    f"`regimes={nightly.training_research_discovery_regime_mix or '—'}`"
                    + (
                        " "
                        f"`target={nightly.training_research_discovery_recommended_refresh_target}`"
                        if nightly.training_research_discovery_recommended_refresh_target
                        else ""
                    )
                ),
                (
                    "- Nightly promotion reviews: "
                    f"`{len(nightly.promotion_review_paths)}` "
                    f"(kind=`{review_label}` "
                    f"promoted=`{nightly.promoted_model_kind or '—'}`)"
                ),
            ]
        )
        if nightly.training_research_discovery_summary:
            lines.append(
                f"- Nightly discovery summary: `{nightly.training_research_discovery_summary}`"
            )
        if (
            nightly.training_research_discovery_follow_up_target
            or nightly.training_research_research_follow_up_target
            or nightly.training_research_execution_follow_up_target
        ):
            lines.append(
                "- Nightly follow-up guidance: "
                + " ".join(
                    part
                    for part in (
                        (
                            f"`discovery={nightly.training_research_discovery_follow_up_target}`"
                            if nightly.training_research_discovery_follow_up_target
                            else ""
                        ),
                        (
                            f"`research={nightly.training_research_research_follow_up_target}`"
                            if nightly.training_research_research_follow_up_target
                            else ""
                        ),
                        (
                            f"`execution={nightly.training_research_execution_follow_up_target}`"
                            if nightly.training_research_execution_follow_up_target
                            else ""
                        ),
                    )
                    if part
                )
            )
    if bundle.recommended_refresh_target:
        lines.extend(
            [
                f"- Recommended remediation target: `{bundle.recommended_refresh_target}`",
                f"- Recommended force refresh: `{bundle.recommended_force_refresh}`",
            ]
        )
    if bundle.effective_refresh_target or bundle.refresh_target_source:
        lane_kwargs = {
            "nightly_report": bundle.nightly_report,
            "workflow_summary": bundle.workflow_snapshot,
        }
        refresh_context = (
            f"effective_target={bundle.effective_refresh_target or '—'} "
            f"source={bundle.refresh_target_source or '—'} "
            f"discovery={follow_up_lane_target(**lane_kwargs, lane='discovery') or '—'} "
            f"research={follow_up_lane_target(**lane_kwargs, lane='research') or '—'} "
            f"execution={follow_up_lane_target(**lane_kwargs, lane='execution') or '—'} "
            f"scouts={bundle.scout_support_target or '—'}"
        )
        if bundle.scout_support_summary:
            refresh_context += f" ({bundle.scout_support_summary})"
        lines.append(f"- Refresh context: `{refresh_context}`")
    if bundle.recommended_follow_up_lane:
        lines.append(f"- Recommended follow-up lane: `{bundle.recommended_follow_up_lane}`")
    if bundle.cross_artifact_alignment is not None:
        alignment = bundle.cross_artifact_alignment
        lines.append(
            "- Cross-artifact alignment: "
            f"`status={alignment.overall_status} summary={alignment.summary or '—'}`"
        )
    if bundle.dual_promotion_review is not None:
        dual = bundle.dual_promotion_review
        lines.append(
            "- Dual promotion review: "
            f"`posture={dual.posture} warning={dual.warning} "
            f"kinds={','.join(dual.nightly_review_kinds) or '—'} "
            f"summary={dual.summary or '—'}`"
        )
    if bundle.replay_linkage is not None:
        linkage = bundle.replay_linkage
        lines.append(f"- Replay linkage: `{_format_replay_linkage(linkage)}`")
    if bundle.recommended_cli_command:
        lines.append(f"- Recommended command: `{bundle.recommended_cli_command}`")
    if bundle.recommended_discovery_action:
        lines.append(f"- Recommended discovery follow-up: `{bundle.recommended_discovery_action}`")
    if bundle.recommended_discovery_cli_command:
        lines.append(
            f"- Recommended discovery command: `{bundle.recommended_discovery_cli_command}`"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
    )
    for check in bundle.checks:
        lines.append(f"| `{check.name}` | `{check.status}` | {check.detail} |")
    if bundle.workflow_snapshot is not None:
        wf = bundle.workflow_snapshot
        lines.extend(
            [
                "",
                "## Workflow Snapshot",
                "",
                f"- Path: `{wf.path}`",
                f"- Source: `{wf.source or '—'}`",
                f"- Timeframe: `{wf.timeframe or '—'}`",
                f"- Lookback: `{wf.lookback_days or 0}d`",
                f"- Universe: `{wf.universe_count}`",
                f"- Shortlist: `{wf.shortlist_count}`",
                f"- Briefing candidates: `{wf.briefing_candidates}`",
                (
                    f"- Allocation: `{wf.allocation_selected}/{wf.allocation_max_positions}` "
                    f"selected, `{wf.allocation_skipped}` skipped"
                ),
                f"- Training candidates: `ml={wf.ml_count}` `rl={wf.rl_count}`",
            ]
        )
        if wf.team_role_count > 0:
            lines.extend(
                [
                    (
                        f"- Team posture: `{wf.team_ok_role_count}/{wf.team_role_count}` "
                        "roles ready"
                    ),
                    (
                        "- Team nightly posture: "
                        f"`{wf.team_nightly_aligned_reports}/{wf.team_nightly_enabled_reports}` "
                        "aligned"
                    ),
                    (
                        "- Team remediation posture: "
                        f"`target={wf.team_nightly_recommended_target or '—'} "
                        f"force_refresh={1 if wf.team_nightly_recommended_force_refresh else 0}`"
                    ),
                ]
            )
            if wf.team_research_alignment_summary:
                lines.append(
                    "- Team basket/research posture: "
                    f"`{wf.team_research_alignment_summary}"
                    + (
                        f" mix={wf.team_research_alignment_target_mix}"
                        if wf.team_research_alignment_target_mix
                        else ""
                    )
                    + "`"
                )
            if wf.team_research_scout_support_summary:
                lines.append(
                    "- Team scout-supported research rows: "
                    f"`{wf.team_research_scout_support_summary}`"
                )
            if wf.team_discovery_alignment_summary:
                lines.append(
                    "- Team discovery posture: "
                    f"`{wf.team_discovery_alignment_summary}"
                    + (
                        f" overlap={wf.team_discovery_overlap_symbols}"
                        if wf.team_discovery_overlap_symbols
                        else ""
                    )
                    + "`"
                )
            if wf.team_discovery_refresh_requested:
                lines.append(
                    "- Team discovery refresh: `"
                    + f"refreshed={wf.team_discovery_refreshed_count}"
                    + (
                        f" source={wf.team_discovery_refresh_source}"
                        if wf.team_discovery_refresh_source
                        else ""
                    )
                    + (
                        f" timeframe={wf.team_discovery_refresh_timeframe}"
                        if wf.team_discovery_refresh_timeframe
                        else ""
                    )
                    + (
                        f" days={wf.team_discovery_refresh_days}"
                        if wf.team_discovery_refresh_days > 0
                        else ""
                    )
                    + "`"
                )
            if (
                wf.team_research_discovery_follow_up_target
                or wf.team_research_research_follow_up_target
                or wf.team_research_execution_follow_up_target
            ):
                lines.append(
                    "- Team follow-up guidance: "
                    + " ".join(
                        part
                        for part in (
                            (
                                f"`discovery={wf.team_research_discovery_follow_up_target}`"
                                if wf.team_research_discovery_follow_up_target
                                else ""
                            ),
                            (
                                f"`research={wf.team_research_research_follow_up_target}`"
                                if wf.team_research_research_follow_up_target
                                else ""
                            ),
                            (
                                f"`execution={wf.team_research_execution_follow_up_target}`"
                                if wf.team_research_execution_follow_up_target
                                else ""
                            ),
                        )
                        if part
                    )
                )
            if wf.team_nightly_recent_window > 0:
                lines.append(
                    "- Team recent trend: "
                    f"`{wf.team_nightly_recent_aligned}/{wf.team_nightly_recent_enabled} "
                    f"over {wf.team_nightly_recent_window} run(s) "
                    f"latest={wf.team_nightly_recent_latest_status or '—'} "
                    f"basket={wf.team_nightly_recent_latest_basket_size}`"
                )
        if wf.universe_top_adaptive_note:
            lines.append(f"- Adaptive note: `{wf.universe_top_adaptive_note}`")
    for title, review in (
        ("RL Promotion Review", bundle.rl_review),
        ("ML Promotion Review", bundle.ml_review),
    ):
        if review is None:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"- Run ID: `{review.record.run_id}`",
                f"- Symbol: `{review.record.symbol or '—'}`",
                f"- Pointer: `{review.pointer_path}`",
                f"- Advisory ready: `{review.record.advisory_ready}`",
                f"- Verdict passed: `{review.record.verdict_passed}`",
            ]
        )
        if review.workflow_summary_text:
            lines.append(f"- Workflow summary: `{review.workflow_summary_text}`")
    if bundle.model_health is not None:
        health = bundle.model_health
        lines.extend(
            [
                "",
                "## Model Health",
                "",
                f"- Registry enabled: `{health.registry_enabled}`",
                f"- Promotion required: `{health.promotion_required}`",
                f"- RL available: `{health.rl.available}`",
                f"- RL advisory ready: `{health.rl.advisory_ready}`",
                f"- ML available: `{health.ml.available}`",
                f"- ML enabled: `{health.ml.enabled}`",
                f"- Learning rows: `{health.agentic.learning_summary.total_rows}`",
                f"- Paper closed rows: `{health.agentic.learning_summary.paper_closed_rows}`",
            ]
        )
    return "\n".join(lines) + "\n"


def export_acceptance_bundle(
    bundle: AcceptanceBundle,
    out_path: Path | str,
    *,
    fmt: str,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    elif fmt == "md":
        path.write_text(format_acceptance_bundle_markdown(bundle), encoding="utf-8")
    else:
        raise ValueError(f"unsupported acceptance bundle format: {fmt}")
    return path


def build_and_export_acceptance_bundle(
    *,
    settings,
    out_dir: Path | str,
    basename: str = "acceptance_bundle",
    engine: Any | None = None,
    models_root: Path | str = "models",
    ml_base: Optional[Path | str] = None,
    symbol: Optional[str] = None,
    nightly_report_path: Optional[Path | str] = None,
    nightly_report_dir: Optional[Path | str] = None,
    workflow_snapshot_path: Optional[Path | str] = None,
    include_model_health: bool = True,
) -> tuple[AcceptanceBundle, Path, Path]:
    with workflow_boundary(
        settings,
        event_name="acceptance_bundle",
        module="fortuna.app.acceptance_bundle",
        workflow_id="acceptance_bundle",
        symbol=(symbol or "").upper().strip() or None,
    ) as span:
        bundle = gather_acceptance_bundle(
            settings=settings,
            engine=engine,
            models_root=models_root,
            ml_base=ml_base,
            symbol=symbol,
            nightly_report_path=nightly_report_path,
            nightly_report_dir=nightly_report_dir,
            workflow_snapshot_path=workflow_snapshot_path,
            include_model_health=include_model_health,
        )
        target_dir = Path(out_dir)
        md_path = export_acceptance_bundle(bundle, target_dir / f"{basename}.md", fmt="md")
        json_path = export_acceptance_bundle(bundle, target_dir / f"{basename}.json", fmt="json")
        alignment = bundle.cross_artifact_alignment
        span.set_context(
            md_path=str(md_path),
            json_path=str(json_path),
            alignment_status=getattr(alignment, "overall_status", None) if alignment else None,
        )
        return bundle, md_path, json_path


def load_nightly_report_evidence(
    *,
    nightly_report_path: Optional[Path | str] = None,
    nightly_report_dir: Optional[Path | str] = None,
) -> Optional[NightlyReportEvidence]:
    path = _resolve_nightly_report_path(
        nightly_report_path=nightly_report_path,
        nightly_report_dir=nightly_report_dir,
    )
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    steps = payload.get("steps")
    step_rows = steps if isinstance(steps, list) else []
    basket_detail = _step_detail(step_rows, "basket_resolution")
    candidate_detail = _step_detail(step_rows, "training_candidates")
    research_detail = _step_detail(step_rows, "training_research")
    execution_detail = _step_detail(step_rows, "training_execution_target")
    workflow_detail = _step_detail(step_rows, "workflow_snapshot")
    review_detail = _step_detail(step_rows, "promotion_reviews")
    promote_detail = _step_detail(step_rows, "promote_best")
    promotion_paths: tuple[str, ...] = ()
    if isinstance(review_detail, dict):
        raw_paths = review_detail.get("paths")
        if isinstance(raw_paths, list):
            promotion_paths = tuple(str(row) for row in raw_paths if str(row or "").strip())
    workflow_path = None
    if isinstance(workflow_detail, dict):
        raw_path = str(workflow_detail.get("path", "") or "").strip()
        workflow_path = raw_path or None
    workflow_exists = (
        _resolve_report_relative_path(path, workflow_path).is_file()
        if workflow_path
        else False
    )
    training_candidate_path = None
    training_candidate_exists = False
    training_candidate_count = 0
    training_candidate_ml_count = 0
    training_candidate_rl_count = 0
    training_candidate_selection_policy = None
    if isinstance(candidate_detail, dict):
        raw_path = str(candidate_detail.get("path", "") or "").strip()
        training_candidate_path = raw_path or None
        training_candidate_exists = (
            _resolve_report_relative_path(path, training_candidate_path).is_file()
            if training_candidate_path
            else False
        )
        training_candidate_count = _coerce_int(candidate_detail.get("count")) or 0
        training_candidate_ml_count = _coerce_int(candidate_detail.get("ml_count")) or 0
        training_candidate_rl_count = _coerce_int(candidate_detail.get("rl_count")) or 0
        raw_policy = str(candidate_detail.get("selection_policy", "") or "").strip()
        training_candidate_selection_policy = raw_policy or None
    training_research_path = None
    training_research_exists = False
    training_research_count = 0
    training_research_ml_count = 0
    training_research_rl_count = 0
    training_research_discovery_preferred_count = 0
    training_research_discovery_regime_mix = None
    training_research_discovery_summary = None
    training_research_discovery_recommended_refresh_target = None
    training_research_effective_refresh_target = None
    training_research_scout_support_target = None
    training_research_discovery_follow_up_target = None
    training_research_discovery_follow_up_summary = None
    training_research_discovery_follow_up_action = None
    training_research_research_follow_up_target = None
    training_research_research_follow_up_summary = None
    training_research_research_follow_up_action = None
    training_research_execution_follow_up_target = None
    training_research_execution_follow_up_summary = None
    training_research_execution_follow_up_action = None
    training_execution_target = None
    training_execution_model_family = None
    training_execution_selection_source = None
    if isinstance(research_detail, dict):
        raw_path = str(research_detail.get("path", "") or "").strip()
        training_research_path = raw_path or None
        training_research_exists = (
            _resolve_report_relative_path(path, training_research_path).is_file()
            if training_research_path
            else False
        )
        training_research_count = _coerce_int(research_detail.get("count")) or 0
        training_research_ml_count = _coerce_int(research_detail.get("ml_count")) or 0
        training_research_rl_count = _coerce_int(research_detail.get("rl_count")) or 0
        training_research_discovery_preferred_count = (
            _coerce_int(research_detail.get("discovery_preferred_count")) or 0
        )
        raw_regime_mix = str(research_detail.get("discovery_regime_mix", "") or "").strip()
        training_research_discovery_regime_mix = raw_regime_mix or None
        raw_discovery_summary = str(research_detail.get("discovery_summary", "") or "").strip()
        training_research_discovery_summary = raw_discovery_summary or None
        raw_discovery_target = str(
            research_detail.get("discovery_recommended_refresh_target", "") or ""
        ).strip()
        training_research_discovery_recommended_refresh_target = raw_discovery_target or None
        raw_effective_target = str(
            research_detail.get("effective_refresh_target", "") or ""
        ).strip()
        training_research_effective_refresh_target = raw_effective_target or None
        raw_scout_target = str(
            research_detail.get("scout_support_target", "") or ""
        ).strip()
        training_research_scout_support_target = raw_scout_target or None
        training_research_discovery_follow_up_target = (
            str(research_detail.get("discovery_follow_up_target", "") or "").strip() or None
        )
        training_research_discovery_follow_up_summary = (
            str(research_detail.get("discovery_follow_up_summary", "") or "").strip() or None
        )
        training_research_discovery_follow_up_action = (
            str(research_detail.get("discovery_follow_up_action", "") or "").strip() or None
        )
        training_research_research_follow_up_target = (
            str(research_detail.get("research_follow_up_target", "") or "").strip() or None
        )
        training_research_research_follow_up_summary = (
            str(research_detail.get("research_follow_up_summary", "") or "").strip() or None
        )
        training_research_research_follow_up_action = (
            str(research_detail.get("research_follow_up_action", "") or "").strip() or None
        )
        training_research_execution_follow_up_target = (
            str(research_detail.get("execution_follow_up_target", "") or "").strip() or None
        )
        training_research_execution_follow_up_summary = (
            str(research_detail.get("execution_follow_up_summary", "") or "").strip() or None
        )
        training_research_execution_follow_up_action = (
            str(research_detail.get("execution_follow_up_action", "") or "").strip() or None
        )
    if isinstance(execution_detail, dict):
        raw_execution_target = str(execution_detail.get("target", "") or "").strip().lower()
        if raw_execution_target in {"all", "ml", "rl"}:
            training_execution_target = raw_execution_target
        raw_run_ml = bool(execution_detail.get("run_ml"))
        raw_run_rl = bool(execution_detail.get("run_rl"))
        if raw_run_ml and not raw_run_rl:
            training_execution_model_family = "ml"
        elif raw_run_rl and not raw_run_ml:
            training_execution_model_family = "rl"
        elif raw_run_ml and raw_run_rl:
            training_execution_model_family = "hybrid"
        raw_selection_source = str(execution_detail.get("selection_source", "") or "").strip()
        training_execution_selection_source = raw_selection_source or None
    review_exists = tuple(
        _resolve_report_relative_path(path, row).is_file()
        for row in promotion_paths
    )
    basket = payload.get("basket")
    basket_mode = None
    if isinstance(basket_detail, dict):
        raw_mode = str(basket_detail.get("mode", "") or "").strip()
        basket_mode = raw_mode or None
    lane_detail = _step_detail(step_rows, "lane_readiness")
    lane_global_blockers: tuple[str, ...] = ()
    lane_should_abort = False
    if isinstance(lane_detail, dict):
        raw_blockers = lane_detail.get("global_blockers")
        if isinstance(raw_blockers, list):
            lane_global_blockers = tuple(
                str(row).strip() for row in raw_blockers if str(row or "").strip()
            )
        lane_should_abort = bool(lane_detail.get("should_abort"))
    step_statuses = {
        str(row.get("name", "")).strip(): str(row.get("status", "")).strip()
        for row in step_rows
        if isinstance(row, dict) and str(row.get("name", "")).strip()
    }
    if isinstance(promote_detail, dict) and "count" in promote_detail:
        step_statuses["promote_best_count"] = str(promote_detail.get("count", ""))
    promoted_model_kind = None
    if isinstance(promote_detail, dict):
        raw_promote_kind = str(promote_detail.get("model_kind", "") or "").strip().lower()
        promoted_model_kind = raw_promote_kind or None
    promotion_review_model_kind = None
    promotion_review_model_kinds: tuple[str, ...] = ()
    if isinstance(review_detail, dict):
        raw_review_kinds = review_detail.get("model_kinds")
        if isinstance(raw_review_kinds, list):
            promotion_review_model_kinds = tuple(
                kind
                for kind in (
                    str(item or "").strip().lower() for item in raw_review_kinds
                )
                if kind
            )
        raw_review_kind = str(review_detail.get("model_kind", "") or "").strip().lower()
        promotion_review_model_kind = raw_review_kind or None
        if promotion_review_model_kind is None and promotion_review_model_kinds:
            promotion_review_model_kind = promotion_review_model_kinds[0]
    return NightlyReportEvidence(
        path=str(path),
        overall_status=str(payload.get("overall_status", "") or ""),
        basket=tuple(str(row) for row in basket) if isinstance(basket, list) else (),
        basket_mode=basket_mode,
        training_execution_target=training_execution_target,
        training_execution_model_family=training_execution_model_family,
        training_execution_selection_source=training_execution_selection_source,
        training_candidate_path=training_candidate_path,
        training_candidate_exists=training_candidate_exists,
        training_candidate_count=training_candidate_count,
        training_candidate_ml_count=training_candidate_ml_count,
        training_candidate_rl_count=training_candidate_rl_count,
        training_candidate_selection_policy=training_candidate_selection_policy,
        training_research_path=training_research_path,
        training_research_exists=training_research_exists,
        training_research_count=training_research_count,
        training_research_ml_count=training_research_ml_count,
        training_research_rl_count=training_research_rl_count,
        training_research_discovery_preferred_count=training_research_discovery_preferred_count,
        training_research_discovery_regime_mix=training_research_discovery_regime_mix,
        training_research_discovery_summary=training_research_discovery_summary,
        training_research_discovery_recommended_refresh_target=(
            training_research_discovery_recommended_refresh_target
        ),
        training_research_effective_refresh_target=training_research_effective_refresh_target,
        training_research_scout_support_target=training_research_scout_support_target,
        training_research_discovery_follow_up_target=training_research_discovery_follow_up_target,
        training_research_discovery_follow_up_summary=training_research_discovery_follow_up_summary,
        training_research_discovery_follow_up_action=training_research_discovery_follow_up_action,
        training_research_research_follow_up_target=training_research_research_follow_up_target,
        training_research_research_follow_up_summary=training_research_research_follow_up_summary,
        training_research_research_follow_up_action=training_research_research_follow_up_action,
        training_research_execution_follow_up_target=training_research_execution_follow_up_target,
        training_research_execution_follow_up_summary=training_research_execution_follow_up_summary,
        training_research_execution_follow_up_action=training_research_execution_follow_up_action,
        workflow_snapshot_path=workflow_path,
        workflow_snapshot_exists=workflow_exists,
        promoted_model_kind=promoted_model_kind,
        promotion_review_paths=promotion_paths,
        promotion_review_exists=review_exists,
        promotion_review_model_kind=promotion_review_model_kind,
        promotion_review_model_kinds=promotion_review_model_kinds,
        step_statuses=step_statuses,
        lane_readiness_global_blockers=lane_global_blockers,
        lane_readiness_should_abort=lane_should_abort,
    )


def _build_checks(
    *,
    settings,
    nightly_report: Optional[NightlyReportEvidence],
    workflow_summary: Optional[WorkflowSnapshotSummary],
    rl_review: Optional[PromotionReview],
    ml_review: Optional[PromotionReview],
    model_health: Optional[ModelHealthResponse],
    dual_promotion_review: Optional[DualPromotionReviewSummary] = None,
    linkage_summary: Optional[NightlyArtifactLinkageSummary] = None,
) -> tuple[AcceptanceCheck, ...]:
    checks: list[AcceptanceCheck] = []
    if nightly_report is None:
        checks.append(
            AcceptanceCheck(
                name="nightly_report",
                status="warn",
                detail="No nightly report evidence was supplied to this acceptance bundle.",
            )
        )
    else:
        review_label = _review_label(
            nightly_report.promotion_review_model_kind,
            nightly_report.promotion_review_model_kinds,
        )
        checks.append(
            AcceptanceCheck(
                name="nightly_report",
                status="pass",
                detail=(
                    f"Nightly report status={nightly_report.overall_status or '—'} "
                    f"target={nightly_report.training_execution_target or '—'} "
                    f"family={nightly_report.training_execution_model_family or '—'} "
                    f"source={nightly_report.training_execution_selection_source or '—'} "
                    f"basket={len(nightly_report.basket)} "
                    f"training_candidates={nightly_report.training_candidate_count} "
                    f"review_artifacts={len(nightly_report.promotion_review_paths)} "
                    f"review_kind={review_label}."
                ),
            )
        )
        if (
            str(nightly_report.overall_status or "").strip().lower() == "ok"
            and nightly_report.lane_readiness_global_blockers
        ):
            checks.append(
                AcceptanceCheck(
                    name="nightly_lane_readiness",
                    status="warn",
                    detail=(
                        "Nightly report overall_status=ok but lane readiness recorded "
                        "global blockers: "
                        + ", ".join(nightly_report.lane_readiness_global_blockers)
                    ),
                )
            )
        elif nightly_report.lane_readiness_global_blockers:
            checks.append(
                AcceptanceCheck(
                    name="nightly_lane_readiness",
                    status="pass",
                    detail=(
                        "Lane readiness blockers recorded: "
                        + ", ".join(nightly_report.lane_readiness_global_blockers)
                    ),
                )
            )
        if nightly_report.basket_mode in {"candidate_manifest", "derived_training_candidates"}:
            checks.append(_candidate_nightly_shape_check(nightly_report))
            checks.append(
                AcceptanceCheck(
                    name="nightly_training_candidates_reference",
                    status="pass" if nightly_report.training_candidate_exists else "fail",
                    detail=(
                        "Nightly report references a training-candidate artifact and it exists."
                        if nightly_report.training_candidate_exists
                        else (
                            "Nightly report is missing a training-candidate artifact "
                            "reference or the referenced file is missing."
                        )
                    ),
                )
            )
            checks.append(
                AcceptanceCheck(
                    name="nightly_training_research_reference",
                    status="pass" if nightly_report.training_research_exists else "fail",
                    detail=(
                        "Nightly report references a training-research artifact and it exists."
                        if nightly_report.training_research_exists
                        else (
                            "Nightly report is missing a training-research artifact "
                            "reference or the referenced file is missing."
                        )
                    ),
                )
            )
        if nightly_report.workflow_snapshot_path:
            checks.append(
                AcceptanceCheck(
                    name="nightly_workflow_reference",
                    status="pass" if nightly_report.workflow_snapshot_exists else "fail",
                    detail=(
                        "Nightly report references a workflow snapshot artifact "
                        "and it exists."
                        if nightly_report.workflow_snapshot_exists
                        else (
                            "Nightly report references a workflow snapshot "
                            "artifact, but the file is missing."
                        )
                    ),
                )
            )
        else:
            checks.append(
                AcceptanceCheck(
                    name="nightly_workflow_reference",
                    status="fail",
                    detail="Nightly report does not reference a workflow snapshot artifact.",
                )
            )
    if workflow_summary is None:
        checks.append(
            AcceptanceCheck(
                name="workflow_snapshot",
                status="fail",
                detail="No workflow snapshot is available for acceptance review.",
            )
        )
    else:
        checks.append(
            AcceptanceCheck(
                name="workflow_snapshot",
                status="pass",
                detail=(
                    f"Universe={workflow_summary.universe_count}, "
                    f"shortlist={workflow_summary.shortlist_count}, "
                    "allocation="
                    f"{workflow_summary.allocation_selected}/"
                    f"{workflow_summary.allocation_max_positions}, "
                    f"ml={workflow_summary.ml_count}, rl={workflow_summary.rl_count}."
                    + (
                        " training_research="
                        f"{workflow_summary.research_count}"
                        f" (ml={workflow_summary.research_ml_count} "
                        f"rl={workflow_summary.research_rl_count})"
                        + (
                            f" refreshed={workflow_summary.research_refreshed_count}."
                            if workflow_summary.research_refresh_requested
                            else "."
                        )
                        if (
                            workflow_summary.research_count > 0
                            or workflow_summary.research_refresh_requested
                        )
                        else ""
                    )
                    + (
                        " allocation_research="
                        f"{workflow_summary.allocation_research_target_mix}."
                        if workflow_summary.allocation_research_target_mix
                        else ""
                    )
                    + (
                        " refreshed_allocation_research="
                        f"{workflow_summary.allocation_refreshed_research_target_mix}."
                        if workflow_summary.allocation_refreshed_research_target_mix
                        else ""
                    )
                ),
            )
        )
        if nightly_report is not None:
            checks.append(
                _nightly_workflow_execution_consistency_check(
                    nightly_report=nightly_report,
                    workflow_summary=workflow_summary,
                )
            )
        if workflow_summary.universe_count <= 0 or workflow_summary.shortlist_count <= 0:
            checks.append(
                AcceptanceCheck(
                    name="workflow_population",
                    status="fail",
                    detail=(
                        "Workflow snapshot does not contain a populated "
                        "universe/shortlist stage."
                    ),
                )
            )
        else:
            checks.append(
                AcceptanceCheck(
                    name="workflow_population",
                    status="pass",
                    detail="Workflow snapshot carries populated universe and shortlist stages.",
                )
            )
        if workflow_summary.allocation_research_alignment_enabled:
            checks.append(
                AcceptanceCheck(
                    name="allocation_research_alignment",
                    status=(
                        "pass"
                        if workflow_summary.allocation_research_target_mix
                        else "warn"
                    ),
                    detail=(
                        "Workflow snapshot shows allocation research alignment: "
                        f"{workflow_summary.allocation_research_target_mix}."
                        if workflow_summary.allocation_research_target_mix
                        else (
                            "Workflow snapshot shows research alignment enabled, "
                            "but no allocation research-target mix was captured."
                        )
                    ),
                )
            )
        if workflow_summary.research_refresh_requested:
            checks.append(
                AcceptanceCheck(
                    name="training_research_refresh",
                    status=(
                        "pass"
                        if workflow_summary.research_refreshed_count > 0
                        else "warn"
                    ),
                    detail=(
                        "Workflow snapshot shows training-research refresh requested: "
                        f"target={workflow_summary.research_refresh_target or '—'}, "
                        f"refreshed={workflow_summary.research_refreshed_count}, "
                        f"rows={workflow_summary.research_count}."
                    ),
                )
            )
        if workflow_summary.team_role_count > 0:
            checks.append(
                AcceptanceCheck(
                    name="multi_agent_team_posture",
                    status=(
                        "pass"
                        if workflow_summary.team_nightly_report_count > 0
                        else "warn"
                    ),
                    detail=(
                        "Workflow snapshot preserves multi-agent team posture: "
                        f"roles={workflow_summary.team_ok_role_count}/"
                        f"{workflow_summary.team_role_count}, "
                        f"nightly={workflow_summary.team_nightly_aligned_reports}/"
                        f"{workflow_summary.team_nightly_enabled_reports}, "
                        f"target={workflow_summary.team_nightly_recommended_target or '—'}, "
                        "force_refresh="
                        f"{1 if workflow_summary.team_nightly_recommended_force_refresh else 0}."
                        + (
                            " recent="
                            f"{workflow_summary.team_nightly_recent_aligned}/"
                            f"{workflow_summary.team_nightly_recent_enabled}"
                            f" over {workflow_summary.team_nightly_recent_window} run(s)"
                            f" latest={workflow_summary.team_nightly_recent_latest_status or '—'}"
                            f":{workflow_summary.team_nightly_recent_latest_basket_size}."
                            if workflow_summary.team_nightly_recent_window > 0
                            else ""
                        )
                        + (
                            " basket_research="
                            f"{workflow_summary.team_research_alignment_summary}"
                            + (
                                f" mix={workflow_summary.team_research_alignment_target_mix}"
                                if workflow_summary.team_research_alignment_target_mix
                                else ""
                            )
                            + "."
                            if workflow_summary.team_research_alignment_summary
                            else ""
                        )
                        + (
                            " discovery="
                            f"{workflow_summary.team_discovery_alignment_summary}"
                            + (
                                f" overlap={workflow_summary.team_discovery_overlap_symbols}"
                                if workflow_summary.team_discovery_overlap_symbols
                                else ""
                            )
                            + "."
                            if workflow_summary.team_discovery_alignment_summary
                            else ""
                        )
                    ),
                )
            )
        if workflow_summary.team_discovery_refresh_requested:
            checks.append(
                AcceptanceCheck(
                    name="workflow_discovery_refresh",
                    status=(
                        "pass"
                        if workflow_summary.team_discovery_refreshed_count > 0
                        else "warn"
                    ),
                    detail=(
                        "Workflow snapshot shows discovery refresh requested: "
                        f"source={workflow_summary.team_discovery_refresh_source or '—'}, "
                        f"timeframe={workflow_summary.team_discovery_refresh_timeframe or '—'}, "
                        f"days={workflow_summary.team_discovery_refresh_days or 0}, "
                        f"refreshed={workflow_summary.team_discovery_refreshed_count}."
                    ),
                )
            )
        if workflow_summary.team_research_alignment_selected_count > 0:
            overlap = int(workflow_summary.team_research_alignment_overlap_count or 0)
            selected = int(workflow_summary.team_research_alignment_selected_count or 0)
            ratio = float(overlap) / float(selected) if selected > 0 else 0.0
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
            checks.append(
                AcceptanceCheck(
                    name="workflow_basket_research_alignment",
                    status="pass" if ratio >= warn_ratio else "warn",
                    detail=(
                        "Workflow snapshot shows basket/research alignment: "
                        f"ratio={ratio:.2f} ({overlap}/{selected}) "
                        f"{workflow_summary.team_research_alignment_summary or '—'}"
                        + (
                            f" mix={workflow_summary.team_research_alignment_target_mix}."
                            if workflow_summary.team_research_alignment_target_mix
                            else "."
                        )
                        + (
                            " recommended_target="
                            f"{_workflow_research_alignment_target(workflow_summary)}."
                            if _workflow_research_alignment_target(workflow_summary)
                            else ""
                        )
                    ),
                )
            )
        if workflow_summary.team_discovery_alignment_compare_count > 0:
            overlap = int(workflow_summary.team_discovery_alignment_overlap_count or 0)
            compare_count = int(workflow_summary.team_discovery_alignment_compare_count or 0)
            ratio = float(overlap) / float(compare_count) if compare_count > 0 else 0.0
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
            checks.append(
                AcceptanceCheck(
                    name="workflow_discovery_alignment",
                    status="pass" if ratio >= warn_ratio else "warn",
                    detail=(
                        "Workflow snapshot shows discovery alignment: "
                        f"ratio={ratio:.2f} ({overlap}/{compare_count}) "
                        f"{workflow_summary.team_discovery_alignment_summary or '—'}"
                        + (
                            f" overlap={workflow_summary.team_discovery_overlap_symbols}."
                            if workflow_summary.team_discovery_overlap_symbols
                            else "."
                        )
                    ),
                )
            )

    reviews = [row for row in (rl_review, ml_review) if row is not None]
    if not reviews:
        checks.append(
            AcceptanceCheck(
                name="promotion_reviews",
                status="warn",
                detail="No promoted RL/ML artifact review is available yet.",
            )
        )
    else:
        checks.append(
            AcceptanceCheck(
                name="promotion_reviews",
                status="pass",
                detail="Promotion review evidence exists for "
                + ", ".join(review.record.model_kind.value for review in reviews)
                + ".",
            )
        )

    if nightly_report is not None and nightly_report.promotion_review_paths:
        missing_review_count = sum(
            1 for row in nightly_report.promotion_review_exists if not row
        )
        review_label = _review_label(
            nightly_report.promotion_review_model_kind,
            nightly_report.promotion_review_model_kinds,
        )
        checks.append(
            AcceptanceCheck(
                name="nightly_review_artifacts",
                status="pass" if missing_review_count == 0 else "fail",
                detail=(
                    "Nightly report references "
                    f"{len(nightly_report.promotion_review_paths)} promotion-review artifact(s)"
                    + (
                        f" for {review_label}."
                        if (
                            nightly_report.promotion_review_model_kind
                            or nightly_report.promotion_review_model_kinds
                        )
                        else "."
                    )
                    + (
                        " They all exist."
                        if missing_review_count == 0
                        else f" But {missing_review_count} file(s) are missing."
                    )
                ),
            )
        )

    if dual_promotion_review is not None and dual_promotion_review.both_present:
        status = "warn" if dual_promotion_review.warning else "pass"
        checks.append(
            AcceptanceCheck(
                name="dual_promotion_posture",
                status=status,
                detail=dual_promotion_review.summary or dual_promotion_review.posture,
            )
        )
    elif dual_promotion_review is not None and dual_promotion_review.posture == "single_lane":
        checks.append(
            AcceptanceCheck(
                name="dual_promotion_posture",
                status="pass",
                detail=dual_promotion_review.summary or "Single-lane promotion review posture.",
            )
        )

    if linkage_summary is not None:
        linkage_status = str(linkage_summary.overall_status or "unknown").strip().lower()
        if linkage_status == "linked":
            linkage_check_status = "pass"
        elif linkage_status == "partial":
            linkage_check_status = "warn"
        elif linkage_status == "broken":
            linkage_check_status = "fail"
        else:
            linkage_check_status = "warn"
        checks.append(
            AcceptanceCheck(
                name="nightly_replay_linkage",
                status=linkage_check_status,
                detail=linkage_summary.summary or f"replay linkage status={linkage_status}",
            )
        )

    if workflow_summary is not None and reviews:
        missing_link = [
            review.record.model_kind.value
            for review in reviews
            if not str(review.record.workflow_snapshot_path or "").strip()
        ]
        mismatched = [
            review.record.model_kind.value
            for review in reviews
            if str(review.record.workflow_snapshot_path or "").strip()
            and str(review.record.workflow_snapshot_path).strip() != workflow_summary.path
        ]
        if missing_link:
            checks.append(
                AcceptanceCheck(
                    name="promotion_workflow_linkage",
                    status="fail",
                    detail=(
                        "Promotion reviews are missing a linked workflow snapshot for "
                        + ", ".join(missing_link)
                        + "."
                    ),
                )
            )
        elif mismatched:
            checks.append(
                AcceptanceCheck(
                    name="promotion_workflow_linkage",
                    status="fail",
                    detail=(
                        "Promotion reviews reference a different workflow snapshot for "
                        + ", ".join(mismatched)
                        + "."
                    ),
                )
            )
        else:
            checks.append(
                AcceptanceCheck(
                    name="promotion_workflow_linkage",
                    status="pass",
                    detail="Promotion reviews point to the same workflow snapshot being accepted.",
                )
            )

    if model_health is None:
        checks.append(
            AcceptanceCheck(
                name="model_health_visibility",
                status="warn",
                detail="Model-health visibility was not captured in this acceptance bundle.",
            )
        )
    elif workflow_summary is None:
        checks.append(
            AcceptanceCheck(
                name="model_health_visibility",
                status="pass",
                detail=(
                    "Model-health snapshot was captured, but workflow linkage "
                    "could not be checked."
                ),
            )
        )
    else:
        linked = False
        for promotion in (
            model_health.rl.last_promotion,
            model_health.ml.last_promotion,
        ):
            if promotion and promotion.workflow_snapshot is not None:
                if promotion.workflow_snapshot.path == workflow_summary.path:
                    linked = True
                    break
        checks.append(
            AcceptanceCheck(
                name="model_health_visibility",
                status="pass" if linked else "warn",
                detail=(
                    "Model-health surfaces show linked workflow context."
                    if linked
                    else (
                        "Model-health was captured, but the linked workflow "
                        "snapshot is not visible in the current promotion payload."
                    )
                ),
            )
        )
        nightly_alignment = model_health.agentic.nightly_alignment
        min_enabled_reports = max(
            1,
            int(
                getattr(
                    settings,
                    "acceptance_bundle_nightly_alignment_min_enabled_reports",
                    2,
                )
            ),
        )
        warn_ratio = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        settings,
                        "acceptance_bundle_nightly_alignment_warn_ratio",
                        0.6,
                    )
                ),
            ),
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
        if nightly_alignment.enabled_reports >= min_enabled_reports:
            ratio = (
                float(nightly_alignment.aligned_reports) / float(nightly_alignment.enabled_reports)
                if nightly_alignment.enabled_reports > 0
                else 0.0
            )
            checks.append(
                AcceptanceCheck(
                    name="recent_nightly_alignment",
                    status="pass" if ratio >= warn_ratio else "warn",
                    detail=(
                        "Recent nightly allocation-research alignment ratio="
                        f"{ratio:.2f} ({nightly_alignment.aligned_reports}/"
                        f"{nightly_alignment.enabled_reports})"
                        + (
                            f" latest={nightly_alignment.latest_target_mix}."
                            if nightly_alignment.latest_target_mix
                            else "."
                        )
                        + (
                            f" refreshed={nightly_alignment.latest_refreshed_target_mix}."
                            if nightly_alignment.latest_refreshed_target_mix
                            else ""
                        )
                    ),
                )
            )
            refreshed_ratio = (
                float(nightly_alignment.refreshed_aligned_reports)
                / float(nightly_alignment.enabled_reports)
                if nightly_alignment.enabled_reports > 0
                else 0.0
            )
            checks.append(
                AcceptanceCheck(
                    name="recent_refreshed_nightly_alignment",
                    status="pass" if refreshed_ratio >= refreshed_warn_ratio else "warn",
                    detail=(
                        "Recent refreshed allocation-research alignment ratio="
                        f"{refreshed_ratio:.2f} ({nightly_alignment.refreshed_aligned_reports}/"
                        f"{nightly_alignment.enabled_reports})"
                        + (
                            f" latest={nightly_alignment.latest_refreshed_target_mix}."
                            if nightly_alignment.latest_refreshed_target_mix
                            else "."
                        )
                        + (
                            f" recommended_target={nightly_alignment.recommended_refresh_target}."
                            if nightly_alignment.recommended_refresh_target
                            else ""
                        )
                        + (
                            " force_refresh=1."
                            if nightly_alignment.recommended_force_refresh
                            else ""
                        )
                    ),
                )
            )

    return tuple(checks)


def _resolve_nightly_report_path(
    *,
    nightly_report_path: Optional[Path | str],
    nightly_report_dir: Optional[Path | str],
) -> Optional[Path]:
    if nightly_report_path is not None and str(nightly_report_path).strip():
        return Path(str(nightly_report_path))
    if nightly_report_dir is None or not str(nightly_report_dir).strip():
        return None
    root = Path(str(nightly_report_dir))
    if not root.is_dir():
        return None
    candidates = sorted(
        root.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _step_detail(steps: list[Any], name: str) -> Optional[dict[str, Any]]:
    for row in steps:
        if not isinstance(row, dict):
            continue
        if str(row.get("name", "")).strip() != name:
            continue
        detail = row.get("detail")
        return detail if isinstance(detail, dict) else None
    return None


def _candidate_nightly_shape_check(nightly_report: NightlyReportEvidence) -> AcceptanceCheck:
    statuses = dict(nightly_report.step_statuses or {})
    missing: list[str] = []
    failed: list[str] = []
    for name in (
        "basket_resolution",
        "training_candidates",
        "training_research",
        "training_execution_target",
        "workflow_snapshot",
        "promote_best",
    ):
        status = statuses.get(name, "")
        if not status:
            missing.append(name)
        elif status == "fail":
            failed.append(name)
    promote_count_text = statuses.get("promote_best_count", "").strip()
    promote_count = _coerce_int(promote_count_text) or 0
    review_status = statuses.get("promotion_reviews", "")
    if promote_count > 0:
        if not review_status:
            missing.append("promotion_reviews")
        elif review_status == "fail":
            failed.append("promotion_reviews")
    if missing or failed:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append("missing=" + ",".join(missing))
        if failed:
            detail_parts.append("failed=" + ",".join(failed))
        return AcceptanceCheck(
            name="candidate_nightly_shape",
            status="fail",
            detail=(
                "Candidate-driven nightly report is missing expected step coverage: "
                + " ".join(detail_parts)
            ),
        )
    return AcceptanceCheck(
        name="candidate_nightly_shape",
        status="pass",
        detail=(
            f"Candidate-driven nightly mode `{nightly_report.basket_mode}` carries "
            "basket/training-candidate/training-research/workflow/promotion steps."
        ),
    )


def _review_label(
    review_kind: Optional[str],
    review_kinds: tuple[str, ...],
) -> str:
    normalized = tuple(
        str(kind or "").strip() for kind in review_kinds if str(kind or "").strip()
    )
    if normalized:
        return ", ".join(normalized)
    kind = str(review_kind or "").strip()
    return kind or "—"


def _nightly_workflow_execution_consistency_check(
    *,
    nightly_report: NightlyReportEvidence,
    workflow_summary: WorkflowSnapshotSummary,
) -> AcceptanceCheck:
    comparisons = (
        (
            "target",
            nightly_report.training_execution_target,
            workflow_summary.team_nightly_latest_execution_target,
        ),
        (
            "family",
            nightly_report.training_execution_model_family,
            workflow_summary.team_nightly_latest_execution_model_family,
        ),
        (
            "source",
            nightly_report.training_execution_selection_source,
            workflow_summary.team_nightly_latest_execution_selection_source,
        ),
        (
            "review_kind",
            nightly_report.promotion_review_model_kind,
            workflow_summary.team_nightly_latest_promotion_review_model_kind,
        ),
        (
            "review_kinds",
            ", ".join(nightly_report.promotion_review_model_kinds),
            workflow_summary.team_nightly_latest_promotion_review_model_kinds,
        ),
    )
    matched: list[str] = []
    mismatched: list[str] = []
    for label, nightly_value, workflow_value in comparisons:
        nightly_text = str(nightly_value or "").strip()
        workflow_text = str(workflow_value or "").strip()
        if not nightly_text or not workflow_text:
            continue
        if nightly_text == workflow_text:
            matched.append(f"{label}={nightly_text}")
        else:
            mismatched.append(
                f"{label} nightly={nightly_text} workflow={workflow_text}"
            )
    if mismatched:
        return AcceptanceCheck(
            name="nightly_workflow_execution_consistency",
            status="fail",
            detail=(
                "Nightly execution provenance disagrees with the workflow snapshot: "
                + "; ".join(mismatched)
                + "."
            ),
        )
    if matched:
        return AcceptanceCheck(
            name="nightly_workflow_execution_consistency",
            status="pass",
            detail=(
                "Nightly execution provenance matches the workflow snapshot for "
                + ", ".join(matched)
                + "."
            ),
        )
    return AcceptanceCheck(
        name="nightly_workflow_execution_consistency",
        status="warn",
        detail=(
            "Nightly and workflow artifacts were present, but no shared execution "
            "provenance fields were populated strongly enough to compare."
        ),
    )


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_report_relative_path(report_path: Path, artifact_path: str) -> Path:
    candidate = Path(str(artifact_path))
    if candidate.is_absolute():
        return candidate
    base = report_path.parent
    if len(report_path.parents) >= 3:
        base = report_path.parents[2]
    return base / candidate


def _resolve_nightly_artifact_path(
    nightly_report: NightlyReportEvidence,
    artifact_path: Optional[str],
) -> str:
    raw = str(artifact_path or "").strip()
    if not raw:
        return ""
    return str(_resolve_report_relative_path(Path(nightly_report.path), raw))


def _overall_status(checks: tuple[AcceptanceCheck, ...]) -> str:
    statuses = {row.status for row in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _workflow_research_alignment_target(
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


__all__ = [
    "AcceptanceBundle",
    "AcceptanceCheck",
    "NightlyReportEvidence",
    "build_and_export_acceptance_bundle",
    "export_acceptance_bundle",
    "format_acceptance_bundle",
    "format_acceptance_bundle_markdown",
    "gather_acceptance_bundle",
    "load_nightly_report_evidence",
]
