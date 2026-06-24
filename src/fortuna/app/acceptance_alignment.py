"""Shared acceptance refresh and cross-artifact alignment helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from fortuna.agentic.contracts import (
    CrossArtifactAlignmentSummary,
    DualPromotionReviewSummary,
    WorkflowSnapshotSummary,
)

if TYPE_CHECKING:
    from fortuna.agentic.contracts import ModelHealthResponse
    from fortuna.app.acceptance_bundle import NightlyReportEvidence
    from fortuna.app.promotion_review import PromotionReview


def follow_up_lane_target(
    *,
    nightly_report: Optional[NightlyReportEvidence],
    workflow_summary: Optional[WorkflowSnapshotSummary],
    lane: str,
) -> Optional[str]:
    nightly = nightly_report
    wf = workflow_summary
    if lane == "discovery":
        if nightly is not None and nightly.training_research_discovery_follow_up_target:
            return nightly.training_research_discovery_follow_up_target
        if wf is not None and wf.team_research_discovery_follow_up_target:
            return wf.team_research_discovery_follow_up_target
    elif lane == "research":
        if nightly is not None and nightly.training_research_research_follow_up_target:
            return nightly.training_research_research_follow_up_target
        if wf is not None and wf.team_research_research_follow_up_target:
            return wf.team_research_research_follow_up_target
    elif lane == "execution":
        if nightly is not None and nightly.training_research_execution_follow_up_target:
            return nightly.training_research_execution_follow_up_target
        if wf is not None and wf.team_research_execution_follow_up_target:
            return wf.team_research_execution_follow_up_target
    return None


def follow_up_lane_action(
    *,
    nightly_report: Optional[NightlyReportEvidence],
    workflow_summary: Optional[WorkflowSnapshotSummary],
    lane: str,
) -> Optional[str]:
    if nightly_report is not None:
        if lane == "discovery":
            return nightly_report.training_research_discovery_follow_up_action
        if lane == "research":
            return nightly_report.training_research_research_follow_up_action
        if lane == "execution":
            return nightly_report.training_research_execution_follow_up_action
    if workflow_summary is not None:
        if lane == "discovery":
            return workflow_summary.team_research_discovery_follow_up_action
        if lane == "research":
            return workflow_summary.team_research_research_follow_up_action
        if lane == "execution":
            return workflow_summary.team_research_execution_follow_up_action
    return None


def resolve_refresh_target_source(
    *,
    effective: Optional[str],
    nightly_recommended: Optional[str],
    discovery_target: Optional[str],
    scout_target: Optional[str],
    workflow_target: Optional[str],
) -> Optional[str]:
    if not effective:
        return None
    scout = str(scout_target or "").strip().lower()
    effective_norm = str(effective or "").strip().lower()
    if scout in {"ml", "rl"} and effective_norm == scout:
        concrete = {
            str(target or "").strip().lower()
            for target in (nightly_recommended, workflow_target, discovery_target)
            if str(target or "").strip().lower() in {"ml", "rl"}
        }
        if len(concrete) >= 2:
            return "scout_tiebreak"
    discovery = str(discovery_target or "").strip().lower()
    nightly = str(nightly_recommended or "").strip().lower()
    if discovery and effective_norm == discovery and effective_norm != nightly:
        return "discovery"
    workflow = str(workflow_target or "").strip().lower()
    if workflow and effective_norm == workflow and not nightly:
        return "workflow"
    return "nightly"


def resolve_recommended_follow_up_lane(
    *,
    effective: Optional[str],
    nightly_report: Optional[NightlyReportEvidence],
    workflow_summary: Optional[WorkflowSnapshotSummary],
) -> Optional[str]:
    effective_norm = str(effective or "").strip().lower()
    lane_kwargs = {
        "nightly_report": nightly_report,
        "workflow_summary": workflow_summary,
    }
    lanes = (
        ("execution", follow_up_lane_target(**lane_kwargs, lane="execution")),
        ("research", follow_up_lane_target(**lane_kwargs, lane="research")),
        ("discovery", follow_up_lane_target(**lane_kwargs, lane="discovery")),
    )
    if effective_norm:
        for lane, target in lanes:
            if str(target or "").strip().lower() == effective_norm:
                return lane
    for lane, _target in lanes:
        if follow_up_lane_action(**lane_kwargs, lane=lane):
            return lane
        if _target:
            return lane
    return None


def resolve_acceptance_refresh_context(
    *,
    model_health: Optional[ModelHealthResponse],
    nightly_report: Optional[NightlyReportEvidence],
    workflow_summary: Optional[WorkflowSnapshotSummary],
    workflow_research_alignment_target,
) -> dict[str, Optional[str]]:
    nightly_recommended = None
    effective = None
    scout_target = None
    scout_summary = None
    discovery_target = None
    workflow_target = None

    if model_health is not None:
        nightly = model_health.agentic.nightly_alignment
        nightly_recommended = nightly.recommended_refresh_target
        effective = nightly.effective_refresh_target
        scout_target = (
            nightly.latest_workflow_research_scout_support_target
            or nightly.latest_training_research_scout_support_target
        )
        discovery_target = nightly.latest_training_research_discovery_recommended_target
        workflow_target = nightly.latest_workflow_research_recommended_target

    if nightly_report is not None:
        if not effective:
            effective = nightly_report.training_research_effective_refresh_target
        if not discovery_target:
            discovery_target = nightly_report.training_research_discovery_recommended_refresh_target
        if not scout_target:
            scout_target = nightly_report.training_research_scout_support_target

    if workflow_summary is not None:
        if not effective:
            effective = (
                workflow_summary.research_effective_target
                or workflow_summary.team_nightly_recommended_target
            )
        if not scout_target:
            scout_target = workflow_summary.team_research_scout_support_target
        if not scout_summary:
            scout_summary = workflow_summary.team_research_scout_support_summary
        if not workflow_target:
            workflow_target = workflow_research_alignment_target(workflow_summary)

    if not effective and discovery_target:
        effective = discovery_target

    source = resolve_refresh_target_source(
        effective=effective,
        nightly_recommended=nightly_recommended,
        discovery_target=discovery_target,
        scout_target=scout_target,
        workflow_target=workflow_target,
    )
    follow_up_lane = resolve_recommended_follow_up_lane(
        effective=effective,
        nightly_report=nightly_report,
        workflow_summary=workflow_summary,
    )
    return {
        "effective_refresh_target": effective,
        "refresh_target_source": source,
        "recommended_follow_up_lane": follow_up_lane,
        "scout_support_target": scout_target,
        "scout_support_summary": scout_summary,
    }


def targets_aligned(left: Optional[str], right: Optional[str]) -> Optional[bool]:
    left_norm = str(left or "").strip().lower()
    right_norm = str(right or "").strip().lower()
    if not left_norm or not right_norm:
        return None
    if left_norm == right_norm:
        return True
    if left_norm == "all" or right_norm == "all":
        return left_norm in {"all", right_norm} or right_norm in {"all", left_norm}
    return False


def build_cross_artifact_alignment(
    *,
    nightly_report: Optional[NightlyReportEvidence],
    workflow_summary: Optional[WorkflowSnapshotSummary],
    rl_review: Optional[PromotionReview],
    ml_review: Optional[PromotionReview],
    model_health: Optional[ModelHealthResponse],
    effective_refresh_target: Optional[str],
    nightly_alignment: Any | None = None,
) -> CrossArtifactAlignmentSummary:
    discovery_target = None
    research_target = None
    execution_target = None
    if nightly_report is not None:
        discovery_target = nightly_report.training_research_discovery_follow_up_target
        research_target = nightly_report.training_research_research_follow_up_target
        execution_target = nightly_report.training_research_execution_follow_up_target
        if not execution_target:
            execution_target = nightly_report.training_execution_target
    if workflow_summary is not None:
        wf = workflow_summary
        discovery_target = discovery_target or wf.team_research_discovery_follow_up_target
        research_target = research_target or wf.team_research_research_follow_up_target
        execution_target = execution_target or wf.team_research_execution_follow_up_target
        if not execution_target:
            execution_target = wf.team_nightly_latest_execution_target

    discovery_research_aligned = targets_aligned(discovery_target, research_target)
    research_execution_aligned = targets_aligned(research_target, execution_target)

    promotion_target = None
    for review in (rl_review, ml_review):
        if review is not None and review.workflow_research_alignment_recommended_target:
            promotion_target = review.workflow_research_alignment_recommended_target
            break
    if model_health is not None:
        promotion_target = (
            promotion_target
            or model_health.agentic.nightly_alignment.latest_workflow_research_recommended_target
        )
    elif nightly_alignment is not None:
        promotion_target = (
            promotion_target
            or getattr(nightly_alignment, "latest_workflow_research_recommended_target", None)
        )
    promotion_research_aligned = targets_aligned(
        promotion_target or effective_refresh_target,
        research_target or effective_refresh_target,
    )

    refreshed_basket_aligned = None
    if model_health is not None:
        nightly = model_health.agentic.nightly_alignment
        if nightly.enabled_reports > 0:
            refreshed_basket_aligned = nightly.refreshed_aligned_reports >= nightly.enabled_reports
    elif nightly_alignment is not None and getattr(nightly_alignment, "enabled_reports", 0) > 0:
        refreshed_basket_aligned = (
            getattr(nightly_alignment, "refreshed_aligned_reports", 0)
            >= getattr(nightly_alignment, "enabled_reports", 0)
        )
    elif workflow_summary is not None and workflow_summary.team_nightly_enabled_reports > 0:
        refreshed_basket_aligned = (
            workflow_summary.team_nightly_aligned_reports
            >= workflow_summary.team_nightly_enabled_reports
        )

    flags = [
        discovery_research_aligned,
        research_execution_aligned,
        promotion_research_aligned,
        refreshed_basket_aligned,
    ]
    known = [flag for flag in flags if flag is not None]
    if not known:
        return CrossArtifactAlignmentSummary(
            discovery_research_aligned=discovery_research_aligned,
            research_execution_aligned=research_execution_aligned,
            promotion_research_aligned=promotion_research_aligned,
            refreshed_basket_aligned=refreshed_basket_aligned,
            overall_status="unknown",
            summary="Insufficient artifact evidence for cross-layer alignment.",
        )
    if all(flag is True for flag in known):
        status = "aligned"
        summary = "Discovery, research, execution, and promotion guidance are aligned."
    elif any(flag is False for flag in known):
        status = "drifting"
        parts: list[str] = []
        if discovery_research_aligned is False:
            parts.append("discovery vs research")
        if research_execution_aligned is False:
            parts.append("research vs execution")
        if promotion_research_aligned is False:
            parts.append("promotion vs research")
        if refreshed_basket_aligned is False:
            parts.append("refreshed basket")
        summary = "Drift detected: " + ", ".join(parts) + "."
    else:
        status = "unknown"
        summary = "Partial artifact evidence; alignment inconclusive."
    return CrossArtifactAlignmentSummary(
        discovery_research_aligned=discovery_research_aligned,
        research_execution_aligned=research_execution_aligned,
        promotion_research_aligned=promotion_research_aligned,
        refreshed_basket_aligned=refreshed_basket_aligned,
        overall_status=status,
        summary=summary,
    )


_ML_KINDS = frozenset({"ml_scorer", "ml"})
_RL_KINDS = frozenset({"rl_policy", "rl"})


def _normalize_review_kind(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text or text == "mixed":
        return None
    if text in _ML_KINDS:
        return "ml_scorer"
    if text in _RL_KINDS:
        return "rl_policy"
    return text


def _collect_review_kinds(
    *,
    nightly_report: Optional[NightlyReportEvidence],
    rl_review: Optional[PromotionReview],
    ml_review: Optional[PromotionReview],
    extra_review_kinds: tuple[str, ...] = (),
) -> tuple[str, ...]:
    kinds: set[str] = set()
    if nightly_report is not None:
        for kind in nightly_report.promotion_review_model_kinds:
            normalized = _normalize_review_kind(kind)
            if normalized:
                kinds.add(normalized)
        normalized = _normalize_review_kind(nightly_report.promotion_review_model_kind)
        if normalized:
            kinds.add(normalized)
    for kind in extra_review_kinds:
        normalized = _normalize_review_kind(kind)
        if normalized:
            kinds.add(normalized)
    if rl_review is not None:
        kinds.add("rl_policy")
    if ml_review is not None:
        kinds.add("ml_scorer")
    return tuple(sorted(kinds))


def _review_research_target(review: Optional[PromotionReview]) -> Optional[str]:
    if review is None:
        return None
    return review.workflow_research_alignment_recommended_target


def _execution_supports_dual(
    *,
    execution_target: Optional[str],
    execution_family: Optional[str],
) -> bool:
    target = str(execution_target or "").strip().lower()
    family = str(execution_family or "").strip().lower()
    if target == "all" or family == "hybrid":
        return True
    if target in {"ml", "rl"} and family in {"", target}:
        return False
    return family == "hybrid"


def build_dual_promotion_review_summary(
    *,
    nightly_report: Optional[NightlyReportEvidence],
    workflow_summary: Optional[WorkflowSnapshotSummary],
    rl_review: Optional[PromotionReview],
    ml_review: Optional[PromotionReview],
    effective_refresh_target: Optional[str],
    extra_review_kinds: tuple[str, ...] = (),
    execution_target_override: Optional[str] = None,
    execution_family_override: Optional[str] = None,
) -> DualPromotionReviewSummary:
    review_kinds = _collect_review_kinds(
        nightly_report=nightly_report,
        rl_review=rl_review,
        ml_review=ml_review,
        extra_review_kinds=extra_review_kinds,
    )
    ml_present = "ml_scorer" in review_kinds or ml_review is not None
    rl_present = "rl_policy" in review_kinds or rl_review is not None
    both_present = ml_present and rl_present

    execution_target = execution_target_override
    execution_family = execution_family_override
    if execution_target is None and execution_family is None:
        if nightly_report is not None:
            execution_target = nightly_report.training_execution_target
            execution_family = nightly_report.training_execution_model_family
        elif workflow_summary is not None:
            execution_target = workflow_summary.team_nightly_latest_execution_target
            execution_family = workflow_summary.team_nightly_latest_execution_model_family

    effective = effective_refresh_target
    if not effective and workflow_summary is not None:
        effective = (
            workflow_summary.research_effective_target
            or workflow_summary.team_nightly_recommended_target
        )

    if not both_present:
        if ml_present or rl_present:
            lane = "ml_scorer" if ml_present else "rl_policy"
            return DualPromotionReviewSummary(
                both_present=False,
                ml_present=ml_present,
                rl_present=rl_present,
                nightly_review_kinds=review_kinds,
                execution_target=execution_target,
                effective_refresh_target=effective,
                posture="single_lane",
                summary=f"Single-lane promotion review posture ({lane}).",
                warning=False,
            )
        return DualPromotionReviewSummary(
            nightly_review_kinds=review_kinds,
            execution_target=execution_target,
            effective_refresh_target=effective,
            posture="unknown",
            summary="Insufficient dual-promotion review evidence.",
        )

    ml_target = _review_research_target(ml_review)
    rl_target = _review_research_target(rl_review)
    research_targets_conflict = (
        ml_target is not None
        and rl_target is not None
        and not targets_aligned(ml_target, rl_target)
    )
    execution_conflict = not _execution_supports_dual(
        execution_target=execution_target,
        execution_family=execution_family,
    )
    dual_execution_ok = _execution_supports_dual(
        execution_target=execution_target,
        execution_family=execution_family,
    )
    effective_allows_mixed = str(effective or "").strip().lower() in {"all", "ml", "rl", ""}

    if execution_conflict or research_targets_conflict:
        parts: list[str] = []
        if execution_conflict:
            parts.append(
                f"execution={execution_target or '—'}/{execution_family or '—'}"
            )
        if research_targets_conflict:
            parts.append(f"ml_target={ml_target} rl_target={rl_target}")
        return DualPromotionReviewSummary(
            both_present=True,
            ml_present=True,
            rl_present=True,
            nightly_review_kinds=review_kinds,
            execution_target=execution_target,
            effective_refresh_target=effective,
            posture="conflicting",
            summary="Dual promotion posture conflicts: " + ", ".join(parts) + ".",
            warning=True,
        )

    if dual_execution_ok and effective_allows_mixed:
        posture = "expected_dual"
        summary = (
            "Both ML and RL promotion reviews are present with mixed execution posture "
            f"(target={execution_target or '—'}, family={execution_family or '—'})."
        )
    else:
        posture = "coherent_mixed"
        summary = (
            "Both ML and RL promotion reviews are present and review kinds agree "
            f"with nightly execution posture (target={execution_target or '—'})."
        )

    return DualPromotionReviewSummary(
        both_present=True,
        ml_present=True,
        rl_present=True,
        nightly_review_kinds=review_kinds,
        execution_target=execution_target,
        effective_refresh_target=effective,
        posture=posture,
        summary=summary,
        warning=False,
    )
