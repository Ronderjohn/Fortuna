"""Helpers for dashboard-friendly multi-agent workflow summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from fortuna.agentic.contracts import (
    MarketUniverseResponse,
    MultiAgentWorkflowResponse,
    PortfolioAllocationResponse,
    ShortlistAnalysisResponse,
    ShortlistBriefingResponse,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
)


@dataclass(frozen=True)
class WorkflowSummary:
    metrics: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    notes: tuple[str, ...] = ()
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "rows": [dict(row) for row in self.rows],
            "notes": list(self.notes),
            "error": self.error,
        }


@dataclass(frozen=True)
class WorkflowSnapshot:
    source: str
    timeframe: str
    lookback_days: int
    team: WorkflowSummary
    universe: WorkflowSummary
    shortlist: WorkflowSummary
    briefing: WorkflowSummary
    candidates: WorkflowSummary
    research: WorkflowSummary
    allocation: WorkflowSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "timeframe": self.timeframe,
            "lookback_days": self.lookback_days,
            "team": self.team.to_dict(),
            "universe": self.universe.to_dict(),
            "shortlist": self.shortlist.to_dict(),
            "briefing": self.briefing.to_dict(),
            "candidates": self.candidates.to_dict(),
            "research": self.research.to_dict(),
            "allocation": self.allocation.to_dict(),
        }


def summarize_multi_agent_workflow(response: MultiAgentWorkflowResponse) -> WorkflowSnapshot:
    return WorkflowSnapshot(
        source=response.source,
        timeframe=response.timeframe,
        lookback_days=int(response.lookback_days),
        team=summarize_team_roles(response),
        universe=(
            summarize_market_universe(response.universe)
            if response.universe is not None
            else WorkflowSummary(metrics={"count": 0, "source": response.source}, rows=())
        ),
        shortlist=(
            summarize_shortlist_analysis(response.shortlist)
            if response.shortlist is not None
            else WorkflowSummary(metrics={"count": 0, "source": response.source}, rows=())
        ),
        briefing=(
            summarize_shortlist_briefing(response.briefing)
            if response.briefing is not None
            else WorkflowSummary(metrics={"count": 0, "source": response.source}, rows=())
        ),
        candidates=(
            summarize_training_candidates(response.candidates)
            if response.candidates is not None
            else WorkflowSummary(metrics={"count": 0, "source": response.source}, rows=())
        ),
        research=(
            summarize_training_research(response.research)
            if response.research is not None
            else WorkflowSummary(metrics={"count": 0, "source": response.source}, rows=())
        ),
        allocation=(
            summarize_portfolio_allocation(response.allocation)
            if response.allocation is not None
            else WorkflowSummary(metrics={"count": 0, "source": response.source}, rows=())
        ),
    )


def annotate_discovery_refresh(
    snapshot: WorkflowSnapshot,
    *,
    source: Optional[str],
    timeframe: Optional[str],
    days: Optional[int],
    refreshed_count: Optional[int] = None,
) -> WorkflowSnapshot:
    count = int(
        refreshed_count
        if refreshed_count is not None
        else snapshot.universe.metrics.get("count", 0) or 0
    )
    team_metrics = dict(snapshot.team.metrics)
    team_metrics.update(
        {
            "discovery_refresh_source": str(source or "").strip() or None,
            "discovery_refresh_timeframe": str(timeframe or "").strip() or None,
            "discovery_refresh_days": int(days or 0),
            "discovery_refresh_requested": True,
            "discovery_refreshed_count": count,
        }
    )
    refresh_note = (
        "discovery_refresh: "
        + f"source={team_metrics['discovery_refresh_source'] or 'auto'} "
        + f"timeframe={team_metrics['discovery_refresh_timeframe'] or '—'} "
        + (
            f"days={team_metrics['discovery_refresh_days']} "
            if int(team_metrics["discovery_refresh_days"] or 0) > 0
            else ""
        )
        + f"refreshed={count}"
    ).strip()
    team_notes = tuple(
        note
        for note in snapshot.team.notes
        if not str(note or "").startswith("discovery_refresh: ")
    ) + (refresh_note,)
    return replace(
        snapshot,
        team=replace(snapshot.team, metrics=team_metrics, notes=team_notes),
    )


def annotate_artifact_recovery_posture(
    snapshot: WorkflowSnapshot,
    *,
    action_label: Optional[str],
    target: Optional[str],
    posture_text: Optional[str],
) -> WorkflowSnapshot:
    action = str(action_label or "").strip()
    normalized_target = str(target or "").strip() or None
    posture = str(posture_text or "").strip()
    if not action and not normalized_target and not posture:
        return snapshot
    team_metrics = dict(snapshot.team.metrics)
    team_metrics.update(
        {
            "artifact_follow_up_action": action or None,
            "artifact_follow_up_target": normalized_target,
            "artifact_recovery_posture": posture or None,
        }
    )
    artifact_note = (
        "artifact_recovery: "
        + f"action={action or 'review'}"
        + (f" target={normalized_target}" if normalized_target else "")
        + (f" posture={posture}" if posture else "")
    )
    team_notes = tuple(
        note
        for note in snapshot.team.notes
        if not str(note or "").startswith("artifact_recovery: ")
    ) + (artifact_note,)
    return replace(
        snapshot,
        team=replace(snapshot.team, metrics=team_metrics, notes=team_notes),
    )


def replace_workflow_snapshot_sections(
    snapshot: WorkflowSnapshot,
    *,
    team: Optional[WorkflowSummary] = None,
    universe: Optional[WorkflowSummary] = None,
    shortlist: Optional[WorkflowSummary] = None,
    briefing: Optional[WorkflowSummary] = None,
    candidates: Optional[WorkflowSummary] = None,
    research: Optional[WorkflowSummary] = None,
    allocation: Optional[WorkflowSummary] = None,
) -> WorkflowSnapshot:
    return replace(
        snapshot,
        team=team or snapshot.team,
        universe=universe or snapshot.universe,
        shortlist=shortlist or snapshot.shortlist,
        briefing=briefing or snapshot.briefing,
        candidates=candidates or snapshot.candidates,
        research=research or snapshot.research,
        allocation=allocation or snapshot.allocation,
    )


def summarize_team_roles(response: MultiAgentWorkflowResponse) -> WorkflowSummary:
    nightly_alignment = getattr(response, "nightly_alignment", None)
    research = getattr(response, "research", None)
    allocation = getattr(response, "allocation", None)
    rows = tuple(
        {
            "name": row.name,
            "ok": row.ok,
            "headline": row.headline,
            "focus_count": len(row.focus_symbols),
            "focus_symbols": list(row.focus_symbols[:5]),
            "notes": list(row.notes[:5]),
        }
        for row in response.roles
    )
    ok_roles = sum(1 for row in response.roles if row.ok)
    notes: list[str] = []
    recent_metric_window = 0
    recent_metric_enabled = 0
    recent_metric_aligned = 0
    recent_metric_latest_status = None
    recent_metric_latest_basket_size = 0
    research_role = next(
        (row for row in response.roles if str(getattr(row, "name", "")) == "research_planner"),
        None,
    )
    research_headline = str(getattr(research_role, "headline", "") or "").strip() or None
    research_selection_policy = (
        str(getattr(research, "selection_policy", "") or "").strip() or None
        if research is not None
        else None
    )
    research_refresh_target = (
        str(getattr(research, "refresh_target", "") or "").strip() or None
        if research is not None
        else None
    )
    research_discovery_target = (
        str(getattr(research, "discovery_recommended_refresh_target", "") or "").strip() or None
        if research is not None
        else None
    )
    research_effective_target = (
        str(getattr(research, "effective_refresh_target", "") or "").strip() or None
        if research is not None
        else None
    ) or _effective_research_refresh_target(
        requested_target=research_refresh_target,
        discovery_target=research_discovery_target,
        nightly_target=(
            getattr(research, "nightly_alignment_target", None) if research is not None else None
        ),
    )
    research_target_mismatch = _targets_mismatch(
        requested_target=research_refresh_target,
        discovery_target=research_discovery_target,
    )
    research_refresh_requested = bool(getattr(research, "refresh_requested", False))
    research_refreshed_count = (
        sum(1 for row in getattr(research, "rows", ()) if bool(getattr(row, "refreshed", False)))
        if research is not None
        else 0
    )
    research_ml_count = (
        len(getattr(research, "ml_symbols", ()) or ()) if research is not None else 0
    )
    research_rl_count = (
        len(getattr(research, "rl_symbols", ()) or ()) if research is not None else 0
    )
    research_discovery_posture = (
        getattr(research, "discovery_posture", None) if research is not None else None
    )
    research_strategy_posture = (
        getattr(research, "strategy_posture", None) if research is not None else None
    )
    research_candidate_posture = (
        getattr(research, "candidate_posture", None) if research is not None else None
    )
    research_nightly_posture = (
        getattr(research, "nightly_posture", None) if research is not None else None
    )
    research_effective_posture = (
        getattr(research, "effective_posture", None) if research is not None else None
    )
    research_refresh_urgency = (
        getattr(research, "refresh_urgency", None) if research is not None else None
    )
    research_refresh_urgency_target = (
        getattr(research, "refresh_urgency_target", None) if research is not None else None
    )
    research_refresh_urgency_score = (
        getattr(research, "refresh_urgency_score", None) if research is not None else None
    )
    research_refresh_urgency_rows = (
        int(getattr(research, "refresh_urgency_rows", 0) or 0) if research is not None else 0
    )
    research_refresh_urgency_summary = (
        getattr(research, "refresh_urgency_summary", None) if research is not None else None
    )
    research_follow_up_action = (
        getattr(research, "follow_up_action", None) if research is not None else None
    )
    research_discovery_follow_up_target = (
        getattr(research, "discovery_follow_up_target", None) if research is not None else None
    )
    research_discovery_follow_up_summary = (
        getattr(research, "discovery_follow_up_summary", None) if research is not None else None
    )
    research_discovery_follow_up_action = (
        getattr(research, "discovery_follow_up_action", None) if research is not None else None
    )
    research_research_follow_up_target = (
        getattr(research, "research_follow_up_target", None) if research is not None else None
    )
    research_research_follow_up_summary = (
        getattr(research, "research_follow_up_summary", None) if research is not None else None
    )
    research_research_follow_up_action = (
        getattr(research, "research_follow_up_action", None) if research is not None else None
    )
    research_execution_follow_up_target = (
        getattr(research, "execution_follow_up_target", None) if research is not None else None
    )
    research_execution_follow_up_summary = (
        getattr(research, "execution_follow_up_summary", None) if research is not None else None
    )
    research_execution_follow_up_action = (
        getattr(research, "execution_follow_up_action", None) if research is not None else None
    )
    research_selected_target_mix = (
        getattr(research, "selected_target_mix", None) if research is not None else None
    )
    research_selected_regime_mix = (
        getattr(research, "selected_regime_mix", None) if research is not None else None
    )
    (
        research_scout_support_target,
        research_scout_support_summary,
        research_scout_supported_symbols,
        research_scout_overlap_selected_count,
        research_scout_volume_dense_selected_count,
    ) = _research_scout_support(research)
    research_alignment_summary, research_alignment_target_mix = _basket_research_alignment(
        allocation=allocation,
        research=research,
    )
    research_alignment_overlap_count, research_alignment_selected_count = _alignment_counts(
        research_alignment_summary
    )
    discovery_alignment_summary, discovery_overlap_symbols = _discovery_alignment(
        getattr(response, "universe", None)
    )
    discovery_alignment_overlap_count, discovery_alignment_compare_count = _alignment_counts(
        discovery_alignment_summary
    )
    if nightly_alignment is not None and (
        int(getattr(nightly_alignment, "report_count", 0) or 0) > 0
    ):
        notes.append(
            "nightly_alignment: "
            f"reports={int(getattr(nightly_alignment, 'report_count', 0) or 0)} "
            f"enabled={int(getattr(nightly_alignment, 'enabled_reports', 0) or 0)} "
            f"aligned={int(getattr(nightly_alignment, 'aligned_reports', 0) or 0)}"
        )
        latest_target = str(getattr(nightly_alignment, "latest_target_mix", "") or "").strip()
        if latest_target:
            notes.append(f"nightly_alignment_latest: {latest_target}")
        latest_refreshed = str(
            getattr(nightly_alignment, "latest_refreshed_target_mix", "") or ""
        ).strip()
        if latest_refreshed:
            notes.append(f"nightly_alignment_refreshed: {latest_refreshed}")
        latest_execution_target = str(
            getattr(nightly_alignment, "latest_execution_target", "") or ""
        ).strip()
        latest_execution_family = str(
            getattr(nightly_alignment, "latest_execution_model_family", "") or ""
        ).strip()
        latest_execution_source = str(
            getattr(nightly_alignment, "latest_execution_selection_source", "") or ""
        ).strip()
        if latest_execution_target or latest_execution_family or latest_execution_source:
            notes.append(
                "nightly_alignment_execution: "
                + (
                    f"target={latest_execution_target} "
                    if latest_execution_target
                    else ""
                )
                + (
                    f"family={latest_execution_family}"
                    if latest_execution_family
                    else ""
                )
                + (
                    f" source={latest_execution_source}"
                    if latest_execution_source
                    else ""
                )
            )
        latest_review_kind = str(
            getattr(nightly_alignment, "latest_promotion_review_model_kind", "") or ""
        ).strip()
        if latest_review_kind:
            notes.append(f"nightly_alignment_review_kind={latest_review_kind}")
        recommended_target = str(
            getattr(nightly_alignment, "recommended_refresh_target", "") or ""
        ).strip()
        if recommended_target:
            notes.append(f"nightly_alignment_target: {recommended_target}")
        if bool(getattr(nightly_alignment, "recommended_force_refresh", False)):
            notes.append("nightly_alignment_force_refresh=1")
        recommended_discovery_action = str(
            getattr(nightly_alignment, "recommended_discovery_action", "") or ""
        ).strip()
        if recommended_discovery_action:
            notes.append("nightly_alignment_discovery_follow_up=1")
        recommended_discovery_command = str(
            getattr(nightly_alignment, "recommended_discovery_cli_command", "") or ""
        ).strip()
        if recommended_discovery_command:
            notes.append("nightly_alignment_discovery_command=1")
        recent_window = int(getattr(nightly_alignment, "recent_trend_window", 0) or 0)
        recent_enabled = int(getattr(nightly_alignment, "recent_trend_enabled", 0) or 0)
        recent_aligned = int(getattr(nightly_alignment, "recent_trend_aligned", 0) or 0)
        recent_metric_latest_status = getattr(
            nightly_alignment,
            "recent_trend_latest_status",
            None,
        )
        recent_metric_latest_basket_size = int(
            getattr(nightly_alignment, "recent_trend_latest_basket_size", 0) or 0
        )
        if recent_window <= 0:
            recent_rows = tuple(getattr(nightly_alignment, "recent_rows", ()) or ())
            recent_window_rows = recent_rows[:3]
            if recent_window_rows:
                recent_window = len(recent_window_rows)
                recent_enabled = sum(
                    1 for row in recent_window_rows if getattr(row, "alignment_enabled", False)
                )
                recent_aligned = sum(
                    1
                    for row in recent_window_rows
                    if getattr(row, "alignment_enabled", False)
                    and str(getattr(row, "target_mix", "") or "").strip()
                )
                recent_metric_latest_status = str(
                    getattr(recent_window_rows[0], "overall_status", "") or ""
                ).strip() or None
                recent_metric_latest_basket_size = int(
                    getattr(recent_window_rows[0], "basket_size", 0) or 0
                )
        recent_metric_window = recent_window
        recent_metric_enabled = recent_enabled
        recent_metric_aligned = recent_aligned
        if recent_window > 0:
            notes.append(f"nightly_alignment_recent: {recent_aligned}/{recent_enabled}")
        nightly_posture = str(getattr(nightly_alignment, "nightly_posture", "") or "").strip()
        if nightly_posture:
            notes.append(f"nightly_alignment_posture: {nightly_posture}")
    if research_headline:
        notes.append(f"research_planner: {research_headline}")
    if discovery_alignment_summary:
        notes.append(discovery_alignment_summary)
    if discovery_overlap_symbols:
        notes.append(f"discovery_overlap={discovery_overlap_symbols}")
    if research_selection_policy:
        notes.append(
            "research_posture: "
            f"policy={research_selection_policy} "
            f"target={research_refresh_target or 'all'} "
            + (
                f"discovery_target={research_discovery_target} "
                if research_discovery_target
                else ""
            )
            + (
                f"effective_target={research_effective_target} "
                if research_effective_target
                else ""
            )
            +
            f"ml={research_ml_count} rl={research_rl_count}"
            + (
                f" refreshed={research_refreshed_count}"
                if research_refresh_requested
                else ""
            )
        )
    if research_target_mismatch:
        notes.append(
            "research_target_mismatch: "
            f"requested={research_refresh_target or 'all'} "
            f"discovery={research_discovery_target or 'all'} "
            f"effective={research_effective_target or 'all'}"
        )
    if research_refresh_urgency_summary:
        notes.append(f"research_refresh_urgency: {research_refresh_urgency_summary}")
    if research_follow_up_action:
        notes.append(f"research_follow_up: {research_follow_up_action}")
    if research_discovery_follow_up_summary or research_discovery_follow_up_action:
        notes.append(
            "research_discovery_follow_up: "
            + " ".join(
                part
                for part in (
                    research_discovery_follow_up_summary or "",
                    (
                        f"action={research_discovery_follow_up_action}"
                        if research_discovery_follow_up_action
                        else ""
                    ),
                )
                if part
            )
        )
    if research_research_follow_up_summary or research_research_follow_up_action:
        notes.append(
            "research_research_follow_up: "
            + " ".join(
                part
                for part in (
                    research_research_follow_up_summary or "",
                    (
                        f"action={research_research_follow_up_action}"
                        if research_research_follow_up_action
                        else ""
                    ),
                )
                if part
            )
        )
    if research_execution_follow_up_summary or research_execution_follow_up_action:
        notes.append(
            "research_execution_follow_up: "
            + " ".join(
                part
                for part in (
                    research_execution_follow_up_summary or "",
                    (
                        f"action={research_execution_follow_up_action}"
                        if research_execution_follow_up_action
                        else ""
                    ),
                )
                if part
            )
        )
    if (
        research_discovery_posture
        or research_strategy_posture
        or research_candidate_posture
        or research_nightly_posture
        or research_effective_posture
    ):
        notes.append(
            "research_posture_mix: "
            + " ".join(
                part
                for part in (
                    (
                        f"discovery={research_discovery_posture}"
                        if research_discovery_posture
                        else ""
                    ),
                    f"setup={research_strategy_posture}" if research_strategy_posture else "",
                    (
                        f"candidate={research_candidate_posture}"
                        if research_candidate_posture
                        else ""
                    ),
                    (
                        f"nightly={research_nightly_posture}"
                        if research_nightly_posture
                        else ""
                    ),
                    (
                        f"effective={research_effective_posture}"
                        if research_effective_posture
                        else ""
                    ),
                )
                if part
            )
        )
    if research_selected_target_mix or research_selected_regime_mix:
        notes.append(
            "research_cohorts: "
            + " ".join(
                part
                for part in (
                    (
                        f"targets={research_selected_target_mix}"
                        if research_selected_target_mix
                        else ""
                    ),
                    (
                        f"regimes={research_selected_regime_mix}"
                        if research_selected_regime_mix
                        else ""
                    ),
                )
                if part
            )
        )
    if research_scout_support_summary:
        notes.append(f"research_scouts: {research_scout_support_summary}")
    if research_alignment_summary:
        notes.append(research_alignment_summary)
    if research_alignment_target_mix:
        notes.append(f"basket_research_mix={research_alignment_target_mix}")
    return WorkflowSummary(
        metrics={
            "count": len(rows),
            "ok_count": ok_roles,
            "headline": response.headline,
            "nightly_report_count": (
                int(getattr(nightly_alignment, "report_count", 0) or 0)
                if nightly_alignment is not None
                else 0
            ),
            "nightly_enabled_reports": (
                int(getattr(nightly_alignment, "enabled_reports", 0) or 0)
                if nightly_alignment is not None
                else 0
            ),
            "nightly_aligned_reports": (
                int(getattr(nightly_alignment, "aligned_reports", 0) or 0)
                if nightly_alignment is not None
                else 0
            ),
            "nightly_latest_target_mix": (
                getattr(nightly_alignment, "latest_target_mix", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_latest_execution_target": (
                getattr(nightly_alignment, "latest_execution_target", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_latest_execution_model_family": (
                getattr(nightly_alignment, "latest_execution_model_family", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_latest_execution_selection_source": (
                getattr(nightly_alignment, "latest_execution_selection_source", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_latest_refreshed_target_mix": (
                getattr(nightly_alignment, "latest_refreshed_target_mix", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_posture": (
                getattr(nightly_alignment, "nightly_posture", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_latest_promotion_review_model_kind": (
                getattr(nightly_alignment, "latest_promotion_review_model_kind", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_latest_promotion_review_model_kinds": (
                ", ".join(
                    str(kind or "").strip()
                    for kind in getattr(
                        nightly_alignment,
                        "latest_promotion_review_model_kinds",
                        (),
                    )
                    if str(kind or "").strip()
                )
                or None
            )
            if nightly_alignment is not None
            else None,
            "nightly_recommended_target": (
                getattr(nightly_alignment, "recommended_refresh_target", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_recommended_force_refresh": (
                bool(getattr(nightly_alignment, "recommended_force_refresh", False))
                if nightly_alignment is not None
                else False
            ),
            "nightly_recommended_discovery_action": (
                getattr(nightly_alignment, "recommended_discovery_action", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_recommended_discovery_cli_command": (
                getattr(nightly_alignment, "recommended_discovery_cli_command", None)
                if nightly_alignment is not None
                else None
            ),
            "nightly_recent_window": (
                recent_metric_window
            ),
            "nightly_recent_enabled": (
                recent_metric_enabled
            ),
            "nightly_recent_aligned": (
                recent_metric_aligned
            ),
            "nightly_recent_latest_status": (
                recent_metric_latest_status
            ),
            "nightly_recent_latest_basket_size": (
                recent_metric_latest_basket_size
            ),
            "research_headline": research_headline,
            "research_selection_policy": research_selection_policy,
            "research_refresh_target": research_refresh_target,
            "research_discovery_summary": (
                getattr(research, "discovery_summary", None) if research is not None else None
            ),
            "research_discovery_recommended_target": research_discovery_target,
            "research_discovery_posture": research_discovery_posture,
            "research_strategy_posture": research_strategy_posture,
            "research_candidate_posture": research_candidate_posture,
            "research_nightly_posture": research_nightly_posture,
            "research_effective_posture": research_effective_posture,
            "research_effective_target": research_effective_target,
            "research_refresh_urgency": research_refresh_urgency,
            "research_refresh_urgency_target": research_refresh_urgency_target,
            "research_refresh_urgency_score": _round_or_none(research_refresh_urgency_score),
            "research_refresh_urgency_rows": research_refresh_urgency_rows,
            "research_refresh_urgency_summary": research_refresh_urgency_summary,
            "research_follow_up_action": research_follow_up_action,
            "research_discovery_follow_up_target": research_discovery_follow_up_target,
            "research_discovery_follow_up_summary": research_discovery_follow_up_summary,
            "research_discovery_follow_up_action": research_discovery_follow_up_action,
            "research_research_follow_up_target": research_research_follow_up_target,
            "research_research_follow_up_summary": research_research_follow_up_summary,
            "research_research_follow_up_action": research_research_follow_up_action,
            "research_execution_follow_up_target": research_execution_follow_up_target,
            "research_execution_follow_up_summary": research_execution_follow_up_summary,
            "research_execution_follow_up_action": research_execution_follow_up_action,
            "research_target_mismatch": research_target_mismatch,
            "research_refresh_requested": research_refresh_requested,
            "research_refreshed_count": research_refreshed_count,
            "research_ml_count": research_ml_count,
            "research_rl_count": research_rl_count,
            "research_selected_target_mix": research_selected_target_mix,
            "research_selected_regime_mix": research_selected_regime_mix,
            "research_scout_support_target": research_scout_support_target,
            "research_scout_support_summary": research_scout_support_summary,
            "research_scout_supported_symbols": research_scout_supported_symbols,
            "research_scout_overlap_selected_count": research_scout_overlap_selected_count,
            "research_scout_volume_dense_selected_count": (
                research_scout_volume_dense_selected_count
            ),
            "research_alignment_summary": research_alignment_summary,
            "research_alignment_target_mix": research_alignment_target_mix,
            "research_alignment_overlap_count": research_alignment_overlap_count,
            "research_alignment_selected_count": research_alignment_selected_count,
            "discovery_alignment_summary": discovery_alignment_summary,
            "discovery_overlap_symbols": discovery_overlap_symbols,
            "discovery_alignment_overlap_count": discovery_alignment_overlap_count,
            "discovery_alignment_compare_count": discovery_alignment_compare_count,
        },
        rows=rows,
        notes=tuple(notes[:12]),
    )


def _basket_research_alignment(*, allocation, research) -> tuple[str | None, str | None]:
    if allocation is None or research is None:
        return None, None
    selected_symbols = tuple(getattr(allocation, "selected_symbols", ()) or ())
    if not selected_symbols:
        return None, None
    ml_symbols = {str(symbol) for symbol in getattr(research, "ml_symbols", ()) or ()}
    rl_symbols = {str(symbol) for symbol in getattr(research, "rl_symbols", ()) or ()}
    if not ml_symbols and not rl_symbols:
        return None, None

    target_counts: dict[str, int] = {}
    overlap_count = 0
    for symbol in selected_symbols:
        target = "none"
        in_ml = symbol in ml_symbols
        in_rl = symbol in rl_symbols
        if in_ml and in_rl:
            target = "both"
            overlap_count += 1
        elif in_rl:
            target = "rl"
            overlap_count += 1
        elif in_ml:
            target = "ml"
            overlap_count += 1
        target_counts[target] = target_counts.get(target, 0) + 1

    selected_count = len(selected_symbols)
    if overlap_count >= selected_count:
        status = "aligned"
    elif overlap_count > 0:
        status = "partial"
    else:
        status = "drifting"
    summary = f"basket/research {status} {overlap_count}/{selected_count}"
    target_mix = ", ".join(f"{key}={value}" for key, value in sorted(target_counts.items()))
    return summary, target_mix or None


def _alignment_counts(summary: str | None) -> tuple[int, int]:
    text = str(summary or "").strip()
    if not text:
        return 0, 0
    last_token = text.rsplit(" ", 1)[-1]
    if "/" not in last_token:
        return 0, 0
    left, right = last_token.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return 0, 0


def _discovery_alignment(universe) -> tuple[str | None, str | None]:
    if universe is None:
        return None, None
    candidates = tuple(getattr(universe, "candidates", ()) or ())
    if not candidates:
        return None, None
    liquidity_ranked = sorted(
        candidates,
        key=lambda row: (
            -(float(getattr(row, "liquidity_score", 0.0) or 0.0)),
            -(float(getattr(row, "avg_turnover", 0.0) or 0.0)),
            row.symbol,
        ),
    )
    activity_ranked = [
        row
        for row in candidates
        if (
            getattr(row, "activity_score", None) is not None
            or getattr(row, "trend_pct", None) is not None
            or getattr(row, "volume_ratio", None) is not None
            or str(getattr(row, "regime", "") or "").strip()
        )
    ]
    activity_ranked = sorted(
        activity_ranked,
        key=lambda row: (
            -(float(getattr(row, "activity_score", 0.0) or 0.0)),
            -(abs(float(getattr(row, "trend_pct", 0.0) or 0.0))),
            row.symbol,
        ),
    )
    if not liquidity_ranked or not activity_ranked:
        return None, None
    compare_count = min(3, len(liquidity_ranked), len(activity_ranked))
    if compare_count <= 0:
        return None, None
    liquidity_top = tuple(row.symbol for row in liquidity_ranked[:compare_count])
    activity_top = tuple(row.symbol for row in activity_ranked[:compare_count])
    overlap = tuple(symbol for symbol in liquidity_top if symbol in activity_top)
    overlap_count = len(overlap)
    if overlap_count >= compare_count:
        status = "aligned"
    elif overlap_count > 0:
        status = "partial"
    else:
        status = "drifting"
    summary = f"discovery {status} {overlap_count}/{compare_count}"
    overlap_text = ", ".join(overlap) if overlap else None
    return summary, overlap_text


def summarize_market_universe(response: MarketUniverseResponse) -> WorkflowSummary:
    if not response.ok:
        return WorkflowSummary(
            metrics={"count": 0, "source": response.source},
            rows=(),
            error=response.error.message if response.error is not None else "Universe scan failed",
        )
    adaptive_notes = tuple(
        note
        for row in response.candidates
        for note in row.notes
        if str(note).startswith("adaptive_")
    )
    rows = tuple(
        {
            "symbol": row.symbol,
            "display": row.display_name,
            "liquidity_score": _round_or_none(row.liquidity_score),
            "trend_pct": _round_or_none(row.trend_pct),
            "regime": row.regime or None,
            "volume_ratio": _round_or_none(row.volume_ratio),
            "activity_score": _round_or_none(row.activity_score),
            "adaptive_score_adjustment": _round_or_none(row.adaptive_score_adjustment),
            "adaptive_row_count": row.adaptive_row_count or 0,
            "avg_turnover": _round_or_none(row.avg_turnover),
            "avg_volume": _round_or_none(row.avg_volume),
            "adaptive_note": next(
                (note for note in row.notes if str(note).startswith("adaptive_")),
                None,
            ),
        }
        for row in response.candidates
    )
    top_score = (
        max((row.liquidity_score or 0.0) for row in response.candidates)
        if response.candidates
        else 0.0
    )
    scout_summary = str(getattr(response, "scout_summary", "") or "").strip() or None
    scout_liquidity = ",".join(getattr(response, "scout_liquidity_symbols", ()) or ()) or None
    scout_activity = ",".join(getattr(response, "scout_activity_symbols", ()) or ()) or None
    scout_volume_dense = (
        ",".join(getattr(response, "scout_volume_dense_symbols", ()) or ()) or None
    )
    scout_overlap = ",".join(getattr(response, "scout_overlap_symbols", ()) or ()) or None
    return WorkflowSummary(
        metrics={
            "count": len(rows),
            "source": response.source,
            "top_liquidity_score": round(top_score, 2),
            "adaptive_count": len(adaptive_notes),
            "scout_summary": scout_summary,
            "scout_liquidity_symbols": scout_liquidity,
            "scout_activity_symbols": scout_activity,
            "scout_volume_dense_symbols": scout_volume_dense,
            "scout_overlap_symbols": scout_overlap,
            "seed_source": response.seed_source or response.source,
            "scoring_source": response.scoring_source,
            "provider_summary": response.provider_summary,
            "fallback_from": response.fallback_from,
            "nightly_alignment_target": response.nightly_alignment_target,
            "nightly_alignment_force_refresh": response.nightly_alignment_force_refresh,
            "nightly_recent_window": response.nightly_recent_window,
            "nightly_recent_enabled": response.nightly_recent_enabled,
            "nightly_recent_aligned": response.nightly_recent_aligned,
            "nightly_recent_latest_status": response.nightly_recent_latest_status,
            "nightly_recent_latest_basket_size": response.nightly_recent_latest_basket_size,
        },
        rows=rows,
        notes=(
            ((response.provider_summary,) if response.provider_summary else ())
            + ((scout_summary,) if scout_summary else ())
            + tuple(adaptive_notes[:4])
            + ((response.nightly_alignment_summary,) if response.nightly_alignment_summary else ())
        )[:5],
    )


def summarize_shortlist_briefing(response: ShortlistBriefingResponse) -> WorkflowSummary:
    if not response.ok:
        return WorkflowSummary(
            metrics={"count": 0, "source": response.source},
            rows=(),
            error=(
                response.error.message
                if response.error is not None
                else "Shortlist briefing failed"
            ),
        )
    rows = tuple(
        {
            "symbol": row.symbol,
            "selection_rank": row.selection_rank,
            "action": row.action,
            "verdict": row.verdict,
            "confidence_pct": round(row.confidence * 100, 1),
            "priority_score": _round_or_none(row.priority_score),
            "exposure_penalty": _round_or_none(row.exposure_penalty),
            "liquidity_score": _round_or_none(row.liquidity_score),
            "regime": row.regime or None,
            "volume_ratio": _round_or_none(row.volume_ratio),
            "activity_score": _round_or_none(row.activity_score),
            "winning_strategy": row.winning_strategy,
            "summary": row.summary,
        }
        for row in response.items
    )
    candidate_count = sum(1 for row in response.items if row.verdict == "candidate")
    notes = tuple(note.message for note in (response.portfolio.notes if response.portfolio else ()))
    return WorkflowSummary(
        metrics={
            "count": len(rows),
            "source": response.source,
            "candidates": candidate_count,
            "headline": response.headline,
        },
        rows=rows,
        notes=notes,
    )


def summarize_shortlist_analysis(response: ShortlistAnalysisResponse) -> WorkflowSummary:
    if not response.ok:
        return WorkflowSummary(
            metrics={"count": 0, "source": response.source},
            rows=(),
            error=(
                response.error.message
                if response.error is not None
                else "Shortlist analysis failed"
            ),
        )
    rows = tuple(
        {
            "symbol": row.symbol,
            "selection_rank": row.selection_rank,
            "action": row.decision.action if row.decision is not None else None,
            "confidence_pct": (
                round(float(row.decision.confidence) * 100, 1)
                if row.decision is not None
                else None
            ),
            "verdict": row.critique.verdict if row.critique is not None else None,
            "priority_score": _round_or_none(row.priority_score),
            "exposure_penalty": _round_or_none(row.exposure_penalty),
            "liquidity_score": _round_or_none(row.liquidity_score),
            "regime": row.regime or None,
            "volume_ratio": _round_or_none(row.volume_ratio),
            "activity_score": _round_or_none(row.activity_score),
        }
        for row in response.items
    )
    return WorkflowSummary(
        metrics={
            "count": len(rows),
            "source": response.source,
        },
        rows=rows,
    )


def summarize_training_candidates(response: TrainingCandidateResponse) -> WorkflowSummary:
    if not response.ok:
        return WorkflowSummary(
            metrics={"count": 0, "source": response.source},
            rows=(),
            error=(
                response.error.message
                if response.error is not None
                else "Training candidate scan failed"
            ),
        )
    rows = tuple(
        {
            "symbol": row.symbol,
            "selection_rank": row.selection_rank,
            "rank": row.shortlist_rank,
            "winning_strategy": row.winning_strategy,
            "action": row.decision_action,
            "confidence_pct": (
                round((row.decision_confidence or 0.0) * 100, 1)
                if row.decision_confidence is not None
                else None
            ),
            "verdict": row.critique_verdict,
            "ml_candidate": row.ml_candidate,
            "rl_candidate": row.rl_candidate,
            "priority_score": _round_or_none(row.priority_score),
            "remediation_target": row.remediation_target or None,
            "remediation_pressure": _round_or_none(row.remediation_pressure),
            "setup_family_reinforcement": _round_or_none(row.setup_family_reinforcement),
            "setup_family_verdict": row.setup_family_verdict or None,
            "exposure_penalty": _round_or_none(row.exposure_penalty),
            "liquidity_score": _round_or_none(row.liquidity_score),
            "market_regime": row.market_regime or None,
            "volume_ratio": _round_or_none(row.volume_ratio),
            "adaptive_score_adjustment": _round_or_none(row.adaptive_score_adjustment),
        }
        for row in response.candidates
    )
    ml_count = sum(1 for row in response.candidates if row.ml_candidate)
    rl_count = sum(1 for row in response.candidates if row.rl_candidate)
    return WorkflowSummary(
        metrics={
            "count": len(rows),
            "source": response.source,
            "ml_count": ml_count,
            "rl_count": rl_count,
        },
        rows=rows,
    )


def summarize_training_research(response: TrainingResearchPlanResponse) -> WorkflowSummary:
    if not response.ok:
        return WorkflowSummary(
            metrics={"count": 0, "source": response.source},
            rows=(),
            error=(
                response.error.message
                if response.error is not None
                else "Training research plan failed"
            ),
        )
    rows = tuple(
        {
            "symbol": row.symbol,
            "target": row.target,
            "universe_rank": row.universe_rank,
            "shortlist_rank": row.shortlist_rank,
            "selection_rank": row.selection_rank,
            "winning_strategy": row.winning_strategy,
            "action": row.decision_action,
            "verdict": row.critique_verdict,
            "market_regime": row.market_regime or None,
            "priority_score": _round_or_none(row.priority_score),
            "adaptive_score_adjustment": _round_or_none(row.adaptive_score_adjustment),
            "remediation_target": row.remediation_target or None,
            "remediation_pressure": _round_or_none(row.remediation_pressure),
            "setup_family_reinforcement": _round_or_none(row.setup_family_reinforcement),
            "setup_family_verdict": row.setup_family_verdict or None,
            "refreshed": row.refreshed,
        }
        for row in response.rows
    )
    (
        scout_support_target,
        scout_support_summary,
        scout_supported_symbols,
        scout_overlap_selected_count,
        scout_volume_dense_selected_count,
    ) = _research_scout_support(response)
    return WorkflowSummary(
        metrics={
            "count": len(rows),
            "source": response.source,
            "ml_count": len(response.ml_symbols),
            "rl_count": len(response.rl_symbols),
            "selection_policy": response.selection_policy,
            "refresh_target": response.refresh_target,
            "discovery_summary": response.discovery_summary,
            "discovery_posture": response.discovery_posture,
            "discovery_volume_dense_symbols": (
                ",".join(response.discovery_volume_dense_symbols)
                if response.discovery_volume_dense_symbols
                else None
            ),
            "strategy_posture": response.strategy_posture,
            "candidate_posture": response.candidate_posture,
            "nightly_posture": response.nightly_posture,
            "effective_posture": response.effective_posture,
            "discovery_recommended_refresh_target": (
                response.discovery_recommended_refresh_target
            ),
            "effective_refresh_target": (
                response.effective_refresh_target
                or _effective_research_refresh_target(
                    requested_target=response.refresh_target,
                    discovery_target=response.discovery_recommended_refresh_target,
                    nightly_target=response.nightly_alignment_target,
                )
            ),
            "refresh_urgency": response.refresh_urgency,
            "refresh_urgency_target": response.refresh_urgency_target,
            "refresh_urgency_score": _round_or_none(response.refresh_urgency_score),
            "refresh_urgency_rows": int(response.refresh_urgency_rows or 0),
            "refresh_urgency_summary": response.refresh_urgency_summary,
            "follow_up_action": response.follow_up_action,
            "discovery_follow_up_target": response.discovery_follow_up_target,
            "discovery_follow_up_summary": response.discovery_follow_up_summary,
            "discovery_follow_up_action": response.discovery_follow_up_action,
            "research_follow_up_target": response.research_follow_up_target,
            "research_follow_up_summary": response.research_follow_up_summary,
            "research_follow_up_action": response.research_follow_up_action,
            "execution_follow_up_target": response.execution_follow_up_target,
            "execution_follow_up_summary": response.execution_follow_up_summary,
            "execution_follow_up_action": response.execution_follow_up_action,
            "target_mismatch": _targets_mismatch(
                requested_target=response.refresh_target,
                discovery_target=response.discovery_recommended_refresh_target,
            ),
            "refresh_requested": response.refresh_requested,
            "refreshed_count": sum(1 for row in response.rows if row.refreshed),
            "selected_target_mix": response.selected_target_mix,
            "selected_regime_mix": response.selected_regime_mix,
            "scout_support_target": scout_support_target,
            "scout_support_summary": scout_support_summary,
            "scout_supported_symbols": scout_supported_symbols,
            "scout_overlap_selected_count": scout_overlap_selected_count,
            "scout_volume_dense_selected_count": scout_volume_dense_selected_count,
            "nightly_alignment_summary": response.nightly_alignment_summary,
            "nightly_alignment_target": response.nightly_alignment_target,
            "nightly_alignment_force_refresh": response.nightly_alignment_force_refresh,
            "nightly_recent_window": response.nightly_recent_window,
            "nightly_recent_enabled": response.nightly_recent_enabled,
            "nightly_recent_aligned": response.nightly_recent_aligned,
            "nightly_recent_latest_status": response.nightly_recent_latest_status,
            "nightly_recent_latest_basket_size": response.nightly_recent_latest_basket_size,
        },
        rows=rows,
        notes=tuple(
            note
            for note in (
                response.nightly_alignment_summary,
                response.refresh_urgency_summary,
                scout_support_summary,
                response.follow_up_action,
                response.discovery_follow_up_summary,
                response.research_follow_up_summary,
                response.execution_follow_up_summary,
            )
            if note
        ),
    )


def _research_scout_support(
    research,
) -> tuple[str | None, str | None, str | None, int, int]:
    if research is None:
        return None, None, None, 0, 0
    preferred = {
        str(symbol or "").strip().upper()
        for symbol in (getattr(research, "discovery_preferred_symbols", ()) or ())
        if str(symbol or "").strip()
    }
    volume_dense = {
        str(symbol or "").strip().upper()
        for symbol in (getattr(research, "discovery_volume_dense_symbols", ()) or ())
        if str(symbol or "").strip()
    }
    overlap_selected = 0
    volume_dense_selected = 0
    supported: list[str] = []
    for row in getattr(research, "rows", ()) or ():
        symbol = str(getattr(row, "symbol", "") or "").strip().upper()
        if not symbol:
            continue
        overlap_hit = symbol in preferred
        volume_dense_hit = symbol in volume_dense
        if overlap_hit:
            overlap_selected += 1
        if volume_dense_hit:
            volume_dense_selected += 1
        if (overlap_hit or volume_dense_hit) and symbol not in supported:
            supported.append(symbol)
    if not supported:
        return None, None, None, overlap_selected, volume_dense_selected
    scout_target = _scout_supported_target(
        getattr(research, "rows", ()) or (),
        preferred,
        volume_dense,
    )
    parts = [f"supported={len(supported)}"]
    if scout_target:
        parts.append(f"target={scout_target}")
    if overlap_selected > 0:
        parts.append(f"overlap={overlap_selected}")
    if volume_dense_selected > 0:
        parts.append(f"volume_dense={volume_dense_selected}")
    parts.append(f"symbols={','.join(supported[:3])}")
    return (
        scout_target,
        ", ".join(parts),
        ",".join(supported),
        overlap_selected,
        volume_dense_selected,
    )


def _scout_supported_target(
    rows,
    preferred_symbols: set[str],
    volume_dense_symbols: set[str],
) -> str | None:
    ml_votes = 0
    rl_votes = 0
    for row in rows:
        symbol = str(getattr(row, "symbol", "") or "").strip().upper()
        if symbol not in preferred_symbols and symbol not in volume_dense_symbols:
            continue
        target = str(getattr(row, "target", "") or "").strip().lower()
        if target in {"both", "all"}:
            ml_votes += 1
            rl_votes += 1
        elif target == "ml":
            ml_votes += 1
        elif target == "rl":
            rl_votes += 1
    if ml_votes <= 0 and rl_votes <= 0:
        return None
    if ml_votes == rl_votes:
        return "all"
    return "ml" if ml_votes > rl_votes else "rl"


def _effective_research_refresh_target(
    *,
    requested_target: Optional[str],
    discovery_target: Optional[str],
    nightly_target: Optional[str],
) -> Optional[str]:
    requested = str(requested_target or "").strip().lower()
    if requested in {"ml", "rl"}:
        return requested
    normalized = [
        str(target or "").strip().lower()
        for target in (discovery_target, nightly_target)
        if str(target or "").strip()
    ]
    if not normalized:
        return None
    concrete = {target for target in normalized if target in {"ml", "rl"}}
    if len(concrete) >= 2:
        return "all"
    if len(concrete) == 1:
        return next(iter(concrete))
    if "all" in normalized:
        return "all"
    return normalized[0]


def _targets_mismatch(
    *,
    requested_target: Optional[str],
    discovery_target: Optional[str],
) -> bool:
    requested = str(requested_target or "").strip().lower()
    discovery = str(discovery_target or "").strip().lower()
    if requested not in {"ml", "rl"} or discovery not in {"ml", "rl"}:
        return False
    return requested != discovery


def summarize_portfolio_allocation(response: PortfolioAllocationResponse) -> WorkflowSummary:
    if not response.ok:
        return WorkflowSummary(
            metrics={"count": 0, "source": response.source},
            rows=(),
            error=(
                response.error.message
                if response.error is not None
                else "Portfolio allocation failed"
            ),
        )
    rows = tuple(
        {
            "symbol": row.symbol,
            "selected": row.selected,
            "allocation_rank": row.allocation_rank or None,
            "allocation_weight": _round_or_none(row.allocation_weight),
            "action": row.action,
            "verdict": row.verdict,
            "selection_rank": row.selection_rank,
            "priority_score": _round_or_none(row.priority_score),
            "allocation_score": _round_or_none(row.allocation_score),
            "exposure_penalty": _round_or_none(row.exposure_penalty),
            "exposure_key": row.exposure_key,
            "winning_strategy": row.winning_strategy,
            "regime": row.regime or None,
            "risk_bucket": row.risk_bucket or None,
            "sizing_hint": row.sizing_hint or None,
            "research_target": row.research_target or None,
            "research_priority_score": _round_or_none(row.research_priority_score),
            "research_refreshed": bool(row.research_refreshed),
            "research_nightly_reports": int(getattr(row, "research_nightly_reports", 0) or 0),
            "research_nightly_refreshed_reports": int(
                getattr(row, "research_nightly_refreshed_reports", 0) or 0
            ),
            "research_nightly_promoted_reports": int(
                getattr(row, "research_nightly_promoted_reports", 0) or 0
            ),
            "research_nightly_score": _round_or_none(
                getattr(row, "research_nightly_score", None)
            ),
            "critic_penalty": _round_or_none(getattr(row, "critic_penalty", None)),
            "critic_note": row.critic_note or None,
            "reason": row.reason,
        }
        for row in response.items
    )
    selected_count = sum(1 for row in response.items if row.selected)
    skipped_count = sum(1 for row in response.items if not row.selected)
    selected_rows = tuple(row for row in response.items if row.selected)
    regime_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for row in selected_rows:
        regime = str(row.regime or "").strip().upper()
        if regime:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        strategy = str(row.winning_strategy or "").strip().upper()
        if strategy:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        risk_bucket = str(row.risk_bucket or "").strip().lower()
        if risk_bucket:
            risk_counts[risk_bucket] = risk_counts.get(risk_bucket, 0) + 1

    def _mix_text(counts: dict[str, int]) -> str | None:
        if not counts:
            return None
        return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))

    return WorkflowSummary(
        metrics={
            "count": len(rows),
            "source": response.source,
            "selected_count": selected_count,
            "skipped_count": skipped_count,
            "headline": response.headline,
            "max_positions": response.max_positions,
            "critic_enabled": any(note == "portfolio_critic=enabled" for note in response.notes),
            "regime_mix": _mix_text(regime_counts),
            "strategy_mix": _mix_text(strategy_counts),
            "risk_mix": _mix_text(risk_counts),
            "freshness_mix": _prefixed_allocation_note(response.notes, "allocation_freshness_mix:"),
            "target_balance_summary": _prefixed_allocation_note(
                response.notes,
                "allocation_target_balance_summary:",
            ),
            "critic_summary": _prefixed_allocation_note(
                response.notes,
                "allocation_critic_summary:",
            ),
        },
        rows=rows,
        notes=tuple(response.notes),
    )


def build_workflow_snapshot(
    *,
    settings,
    engine_factory=None,
    registry=None,
    universe_limit: int = 10,
    analysis_limit: int = 5,
    candidate_limit: int = 8,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
) -> WorkflowSnapshot:
    from fortuna.app.multi_agent_team import build_multi_agent_workflow

    workflow = build_multi_agent_workflow(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        universe_limit=max(universe_limit, analysis_limit * 2, candidate_limit * 2),
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
        ml_top_n=max(1, candidate_limit),
        rl_top_n=max(1, min(3, candidate_limit)),
        selection_policy="diversified",
        refresh_research_data=False,
    )
    return summarize_multi_agent_workflow(workflow)


def export_workflow_snapshot(snapshot: WorkflowSnapshot, out_path: Path | str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    return path


def _prefixed_allocation_note(notes: tuple[str, ...], prefix: str) -> Optional[str]:
    needle = str(prefix or "").strip()
    if not needle:
        return None
    for note in notes:
        text = str(note or "").strip()
        if text.startswith(needle):
            return text[len(needle) :].strip()
    return None


def _round_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)
