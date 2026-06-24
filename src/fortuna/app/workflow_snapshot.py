"""Helpers for loading and formatting workflow snapshot artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fortuna.agentic.contracts import WorkflowSnapshotSummary


def load_workflow_snapshot_summary(
    path_text: Optional[str],
) -> Optional[WorkflowSnapshotSummary]:
    path_str = str(path_text or "").strip()
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    universe = payload.get("universe") or {}
    shortlist = payload.get("shortlist") or {}
    briefing = payload.get("briefing") or {}
    allocation = payload.get("allocation") or {}
    candidates = payload.get("candidates") or {}
    research = payload.get("research") or {}
    team = payload.get("team") or {}
    return WorkflowSnapshotSummary(
        path=str(path),
        source=payload.get("source"),
        timeframe=payload.get("timeframe"),
        lookback_days=_coerce_int(payload.get("lookback_days")),
        team_role_count=_metric_int(team, "count"),
        team_ok_role_count=_metric_int(team, "ok_count"),
        team_headline=_metric_str(team, "headline"),
        team_nightly_report_count=_metric_int(team, "nightly_report_count"),
        team_nightly_aligned_reports=_metric_int(team, "nightly_aligned_reports"),
        team_nightly_enabled_reports=_metric_int(team, "nightly_enabled_reports"),
        team_nightly_latest_target_mix=_metric_str(team, "nightly_latest_target_mix"),
        team_nightly_latest_execution_target=_metric_str(
            team,
            "nightly_latest_execution_target",
        ),
        team_nightly_latest_execution_model_family=_metric_str(
            team,
            "nightly_latest_execution_model_family",
        ),
        team_nightly_latest_execution_selection_source=_metric_str(
            team,
            "nightly_latest_execution_selection_source",
        ),
        team_nightly_latest_promotion_review_model_kind=_metric_str(
            team,
            "nightly_latest_promotion_review_model_kind",
        ),
        team_nightly_latest_promotion_review_model_kinds=_metric_str(
            team,
            "nightly_latest_promotion_review_model_kinds",
        ),
        team_nightly_latest_refreshed_target_mix=_metric_str(
            team,
            "nightly_latest_refreshed_target_mix",
        ),
        team_nightly_posture=_metric_str(team, "nightly_posture"),
        team_nightly_recommended_target=_metric_str(team, "nightly_recommended_target"),
        team_nightly_recommended_force_refresh=_metric_bool(
            team,
            "nightly_recommended_force_refresh",
        ),
        team_nightly_recent_window=_metric_int(team, "nightly_recent_window"),
        team_nightly_recent_enabled=_metric_int(team, "nightly_recent_enabled"),
        team_nightly_recent_aligned=_metric_int(team, "nightly_recent_aligned"),
        team_nightly_recent_latest_status=_metric_str(team, "nightly_recent_latest_status"),
        team_nightly_recent_latest_basket_size=_metric_int(
            team,
            "nightly_recent_latest_basket_size",
        ),
        team_research_headline=_metric_str(team, "research_headline"),
        team_research_selection_policy=_metric_str(team, "research_selection_policy"),
        team_research_refresh_target=_metric_str(team, "research_refresh_target"),
        team_research_discovery_summary=_metric_str(team, "research_discovery_summary"),
        team_research_discovery_recommended_target=_metric_str(
            team,
            "research_discovery_recommended_target",
        ),
        team_research_discovery_posture=_metric_str(team, "research_discovery_posture"),
        team_research_strategy_posture=_metric_str(team, "research_strategy_posture"),
        team_research_candidate_posture=_metric_str(team, "research_candidate_posture"),
        team_research_nightly_posture=_metric_str(team, "research_nightly_posture"),
        team_research_effective_posture=_metric_str(team, "research_effective_posture"),
        team_research_effective_target=_metric_str(team, "research_effective_target"),
        team_research_refresh_urgency=_metric_str(team, "research_refresh_urgency"),
        team_research_refresh_urgency_target=_metric_str(
            team,
            "research_refresh_urgency_target",
        ),
        team_research_refresh_urgency_score=_metric_float(
            team,
            "research_refresh_urgency_score",
        ),
        team_research_refresh_urgency_rows=_metric_int(team, "research_refresh_urgency_rows"),
        team_research_refresh_urgency_summary=_metric_str(
            team,
            "research_refresh_urgency_summary",
        ),
        team_research_follow_up_action=_metric_str(team, "research_follow_up_action"),
        team_research_discovery_follow_up_target=_metric_str(
            team,
            "research_discovery_follow_up_target",
        ),
        team_research_discovery_follow_up_summary=_metric_str(
            team,
            "research_discovery_follow_up_summary",
        ),
        team_research_discovery_follow_up_action=_metric_str(
            team,
            "research_discovery_follow_up_action",
        ),
        team_research_research_follow_up_target=_metric_str(
            team,
            "research_research_follow_up_target",
        ),
        team_research_research_follow_up_summary=_metric_str(
            team,
            "research_research_follow_up_summary",
        ),
        team_research_research_follow_up_action=_metric_str(
            team,
            "research_research_follow_up_action",
        ),
        team_research_execution_follow_up_target=_metric_str(
            team,
            "research_execution_follow_up_target",
        ),
        team_research_execution_follow_up_summary=_metric_str(
            team,
            "research_execution_follow_up_summary",
        ),
        team_research_execution_follow_up_action=_metric_str(
            team,
            "research_execution_follow_up_action",
        ),
        team_research_target_mismatch=_metric_bool(team, "research_target_mismatch"),
        team_research_refresh_requested=_metric_bool(team, "research_refresh_requested"),
        team_research_refreshed_count=_metric_int(team, "research_refreshed_count"),
        team_research_ml_count=_metric_int(team, "research_ml_count"),
        team_research_rl_count=_metric_int(team, "research_rl_count"),
        team_research_selected_target_mix=_metric_str(team, "research_selected_target_mix"),
        team_research_selected_regime_mix=_metric_str(team, "research_selected_regime_mix"),
        team_research_scout_support_target=_metric_str(team, "research_scout_support_target"),
        team_research_scout_support_summary=_metric_str(team, "research_scout_support_summary"),
        team_research_scout_supported_symbols=_metric_str(team, "research_scout_supported_symbols"),
        team_research_scout_overlap_selected_count=_metric_int(
            team,
            "research_scout_overlap_selected_count",
        ),
        team_research_scout_volume_dense_selected_count=_metric_int(
            team,
            "research_scout_volume_dense_selected_count",
        ),
        team_research_alignment_summary=_metric_str(team, "research_alignment_summary"),
        team_research_alignment_target_mix=_metric_str(team, "research_alignment_target_mix"),
        team_research_alignment_overlap_count=_metric_int(
            team,
            "research_alignment_overlap_count",
        ),
        team_research_alignment_selected_count=_metric_int(
            team,
            "research_alignment_selected_count",
        ),
        team_discovery_refresh_source=_metric_str(team, "discovery_refresh_source"),
        team_discovery_refresh_timeframe=_metric_str(team, "discovery_refresh_timeframe"),
        team_discovery_refresh_days=_metric_int(team, "discovery_refresh_days"),
        team_discovery_refresh_requested=_metric_bool(team, "discovery_refresh_requested"),
        team_discovery_refreshed_count=_metric_int(team, "discovery_refreshed_count"),
        team_discovery_alignment_summary=_metric_str(team, "discovery_alignment_summary"),
        team_discovery_overlap_symbols=_metric_str(team, "discovery_overlap_symbols"),
        team_discovery_alignment_overlap_count=_metric_int(
            team,
            "discovery_alignment_overlap_count",
        ),
        team_discovery_alignment_compare_count=_metric_int(
            team,
            "discovery_alignment_compare_count",
        ),
        team_artifact_follow_up_action=_metric_str(team, "artifact_follow_up_action"),
        team_artifact_follow_up_target=_metric_str(team, "artifact_follow_up_target"),
        team_artifact_recovery_posture=_metric_str(team, "artifact_recovery_posture"),
        universe_count=_metric_int(universe, "count"),
        universe_adaptive_count=_metric_int(universe, "adaptive_count"),
        universe_top_adaptive_note=_first_note(universe),
        universe_scout_summary=_metric_str(universe, "scout_summary"),
        universe_liquidity_symbols=_metric_str(universe, "scout_liquidity_symbols"),
        universe_activity_symbols=_metric_str(universe, "scout_activity_symbols"),
        universe_volume_dense_symbols=_metric_str(universe, "scout_volume_dense_symbols"),
        universe_overlap_symbols=_metric_str(universe, "scout_overlap_symbols"),
        universe_seed_source=_metric_str(universe, "seed_source"),
        universe_provider_summary=_metric_str(universe, "provider_summary"),
        shortlist_count=_metric_int(shortlist, "count"),
        briefing_candidates=_metric_int(briefing, "candidates"),
        allocation_selected=_metric_int(allocation, "selected_count"),
        allocation_skipped=_metric_int(allocation, "skipped_count"),
        allocation_max_positions=_metric_int(allocation, "max_positions"),
        allocation_research_alignment_enabled=_has_note(
            allocation,
            "portfolio_research_alignment=enabled",
        ),
        allocation_research_target_mix=_prefixed_note(
            allocation,
            "allocation_research_targets:",
        ),
        allocation_refreshed_research_target_mix=_prefixed_note(
            allocation,
            "allocation_refreshed_research_targets:",
        ),
        allocation_critic_enabled=(
            _metric_bool(allocation, "critic_enabled")
            or _has_note(allocation, "portfolio_critic=enabled")
        ),
        allocation_regime_mix=(
            _metric_str(allocation, "regime_mix")
            or _prefixed_note(allocation, "allocation_regime_mix:")
            or _prefixed_note(allocation, "selected_regimes:")
        ),
        allocation_strategy_mix=(
            _metric_str(allocation, "strategy_mix")
            or _prefixed_note(allocation, "allocation_strategy_mix:")
        ),
        allocation_risk_mix=(
            _metric_str(allocation, "risk_mix")
            or _prefixed_note(allocation, "allocation_risk_mix:")
        ),
        allocation_freshness_mix=_prefixed_note(allocation, "allocation_freshness_mix:"),
        allocation_target_balance_summary=_prefixed_note(
            allocation,
            "allocation_target_balance_summary:",
        ),
        allocation_critic_summary=_prefixed_note(allocation, "allocation_critic_summary:"),
        ml_count=_metric_int(candidates, "ml_count"),
        rl_count=_metric_int(candidates, "rl_count"),
        research_count=_metric_int(research, "count"),
        research_ml_count=_metric_int(research, "ml_count"),
        research_rl_count=_metric_int(research, "rl_count"),
        research_selection_policy=_metric_str(research, "selection_policy"),
        research_refresh_target=_metric_str(research, "refresh_target"),
        research_discovery_summary=_metric_str(research, "discovery_summary"),
        research_discovery_posture=_metric_str(research, "discovery_posture"),
        research_volume_dense_symbols=_metric_str(research, "discovery_volume_dense_symbols"),
        research_strategy_posture=_metric_str(research, "strategy_posture"),
        research_candidate_posture=_metric_str(research, "candidate_posture"),
        research_nightly_posture=_metric_str(research, "nightly_posture"),
        research_effective_posture=_metric_str(research, "effective_posture"),
        research_discovery_recommended_target=_metric_str(
            research,
            "discovery_recommended_refresh_target",
        ),
        research_effective_target=_metric_str(research, "effective_refresh_target"),
        research_refresh_urgency=_metric_str(research, "refresh_urgency"),
        research_refresh_urgency_target=_metric_str(research, "refresh_urgency_target"),
        research_refresh_urgency_score=_metric_float(research, "refresh_urgency_score"),
        research_refresh_urgency_rows=_metric_int(research, "refresh_urgency_rows"),
        research_refresh_urgency_summary=_metric_str(research, "refresh_urgency_summary"),
        research_follow_up_action=_metric_str(research, "follow_up_action"),
        research_discovery_follow_up_target=_metric_str(research, "discovery_follow_up_target"),
        research_discovery_follow_up_summary=_metric_str(
            research,
            "discovery_follow_up_summary",
        ),
        research_discovery_follow_up_action=_metric_str(research, "discovery_follow_up_action"),
        research_research_follow_up_target=_metric_str(research, "research_follow_up_target"),
        research_research_follow_up_summary=_metric_str(
            research,
            "research_follow_up_summary",
        ),
        research_research_follow_up_action=_metric_str(research, "research_follow_up_action"),
        research_execution_follow_up_target=_metric_str(research, "execution_follow_up_target"),
        research_execution_follow_up_summary=_metric_str(
            research,
            "execution_follow_up_summary",
        ),
        research_execution_follow_up_action=_metric_str(research, "execution_follow_up_action"),
        research_target_mismatch=_metric_bool(research, "target_mismatch"),
        research_refresh_requested=_metric_bool(research, "refresh_requested"),
        research_refreshed_count=_metric_int(research, "refreshed_count"),
        research_selected_target_mix=_metric_str(research, "selected_target_mix"),
        research_selected_regime_mix=_metric_str(research, "selected_regime_mix"),
        research_scout_support_target=_metric_str(research, "scout_support_target"),
        research_scout_support_summary=_metric_str(research, "scout_support_summary"),
        research_scout_supported_symbols=_metric_str(research, "scout_supported_symbols"),
        research_scout_overlap_selected_count=_metric_int(research, "scout_overlap_selected_count"),
        research_scout_volume_dense_selected_count=_metric_int(
            research,
            "scout_volume_dense_selected_count",
        ),
    )


def format_workflow_snapshot_summary(summary: WorkflowSnapshotSummary) -> str:
    source = summary.source or "—"
    timeframe = summary.timeframe or "—"
    lookback = f"{summary.lookback_days}d" if summary.lookback_days is not None else "—"
    return (
        "workflow summary: "
        + f"source={source} timeframe={timeframe} lookback={lookback} "
        + (
            f"team={summary.team_ok_role_count}/{summary.team_role_count} "
            if summary.team_role_count > 0
            else ""
        )
        + f"universe={summary.universe_count} shortlist={summary.shortlist_count} "
        + f"briefing={summary.briefing_candidates} "
        + f"allocation={summary.allocation_selected}/{summary.allocation_max_positions} "
        + f"skipped={summary.allocation_skipped} "
        + f"ml={summary.ml_count} rl={summary.rl_count}"
        + (
            f" training_research={summary.research_count}"
            f"({summary.research_ml_count}/{summary.research_rl_count})"
            + (
                f" refreshed={summary.research_refreshed_count}"
                if summary.research_refresh_requested
                else ""
            )
            if summary.research_count > 0 or summary.research_refresh_requested
            else ""
        )
        + (
            f" research={summary.allocation_research_target_mix}"
            if summary.allocation_research_target_mix
            else ""
        )
        + (
            f" refreshed_research={summary.allocation_refreshed_research_target_mix}"
            if summary.allocation_refreshed_research_target_mix
            else ""
        )
        + (
            f" allocation_regimes={summary.allocation_regime_mix}"
            if summary.allocation_regime_mix
            else ""
        )
        + (
            f" allocation_strategies={summary.allocation_strategy_mix}"
            if summary.allocation_strategy_mix
            else ""
        )
        + (
            f" allocation_risk={summary.allocation_risk_mix}"
            if summary.allocation_risk_mix
            else ""
        )
        + (
            f" allocation_freshness={summary.allocation_freshness_mix}"
            if summary.allocation_freshness_mix
            else ""
        )
        + (
            f" allocation_balance={summary.allocation_target_balance_summary}"
            if summary.allocation_target_balance_summary
            else ""
        )
        + (
            " allocation_critic=enabled"
            if summary.allocation_critic_enabled
            else ""
        )
        + (
            f" nightly={summary.team_nightly_aligned_reports}/"
            f"{summary.team_nightly_enabled_reports}"
            if summary.team_nightly_report_count > 0
            else ""
        )
        + (
            f" nightly_recent={summary.team_nightly_recent_aligned}/"
            f"{summary.team_nightly_recent_enabled}"
            if summary.team_nightly_recent_window > 0
            else ""
        )
        + (
            f" nightly_posture={summary.team_nightly_posture}"
            if summary.team_nightly_posture
            else ""
        )
        + (
            f" nightly_target={summary.team_nightly_recommended_target}"
            if summary.team_nightly_recommended_target
            else ""
        )
        + (
            f" nightly_execution={summary.team_nightly_latest_execution_target or '—'}"
            + (
                f"/{summary.team_nightly_latest_execution_model_family}"
                if summary.team_nightly_latest_execution_model_family
                else ""
            )
            + (
                f" source={summary.team_nightly_latest_execution_selection_source}"
                if summary.team_nightly_latest_execution_selection_source
                else ""
            )
            if (
                summary.team_nightly_latest_execution_target
                or summary.team_nightly_latest_execution_model_family
                or summary.team_nightly_latest_execution_selection_source
            )
            else ""
        )
        + (
            " team_research="
            + f"{summary.team_research_ml_count}/{summary.team_research_rl_count}"
            + (
                f" policy={summary.team_research_selection_policy}"
                if summary.team_research_selection_policy
                else ""
            )
            + (
                f" target={summary.team_research_refresh_target}"
                if summary.team_research_refresh_target
                else ""
            )
            + (
                f" discovery={summary.team_research_discovery_summary}"
                if summary.team_research_discovery_summary
                else ""
            )
            + (
                f" discovery_target={summary.team_research_discovery_recommended_target}"
                if summary.team_research_discovery_recommended_target
                else ""
            )
            + (
                f" candidate={summary.team_research_candidate_posture}"
                if summary.team_research_candidate_posture
                else ""
            )
            + (
                f" nightly={summary.team_research_nightly_posture}"
                if summary.team_research_nightly_posture
                else ""
            )
            + (
                f" posture={summary.team_research_effective_posture}"
                if summary.team_research_effective_posture
                else ""
            )
            + (
                f" effective_target={summary.team_research_effective_target}"
                if summary.team_research_effective_target
                else ""
            )
            + (
                f" urgency={summary.team_research_refresh_urgency}"
                if summary.team_research_refresh_urgency
                else ""
            )
            + (
                f"/{summary.team_research_refresh_urgency_target}"
                if summary.team_research_refresh_urgency_target
                else ""
            )
            + (
                f" pressure=+{summary.team_research_refresh_urgency_score:.2f}"
                if summary.team_research_refresh_urgency_score is not None
                else ""
            )
            + (
                f" rows={summary.team_research_refresh_urgency_rows}"
                if summary.team_research_refresh_urgency_rows > 0
                else ""
            )
            + (
                f" targets={summary.team_research_selected_target_mix}"
                if summary.team_research_selected_target_mix
                else ""
            )
            + (
                f" scouts={summary.team_research_scout_support_summary}"
                if summary.team_research_scout_support_summary
                else ""
            )
            + (
                f" regimes={summary.team_research_selected_regime_mix}"
                if summary.team_research_selected_regime_mix
                else ""
            )
            + (
                f" refreshed={summary.team_research_refreshed_count}"
                if summary.team_research_refresh_requested
                else ""
            )
            + (
                f" alignment={summary.team_research_alignment_summary}"
                if summary.team_research_alignment_summary
                else ""
            )
            + (
                " mismatch=1"
                if summary.team_research_target_mismatch
                else ""
            )
            + (
                f" follow_up={summary.team_research_follow_up_action}"
                if summary.team_research_follow_up_action
                else ""
            )
            + (
                f" discovery_follow_up={summary.team_research_discovery_follow_up_target}"
                if summary.team_research_discovery_follow_up_target
                else ""
            )
            + (
                f" research_follow_up={summary.team_research_research_follow_up_target}"
                if summary.team_research_research_follow_up_target
                else ""
            )
            + (
                f" execution_follow_up={summary.team_research_execution_follow_up_target}"
                if summary.team_research_execution_follow_up_target
                else ""
            )
            if summary.team_research_headline
            else ""
        )
        + (
            " nightly_force_refresh=1"
            if summary.team_nightly_recommended_force_refresh
            else ""
        )
        + (
            " discovery_refresh="
            + f"{summary.team_discovery_refreshed_count}"
            + (
                f" source={summary.team_discovery_refresh_source}"
                if summary.team_discovery_refresh_source
                else ""
            )
            + (
                f" timeframe={summary.team_discovery_refresh_timeframe}"
                if summary.team_discovery_refresh_timeframe
                else ""
            )
            + (
                f" days={summary.team_discovery_refresh_days}"
                if summary.team_discovery_refresh_days > 0
                else ""
            )
            if summary.team_discovery_refresh_requested
            else ""
        )
        + (
            f" discovery={summary.team_discovery_alignment_summary}"
            if summary.team_discovery_alignment_summary
            else ""
        )
        + (
            f" artifact_follow_up={summary.team_artifact_follow_up_action}"
            if summary.team_artifact_follow_up_action
            else ""
        )
        + (
            f" artifact_target={summary.team_artifact_follow_up_target}"
            if summary.team_artifact_follow_up_target
            else ""
        )
        + (
            f" artifact_recovery={summary.team_artifact_recovery_posture}"
            if summary.team_artifact_recovery_posture
            else ""
        )
        + (
            f" provider={summary.universe_provider_summary}"
            if summary.universe_provider_summary
            else ""
        )
        + (
            f" scouts={summary.universe_scout_summary}"
            if summary.universe_scout_summary
            else ""
        )
        + (
            f" overlap={summary.team_discovery_overlap_symbols}"
            if summary.team_discovery_overlap_symbols
            else ""
        )
        + (
            f" adaptive={summary.universe_top_adaptive_note}"
            if summary.universe_top_adaptive_note
            else ""
        )
    )


def _metric_int(section: dict[str, Any], key: str) -> int:
    metrics = section.get("metrics") if isinstance(section, dict) else None
    if not isinstance(metrics, dict):
        return 0
    return _coerce_int(metrics.get(key)) or 0


def _metric_str(section: dict[str, Any], key: str) -> Optional[str]:
    metrics = section.get("metrics") if isinstance(section, dict) else None
    if not isinstance(metrics, dict):
        return None
    value = str(metrics.get(key, "") or "").strip()
    return value or None


def _metric_float(section: dict[str, Any], key: str) -> Optional[float]:
    metrics = section.get("metrics") if isinstance(section, dict) else None
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _metric_bool(section: dict[str, Any], key: str) -> bool:
    metrics = section.get("metrics") if isinstance(section, dict) else None
    if not isinstance(metrics, dict):
        return False
    value = metrics.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _first_note(section: dict[str, Any]) -> Optional[str]:
    notes = section.get("notes") if isinstance(section, dict) else None
    if not isinstance(notes, list) or not notes:
        return None
    note = str(notes[0] or "").strip()
    return note or None


def _has_note(section: dict[str, Any], note_text: str) -> bool:
    notes = section.get("notes") if isinstance(section, dict) else None
    if not isinstance(notes, list):
        return False
    target = str(note_text or "").strip()
    return any(str(note or "").strip() == target for note in notes)


def _prefixed_note(section: dict[str, Any], prefix: str) -> Optional[str]:
    notes = section.get("notes") if isinstance(section, dict) else None
    if not isinstance(notes, list):
        return None
    target = str(prefix or "").strip()
    for note in notes:
        text = str(note or "").strip()
        if text.startswith(target):
            return text[len(target) :].strip() or None
    return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
