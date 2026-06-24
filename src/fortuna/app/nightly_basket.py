"""Resolve nightly training baskets without importing the full RL stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fortuna.config.settings import get_settings


def load_training_candidates_manifest(path: Path):
    from fortuna.app.training_candidates import load_training_candidates_manifest as _load

    return _load(path)


def export_training_candidates(response, out_path: Path | str):
    from fortuna.app.training_candidates import export_training_candidates as _export

    return _export(response, Path(out_path))


def select_training_symbols(
    manifest,
    *,
    target: str,
    top_n: Optional[int] = None,
    selection_policy: str = "ranked",
):
    from fortuna.app.training_candidates import select_training_symbols as _select

    return _select(
        manifest,
        target=target,
        top_n=top_n,
        selection_policy=selection_policy,
    )


def build_training_candidates(**kwargs):
    from fortuna.app.training_candidates import build_training_candidates as _build

    return _build(**kwargs)


def build_training_research_plan(**kwargs):
    from fortuna.app.training_research import build_training_research_plan as _build

    return _build(**kwargs)


def export_training_research_plan(response, out_path: Path | str):
    from fortuna.app.training_research import export_training_research_plan as _export

    return _export(response, Path(out_path))


def build_workflow_snapshot(**kwargs):
    from fortuna.app.operator_workflow import build_workflow_snapshot as _build

    return _build(**kwargs)


def export_workflow_snapshot(snapshot, out_path: Path | str):
    from fortuna.app.operator_workflow import export_workflow_snapshot as _export

    return _export(snapshot, out_path)


def resolve_training_candidate_response(args: Any):
    if str(getattr(args, "candidate_manifest", "") or "").strip():
        return load_training_candidates_manifest(Path(args.candidate_manifest))
    if bool(getattr(args, "use_training_candidates", False)):
        settings = get_settings()
        response = build_training_candidates(
            settings=settings,
            universe_limit=args.candidate_universe_limit,
            analysis_limit=args.candidate_analysis_limit,
            timeframe=args.timeframe,
            days=args.days,
            source=args.candidate_source,
        )
        if not response.ok:
            message = (
                response.error.message
                if response.error is not None
                else "candidate build failed"
            )
            raise RuntimeError(message)
        return response
    return None


def _resolved_candidate_target(
    args: Any,
    *,
    allow_auto_narrow: bool = True,
) -> str:
    requested_target = str(getattr(args, "candidate_target", "all") or "all").strip().lower()
    if requested_target in {"ml", "rl"}:
        return requested_target
    if (
        not allow_auto_narrow
        or not bool(getattr(args, "use_training_candidates", False))
        or str(getattr(args, "candidate_manifest", "") or "").strip()
    ):
        return "all"
    response = _candidate_target_research_probe(args)
    if response is None:
        return "all"
    discovery_target = str(
        (
            getattr(response, "effective_refresh_target", None)
            or getattr(response, "discovery_recommended_refresh_target", "")
            or ""
        )
    ).strip().lower()
    if discovery_target in {"ml", "rl"}:
        return discovery_target
    return "all"


def _candidate_target_research_probe(args: Any):
    settings = get_settings()
    try:
        response = build_training_research_plan(
            settings=settings,
            universe_limit=int(getattr(args, "candidate_universe_limit", 15)),
            analysis_limit=int(getattr(args, "candidate_analysis_limit", 8)),
            timeframe=str(getattr(args, "timeframe", "5m")),
            days=int(getattr(args, "days", 30)),
            source=str(getattr(args, "candidate_source", "auto")),
            ml_top_n=max(
                1,
                int(
                    getattr(args, "candidate_top_n", 0)
                    or getattr(args, "candidate_analysis_limit", 8)
                ),
            ),
            rl_top_n=max(
                1,
                int(
                    getattr(args, "candidate_top_n", 0)
                    or min(3, getattr(args, "candidate_analysis_limit", 8))
                ),
            ),
            selection_policy=str(getattr(args, "candidate_selection_policy", "ranked")),
            refresh_data=False,
            refresh_target="all",
        )
    except Exception:
        return None
    if not getattr(response, "ok", False):
        return None
    return response


def resolve_symbol_basket(
    args: Any,
    *,
    candidate_response=None,
    research_response=None,
) -> tuple[list[str], dict[str, object]]:
    """Resolve the nightly training basket from explicit symbols or candidate inputs."""
    basket_meta: dict[str, object]
    requested_target = str(getattr(args, "candidate_target", "all") or "all").strip().lower()
    planning_response = research_response
    if planning_response is None and _candidate_target_probe_enabled(args):
        planning_response = _candidate_target_research_probe(args)
    resolved_target = _resolved_candidate_target_from_response(
        requested_target=requested_target,
        response=planning_response,
    )
    if str(getattr(args, "candidate_manifest", "") or "").strip():
        manifest = candidate_response or resolve_training_candidate_response(args)
        cash_symbols = select_training_symbols(
            manifest,
            target=resolved_target,
            top_n=(args.candidate_top_n or None),
            selection_policy=getattr(args, "candidate_selection_policy", "ranked"),
        )
        basket_meta = {
            "mode": "candidate_manifest",
            "source": manifest.source,
            "target": resolved_target,
            "requested_target": requested_target,
            "target_auto_narrowed": resolved_target != requested_target,
            "selection_policy": getattr(args, "candidate_selection_policy", "ranked"),
            "count": len(cash_symbols),
            "candidate_count": len(getattr(manifest, "candidates", ()) or ()),
        }
    elif bool(getattr(args, "use_training_candidates", False)):
        response = candidate_response or resolve_training_candidate_response(args)
        research_symbols = _resolve_research_selected_symbols(
            planning_response,
            target=resolved_target,
            top_n=(args.candidate_top_n or None),
        )
        cash_symbols = research_symbols or select_training_symbols(
            response,
            target=resolved_target,
            top_n=(args.candidate_top_n or None),
            selection_policy=getattr(args, "candidate_selection_policy", "ranked"),
        )
        basket_meta = {
            "mode": "derived_training_candidates",
            "source": response.source,
            "target": resolved_target,
            "requested_target": requested_target,
            "target_auto_narrowed": resolved_target != requested_target,
            "selection_policy": getattr(args, "candidate_selection_policy", "ranked"),
            "count": len(cash_symbols),
            "candidate_count": len(getattr(response, "candidates", ()) or ()),
            "selection_source": "training_research" if research_symbols else "training_candidates",
            "discovery_summary": getattr(planning_response, "discovery_summary", None),
            "discovery_recommended_refresh_target": getattr(
                planning_response,
                "discovery_recommended_refresh_target",
                None,
            ),
        }
    else:
        cash_symbols = args.symbols[: args.max_symbols] if args.max_symbols else args.symbols
        basket_meta = {"mode": "explicit", "count": len(cash_symbols)}

    if not cash_symbols:
        raise RuntimeError("No training symbols resolved for nightly basket")

    if args.include_futures:
        if args.futures_symbols is not None:
            futures_symbols = args.futures_symbols
        else:
            futures_symbols = [
                s.replace(".NS", ".FUT") if s.endswith(".NS") else s
                for s in cash_symbols
            ]
        symbols = list(cash_symbols) + list(futures_symbols)
    else:
        symbols = list(cash_symbols)
    basket_meta["total"] = len(symbols)
    return symbols, basket_meta


def _candidate_target_probe_enabled(args: Any) -> bool:
    return bool(getattr(args, "use_training_candidates", False)) and not str(
        getattr(args, "candidate_manifest", "") or ""
    ).strip()


def _resolved_candidate_target_from_response(
    requested_target: str,
    *,
    response=None,
) -> str:
    normalized_requested = str(requested_target or "all").strip().lower()
    if normalized_requested in {"ml", "rl"}:
        return normalized_requested
    if response is None:
        return "all"
    discovery_target = str(
        (
            getattr(response, "effective_refresh_target", None)
            or getattr(response, "discovery_recommended_refresh_target", "")
            or ""
        )
    ).strip().lower()
    if discovery_target in {"ml", "rl"}:
        return discovery_target
    return "all"


def _resolve_research_selected_symbols(
    response,
    *,
    target: str,
    top_n: Optional[int],
) -> list[str]:
    if response is None:
        return []
    rows = tuple(getattr(response, "rows", ()) or ())
    if not rows:
        return []
    target_key = str(target or "all").strip().lower()
    eligible_rows = [
        row
        for row in rows
        if _research_row_matches_target(
            str(getattr(row, "target", "") or ""),
            target_key,
        )
    ]
    if not eligible_rows:
        return []
    eligible_rows = sorted(
        eligible_rows,
        key=lambda row: (
            int(getattr(row, "selection_rank", 0) or 0) <= 0,
            int(getattr(row, "selection_rank", 0) or 0),
            int(getattr(row, "shortlist_rank", 0) or 0),
            int(getattr(row, "universe_rank", 0) or 0),
            str(getattr(row, "symbol", "") or ""),
        ),
    )
    limit = int(top_n or 0)
    if limit > 0:
        eligible_rows = eligible_rows[:limit]
    return [
        str(getattr(row, "symbol", "") or "")
        for row in eligible_rows
        if str(getattr(row, "symbol", "") or "").strip()
    ]


def _research_row_matches_target(row_target: str, target: str) -> bool:
    normalized_row = str(row_target or "").strip().lower()
    normalized_target = str(target or "all").strip().lower()
    if normalized_target == "all":
        return normalized_row in {"ml", "rl", "both"}
    if normalized_target == "ml":
        return normalized_row in {"ml", "both"}
    if normalized_target == "rl":
        return normalized_row in {"rl", "both"}
    return False


def emit_training_candidate_manifest(
    args: Any,
    *,
    report_dir: Path | str,
    candidate_response=None,
) -> Optional[dict[str, object]]:
    explicit_out = str(getattr(args, "training_candidate_out", "") or "").strip()
    if (
        not str(getattr(args, "candidate_manifest", "") or "").strip()
        and not bool(getattr(args, "use_training_candidates", False))
        and not explicit_out
    ):
        return None

    response = candidate_response or resolve_training_candidate_response(args)
    if response is None:
        return None

    settings = get_settings()
    out_path = (
        settings.resolve_path(Path(explicit_out))
        if explicit_out
        else Path(report_dir) / "training_candidates.json"
    )
    written = export_training_candidates(response, out_path)
    ml_count = sum(1 for row in response.candidates if getattr(row, "ml_candidate", False))
    rl_count = sum(1 for row in response.candidates if getattr(row, "rl_candidate", False))
    return {
        "path": str(written),
        "source": response.source,
        "count": len(response.candidates),
        "ml_count": ml_count,
        "rl_count": rl_count,
        "selection_policy": getattr(args, "candidate_selection_policy", "ranked"),
    }


def emit_training_research_plan(
    args: Any,
    *,
    report_dir: Path | str,
) -> Optional[dict[str, object]]:
    explicit_out = str(getattr(args, "training_research_out", "") or "").strip()
    if not bool(getattr(args, "use_training_candidates", False)) and not explicit_out:
        return None

    settings = get_settings()
    resolved_target = _resolved_candidate_target(args)
    requested_target = str(getattr(args, "candidate_target", "all") or "all").strip().lower()
    response = build_training_research_plan(
        settings=settings,
        universe_limit=int(getattr(args, "candidate_universe_limit", 15)),
        analysis_limit=int(getattr(args, "candidate_analysis_limit", 8)),
        timeframe=str(getattr(args, "timeframe", "5m")),
        days=int(getattr(args, "days", 30)),
        source=str(getattr(args, "candidate_source", "auto")),
        ml_top_n=max(
            1,
            int(
                getattr(args, "candidate_top_n", 0)
                or getattr(args, "candidate_analysis_limit", 8)
            ),
        ),
        rl_top_n=max(
            1,
            int(
                getattr(args, "candidate_top_n", 0)
                or min(3, getattr(args, "candidate_analysis_limit", 8))
            ),
        ),
        selection_policy=str(getattr(args, "candidate_selection_policy", "ranked")),
        refresh_data=False,
        refresh_target=resolved_target,
    )
    if not response.ok:
        message = (
            response.error.message
            if response.error is not None
            else "training research plan build failed"
        )
        raise RuntimeError(message)

    out_path = (
        settings.resolve_path(Path(explicit_out))
        if explicit_out
        else Path(report_dir) / "training_research_plan.json"
    )
    written = export_training_research_plan(response, out_path)
    return {
        "path": str(written),
        "source": response.source,
        "count": len(response.rows),
        "ml_count": len(response.ml_symbols),
        "rl_count": len(response.rl_symbols),
        "selection_policy": response.selection_policy,
        "requested_refresh_target": requested_target,
        "refresh_target": response.refresh_target,
        "effective_refresh_target": (
            getattr(response, "effective_refresh_target", None) or response.refresh_target
        ),
        "target_auto_narrowed": response.refresh_target != requested_target,
        "refresh_requested": response.refresh_requested,
        "refreshed_count": sum(1 for row in response.rows if row.refreshed),
        "discovery_posture": getattr(response, "discovery_posture", None),
        "strategy_posture": getattr(response, "strategy_posture", None),
        "effective_posture": getattr(response, "effective_posture", None),
        "selected_target_mix": getattr(response, "selected_target_mix", None),
        "selected_regime_mix": getattr(response, "selected_regime_mix", None),
        "scout_support_target": getattr(response, "scout_support_target", None),
        "discovery_preferred_count": len(
            getattr(response, "discovery_preferred_symbols", ()) or ()
        ),
        "discovery_regime_mix": getattr(response, "discovery_regime_mix", None),
        "discovery_summary": getattr(response, "discovery_summary", None),
        "discovery_recommended_refresh_target": getattr(
            response,
            "discovery_recommended_refresh_target",
            None,
        ),
    }


def emit_workflow_snapshot(
    args: Any,
    *,
    report_dir: Path | str,
) -> Optional[dict[str, object]]:
    """Optionally emit a workflow snapshot for nightly candidate-driven runs."""
    explicit_out = str(getattr(args, "workflow_snapshot_out", "") or "").strip()
    if not bool(getattr(args, "use_training_candidates", False)) and not explicit_out:
        return None

    settings = get_settings()
    snapshot = build_workflow_snapshot(
        settings=settings,
        universe_limit=int(getattr(args, "candidate_universe_limit", 15)),
        analysis_limit=int(getattr(args, "candidate_analysis_limit", 8)),
        candidate_limit=max(
            int(getattr(args, "candidate_analysis_limit", 8)),
            int(getattr(args, "candidate_top_n", 0) or 0),
        )
        or 8,
        timeframe=str(getattr(args, "timeframe", "5m")),
        days=int(getattr(args, "days", 30)),
        source=str(getattr(args, "candidate_source", "auto")),
    )
    out_path = (
        settings.resolve_path(Path(explicit_out))
        if explicit_out
        else Path(report_dir) / "workflow_snapshot.json"
    )
    written = export_workflow_snapshot(snapshot, out_path)
    return {
        "path": str(written),
        "source": snapshot.source,
        "team_role_count": snapshot.team.metrics.get("count", 0),
        "team_ok_role_count": snapshot.team.metrics.get("ok_count", 0),
        "team_headline": snapshot.team.metrics.get("headline"),
        "team_nightly_report_count": snapshot.team.metrics.get("nightly_report_count", 0),
        "team_nightly_enabled_reports": snapshot.team.metrics.get(
            "nightly_enabled_reports",
            0,
        ),
        "team_nightly_aligned_reports": snapshot.team.metrics.get(
            "nightly_aligned_reports",
            0,
        ),
        "team_nightly_latest_target_mix": snapshot.team.metrics.get(
            "nightly_latest_target_mix"
        ),
        "team_nightly_latest_refreshed_target_mix": snapshot.team.metrics.get(
            "nightly_latest_refreshed_target_mix"
        ),
        "team_nightly_recommended_target": snapshot.team.metrics.get(
            "nightly_recommended_target"
        ),
        "team_nightly_recommended_force_refresh": snapshot.team.metrics.get(
            "nightly_recommended_force_refresh",
            False,
        ),
        "team_research_headline": snapshot.team.metrics.get("research_headline"),
        "team_research_selection_policy": snapshot.team.metrics.get(
            "research_selection_policy"
        ),
        "team_research_refresh_target": snapshot.team.metrics.get("research_refresh_target"),
        "team_research_discovery_recommended_target": snapshot.team.metrics.get(
            "research_discovery_recommended_target"
        ),
        "team_research_discovery_posture": snapshot.team.metrics.get(
            "research_discovery_posture"
        ),
        "team_research_strategy_posture": snapshot.team.metrics.get(
            "research_strategy_posture"
        ),
        "team_research_effective_posture": snapshot.team.metrics.get(
            "research_effective_posture"
        ),
        "team_research_effective_target": snapshot.team.metrics.get("research_effective_target"),
        "team_research_refresh_urgency_summary": snapshot.team.metrics.get(
            "research_refresh_urgency_summary"
        ),
        "team_research_refresh_urgency_target": snapshot.team.metrics.get(
            "research_refresh_urgency_target"
        ),
        "team_research_follow_up_action": snapshot.team.metrics.get("research_follow_up_action"),
        "team_research_discovery_follow_up_target": snapshot.team.metrics.get(
            "research_discovery_follow_up_target"
        ),
        "team_research_discovery_follow_up_summary": snapshot.team.metrics.get(
            "research_discovery_follow_up_summary"
        ),
        "team_research_discovery_follow_up_action": snapshot.team.metrics.get(
            "research_discovery_follow_up_action"
        ),
        "team_research_research_follow_up_target": snapshot.team.metrics.get(
            "research_research_follow_up_target"
        ),
        "team_research_research_follow_up_summary": snapshot.team.metrics.get(
            "research_research_follow_up_summary"
        ),
        "team_research_research_follow_up_action": snapshot.team.metrics.get(
            "research_research_follow_up_action"
        ),
        "team_research_execution_follow_up_target": snapshot.team.metrics.get(
            "research_execution_follow_up_target"
        ),
        "team_research_execution_follow_up_summary": snapshot.team.metrics.get(
            "research_execution_follow_up_summary"
        ),
        "team_research_execution_follow_up_action": snapshot.team.metrics.get(
            "research_execution_follow_up_action"
        ),
        "team_research_target_mismatch": snapshot.team.metrics.get(
            "research_target_mismatch",
            False,
        ),
        "team_research_refresh_requested": snapshot.team.metrics.get(
            "research_refresh_requested",
            False,
        ),
        "team_research_refreshed_count": snapshot.team.metrics.get(
            "research_refreshed_count",
            0,
        ),
        "team_research_ml_count": snapshot.team.metrics.get("research_ml_count", 0),
        "team_research_rl_count": snapshot.team.metrics.get("research_rl_count", 0),
        "team_research_selected_target_mix": snapshot.team.metrics.get(
            "research_selected_target_mix"
        ),
        "team_research_selected_regime_mix": snapshot.team.metrics.get(
            "research_selected_regime_mix"
        ),
        "team_research_scout_support_target": snapshot.team.metrics.get(
            "research_scout_support_target"
        ),
        "team_research_scout_support_summary": snapshot.team.metrics.get(
            "research_scout_support_summary"
        ),
        "team_research_scout_supported_symbols": snapshot.team.metrics.get(
            "research_scout_supported_symbols"
        ),
        "team_research_scout_overlap_selected_count": snapshot.team.metrics.get(
            "research_scout_overlap_selected_count",
            0,
        ),
        "team_research_scout_volume_dense_selected_count": snapshot.team.metrics.get(
            "research_scout_volume_dense_selected_count",
            0,
        ),
        "team_research_alignment_summary": snapshot.team.metrics.get(
            "research_alignment_summary"
        ),
        "team_research_alignment_target_mix": snapshot.team.metrics.get(
            "research_alignment_target_mix"
        ),
        "team_research_alignment_overlap_count": snapshot.team.metrics.get(
            "research_alignment_overlap_count",
            0,
        ),
        "team_research_alignment_selected_count": snapshot.team.metrics.get(
            "research_alignment_selected_count",
            0,
        ),
        "team_discovery_alignment_summary": snapshot.team.metrics.get(
            "discovery_alignment_summary"
        ),
        "team_discovery_overlap_symbols": snapshot.team.metrics.get(
            "discovery_overlap_symbols"
        ),
        "team_discovery_alignment_overlap_count": snapshot.team.metrics.get(
            "discovery_alignment_overlap_count",
            0,
        ),
        "team_discovery_alignment_compare_count": snapshot.team.metrics.get(
            "discovery_alignment_compare_count",
            0,
        ),
        "universe_count": snapshot.universe.metrics.get("count", 0),
        "shortlist_count": snapshot.shortlist.metrics.get("count", 0),
        "briefing_candidates": snapshot.briefing.metrics.get("candidates", 0),
        "allocation_research_alignment_enabled": _has_snapshot_note(
            snapshot.allocation,
            "portfolio_research_alignment=enabled",
        ),
        "allocation_research_target_mix": _prefixed_snapshot_note(
            snapshot.allocation,
            "allocation_research_targets:",
        ),
        "allocation_refreshed_research_target_mix": _prefixed_snapshot_note(
            snapshot.allocation,
            "allocation_refreshed_research_targets:",
        ),
        "ml_count": snapshot.candidates.metrics.get("ml_count", 0),
        "rl_count": snapshot.candidates.metrics.get("rl_count", 0),
        "research_count": snapshot.research.metrics.get("count", 0),
        "research_ml_count": snapshot.research.metrics.get("ml_count", 0),
        "research_rl_count": snapshot.research.metrics.get("rl_count", 0),
        "research_selection_policy": snapshot.research.metrics.get("selection_policy"),
        "research_refresh_target": snapshot.research.metrics.get("refresh_target"),
        "research_discovery_posture": snapshot.research.metrics.get("discovery_posture"),
        "research_strategy_posture": snapshot.research.metrics.get("strategy_posture"),
        "research_effective_posture": snapshot.research.metrics.get("effective_posture"),
        "research_discovery_recommended_refresh_target": snapshot.research.metrics.get(
            "discovery_recommended_refresh_target"
        ),
        "research_effective_refresh_target": snapshot.research.metrics.get(
            "effective_refresh_target"
        ),
        "research_refresh_urgency_summary": snapshot.research.metrics.get(
            "refresh_urgency_summary"
        ),
        "research_refresh_urgency_target": snapshot.research.metrics.get(
            "refresh_urgency_target"
        ),
        "research_follow_up_action": snapshot.research.metrics.get("follow_up_action"),
        "research_discovery_follow_up_target": snapshot.research.metrics.get(
            "discovery_follow_up_target"
        ),
        "research_discovery_follow_up_summary": snapshot.research.metrics.get(
            "discovery_follow_up_summary"
        ),
        "research_discovery_follow_up_action": snapshot.research.metrics.get(
            "discovery_follow_up_action"
        ),
        "research_research_follow_up_target": snapshot.research.metrics.get(
            "research_follow_up_target"
        ),
        "research_research_follow_up_summary": snapshot.research.metrics.get(
            "research_follow_up_summary"
        ),
        "research_research_follow_up_action": snapshot.research.metrics.get(
            "research_follow_up_action"
        ),
        "research_execution_follow_up_target": snapshot.research.metrics.get(
            "execution_follow_up_target"
        ),
        "research_execution_follow_up_summary": snapshot.research.metrics.get(
            "execution_follow_up_summary"
        ),
        "research_execution_follow_up_action": snapshot.research.metrics.get(
            "execution_follow_up_action"
        ),
        "research_target_mismatch": snapshot.research.metrics.get("target_mismatch", False),
        "research_refresh_requested": snapshot.research.metrics.get("refresh_requested", False),
        "research_refreshed_count": snapshot.research.metrics.get("refreshed_count", 0),
        "research_selected_target_mix": snapshot.research.metrics.get("selected_target_mix"),
        "research_selected_regime_mix": snapshot.research.metrics.get("selected_regime_mix"),
        "research_scout_support_target": snapshot.research.metrics.get("scout_support_target"),
        "research_scout_support_summary": snapshot.research.metrics.get("scout_support_summary"),
        "research_scout_supported_symbols": snapshot.research.metrics.get(
            "scout_supported_symbols"
        ),
        "research_scout_overlap_selected_count": snapshot.research.metrics.get(
            "scout_overlap_selected_count",
            0,
        ),
        "research_scout_volume_dense_selected_count": snapshot.research.metrics.get(
            "scout_volume_dense_selected_count",
            0,
        ),
    }


def _has_snapshot_note(summary, note_text: str) -> bool:
    notes = getattr(summary, "notes", ()) or ()
    target = str(note_text or "").strip()
    return any(str(note or "").strip() == target for note in notes)


def _prefixed_snapshot_note(summary, prefix: str) -> str | None:
    notes = getattr(summary, "notes", ()) or ()
    target = str(prefix or "").strip()
    for note in notes:
        text = str(note or "").strip()
        if text.startswith(target):
            return text[len(target) :].strip() or None
    return None
