"""Pure helpers for dashboard model health summaries (no Streamlit dependency)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from fortuna.agentic.contracts import (
    AgenticDecisionRow,
    AgenticStatusSummary,
    LearningRowSummary,
    LearningSummaryStatus,
    MlModelStatus,
    ModelHealthResponse,
    NightlyAlignmentRow,
    NightlyAlignmentStatus,
    PromotionRecord,
    RegimeModelStatus,
    RlModelStatus,
)
from fortuna.agentic.learning import LearningExample
from fortuna.app.model_activation import build_model_activation_summary
from fortuna.app.runtime_readiness import build_runtime_readiness_summary
from fortuna.app.scaling_posture import build_scaling_posture_summary
from fortuna.app.workflow_snapshot import load_workflow_snapshot_summary
from fortuna.models.metadata import ModelKind
from fortuna.models.registry import live_pointer_path, read_audit


def build_rl_status(
    engine: Any,
    *,
    models_root: Path | str = "models",
) -> RlModelStatus:
    sym = ""
    try:
        sym = engine.state.symbol or ""
    except Exception:  # noqa: BLE001
        pass

    rl_gen = getattr(engine, "rl_generator", None)
    available = bool(rl_gen and getattr(rl_gen, "is_available", False))
    meta = getattr(rl_gen, "metadata", None) if rl_gen else None
    pointer = None
    if sym:
        pointer = live_pointer_path(
            ModelKind.RL_POLICY,
            models_root,
            symbol=sym,
        )
    audit = _latest_promotion(ModelKind.RL_POLICY, models_root, symbol=sym or None)
    last_promotion = _enrich_promotion(PromotionRecord.from_audit_row(audit))

    base = RlModelStatus(
        symbol=sym,
        available=available,
        advisory_ready=bool(getattr(rl_gen, "is_advisory_ready", False)) if rl_gen else False,
        load_error=getattr(rl_gen, "load_error", None) if rl_gen else "no_generator",
        live_pointer=str(pointer) if pointer else None,
        checkpoint_dir=str(getattr(rl_gen, "checkpoint_dir", None) or ""),
        last_promotion=last_promotion,
    )
    if meta is None:
        return base

    return RlModelStatus(
        symbol=base.symbol,
        available=base.available,
        advisory_ready=base.advisory_ready,
        load_error=base.load_error,
        live_pointer=base.live_pointer,
        checkpoint_dir=base.checkpoint_dir,
        last_promotion=base.last_promotion,
        run_id=meta.run_id,
        verdict_passed=bool(meta.verdict_passed),
        policy_type=meta.policy_type,
        baseline_sharpe=meta.baseline_sharpe,
        beats_baseline=meta.beats_baseline,
        oos_sharpe=float(meta.oos_metrics.sharpe_ratio),
        oos_pf=float(meta.oos_metrics.profit_factor),
        oos_trades=int(meta.oos_metrics.total_trades),
    )


def build_ml_status(
    engine: Any,
    *,
    models_root: Path | str = "models",
    ml_base: Optional[Path | str] = None,
) -> MlModelStatus:
    orch = getattr(engine, "_agentic_orchestrator", None)
    scorer = None
    if orch is not None:
        ml_agent = getattr(orch, "_ml", None)
        scorer = getattr(ml_agent, "_scorer", None) if ml_agent else None

    pointer = live_pointer_path(ModelKind.ML_SCORER, models_root, ml_base=ml_base)
    meta = getattr(scorer, "metadata", None) if scorer else None
    available = scorer is not None and meta is not None
    audit = _latest_promotion(ModelKind.ML_SCORER, models_root)
    last_promotion = _enrich_promotion(PromotionRecord.from_audit_row(audit))

    settings = getattr(engine, "settings", None)
    ml_enabled = bool(getattr(settings, "agentic_ml_scorer_enabled", False)) if settings else False
    base = MlModelStatus(
        available=available,
        enabled=ml_enabled,
        live_pointer=str(pointer),
        artifact_dir=str((audit or {}).get("artifact_dir", "") or ""),
        load_error=None if available else "not_loaded",
        last_promotion=last_promotion,
    )
    if meta is None:
        return base

    return MlModelStatus(
        available=base.available,
        enabled=base.enabled,
        live_pointer=base.live_pointer,
        artifact_dir=base.artifact_dir,
        load_error=base.load_error,
        last_promotion=base.last_promotion,
        run_id=meta.run_id,
        verdict_passed=bool(meta.verdict_passed),
        advisory_ready=bool(meta.advisory_ready),
        feature_schema_hash=meta.feature_schema_hash,
        oos_precision=float(meta.oos_metrics.precision),
        oos_roc_auc=float(meta.oos_metrics.roc_auc),
    )


def build_regime_status(
    models_root: Path | str = "models",
) -> RegimeModelStatus:
    root = Path(models_root)
    joblib_path = root / "regime" / "classifier.joblib"
    pointer = live_pointer_path(ModelKind.REGIME_DETECTOR, root)
    return RegimeModelStatus(
        available=joblib_path.is_file(),
        path=str(joblib_path),
        live_pointer=str(pointer),
    )


def build_learning_summary(
    store: Any,
    *,
    symbol: Optional[str] = None,
) -> LearningSummaryStatus:
    rows = _safe_learning_rows(store)
    if symbol is not None:
        sym = str(symbol)
        rows = [r for r in rows if r.symbol == sym]
    if not rows:
        return LearningSummaryStatus()

    resolved = [r for r in rows if r.outcome.status == "resolved"]
    paper_closed = [r for r in rows if r.outcome.paper_closed]
    realized = [
        float(r.outcome.realized_pnl_pct)
        for r in paper_closed
        if r.outcome.realized_pnl_pct is not None
    ]
    recent_rows = tuple(
        LearningRowSummary(
            symbol=row.symbol,
            action=row.action,
            bar_time=row.bar_time or "",
            resolved=row.outcome.status == "resolved",
            paper_closed=bool(row.outcome.paper_closed),
            realized_pnl_pct=row.outcome.realized_pnl_pct,
        )
        for row in rows[-10:]
    )
    return LearningSummaryStatus(
        total_rows=len(rows),
        resolved_rows=len(resolved),
        paper_closed_rows=len(paper_closed),
        avg_realized_pnl_pct=(sum(realized) / len(realized)) if realized else None,
        recent_rows=recent_rows,
    )


def build_agentic_decisions_summary(
    store: Any,
    *,
    limit: int = 10,
) -> tuple[AgenticDecisionRow, ...]:
    if store is None:
        return ()
    try:
        rows = store.read_decisions(limit)
    except Exception:  # noqa: BLE001
        return ()
    out: list[AgenticDecisionRow] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            AgenticDecisionRow(
                symbol=row.get("symbol"),
                action=str(row.get("action", "")),
                confidence=float(row.get("confidence", 0.0) or 0.0),
                bar_time=str(row.get("bar_time", "")),
            )
        )
    return tuple(out)


def build_model_status(engine: Any) -> ModelHealthResponse:
    settings = getattr(engine, "settings", None)
    models_root = Path("models")
    if settings is not None and hasattr(settings, "resolve_path"):
        models_root = settings.resolve_path(Path("models"))
    ml_base = None
    if settings is not None:
        art = settings.resolve_path(settings.ml_scorer_artifact_dir)
        ml_base = art.parent if art.name == "validated" else art

    store = getattr(engine, "_agentic_store", None)
    learning_store = getattr(engine, "_agentic_learning_store", None)
    sym = ""
    try:
        sym = engine.state.symbol or ""
    except Exception:  # noqa: BLE001
        pass
    response = ModelHealthResponse(
        registry_enabled=bool(getattr(settings, "model_registry_enabled", False)),
        promotion_required=bool(getattr(settings, "model_promotion_required", True)),
        rl=build_rl_status(engine, models_root=models_root),
        ml=build_ml_status(engine, models_root=models_root, ml_base=ml_base),
        regime=build_regime_status(models_root),
        agentic=AgenticStatusSummary(
            recent_decisions=build_agentic_decisions_summary(store, limit=10),
            learning_summary=build_learning_summary(learning_store),
            nightly_alignment=build_nightly_alignment_status(settings),
        ),
    )
    if settings is not None:
        response = replace(
            response,
            runtime_readiness=build_runtime_readiness_summary(
                settings,
                model_health=response,
                symbol=sym or None,
            ),
        )
        response = replace(
            response,
            activation=build_model_activation_summary(
                settings,
                engine=engine,
                model_health=response,
                symbol=sym or None,
                models_root=models_root,
            ),
        )
        response = replace(
            response,
            scaling=build_scaling_posture_summary(settings, basket_count=0),
        )
    return response


def _latest_promotion(
    kind: ModelKind,
    models_root: Path | str,
    *,
    symbol: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    events = read_audit(models_root, limit=200)
    symbol_norm = (symbol or "").upper().strip()
    for row in events:
        if row.get("model_kind") != kind.value:
            continue
        if symbol_norm and str(row.get("symbol", "")).upper().strip() != symbol_norm:
            continue
        return row
    return None


def _enrich_promotion(promotion: Optional[PromotionRecord]) -> Optional[PromotionRecord]:
    if promotion is None:
        return None
    summary = load_workflow_snapshot_summary(promotion.workflow_snapshot_path)
    if summary is None:
        return promotion
    return replace(promotion, workflow_snapshot=summary)


def _safe_learning_rows(store: Any) -> list[LearningExample]:
    if store is None:
        return []
    try:
        return list(store.read_all())
    except Exception:  # noqa: BLE001
        return []


def build_nightly_alignment_status(settings: Any) -> NightlyAlignmentStatus:
    if settings is None or not bool(
        getattr(settings, "model_status_nightly_alignment_enabled", True)
    ):
        return NightlyAlignmentStatus()
    report_dir = getattr(settings, "model_status_nightly_report_dir", Path("reports/nightly"))
    root = settings.resolve_path(Path(report_dir))
    if not root.is_dir():
        return NightlyAlignmentStatus()
    lookback = max(1, int(getattr(settings, "model_status_nightly_alignment_lookback_reports", 8)))
    paths = sorted(root.glob("*.json"), reverse=True)[:lookback]
    rows: list[NightlyAlignmentRow] = []
    enabled_reports = 0
    aligned_reports = 0
    latest_execution_target: Optional[str] = None
    latest_execution_model_family: Optional[str] = None
    latest_execution_selection_source: Optional[str] = None
    latest_target_mix: Optional[str] = None
    refreshed_aligned_reports = 0
    latest_refreshed_target_mix: Optional[str] = None
    latest_workflow_research_alignment_summary: Optional[str] = None
    latest_workflow_research_alignment_target_mix: Optional[str] = None
    latest_workflow_research_recommended_target: Optional[str] = None
    latest_workflow_research_refresh_urgency_summary: Optional[str] = None
    latest_workflow_research_refresh_urgency_target: Optional[str] = None
    latest_workflow_research_follow_up_action: Optional[str] = None
    latest_workflow_research_discovery_follow_up_target: Optional[str] = None
    latest_workflow_research_discovery_follow_up_summary: Optional[str] = None
    latest_workflow_research_discovery_follow_up_action: Optional[str] = None
    latest_workflow_research_research_follow_up_target: Optional[str] = None
    latest_workflow_research_research_follow_up_summary: Optional[str] = None
    latest_workflow_research_research_follow_up_action: Optional[str] = None
    latest_workflow_research_execution_follow_up_target: Optional[str] = None
    latest_workflow_research_execution_follow_up_summary: Optional[str] = None
    latest_workflow_research_execution_follow_up_action: Optional[str] = None
    latest_workflow_research_scout_support_target: Optional[str] = None
    latest_workflow_discovery_alignment_summary: Optional[str] = None
    latest_workflow_discovery_overlap_symbols: Optional[str] = None
    latest_workflow_discovery_warning = False
    latest_training_research_discovery_preferred_count = 0
    latest_training_research_discovery_regime_mix: Optional[str] = None
    latest_training_research_discovery_summary: Optional[str] = None
    latest_training_research_discovery_recommended_target: Optional[str] = None
    latest_training_research_scout_support_target: Optional[str] = None
    latest_promotion_review_model_kind: Optional[str] = None
    latest_promotion_review_model_kinds: tuple[str, ...] = ()
    for path in paths:
        row = _load_nightly_alignment_row(path)
        if row is None:
            continue
        rows.append(row)
        if row.alignment_enabled:
            enabled_reports += 1
        if latest_execution_target is None and row.execution_target:
            latest_execution_target = row.execution_target
        if latest_execution_model_family is None and row.execution_model_family:
            latest_execution_model_family = row.execution_model_family
        if latest_execution_selection_source is None and row.execution_selection_source:
            latest_execution_selection_source = row.execution_selection_source
        if row.target_mix:
            aligned_reports += 1
            if latest_target_mix is None:
                latest_target_mix = row.target_mix
        if row.refreshed_target_mix:
            refreshed_aligned_reports += 1
            if latest_refreshed_target_mix is None:
                latest_refreshed_target_mix = row.refreshed_target_mix
        if latest_workflow_research_alignment_summary is None and (
            row.workflow_research_alignment_summary
            or row.workflow_research_alignment_target_mix
            or row.workflow_research_alignment_recommended_target
        ):
            latest_workflow_research_alignment_summary = row.workflow_research_alignment_summary
            latest_workflow_research_alignment_target_mix = (
                row.workflow_research_alignment_target_mix
            )
            latest_workflow_research_recommended_target = (
                row.workflow_research_alignment_recommended_target
            )
        if latest_workflow_research_refresh_urgency_summary is None and (
            row.workflow_research_refresh_urgency_summary
            or row.workflow_research_refresh_urgency_target
            or row.workflow_research_follow_up_action
        ):
            latest_workflow_research_refresh_urgency_summary = (
                row.workflow_research_refresh_urgency_summary
            )
            latest_workflow_research_refresh_urgency_target = (
                row.workflow_research_refresh_urgency_target
            )
            latest_workflow_research_follow_up_action = row.workflow_research_follow_up_action
        if latest_workflow_research_scout_support_target is None and (
            row.workflow_research_scout_support_target
        ):
            latest_workflow_research_scout_support_target = (
                row.workflow_research_scout_support_target
            )
        if latest_workflow_research_discovery_follow_up_summary is None and (
            row.workflow_research_discovery_follow_up_summary
            or row.workflow_research_discovery_follow_up_target
            or row.workflow_research_discovery_follow_up_action
        ):
            latest_workflow_research_discovery_follow_up_target = (
                row.workflow_research_discovery_follow_up_target
            )
            latest_workflow_research_discovery_follow_up_summary = (
                row.workflow_research_discovery_follow_up_summary
            )
            latest_workflow_research_discovery_follow_up_action = (
                row.workflow_research_discovery_follow_up_action
            )
        if latest_workflow_research_research_follow_up_summary is None and (
            row.workflow_research_research_follow_up_summary
            or row.workflow_research_research_follow_up_target
            or row.workflow_research_research_follow_up_action
        ):
            latest_workflow_research_research_follow_up_target = (
                row.workflow_research_research_follow_up_target
            )
            latest_workflow_research_research_follow_up_summary = (
                row.workflow_research_research_follow_up_summary
            )
            latest_workflow_research_research_follow_up_action = (
                row.workflow_research_research_follow_up_action
            )
        if latest_workflow_research_execution_follow_up_summary is None and (
            row.workflow_research_execution_follow_up_summary
            or row.workflow_research_execution_follow_up_target
            or row.workflow_research_execution_follow_up_action
        ):
            latest_workflow_research_execution_follow_up_target = (
                row.workflow_research_execution_follow_up_target
            )
            latest_workflow_research_execution_follow_up_summary = (
                row.workflow_research_execution_follow_up_summary
            )
            latest_workflow_research_execution_follow_up_action = (
                row.workflow_research_execution_follow_up_action
            )
        if (
            row.workflow_discovery_alignment_summary
            and latest_workflow_discovery_alignment_summary is None
        ):
            latest_workflow_discovery_alignment_summary = (
                row.workflow_discovery_alignment_summary
            )
            latest_workflow_discovery_overlap_symbols = (
                row.workflow_discovery_overlap_symbols
            )
            latest_workflow_discovery_warning = _workflow_discovery_alignment_warning(
                overlap_count=row.workflow_discovery_alignment_overlap_count,
                compare_count=row.workflow_discovery_alignment_compare_count,
                settings=settings,
            )
        if latest_training_research_discovery_summary is None and (
            row.training_research_discovery_summary
            or row.training_research_discovery_regime_mix
            or int(row.training_research_discovery_preferred_count or 0) > 0
        ):
            latest_training_research_discovery_preferred_count = int(
                row.training_research_discovery_preferred_count or 0
            )
            latest_training_research_discovery_regime_mix = (
                row.training_research_discovery_regime_mix
            )
            latest_training_research_discovery_summary = (
                row.training_research_discovery_summary
            )
            latest_training_research_discovery_recommended_target = (
                _discovery_refresh_target_hint(
                    preferred_count=latest_training_research_discovery_preferred_count,
                    regime_mix=latest_training_research_discovery_regime_mix,
                )
            )
            latest_training_research_scout_support_target = (
                row.training_research_scout_support_target
            )
        if latest_promotion_review_model_kind is None and row.promotion_review_model_kind:
            latest_promotion_review_model_kind = row.promotion_review_model_kind
        if not latest_promotion_review_model_kinds and row.promotion_review_model_kinds:
            latest_promotion_review_model_kinds = row.promotion_review_model_kinds
    recent_window = tuple(rows[:3])
    nightly_posture = _nightly_alignment_posture(
        recent_rows=recent_window,
        latest_refreshed_target_mix=latest_refreshed_target_mix,
        latest_target_mix=latest_target_mix,
    )
    recommended_action = _nightly_alignment_recommendation(
        enabled_reports=enabled_reports,
        aligned_reports=aligned_reports,
        latest_target_mix=latest_target_mix,
        latest_refreshed_target_mix=latest_refreshed_target_mix,
        settings=settings,
    )
    recommended_refresh_target = (
        _nightly_alignment_refresh_target(
            latest_refreshed_target_mix=latest_refreshed_target_mix,
            latest_target_mix=latest_target_mix,
        )
        if recommended_action is not None
        else None
    )
    recommended_force_refresh = (
        _nightly_alignment_force_refresh(
            enabled_reports=enabled_reports,
            refreshed_aligned_reports=refreshed_aligned_reports,
        )
        if recommended_action is not None
        else False
    )
    effective_refresh_target = (
        _effective_refresh_target(
            nightly_target=recommended_refresh_target,
            workflow_target=latest_workflow_research_recommended_target,
            discovery_target=latest_training_research_discovery_recommended_target,
            scout_target=(
                latest_workflow_research_scout_support_target
                or latest_training_research_scout_support_target
            ),
        )
        if recommended_action is not None
        else None
    )
    workflow_target_mismatch = bool(
        recommended_action is not None
        and recommended_refresh_target
        and latest_workflow_research_recommended_target
        and recommended_refresh_target != latest_workflow_research_recommended_target
    )
    if recommended_action is not None:
        recommended_action = _retarget_nightly_alignment_action(
            effective_refresh_target or recommended_refresh_target or "rl"
        )
        if (
            latest_training_research_discovery_recommended_target
            and effective_refresh_target
            and effective_refresh_target == latest_training_research_discovery_recommended_target
            and effective_refresh_target != (recommended_refresh_target or "")
        ):
            recommended_action += (
                " Discovery context currently leans "
                f"{effective_refresh_target.upper()}."
            )
    recommended_cli_command = (
        _nightly_alignment_recommendation_command(
            settings,
            refresh_target=effective_refresh_target or recommended_refresh_target or "rl",
            force_refresh=recommended_force_refresh,
        )
        if recommended_action is not None
        else None
    )
    recommended_discovery_action = (
        _discovery_alignment_recommendation(
            latest_workflow_discovery_alignment_summary,
            latest_workflow_discovery_overlap_symbols,
        )
        if latest_workflow_discovery_warning
        else None
    )
    recommended_discovery_cli_command = (
        _discovery_alignment_recommendation_command(settings)
        if recommended_discovery_action is not None
        else None
    )
    recent_enabled = sum(1 for row in recent_window if row.alignment_enabled)
    recent_aligned = sum(
        1
        for row in recent_window
        if row.alignment_enabled and str(row.target_mix or "").strip()
    )
    return NightlyAlignmentStatus(
        report_count=len(rows),
        aligned_reports=aligned_reports,
        enabled_reports=enabled_reports,
        latest_execution_target=latest_execution_target,
        latest_execution_model_family=latest_execution_model_family,
        latest_execution_selection_source=latest_execution_selection_source,
        latest_target_mix=latest_target_mix,
        refreshed_aligned_reports=refreshed_aligned_reports,
        latest_refreshed_target_mix=latest_refreshed_target_mix,
        latest_workflow_research_alignment_summary=latest_workflow_research_alignment_summary,
        latest_workflow_research_alignment_target_mix=latest_workflow_research_alignment_target_mix,
        latest_workflow_research_recommended_target=latest_workflow_research_recommended_target,
        latest_workflow_research_refresh_urgency_summary=(
            latest_workflow_research_refresh_urgency_summary
        ),
        latest_workflow_research_refresh_urgency_target=(
            latest_workflow_research_refresh_urgency_target
        ),
        latest_workflow_research_follow_up_action=latest_workflow_research_follow_up_action,
        latest_workflow_research_discovery_follow_up_target=(
            latest_workflow_research_discovery_follow_up_target
        ),
        latest_workflow_research_discovery_follow_up_summary=(
            latest_workflow_research_discovery_follow_up_summary
        ),
        latest_workflow_research_discovery_follow_up_action=(
            latest_workflow_research_discovery_follow_up_action
        ),
        latest_workflow_research_research_follow_up_target=(
            latest_workflow_research_research_follow_up_target
        ),
        latest_workflow_research_research_follow_up_summary=(
            latest_workflow_research_research_follow_up_summary
        ),
        latest_workflow_research_research_follow_up_action=(
            latest_workflow_research_research_follow_up_action
        ),
        latest_workflow_research_execution_follow_up_target=(
            latest_workflow_research_execution_follow_up_target
        ),
        latest_workflow_research_execution_follow_up_summary=(
            latest_workflow_research_execution_follow_up_summary
        ),
        latest_workflow_research_execution_follow_up_action=(
            latest_workflow_research_execution_follow_up_action
        ),
        latest_workflow_research_scout_support_target=(
            latest_workflow_research_scout_support_target
        ),
        latest_workflow_discovery_alignment_summary=latest_workflow_discovery_alignment_summary,
        latest_workflow_discovery_overlap_symbols=latest_workflow_discovery_overlap_symbols,
        latest_workflow_discovery_warning=latest_workflow_discovery_warning,
        latest_training_research_discovery_preferred_count=(
            latest_training_research_discovery_preferred_count
        ),
        latest_training_research_discovery_regime_mix=latest_training_research_discovery_regime_mix,
        latest_training_research_discovery_summary=latest_training_research_discovery_summary,
        latest_training_research_discovery_recommended_target=(
            latest_training_research_discovery_recommended_target
        ),
        latest_training_research_scout_support_target=(
            latest_training_research_scout_support_target
        ),
        latest_promotion_review_model_kind=latest_promotion_review_model_kind,
        latest_promotion_review_model_kinds=latest_promotion_review_model_kinds,
        nightly_posture=nightly_posture,
        recommended_action=recommended_action,
        recommended_refresh_target=recommended_refresh_target,
        effective_refresh_target=effective_refresh_target,
        recommended_force_refresh=recommended_force_refresh,
        workflow_target_mismatch=workflow_target_mismatch,
        recommended_cli_command=recommended_cli_command,
        recommended_discovery_action=recommended_discovery_action,
        recommended_discovery_cli_command=recommended_discovery_cli_command,
        recent_trend_window=len(recent_window),
        recent_trend_enabled=recent_enabled,
        recent_trend_aligned=recent_aligned,
        recent_trend_latest_status=recent_window[0].overall_status if recent_window else None,
        recent_trend_latest_basket_size=recent_window[0].basket_size if recent_window else 0,
        recent_rows=tuple(rows),
    )


def _load_nightly_alignment_row(path: Path) -> Optional[NightlyAlignmentRow]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    workflow_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and str(step.get("name", "")) == "workflow_snapshot"
        ),
        None,
    )
    research_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and str(step.get("name", "")) == "training_research"
        ),
        None,
    )
    execution_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and str(step.get("name", "")) == "training_execution_target"
        ),
        None,
    )
    review_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and str(step.get("name", "")) == "promotion_reviews"
        ),
        None,
    )
    detail = workflow_step.get("detail") if isinstance(workflow_step, dict) else None
    if not isinstance(detail, dict):
        return None
    research_detail = research_step.get("detail") if isinstance(research_step, dict) else None
    execution_detail = execution_step.get("detail") if isinstance(execution_step, dict) else None
    review_detail = review_step.get("detail") if isinstance(review_step, dict) else None
    target_mix = str(detail.get("allocation_research_target_mix", "") or "").strip() or None
    refreshed_target_mix = (
        str(detail.get("allocation_refreshed_research_target_mix", "") or "").strip() or None
    )
    workflow_research_alignment_summary = (
        str(detail.get("team_research_alignment_summary", "") or "").strip() or None
    )
    workflow_research_alignment_target_mix = (
        str(detail.get("team_research_alignment_target_mix", "") or "").strip() or None
    )
    workflow_research_refresh_urgency_summary = (
        str(detail.get("team_research_refresh_urgency_summary", "") or "").strip() or None
    )
    workflow_research_refresh_urgency_target = (
        str(detail.get("team_research_refresh_urgency_target", "") or "").strip() or None
    )
    workflow_research_follow_up_action = (
        str(detail.get("team_research_follow_up_action", "") or "").strip() or None
    )
    workflow_research_discovery_follow_up_target = (
        str(detail.get("team_research_discovery_follow_up_target", "") or "").strip() or None
    )
    workflow_research_discovery_follow_up_summary = (
        str(detail.get("team_research_discovery_follow_up_summary", "") or "").strip() or None
    )
    workflow_research_discovery_follow_up_action = (
        str(detail.get("team_research_discovery_follow_up_action", "") or "").strip() or None
    )
    workflow_research_research_follow_up_target = (
        str(detail.get("team_research_research_follow_up_target", "") or "").strip() or None
    )
    workflow_research_research_follow_up_summary = (
        str(detail.get("team_research_research_follow_up_summary", "") or "").strip() or None
    )
    workflow_research_research_follow_up_action = (
        str(detail.get("team_research_research_follow_up_action", "") or "").strip() or None
    )
    workflow_research_execution_follow_up_target = (
        str(detail.get("team_research_execution_follow_up_target", "") or "").strip() or None
    )
    workflow_research_execution_follow_up_summary = (
        str(detail.get("team_research_execution_follow_up_summary", "") or "").strip() or None
    )
    workflow_research_execution_follow_up_action = (
        str(detail.get("team_research_execution_follow_up_action", "") or "").strip() or None
    )
    workflow_research_scout_support_target = (
        str(detail.get("team_research_scout_support_target", "") or "").strip() or None
    )
    workflow_discovery_alignment_summary = (
        str(detail.get("team_discovery_alignment_summary", "") or "").strip() or None
    )
    workflow_discovery_overlap_symbols = (
        str(detail.get("team_discovery_overlap_symbols", "") or "").strip() or None
    )
    training_research_discovery_preferred_count = 0
    training_research_discovery_regime_mix = None
    training_research_discovery_summary = None
    training_research_scout_support_target = None
    execution_target = None
    execution_model_family = None
    execution_selection_source = None
    promotion_review_model_kind = None
    promotion_review_model_kinds: tuple[str, ...] = ()
    if isinstance(research_detail, dict):
        training_research_discovery_preferred_count = int(
            research_detail.get("discovery_preferred_count", 0) or 0
        )
        training_research_discovery_regime_mix = (
            str(research_detail.get("discovery_regime_mix", "") or "").strip() or None
        )
        training_research_discovery_summary = (
            str(research_detail.get("discovery_summary", "") or "").strip() or None
        )
        training_research_scout_support_target = (
            str(research_detail.get("scout_support_target", "") or "").strip() or None
        )
    if isinstance(execution_detail, dict):
        raw_target = str(execution_detail.get("target", "") or "").strip().lower()
        if raw_target in {"all", "ml", "rl"}:
            execution_target = raw_target
        run_ml = bool(execution_detail.get("run_ml"))
        run_rl = bool(execution_detail.get("run_rl"))
        if run_ml and not run_rl:
            execution_model_family = "ml"
        elif run_rl and not run_ml:
            execution_model_family = "rl"
        elif run_ml and run_rl:
            execution_model_family = "hybrid"
        raw_selection_source = str(execution_detail.get("selection_source", "") or "").strip()
        execution_selection_source = raw_selection_source or None
    if isinstance(review_detail, dict):
        raw_model_kinds = review_detail.get("model_kinds")
        if isinstance(raw_model_kinds, list):
            promotion_review_model_kinds = tuple(
                kind
                for kind in (
                    str(item or "").strip().lower() for item in raw_model_kinds
                )
                if kind
            )
        promotion_review_model_kind = (
            str(review_detail.get("model_kind", "") or "").strip().lower() or None
        )
        if promotion_review_model_kind is None and promotion_review_model_kinds:
            promotion_review_model_kind = promotion_review_model_kinds[0]
    return NightlyAlignmentRow(
        path=str(path),
        overall_status=str(payload.get("overall_status", "") or ""),
        basket_size=len(payload.get("basket", []) or []),
        alignment_enabled=bool(detail.get("allocation_research_alignment_enabled", False)),
        execution_target=execution_target,
        execution_model_family=execution_model_family,
        execution_selection_source=execution_selection_source,
        target_mix=target_mix,
        refreshed_target_mix=refreshed_target_mix,
        workflow_research_alignment_summary=workflow_research_alignment_summary,
        workflow_research_alignment_target_mix=workflow_research_alignment_target_mix,
        workflow_research_alignment_recommended_target=_workflow_target_from_alignment_mix(
            workflow_research_alignment_target_mix
        ),
        workflow_research_refresh_urgency_summary=workflow_research_refresh_urgency_summary,
        workflow_research_refresh_urgency_target=workflow_research_refresh_urgency_target,
        workflow_research_follow_up_action=workflow_research_follow_up_action,
        workflow_research_discovery_follow_up_target=workflow_research_discovery_follow_up_target,
        workflow_research_discovery_follow_up_summary=workflow_research_discovery_follow_up_summary,
        workflow_research_discovery_follow_up_action=workflow_research_discovery_follow_up_action,
        workflow_research_research_follow_up_target=workflow_research_research_follow_up_target,
        workflow_research_research_follow_up_summary=workflow_research_research_follow_up_summary,
        workflow_research_research_follow_up_action=workflow_research_research_follow_up_action,
        workflow_research_execution_follow_up_target=workflow_research_execution_follow_up_target,
        workflow_research_execution_follow_up_summary=workflow_research_execution_follow_up_summary,
        workflow_research_execution_follow_up_action=workflow_research_execution_follow_up_action,
        workflow_research_scout_support_target=workflow_research_scout_support_target,
        workflow_discovery_alignment_summary=workflow_discovery_alignment_summary,
        workflow_discovery_overlap_symbols=workflow_discovery_overlap_symbols,
        workflow_discovery_alignment_overlap_count=int(
            detail.get("team_discovery_alignment_overlap_count", 0) or 0
        ),
        workflow_discovery_alignment_compare_count=int(
            detail.get("team_discovery_alignment_compare_count", 0) or 0
        ),
        training_research_discovery_preferred_count=training_research_discovery_preferred_count,
        training_research_discovery_regime_mix=training_research_discovery_regime_mix,
        training_research_discovery_summary=training_research_discovery_summary,
        training_research_scout_support_target=training_research_scout_support_target,
        promotion_review_model_kind=promotion_review_model_kind,
        promotion_review_model_kinds=promotion_review_model_kinds,
    )


def _nightly_alignment_recommendation(
    *,
    enabled_reports: int,
    aligned_reports: int,
    latest_target_mix: Optional[str],
    latest_refreshed_target_mix: Optional[str],
    settings: Any,
) -> Optional[str]:
    min_enabled_reports = max(
        2,
        int(getattr(settings, "acceptance_bundle_nightly_alignment_min_enabled_reports", 2)),
    )
    if enabled_reports < min_enabled_reports:
        return None
    ratio = float(aligned_reports) / float(enabled_reports) if enabled_reports > 0 else 0.0
    trigger_ratio = max(
        0.0,
        min(
            1.0,
            float(
                getattr(
                    settings,
                    "model_status_nightly_alignment_recommendation_ratio",
                    0.75,
                )
            ),
        ),
    )
    if ratio >= trigger_ratio:
        return None
    target = _nightly_alignment_refresh_target(
        latest_refreshed_target_mix=latest_refreshed_target_mix,
        latest_target_mix=latest_target_mix,
    )
    if target == "ml":
        return (
            "Refresh the ML-oriented training research plan and review "
            "candidate-selection policy before the next promotion or nightly cycle."
        )
    if target == "all":
        return (
            "Refresh the full ML/RL training research plan and review "
            "candidate-selection policy before the next promotion or nightly cycle."
        )
    return (
        "Refresh the RL-oriented training research plan and review "
        "candidate-selection policy before the next promotion or nightly cycle."
    )


def _retarget_nightly_alignment_action(refresh_target: str) -> str:
    target = str(refresh_target or "rl").strip().lower()
    if target == "ml":
        return (
            "Refresh the ML-oriented training research plan and review "
            "candidate-selection policy before the next promotion or nightly cycle."
        )
    if target == "all":
        return (
            "Refresh the full ML/RL training research plan and review "
            "candidate-selection policy before the next promotion or nightly cycle."
        )
    return (
        "Refresh the RL-oriented training research plan and review "
        "candidate-selection policy before the next promotion or nightly cycle."
    )


def _nightly_alignment_recommendation_command(
    settings: Any,
    *,
    refresh_target: str,
    force_refresh: bool,
) -> str:
    source = "auto"
    screener_path = getattr(settings, "market_universe_screener_csv", None)
    try:
        resolved = settings.resolve_path(Path(screener_path)) if screener_path else None
    except Exception:
        resolved = None
    if resolved is not None and resolved.is_file():
        source = "screener"
    timeframe = str(getattr(settings, "default_timeframe", "5m") or "5m")
    days = int(getattr(settings, "default_days", 30) or 30)
    return (
        "uv run python scripts/build_training_research_plan.py "
        f"--source {source} --selection-policy diversified "
        f"--timeframe {timeframe} --days {days} "
        f"--refresh-data --refresh-target {refresh_target}"
        + (" --force-refresh" if force_refresh else "")
    )


def _discovery_alignment_recommendation(
    summary: Optional[str],
    overlap_symbols: Optional[str],
) -> Optional[str]:
    text = str(summary or "").strip()
    if not text:
        return None
    action = (
        "Rebuild the market universe from Screener liquidity and recent activity inputs "
        "before the next promotion or nightly cycle."
    )
    overlap = str(overlap_symbols or "").strip()
    if overlap:
        action += f" Latest overlap: {overlap}."
    return action


def _discovery_alignment_recommendation_command(settings: Any) -> str:
    from fortuna.app.market_universe import resolve_market_universe_cli_source

    source = resolve_market_universe_cli_source(settings)
    timeframe = str(getattr(settings, "default_timeframe", "5m") or "5m")
    days = int(getattr(settings, "default_days", 30) or 30)
    limit = int(getattr(settings, "market_universe_default_limit", 15) or 15)
    return (
        "uv run python scripts/build_market_universe.py "
        f"--source {source} --timeframe {timeframe} --days {days} --limit {limit}"
    )


def _nightly_alignment_refresh_target(
    *,
    latest_refreshed_target_mix: Optional[str],
    latest_target_mix: Optional[str],
) -> str:
    for mix in (latest_refreshed_target_mix, latest_target_mix):
        target = _target_from_mix(mix)
        if target is not None:
            return target
    return "rl"


def _nightly_alignment_posture(
    *,
    recent_rows: tuple[NightlyAlignmentRow, ...],
    latest_refreshed_target_mix: Optional[str],
    latest_target_mix: Optional[str],
) -> Optional[str]:
    rl_votes = 0.0
    ml_votes = 0.0
    for idx, row in enumerate(recent_rows):
        recency_weight = max(0.5, 1.0 - (idx * 0.15))
        mix_ml, mix_rl = _target_mix_votes(row.target_mix, weight=recency_weight)
        refreshed_ml, refreshed_rl = _target_mix_votes(
            row.refreshed_target_mix,
            weight=recency_weight * 1.35,
        )
        ml_votes += mix_ml + refreshed_ml
        rl_votes += mix_rl + refreshed_rl
    if ml_votes <= 0 and rl_votes <= 0:
        return _target_from_mix(latest_refreshed_target_mix or latest_target_mix)
    total_votes = ml_votes + rl_votes
    if total_votes <= 0:
        return None
    if abs(ml_votes - rl_votes) / total_votes <= 0.18:
        return "all"
    return "rl" if rl_votes > ml_votes else "ml"


def _target_mix_votes(mix: Optional[str], *, weight: float) -> tuple[float, float]:
    counts = _parse_target_mix(mix)
    if not counts:
        return 0.0, 0.0
    ml_votes = float(counts.get("ml", 0)) * weight
    rl_votes = float(counts.get("rl", 0)) * weight
    both_votes = float(counts.get("both", 0) + counts.get("all", 0)) * weight
    return ml_votes + both_votes, rl_votes + both_votes


def _parse_target_mix(mix: Optional[str]) -> dict[str, int]:
    text = str(mix or "").strip().lower()
    if not text:
        return {}
    out: dict[str, int] = {}
    for part in text.split(","):
        key_text, _, value_text = part.strip().partition("=")
        key = key_text.strip()
        if not key:
            continue
        try:
            value = int(value_text.strip() or "1")
        except ValueError:
            value = 1
        out[key] = out.get(key, 0) + value
    return out


def _target_from_mix(mix: Optional[str]) -> Optional[str]:
    text = str(mix or "").strip().lower()
    if not text:
        return None
    keys = set(_parse_target_mix(text))
    if not keys:
        return None
    if "both" in keys:
        return "all"
    if "ml" in keys and "rl" in keys:
        return "all"
    if "ml" in keys:
        return "ml"
    if "rl" in keys:
        return "rl"
    return None


def _discovery_refresh_target_hint(
    *,
    preferred_count: int,
    regime_mix: Optional[str],
) -> Optional[str]:
    if int(preferred_count or 0) <= 0:
        return None
    counts = _parse_regime_mix(regime_mix)
    trending_like = counts.get("trending", 0) + counts.get("volatile", 0)
    ranging = counts.get("ranging", 0)
    if trending_like <= 0 and ranging <= 0:
        return None
    if trending_like > ranging:
        return "rl"
    if ranging > trending_like:
        return "ml"
    return "all"


def _parse_regime_mix(mix: Optional[str]) -> dict[str, int]:
    text = str(mix or "").strip()
    if not text:
        return {}
    out: dict[str, int] = {}
    for part in text.split(","):
        key_text, _, value_text = part.strip().partition("=")
        key = key_text.strip().lower()
        if not key:
            continue
        try:
            value = int(value_text.strip() or "0")
        except ValueError:
            value = 0
        out[key] = value
    return out


def _nightly_alignment_force_refresh(
    *,
    enabled_reports: int,
    refreshed_aligned_reports: int,
) -> bool:
    if enabled_reports < 2:
        return False
    refreshed_ratio = (
        float(refreshed_aligned_reports) / float(enabled_reports)
        if enabled_reports > 0
        else 0.0
    )
    return refreshed_ratio <= 0.2


def _effective_refresh_target(
    *,
    nightly_target: Optional[str],
    workflow_target: Optional[str],
    discovery_target: Optional[str],
    scout_target: Optional[str] = None,
) -> Optional[str]:
    scout = str(scout_target or "").strip().lower()
    normalized = [
        str(target or "").strip().lower()
        for target in (nightly_target, workflow_target, discovery_target)
        if str(target or "").strip()
    ]
    if not normalized:
        return scout if scout in {"ml", "rl"} else None
    concrete = {target for target in normalized if target in {"ml", "rl"}}
    if len(concrete) >= 2:
        if scout in concrete:
            return scout
        return "all"
    if len(concrete) == 1:
        return next(iter(concrete))
    if "all" in normalized:
        return scout if scout in {"ml", "rl"} else "all"
    return normalized[0]


def _workflow_discovery_alignment_warning(
    *,
    overlap_count: int,
    compare_count: int,
    settings: Any,
) -> bool:
    if compare_count <= 0:
        return False
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
    ratio = float(overlap_count) / float(compare_count)
    return ratio < warn_ratio


def _workflow_target_from_alignment_mix(mix: Optional[str]) -> Optional[str]:
    text = str(mix or "").strip().lower()
    if not text:
        return None
    keys = set()
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
