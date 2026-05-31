"""Pure helpers for dashboard model health summaries (no Streamlit dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fortuna.agentic.contracts import (
    AgenticDecisionRow,
    AgenticStatusSummary,
    LearningRowSummary,
    LearningSummaryStatus,
    MlModelStatus,
    ModelHealthResponse,
    PromotionRecord,
    RegimeModelStatus,
    RlModelStatus,
)
from fortuna.agentic.learning import LearningExample
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

    base = RlModelStatus(
        symbol=sym,
        available=available,
        advisory_ready=bool(getattr(rl_gen, "is_advisory_ready", False)) if rl_gen else False,
        load_error=getattr(rl_gen, "load_error", None) if rl_gen else "no_generator",
        live_pointer=str(pointer) if pointer else None,
        checkpoint_dir=str(getattr(rl_gen, "checkpoint_dir", None) or ""),
        last_promotion=PromotionRecord.from_audit_row(audit),
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

    settings = getattr(engine, "settings", None)
    ml_enabled = bool(getattr(settings, "agentic_ml_scorer_enabled", False)) if settings else False
    base = MlModelStatus(
        available=available,
        enabled=ml_enabled,
        live_pointer=str(pointer),
        artifact_dir=str((audit or {}).get("artifact_dir", "") or ""),
        load_error=None if available else "not_loaded",
        last_promotion=PromotionRecord.from_audit_row(audit),
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
    return ModelHealthResponse(
        registry_enabled=bool(getattr(settings, "model_registry_enabled", False)),
        promotion_required=bool(getattr(settings, "model_promotion_required", True)),
        rl=build_rl_status(engine, models_root=models_root),
        ml=build_ml_status(engine, models_root=models_root, ml_base=ml_base),
        regime=build_regime_status(models_root),
        agentic=AgenticStatusSummary(
            recent_decisions=build_agentic_decisions_summary(store, limit=10),
            learning_summary=build_learning_summary(learning_store),
        ),
    )


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


def _safe_learning_rows(store: Any) -> list[LearningExample]:
    if store is None:
        return []
    try:
        return list(store.read_all())
    except Exception:  # noqa: BLE001
        return []
