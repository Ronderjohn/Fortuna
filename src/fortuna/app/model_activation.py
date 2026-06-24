"""Typed ML/RL activation path from trained artifact to live advisory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fortuna.agentic.contracts import (
    ActivationBlockerRow,
    LaneActivationSummary,
    MlModelStatus,
    ModelActivationSummary,
    ModelHealthResponse,
    RlModelStatus,
)
from fortuna.config.settings import Settings
from fortuna.ml.artifacts import load_metadata as load_ml_metadata
from fortuna.models.metadata import ModelKind
from fortuna.models.promotion import resolve_live_pointer
from fortuna.models.registry import live_pointer_path
from fortuna.rl.training.checkpoint import PolicyCheckpoint


@dataclass(frozen=True)
class _ValidatedCandidate:
    run_id: str
    artifact_dir: Path
    advisory_ready: bool
    verdict_passed: bool
    symbol: str = ""


def _scan_validated_dirs(
    validated_root: Path,
    *,
    artifact_marker: str,
    metadata_loader,
    symbol_filter: str = "",
) -> list[_ValidatedCandidate]:
    if not validated_root.is_dir():
        return []
    sym_norm = symbol_filter.upper().strip()
    out: list[_ValidatedCandidate] = []
    for child in sorted(validated_root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if not (child / artifact_marker).is_file():
            continue
        meta_path = child / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = metadata_loader(child)
        except Exception:  # noqa: BLE001
            continue
        if meta is None:
            continue
        symbol = str(getattr(meta, "symbol", "") or "").upper().strip()
        if sym_norm and symbol and symbol != sym_norm:
            continue
        out.append(
            _ValidatedCandidate(
                run_id=str(getattr(meta, "run_id", child.name) or child.name),
                artifact_dir=child,
                advisory_ready=bool(getattr(meta, "advisory_ready", False)),
                verdict_passed=bool(getattr(meta, "verdict_passed", False)),
                symbol=symbol,
            )
        )
    return out


def _best_validated_candidate(
    kind: ModelKind,
    *,
    models_root: Path,
    ml_base: Optional[Path] = None,
    symbol: str = "",
) -> Optional[_ValidatedCandidate]:
    if kind == ModelKind.ML_SCORER:
        base = ml_base or models_root / "ml_signal_scorer"
        candidates = _scan_validated_dirs(
            base / "validated",
            artifact_marker="model.joblib",
            metadata_loader=load_ml_metadata,
        )
    else:
        candidates = _scan_validated_dirs(
            models_root / "validated",
            artifact_marker="policy.zip",
            metadata_loader=lambda d: PolicyCheckpoint.read(d / "metadata.json"),
            symbol_filter=symbol,
        )
    if not candidates:
        return None
    for row in candidates:
        if row.advisory_ready and row.verdict_passed:
            return row
    return candidates[0]


def _pointer_exists(path: Path) -> bool:
    return path.is_file()


def _resolve_pointer_artifact(
    pointer: Path,
) -> tuple[Optional[Path], Optional[bool], Optional[bool]]:
    if not pointer.is_file():
        return None, None, None
    resolved = resolve_live_pointer(pointer)
    if resolved is None:
        return None, None, None
    if (resolved / "policy.zip").is_file():
        try:
            meta = PolicyCheckpoint.read(resolved / "metadata.json")
            return resolved, bool(meta.advisory_ready), bool(meta.verdict_passed)
        except Exception:  # noqa: BLE001
            return resolved, None, None
    if (resolved / "model.joblib").is_file():
        meta = load_ml_metadata(resolved)
        if meta is None:
            return resolved, None, None
        return resolved, bool(meta.advisory_ready), bool(meta.verdict_passed)
    return resolved, None, None


def _promote_command_ml(run_id: str) -> str:
    return f"uv run python scripts/promote_model.py --kind ml_scorer --run-id {run_id}"


def _promote_command_rl(run_id: str, symbol: str) -> str:
    base = f"uv run python scripts/promote_policy.py --run-id {run_id}"
    if symbol:
        return f"{base} --symbol {symbol}"
    return base


def _review_command(kind: str, symbol: str = "") -> str:
    if kind == "ml":
        return "uv run python scripts/review_promotion.py --kind ml_scorer"
    cmd = "uv run python scripts/review_promotion.py --kind rl_policy"
    if symbol:
        return f"{cmd} --symbol {symbol}"
    return cmd


def _artifact_dir_for_lane(
    ptr_art: Optional[Path],
    candidate: Optional[_ValidatedCandidate],
) -> Optional[str]:
    if ptr_art is not None:
        return str(ptr_art)
    if candidate is not None:
        return str(candidate.artifact_dir)
    return None


def build_lane_activation_summary(
    settings: Settings,
    *,
    lane: str,
    symbol: Optional[str] = None,
    models_root: Path | str = "models",
    ml_base: Optional[Path | str] = None,
    ml_status: Optional[MlModelStatus] = None,
    rl_status: Optional[RlModelStatus] = None,
) -> LaneActivationSummary:
    lane_norm = str(lane or "").strip().lower()
    sym = (symbol or settings.default_symbol or "").upper().strip()
    root = settings.resolve_path(Path(models_root))
    ml_root = settings.resolve_path(Path(ml_base)) if ml_base else None
    if ml_root is None:
        art = settings.resolve_path(settings.ml_scorer_artifact_dir)
        ml_root = art.parent if art.name == "validated" else art

    registry_strict = bool(settings.model_registry_enabled and settings.model_promotion_required)
    blockers: list[ActivationBlockerRow] = []

    if lane_norm == "ml":
        if not settings.agentic_ml_scorer_enabled:
            return LaneActivationSummary(
                lane="ml",
                stage="disabled",
                active=False,
                summary="ML scorer disabled",
                blockers=(
                    ActivationBlockerRow(
                        code="subsystem_disabled",
                        severity="blocker",
                        detail="FORTUNA_AGENTIC_ML_SCORER_ENABLED=0",
                    ),
                ),
                recommended_command="Set FORTUNA_AGENTIC_ML_SCORER_ENABLED=1",
            )
        kind = ModelKind.ML_SCORER
        pointer = live_pointer_path(kind, root, ml_base=ml_root)
        candidate = _best_validated_candidate(kind, models_root=root, ml_base=ml_root)
        status = ml_status
    else:
        if not settings.agentic_enabled:
            return LaneActivationSummary(
                lane="rl",
                stage="disabled",
                active=False,
                summary="Agentic advisory disabled",
                blockers=(
                    ActivationBlockerRow(
                        code="subsystem_disabled",
                        severity="blocker",
                        detail="FORTUNA_AGENTIC_ENABLED=0",
                    ),
                ),
                recommended_command="Set FORTUNA_AGENTIC_ENABLED=1",
            )
        kind = ModelKind.RL_POLICY
        pointer = live_pointer_path(kind, root, symbol=sym or None)
        candidate = _best_validated_candidate(
            kind,
            models_root=root,
            symbol=sym,
        )
        status = rl_status

    pointer_str = str(pointer)
    has_pointer = _pointer_exists(pointer)
    ptr_art, ptr_advisory_ready, ptr_verdict = _resolve_pointer_artifact(pointer)

    if status is not None and status.available and bool(status.advisory_ready):
        return LaneActivationSummary(
            lane=lane_norm,
            stage="active",
            active=True,
            summary=f"{lane_norm.upper()} loaded and advisory-ready",
            recommended_command=_review_command(lane_norm, sym),
            candidate_run_id=getattr(status, "run_id", None),
            live_pointer=pointer_str if has_pointer else getattr(status, "live_pointer", None),
            artifact_dir=str(
                getattr(status, "checkpoint_dir", None)
                or getattr(status, "artifact_dir", None)
                or ""
            )
            or None,
        )

    if candidate is None and not has_pointer:
        return LaneActivationSummary(
            lane=lane_norm,
            stage="missing_artifact",
            active=False,
            summary=f"No validated {lane_norm.upper()} artifact found",
            blockers=(
                ActivationBlockerRow(
                    code="missing_artifact",
                    severity="blocker",
                    detail="No validated candidate under models/",
                ),
            ),
            recommended_command=(
                "uv run python scripts/train_ml_signal_scorer.py --help"
                if lane_norm == "ml"
                else "uv run --group rl python scripts/run_rl_train.py --help"
            ),
            live_pointer=pointer_str if has_pointer else None,
        )

    if registry_strict and not has_pointer and candidate is not None:
        cmd = (
            _promote_command_ml(candidate.run_id)
            if lane_norm == "ml"
            else _promote_command_rl(candidate.run_id, sym)
        )
        return LaneActivationSummary(
            lane=lane_norm,
            stage="unpromoted",
            active=False,
            summary=f"Validated {lane_norm.upper()} artifact exists but live pointer missing",
            blockers=(
                ActivationBlockerRow(
                    code="missing_live_pointer",
                    severity="blocker",
                    detail=f"Expected pointer at {pointer}",
                ),
            ),
            recommended_command=cmd,
            candidate_run_id=candidate.run_id,
            live_pointer=pointer_str,
            artifact_dir=str(candidate.artifact_dir),
        )

    advisory_ready = ptr_advisory_ready
    if advisory_ready is None and candidate is not None:
        advisory_ready = candidate.advisory_ready
    if advisory_ready is None and status is not None:
        advisory_ready = bool(status.advisory_ready)

    if has_pointer and advisory_ready is False:
        blockers.append(
            ActivationBlockerRow(
                code="not_advisory_ready",
                severity="blocker",
                detail="Promoted artifact metadata is not advisory-ready",
            )
        )
        return LaneActivationSummary(
            lane=lane_norm,
            stage="promoted_not_advisory_ready",
            active=False,
            summary=f"{lane_norm.upper()} promoted but not advisory-ready",
            blockers=tuple(blockers),
            recommended_command=_review_command(lane_norm, sym),
            candidate_run_id=candidate.run_id if candidate else None,
            live_pointer=pointer_str,
            artifact_dir=_artifact_dir_for_lane(ptr_art, candidate),
        )

    loaded = bool(status and status.available and advisory_ready)
    if loaded:
        return LaneActivationSummary(
            lane=lane_norm,
            stage="active",
            active=True,
            summary=f"{lane_norm.upper()} loaded and advisory-ready",
            recommended_command=_review_command(lane_norm, sym),
            candidate_run_id=status.run_id if status else (candidate.run_id if candidate else None),
            live_pointer=pointer_str if has_pointer else None,
            artifact_dir=str(status.checkpoint_dir or status.artifact_dir) if status else None,
        )

    if advisory_ready and (has_pointer or (candidate and not registry_strict)):
        detail = status.load_error if status else "not_loaded"
        blockers.append(
            ActivationBlockerRow(
                code="not_loaded",
                severity="warn",
                detail=str(detail or "Reload ML scorer / restart session"),
            )
        )
        return LaneActivationSummary(
            lane=lane_norm,
            stage="not_loaded",
            active=False,
            summary=f"{lane_norm.upper()} advisory-ready artifact exists but not loaded in session",
            blockers=tuple(blockers),
            recommended_command="Reload from Models tab or restart dashboard session",
            candidate_run_id=candidate.run_id if candidate else None,
            live_pointer=pointer_str if has_pointer else None,
            artifact_dir=_artifact_dir_for_lane(ptr_art, candidate),
        )

    if candidate and not has_pointer and not registry_strict:
        return LaneActivationSummary(
            lane=lane_norm,
            stage="not_loaded",
            active=False,
            summary=f"{lane_norm.upper()} validated artifact available (registry relaxed)",
            blockers=tuple(blockers),
            recommended_command=_promote_command_ml(candidate.run_id)
            if lane_norm == "ml"
            else _promote_command_rl(candidate.run_id, sym),
            candidate_run_id=candidate.run_id,
            artifact_dir=str(candidate.artifact_dir),
        )

    return LaneActivationSummary(
        lane=lane_norm,
        stage="missing_artifact",
        active=False,
        summary=f"{lane_norm.upper()} activation path incomplete",
        blockers=tuple(blockers),
        recommended_command=_review_command(lane_norm, sym),
        live_pointer=pointer_str if has_pointer else None,
    )


def build_model_activation_summary(
    settings: Settings,
    *,
    engine: Any | None = None,
    model_health: Optional[ModelHealthResponse] = None,
    symbol: Optional[str] = None,
    models_root: Path | str = "models",
) -> ModelActivationSummary:
    sym = symbol or ""
    if not sym and engine is not None:
        try:
            sym = engine.state.symbol or ""
        except Exception:  # noqa: BLE001
            sym = ""

    ml_status = model_health.ml if model_health is not None else None
    rl_status = model_health.rl if model_health is not None else None
    if engine is not None and ml_status is None:
        from fortuna.app.model_status import build_ml_status, build_rl_status

        root = settings.resolve_path(Path(models_root))
        art = settings.resolve_path(settings.ml_scorer_artifact_dir)
        ml_base = art.parent if art.name == "validated" else art
        ml_status = build_ml_status(engine, models_root=root, ml_base=ml_base)
        rl_status = build_rl_status(engine, models_root=root)

    ml = build_lane_activation_summary(
        settings,
        lane="ml",
        symbol=sym or None,
        models_root=models_root,
        ml_status=ml_status,
        rl_status=rl_status,
    )
    rl = build_lane_activation_summary(
        settings,
        lane="rl",
        symbol=sym or None,
        models_root=models_root,
        ml_status=ml_status,
        rl_status=rl_status,
    )
    parts = [f"ml={ml.stage}", f"rl={rl.stage}"]
    if ml.active or rl.active:
        parts.append("advisory lanes active")
    return ModelActivationSummary(
        ml=ml,
        rl=rl,
        summary="; ".join(parts),
    )


def format_lane_activation(review: LaneActivationSummary) -> list[str]:
    lines = [
        f"activation: lane={review.lane} stage={review.stage} active={review.active}",
        f"summary: {review.summary}",
    ]
    if review.recommended_command:
        lines.append(f"recommended_command: {review.recommended_command}")
    if review.candidate_run_id:
        lines.append(f"candidate_run_id: {review.candidate_run_id}")
    if review.live_pointer:
        lines.append(f"live_pointer: {review.live_pointer}")
    if review.artifact_dir:
        lines.append(f"artifact_dir: {review.artifact_dir}")
    for row in review.blockers:
        lines.append(f"blocker[{row.severity}]: {row.code} — {row.detail}")
    return lines
