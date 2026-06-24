"""Shared typed runtime readiness summary for operator surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fortuna.agentic.contracts import (
    ModelHealthResponse,
    RuntimeReadinessRow,
    RuntimeReadinessSummary,
)
from fortuna.config.settings import Settings

if TYPE_CHECKING:
    from fortuna.app.operator_preflight import OperatorPreflightResult, PreflightCheck

_CHECK_CATEGORIES = {
    "settings": "settings",
    "agentic_log_dir": "settings",
    "agentic": "advisory",
    "smartapi_env": "data",
    "ml_scorer": "advisory",
    "rl_policy": "advisory",
    "telegram": "advisory",
    "model_registry": "registry",
    "nightly_automation": "nightly",
}


def build_runtime_readiness_summary(
    settings: Settings,
    *,
    preflight: Optional["OperatorPreflightResult"] = None,
    model_health: Optional[ModelHealthResponse] = None,
    strict: bool = False,
    symbol: Optional[str] = None,
    basket_count: int = 0,
    physical_ram_gb: Optional[float] = None,
    n_envs: int = 4,
    use_subproc: bool = True,
) -> RuntimeReadinessSummary:
    from fortuna.app.operator_preflight import run_operator_preflight
    from fortuna.app.scaling_posture import (
        build_scaling_posture_summary,
        merge_scaling_into_readiness,
    )

    pf = preflight or run_operator_preflight(settings, symbol=symbol, strict=strict)
    blockers, warnings = _typed_rows_from_checks(pf.checks)

    agentic_enabled = bool(settings.agentic_enabled)
    ml_enabled = bool(settings.agentic_ml_scorer_enabled)
    nightly_enabled = bool(getattr(settings, "nightly_automation_enabled", False))

    ml_check = _find_check(pf.checks, "ml_scorer")
    rl_check = _find_check(pf.checks, "rl_policy")

    ml_pointer_ok = ml_check is not None and ml_check.status == "pass"
    rl_pointer_ok = rl_check is not None and rl_check.status == "pass"

    ml_advisory_ready = False
    rl_advisory_ready = False
    if model_health is not None:
        ml_advisory_ready = bool(model_health.ml.advisory_ready)
        rl_advisory_ready = bool(model_health.rl.advisory_ready)
    else:
        if ml_pointer_ok and ml_enabled:
            ml_advisory_ready = True
        if rl_pointer_ok and agentic_enabled:
            rl_advisory_ready = True

    nightly_posture, nightly_rows = _resolve_nightly_posture(
        settings,
        nightly_enabled=nightly_enabled,
        blockers=blockers,
    )
    blockers = blockers + nightly_rows

    overall_posture = _resolve_overall_posture(
        settings,
        blockers=blockers,
        agentic_enabled=agentic_enabled,
        ml_enabled=ml_enabled,
        ml_pointer_ok=ml_pointer_ok,
        rl_pointer_ok=rl_pointer_ok,
        ml_advisory_ready=ml_advisory_ready,
        rl_advisory_ready=rl_advisory_ready,
    )

    deterministic_fallback_ok = True
    summary = _build_summary_text(
        overall_posture=overall_posture,
        nightly_posture=nightly_posture,
        blockers=blockers,
        warnings=warnings,
        agentic_enabled=agentic_enabled,
    )

    scaling = build_scaling_posture_summary(
        settings,
        basket_count=basket_count,
        physical_ram_gb=physical_ram_gb,
        n_envs=n_envs,
        use_subproc=use_subproc,
    )

    response = RuntimeReadinessSummary(
        overall_posture=overall_posture,
        nightly_posture=nightly_posture,
        summary=summary,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        deterministic_fallback_ok=deterministic_fallback_ok,
        agentic_enabled=agentic_enabled,
        ml_advisory_ready=ml_advisory_ready,
        rl_advisory_ready=rl_advisory_ready,
        scaling=scaling,
    )
    return merge_scaling_into_readiness(response, scaling)


def _typed_rows_from_checks(
    checks: list["PreflightCheck"],
) -> tuple[list[RuntimeReadinessRow], list[RuntimeReadinessRow]]:
    blockers: list[RuntimeReadinessRow] = []
    warnings: list[RuntimeReadinessRow] = []
    for check in checks:
        category = _CHECK_CATEGORIES.get(check.name, "settings")
        if check.status == "fail":
            blockers.append(
                RuntimeReadinessRow(
                    name=check.name,
                    severity="blocker",
                    category=category,
                    detail=check.detail,
                )
            )
        elif check.status == "warn":
            warnings.append(
                RuntimeReadinessRow(
                    name=check.name,
                    severity="warn",
                    category=category,
                    detail=check.detail,
                )
            )
    return blockers, warnings


def _resolve_nightly_posture(
    settings: Settings,
    *,
    nightly_enabled: bool,
    blockers: list[RuntimeReadinessRow],
) -> tuple[str, list[RuntimeReadinessRow]]:
    extra_blockers: list[RuntimeReadinessRow] = []
    if not nightly_enabled:
        return "manual_only", extra_blockers

    advisory_blockers = [
        row
        for row in blockers
        if row.name in {"ml_scorer", "rl_policy", "smartapi_env", "agentic"}
    ]
    if advisory_blockers:
        extra_blockers.append(
            RuntimeReadinessRow(
                name="nightly_automation",
                severity="blocker",
                category="nightly",
                detail=(
                    "nightly automation enabled but runtime blockers remain: "
                    + ", ".join(row.name for row in advisory_blockers[:3])
                ),
            )
        )
        return "blocked", extra_blockers
    return "enabled", extra_blockers


def _resolve_overall_posture(
    settings: Settings,
    *,
    blockers: list[RuntimeReadinessRow],
    agentic_enabled: bool,
    ml_enabled: bool,
    ml_pointer_ok: bool,
    rl_pointer_ok: bool,
    ml_advisory_ready: bool,
    rl_advisory_ready: bool,
) -> str:
    if any(row.severity == "blocker" for row in blockers):
        return "blocked"

    if not agentic_enabled:
        return "deterministic_only"

    ml_required = ml_enabled
    rl_required = agentic_enabled

    ml_ready = (not ml_required) or (ml_pointer_ok and ml_advisory_ready)
    rl_ready = (not rl_required) or (rl_pointer_ok and rl_advisory_ready)

    if ml_ready and rl_ready:
        return "advisory_ready"

    return "advisory_partial"


def _build_summary_text(
    *,
    overall_posture: str,
    nightly_posture: str,
    blockers: list[RuntimeReadinessRow],
    warnings: list[RuntimeReadinessRow],
    agentic_enabled: bool,
) -> str:
    parts = [f"runtime={overall_posture}", f"nightly={nightly_posture}"]
    if not agentic_enabled:
        parts.append("deterministic fallback healthy")
    if blockers:
        parts.append("blockers=" + ",".join(row.name for row in blockers[:3]))
    elif warnings:
        warn_names = [row.name for row in warnings if row.severity == "warn"][:3]
        if warn_names:
            parts.append("warnings=" + ",".join(warn_names))
    return "; ".join(parts)


def _find_check(checks: list["PreflightCheck"], name: str) -> Optional["PreflightCheck"]:
    for check in checks:
        if check.name == name:
            return check
    return None
