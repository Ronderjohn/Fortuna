"""Typed compute budget and basket-size scaling posture for operator surfaces."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

from fortuna.agentic.contracts import (
    ArtifactRetentionSummary,
    ComputeBudgetSummary,
    RuntimeReadinessRow,
    ScalingPostureSummary,
)
from fortuna.config.settings import Settings


def _small_max(settings: Settings) -> int:
    return max(1, int(getattr(settings, "scaling_basket_small_max", 8)))


def _medium_max(settings: Settings) -> int:
    return max(_small_max(settings), int(getattr(settings, "scaling_basket_medium_max", 15)))


def _machine_tier(
    ram_gb: Optional[float],
    settings: Settings,
) -> tuple[str, int]:
    small = _small_max(settings)
    medium = _medium_max(settings)
    if ram_gb is None:
        return "unknown", medium
    if ram_gb < 12:
        return "low", small
    if ram_gb < 20:
        return "mid", medium
    if ram_gb < 28:
        return "mid_high", medium
    return "high", medium


def build_compute_budget_summary(
    settings: Settings,
    *,
    physical_ram_gb: Optional[float] = None,
    n_envs: int = 4,
    use_subproc: bool = True,
) -> ComputeBudgetSummary:
    from fortuna.utils.nightly_preflight import recommend_rl_env

    ram = physical_ram_gb
    tuned_n, tuned_subproc, _notes = recommend_rl_env(
        max(1, int(n_envs)),
        bool(use_subproc),
        physical_ram=ram,
    )
    machine_class, recommended_max = _machine_tier(ram, settings)
    cpu_count = os.cpu_count()
    ram_text = f"{ram:.1f} GiB" if ram is not None else "unknown"
    summary = (
        f"machine={machine_class} ram={ram_text} "
        f"recommended_max_symbols={recommended_max} rl_n_envs={tuned_n}"
    )
    return ComputeBudgetSummary(
        physical_ram_gb=ram,
        cpu_count=cpu_count,
        machine_class=machine_class,
        rl_n_envs=tuned_n,
        rl_use_subproc=tuned_subproc,
        recommended_max_symbols=recommended_max,
        summary=summary,
    )


def build_scaling_posture_summary(
    settings: Settings,
    *,
    basket_count: int = 0,
    physical_ram_gb: Optional[float] = None,
    n_envs: int = 4,
    use_subproc: bool = True,
) -> ScalingPostureSummary:
    compute = build_compute_budget_summary(
        settings,
        physical_ram_gb=physical_ram_gb,
        n_envs=n_envs,
        use_subproc=use_subproc,
    )
    small = _small_max(settings)
    medium = _medium_max(settings)
    count = max(0, int(basket_count))
    recommended = compute.recommended_max_symbols

    if count > medium:
        posture = "large_not_recommended"
    elif compute.machine_class == "low" and count > small:
        posture = "small_only"
    elif count <= small:
        posture = "medium_ok"
    else:
        posture = "medium_ok"

    within = count <= recommended
    warnings: list[RuntimeReadinessRow] = []
    blockers: list[RuntimeReadinessRow] = []
    if not within and count > 0:
        warnings.append(
            RuntimeReadinessRow(
                name="scaling_basket_exceeds_budget",
                severity="warn",
                category="scaling",
                detail=(
                    f"basket={count} exceeds recommended max {recommended} "
                    f"for machine_class={compute.machine_class}"
                ),
            )
        )
    if posture == "large_not_recommended" and count > 0:
        warnings.append(
            RuntimeReadinessRow(
                name="scaling_large_basket_not_recommended",
                severity="warn",
                category="scaling",
                detail=f"basket={count} above medium tier max {medium}",
            )
        )
    if posture == "small_only" and count > 0:
        warnings.append(
            RuntimeReadinessRow(
                name="scaling_small_basket_only",
                severity="warn",
                category="scaling",
                detail=f"low-RAM machine: keep basket <= {small} (current {count})",
            )
        )

    enforce = bool(getattr(settings, "scaling_enforce_basket_cap", False))
    if enforce and not within and count > 0:
        blockers.append(
            RuntimeReadinessRow(
                name="scaling_basket_exceeds_budget",
                severity="blocker",
                category="scaling",
                detail=f"basket={count} exceeds enforced cap {recommended}",
            )
        )

    summary = (
        f"basket_posture={posture} basket={count} "
        f"recommended_max={recommended} within={within}; {compute.summary}"
    )
    return ScalingPostureSummary(
        basket_posture=posture,
        basket_count=count,
        recommended_max_symbols=recommended,
        within_recommended=within,
        compute=compute,
        summary=summary,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def apply_nightly_report_retention(
    report_dir: Path | str,
    settings: Settings,
    *,
    protect_paths: Optional[set[str]] = None,
) -> ArtifactRetentionSummary:
    enabled = bool(getattr(settings, "nightly_report_retention_enabled", False))
    keep = max(1, int(getattr(settings, "nightly_report_retention_count", 30)))
    policy = f"keep_newest_{keep}_json_reports"
    if not enabled:
        return ArtifactRetentionSummary(
            enabled=False,
            retained_count=0,
            pruned_count=0,
            policy=policy,
            detail="retention disabled",
        )

    root = settings.resolve_path(Path(report_dir))
    if not root.is_dir():
        return ArtifactRetentionSummary(
            enabled=True,
            retained_count=0,
            pruned_count=0,
            policy=policy,
            detail=f"report dir missing: {root}",
        )

    protected = {str(Path(p).resolve()) for p in (protect_paths or set())}
    json_files = sorted(
        (p for p in root.glob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(json_files) <= keep:
        return ArtifactRetentionSummary(
            enabled=True,
            retained_count=len(json_files),
            pruned_count=0,
            policy=policy,
            detail=f"retained all {len(json_files)} reports (under cap {keep})",
        )

    to_keep = json_files[:keep]
    to_prune = json_files[keep:]
    pruned = 0
    for path in to_prune:
        resolved = str(path.resolve())
        if resolved in protected:
            continue
        try:
            path.unlink(missing_ok=True)
            md = path.with_suffix(".md")
            if md.is_file() and str(md.resolve()) not in protected:
                md.unlink(missing_ok=True)
            pruned += 1
        except OSError:
            continue

    return ArtifactRetentionSummary(
        enabled=True,
        retained_count=len(to_keep),
        pruned_count=pruned,
        policy=policy,
        detail=f"retained {len(to_keep)} pruned {pruned} from {len(json_files)} total",
    )


def merge_scaling_into_readiness(
    readiness,
    scaling: ScalingPostureSummary,
):
    """Attach scaling summary and merge scaling warnings into readiness warnings."""
    warnings = list(readiness.warnings) + list(scaling.warnings)
    return replace(
        readiness,
        scaling=scaling,
        warnings=tuple(warnings),
    )
