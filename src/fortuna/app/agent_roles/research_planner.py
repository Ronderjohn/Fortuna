"""Research planner role wrapper."""

from __future__ import annotations

from fortuna.agentic.contracts import AgentRoleStatus
from fortuna.app.agent_roles._shared import (
    basket_research_alignment,
    effective_research_refresh_target,
    targets_mismatch,
)
from fortuna.app.training_research import build_training_research_plan


def run(*, settings, **kwargs):
    response = build_training_research_plan(settings=settings, **kwargs)
    return response, compose_role(response)


def compose_role(response, *, allocation=None) -> AgentRoleStatus:
    rows = getattr(response, "rows", ())
    ml_symbols = tuple(getattr(response, "ml_symbols", ())[:3])
    rl_symbols = tuple(getattr(response, "rl_symbols", ())[:3])
    focus = tuple(dict.fromkeys([*rl_symbols, *ml_symbols]))
    headline = (
        f"{len(rows)} ML/RL research rows prepared"
        if response.ok
        else "Research planner unavailable"
    )
    notes = []
    if response.ok:
        notes.append(f"selection_policy={response.selection_policy}")
        discovery_target = str(
            getattr(response, "discovery_recommended_refresh_target", "") or ""
        ).strip()
        effective_target = str(getattr(response, "effective_refresh_target", "") or "").strip()
        if not effective_target:
            effective_target = effective_research_refresh_target(
                requested_target=str(getattr(response, "refresh_target", "") or "").strip()
                or None,
                discovery_target=discovery_target or None,
            ) or ""
        target_mismatch = targets_mismatch(
            requested_target=str(getattr(response, "refresh_target", "") or "").strip() or None,
            discovery_target=discovery_target or None,
        )
        alignment_summary, alignment_mix = basket_research_alignment(
            allocation=allocation,
            research=response,
        )
        if alignment_summary:
            notes.append(alignment_summary)
        if alignment_mix:
            notes.append(f"basket_research_mix={alignment_mix}")
        if getattr(response, "refresh_requested", False):
            notes.append(
                f"refresh_target={response.refresh_target} refreshed="
                f"{sum(1 for row in rows if bool(getattr(row, 'refreshed', False)))}"
            )
        if discovery_target:
            notes.append(f"discovery_target={discovery_target}")
        if effective_target:
            notes.append(f"effective_target={effective_target}")
        if target_mismatch:
            notes.append(
                f"target_mismatch=requested={response.refresh_target} discovery={discovery_target}"
            )
        nightly_summary = str(getattr(response, "nightly_alignment_summary", "") or "").strip()
        if nightly_summary:
            notes.append(nightly_summary)
        nightly_target = str(getattr(response, "nightly_alignment_target", "") or "").strip()
        if nightly_target:
            notes.append(f"nightly_target={nightly_target}")
        if bool(getattr(response, "nightly_alignment_force_refresh", False)):
            notes.append("nightly_force_refresh=1")
        if str(getattr(response, "nightly_alignment_discovery_action", "") or "").strip():
            notes.append("discovery_follow_up=1")
    return AgentRoleStatus(
        name="research_planner",
        ok=bool(response.ok),
        headline=headline,
        focus_symbols=focus,
        notes=tuple(notes[:9]),
        error=response.error,
    )


def compose_training_data_role(response) -> AgentRoleStatus:
    rows = tuple(getattr(response, "rows", ()) or ())
    focus = tuple(row.symbol for row in rows[:5])
    if not getattr(response, "ok", False):
        return AgentRoleStatus(
            name="training_data_curator",
            ok=False,
            headline="Training data curator unavailable",
            error=getattr(response, "error", None),
        )

    effective_posture = str(getattr(response, "effective_posture", "") or "").strip().lower()
    effective_target = str(getattr(response, "effective_refresh_target", "") or "").strip().lower()
    posture_label = "mixed"
    if effective_posture == "rl" or effective_target == "rl":
        posture_label = "rl-leaning"
    elif effective_posture == "ml" or effective_target == "ml":
        posture_label = "ml-leaning"
    elif effective_posture == "all" or effective_target == "all":
        posture_label = "mixed"
    headline = f"{len(rows)} training rows curated for {posture_label} model refresh"
    notes: list[str] = []
    discovery_posture = str(getattr(response, "discovery_posture", "") or "").strip()
    strategy_posture = str(getattr(response, "strategy_posture", "") or "").strip()
    if discovery_posture:
        notes.append(f"discovery_posture={discovery_posture}")
    if strategy_posture:
        notes.append(f"setup_posture={strategy_posture}")
    if effective_posture:
        notes.append(f"effective_posture={effective_posture}")
    if effective_target:
        notes.append(f"effective_target={effective_target}")
    selected_target_mix = str(getattr(response, "selected_target_mix", "") or "").strip()
    if selected_target_mix:
        notes.append(f"target_mix={selected_target_mix}")
    selected_regime_mix = str(getattr(response, "selected_regime_mix", "") or "").strip()
    if selected_regime_mix:
        notes.append(f"regime_mix={selected_regime_mix}")
    return AgentRoleStatus(
        name="training_data_curator",
        ok=bool(rows),
        headline=headline,
        focus_symbols=focus,
        notes=tuple(notes[:6]),
        error=response.error,
    )
