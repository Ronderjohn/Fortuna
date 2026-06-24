"""Universe scout role wrapper."""

from __future__ import annotations

from fortuna.agentic.contracts import AgentRoleStatus
from fortuna.app.agent_roles._shared import discovery_alignment, universe_blend_note
from fortuna.app.market_universe import build_market_universe


def run(*, settings, registry=None, **kwargs):
    response = build_market_universe(settings=settings, registry=registry, **kwargs)
    return response, compose_role(response)


def compose_role(response) -> AgentRoleStatus:
    focus = tuple(row.symbol for row in getattr(response, "candidates", ())[:5])
    notes: list[str] = []
    candidate_count = len(getattr(response, "candidates", ()) or ())
    discovery_summary, discovery_overlap = discovery_alignment(response)
    if response.ok and getattr(response, "candidates", ()):
        first_notes = tuple(getattr(response.candidates[0], "notes", ()) or ())
        notes.extend(str(note) for note in first_notes[:3])
        notes.append(universe_blend_note(response))
        if discovery_summary:
            notes.append(discovery_summary)
        if discovery_overlap:
            notes.append(f"discovery_overlap={discovery_overlap}")
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
    headline = (
        f"{candidate_count} universe names fused from {response.source} seeds "
        "and local OHLCV context"
        if response.ok
        else "Universe scout unavailable"
    )
    fallback_from = str(getattr(response, "fallback_from", "") or "").strip()
    if response.ok and fallback_from:
        headline = (
            f"{candidate_count} universe names from {response.source} "
            f"(fallback from {fallback_from})"
        )
    return AgentRoleStatus(
        name="universe_scout",
        ok=bool(response.ok),
        headline=headline,
        focus_symbols=focus,
        notes=tuple(notes[:5]),
        error=response.error,
    )
