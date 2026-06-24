"""Briefing agent role wrapper."""

from __future__ import annotations

from fortuna.agentic.contracts import AgentRoleStatus
from fortuna.app.shortlist_briefing import build_shortlist_briefing


def run(*, settings, engine_factory=None, registry=None, **kwargs):
    response = build_shortlist_briefing(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        **kwargs,
    )
    return response, compose_role(response)


def compose_role(response) -> AgentRoleStatus:
    items = getattr(response, "items", ())
    candidate_count = sum(
        1 for row in items if str(getattr(row, "verdict", "")).lower() == "candidate"
    )
    focus = tuple(row.symbol for row in items[:5])
    headline = response.headline if response.ok else "Briefing agent unavailable"
    portfolio = getattr(response, "portfolio", None)
    notes = tuple(str(note.message) for note in getattr(portfolio, "notes", ())[:3])
    if response.ok and candidate_count > 0:
        headline = f"{candidate_count} candidate setups ready for operator briefing"
    return AgentRoleStatus(
        name="briefing_agent",
        ok=bool(response.ok),
        headline=headline,
        focus_symbols=focus,
        notes=notes,
        error=response.error,
    )
