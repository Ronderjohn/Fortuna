"""Portfolio critic role wrapper."""

from __future__ import annotations

from fortuna.agentic.contracts import AgentRoleStatus
from fortuna.app.portfolio_allocator import build_portfolio_allocation


def run(*, settings, engine_factory=None, registry=None, **kwargs):
    response = build_portfolio_allocation(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        **kwargs,
    )
    return response, compose_role(response)


def compose_role(response) -> AgentRoleStatus:
    focus = tuple(getattr(response, "selected_symbols", ())[:5])
    headline = response.headline if response.ok else "Portfolio allocator unavailable"
    notes = [str(note) for note in getattr(response, "notes", ())[:3]]
    return AgentRoleStatus(
        name="portfolio_allocator",
        ok=bool(response.ok),
        headline=headline,
        focus_symbols=focus,
        notes=tuple(notes[:5]),
        error=response.error,
    )
