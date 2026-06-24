"""Instrument analyst role wrapper."""

from __future__ import annotations

from fortuna.agentic.contracts import AgentRoleStatus
from fortuna.app.shortlist_analysis import analyze_market_shortlist


def run(*, settings, engine_factory=None, registry=None, **kwargs):
    response = analyze_market_shortlist(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        **kwargs,
    )
    return response, compose_role(response)


def compose_role(response) -> AgentRoleStatus:
    items = getattr(response, "items", ())
    focus = tuple(row.symbol for row in items[:5])
    headline = (
        f"{len(items)} analyzed setups with deterministic critique"
        if response.ok
        else "Instrument critic unavailable"
    )
    notes = tuple(
        str(getattr(row.critique, "summary", "") or "").strip()
        for row in items[:2]
        if getattr(row, "critique", None) is not None
        and str(getattr(row.critique, "summary", "") or "").strip()
    )
    return AgentRoleStatus(
        name="instrument_critic",
        ok=bool(response.ok),
        headline=headline,
        focus_symbols=focus,
        notes=notes,
        error=response.error,
    )
