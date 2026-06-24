"""Thin agent-role wrappers over existing advisory app services."""

from __future__ import annotations

from fortuna.app.agent_roles import (
    activity_scout,
    briefing_agent,
    instrument_analyst,
    liquidity_scout,
    operations_monitor,
    portfolio_critic,
    research_planner,
    universe_scout,
)
from fortuna.app.agent_roles._shared import basket_research_alignment, discovery_alignment

__all__ = [
    "activity_scout",
    "briefing_agent",
    "basket_research_alignment",
    "discovery_alignment",
    "instrument_analyst",
    "liquidity_scout",
    "operations_monitor",
    "portfolio_critic",
    "research_planner",
    "universe_scout",
]
