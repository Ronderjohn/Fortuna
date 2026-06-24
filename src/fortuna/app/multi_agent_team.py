"""Explicit typed multi-agent workflow over existing advisory services."""

from __future__ import annotations

from dataclasses import replace

from fortuna.agentic.contracts import MultiAgentWorkflowResponse
from fortuna.app.agent_roles import (
    activity_scout,
    basket_research_alignment,
    briefing_agent,
    discovery_alignment,
    instrument_analyst,
    liquidity_scout,
    operations_monitor,
    portfolio_critic,
    research_planner,
    universe_scout,
)
from fortuna.app.market_universe import build_market_universe
from fortuna.app.model_status import build_nightly_alignment_status
from fortuna.app.portfolio_allocator import build_portfolio_allocation
from fortuna.app.shortlist_analysis import analyze_market_shortlist
from fortuna.app.shortlist_briefing import build_shortlist_briefing
from fortuna.app.training_candidates import build_training_candidates
from fortuna.app.training_research import build_training_research_plan


def build_multi_agent_workflow(
    *,
    settings,
    engine_factory=None,
    registry=None,
    universe_limit: int = 15,
    analysis_limit: int = 8,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
    max_positions: int = 3,
    max_per_exposure: int = 1,
    max_same_side: int = 2,
    ml_top_n: int = 5,
    rl_top_n: int = 3,
    selection_policy: str = "diversified",
    refresh_research_data: bool = False,
    research_refresh_target: str = "all",
    research_refresh_timeframe: str | None = None,
    research_refresh_days: int | None = None,
    force_refresh: bool = False,
) -> MultiAgentWorkflowResponse:
    universe = build_market_universe(
        settings=settings,
        registry=registry,
        limit=universe_limit,
        timeframe="1d",
        days=max(days, 20),
        source=source,
    )
    shortlist = analyze_market_shortlist(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    briefing = build_shortlist_briefing(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    allocation = build_portfolio_allocation(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        max_positions=max_positions,
        max_per_exposure=max_per_exposure,
        max_same_side=max_same_side,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    candidates = build_training_candidates(
        settings=settings,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    research = build_training_research_plan(
        settings=settings,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
        ml_top_n=ml_top_n,
        rl_top_n=rl_top_n,
        selection_policy=selection_policy,
        refresh_data=refresh_research_data,
        refresh_target=research_refresh_target,
        refresh_timeframe=research_refresh_timeframe,
        refresh_days=research_refresh_days,
        force_refresh=force_refresh,
    )
    nightly_alignment = build_nightly_alignment_status(settings)

    roles = _compose_roles(
        universe=universe,
        shortlist=shortlist,
        briefing=briefing,
        allocation=allocation,
        research=research,
        nightly_alignment=nightly_alignment,
    )
    ok = any(role.ok for role in roles)
    headline = _workflow_headline(
        universe=universe,
        shortlist=shortlist,
        allocation=allocation,
        research=research,
    )
    return MultiAgentWorkflowResponse(
        ok=ok,
        source=str(source or "auto"),
        timeframe=timeframe,
        lookback_days=int(days),
        headline=headline,
        roles=roles,
        universe=universe,
        shortlist=shortlist,
        briefing=briefing,
        candidates=candidates,
        allocation=allocation,
        research=research,
        nightly_alignment=nightly_alignment,
        error=(
            None
            if ok
            else _first_error(universe, shortlist, briefing, candidates, allocation, research)
        ),
    )


def replace_workflow_universe(
    response: MultiAgentWorkflowResponse,
    *,
    universe,
) -> MultiAgentWorkflowResponse:
    roles = _compose_roles(
        universe=universe,
        shortlist=response.shortlist,
        briefing=response.briefing,
        allocation=response.allocation,
        research=response.research,
        nightly_alignment=response.nightly_alignment,
    )
    headline = _workflow_headline(
        universe=universe,
        shortlist=response.shortlist,
        allocation=response.allocation,
        research=response.research,
    )
    return replace(
        response,
        headline=headline,
        roles=roles,
        universe=universe,
    )


def _compose_roles(*, universe, shortlist, briefing, allocation, research, nightly_alignment):
    return (
        universe_scout.compose_role(universe),
        liquidity_scout.compose_role(universe),
        activity_scout.compose_role(universe),
        instrument_analyst.compose_role(shortlist),
        briefing_agent.compose_role(briefing),
        portfolio_critic.compose_role(allocation),
        research_planner.compose_role(research, allocation=allocation),
        research_planner.compose_training_data_role(research),
        operations_monitor.compose_role(nightly_alignment),
    )


def _workflow_headline(*, universe, shortlist, allocation, research) -> str:
    alignment_summary, _ = basket_research_alignment(
        allocation=allocation,
        research=research,
    )
    discovery_summary, _ = discovery_alignment(universe)
    if allocation.ok:
        return (
            f"Multi-agent flow prepared {len(getattr(universe, 'candidates', ()))} universe names, "
            f"{len(getattr(shortlist, 'items', ()))} analyzed setups, "
            f"{len(getattr(allocation, 'selected_symbols', ()))} selected basket names, and "
            f"{len(getattr(research, 'rows', ()))} ML/RL research rows"
            + (f"; {discovery_summary}" if discovery_summary else "")
            + (f"; {alignment_summary}" if alignment_summary else "")
        )
    if research.ok:
        return (
            f"Multi-agent flow prepared {len(getattr(universe, 'candidates', ()))} "
            "universe names and "
            f"{len(getattr(research, 'rows', ()))} ML/RL research rows"
            + (f"; {discovery_summary}" if discovery_summary else "")
            + (f"; {alignment_summary}" if alignment_summary else "")
        )
    if universe.ok:
        return (
            f"Multi-agent flow prepared {len(getattr(universe, 'candidates', ()))} "
            "universe names"
            + (f"; {discovery_summary}" if discovery_summary else "")
        )
    return "Multi-agent workflow unavailable"


def _first_error(*responses):
    for response in responses:
        error = getattr(response, "error", None)
        if error is not None:
            return error
    return None


__all__ = ["build_multi_agent_workflow", "replace_workflow_universe"]
