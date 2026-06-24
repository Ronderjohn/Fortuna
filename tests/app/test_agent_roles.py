from __future__ import annotations

from pathlib import Path

from fortuna.agentic.contracts import (
    BriefingItem,
    MarketUniverseCandidate,
    MarketUniverseResponse,
    NightlyAlignmentRow,
    NightlyAlignmentStatus,
    PortfolioAllocationResponse,
    ShortlistAnalysisItem,
    ShortlistAnalysisResponse,
    ShortlistBriefingResponse,
    TrainingResearchPlanResponse,
    TrainingResearchRow,
)
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
from fortuna.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )


def _universe_response(**overrides) -> MarketUniverseResponse:
    defaults = dict(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="RELIANCE",
                source="screener",
                liquidity_score=12.4,
                activity_score=1.7,
                trend_pct=2.8,
                volume_ratio=1.35,
                regime="TRENDING",
                notes=("adaptive_boost=+0.10",),
            ),
            MarketUniverseCandidate(
                symbol="SBIN.NS",
                display_name="SBIN",
                source="screener",
                liquidity_score=11.6,
                activity_score=1.2,
                trend_pct=-1.3,
                volume_ratio=1.10,
                regime="TRENDING",
            ),
        ),
    )
    defaults.update(overrides)
    return MarketUniverseResponse(**defaults)


def test_universe_scout_run_returns_response_and_role(tmp_path: Path, monkeypatch):
    expected = _universe_response()
    monkeypatch.setattr(
        "fortuna.app.agent_roles.universe_scout.build_market_universe",
        lambda **kwargs: expected,
    )

    response, role = universe_scout.run(settings=_settings(tmp_path), limit=5, source="screener")

    assert response is expected
    assert role.name == "universe_scout"
    assert role.ok is True
    assert role.focus_symbols == ("RELIANCE.NS", "SBIN.NS")
    assert "blend=screener_liquidity+ohlcv_activity" in role.notes


def test_liquidity_scout_run_returns_response_and_role():
    universe = _universe_response()

    response, role = liquidity_scout.run(universe)

    assert response is universe
    assert role.name == "liquidity_scout"
    assert role.ok is True
    assert role.focus_symbols == ("RELIANCE.NS", "SBIN.NS")
    assert "top_liquidity=RELIANCE.NS:12.40" in role.notes


def test_activity_scout_run_returns_response_and_role():
    universe = _universe_response()

    response, role = activity_scout.run(universe)

    assert response is universe
    assert role.name == "activity_scout"
    assert role.ok is True
    assert role.headline == "2 activity/trend names enriched from local OHLCV context"
    assert "coverage=2/2" in role.notes


def test_instrument_analyst_run_returns_response_and_role(tmp_path: Path, monkeypatch):
    expected = ShortlistAnalysisResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(symbol="RELIANCE.NS", selection_rank=1),
            ShortlistAnalysisItem(symbol="SBIN.NS", selection_rank=2),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.agent_roles.instrument_analyst.analyze_market_shortlist",
        lambda **kwargs: expected,
    )

    response, role = instrument_analyst.run(settings=_settings(tmp_path), source="screener")

    assert response is expected
    assert role.name == "instrument_critic"
    assert role.headline == "2 analyzed setups with deterministic critique"


def test_briefing_agent_run_returns_response_and_role(tmp_path: Path, monkeypatch):
    expected = ShortlistBriefingResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        headline="2 candidate setups, 2 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.84,
                verdict="candidate",
                summary="lead buy",
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.agent_roles.briefing_agent.build_shortlist_briefing",
        lambda **kwargs: expected,
    )

    response, role = briefing_agent.run(settings=_settings(tmp_path), source="screener")

    assert response is expected
    assert role.name == "briefing_agent"
    assert role.headline == "1 candidate setups ready for operator briefing"


def test_portfolio_critic_run_returns_response_and_role(tmp_path: Path, monkeypatch):
    expected = PortfolioAllocationResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        headline="1 selected, 1 skipped under portfolio limits",
        max_positions=1,
        selected_symbols=("RELIANCE.NS",),
        notes=("portfolio_critic=enabled",),
    )
    monkeypatch.setattr(
        "fortuna.app.agent_roles.portfolio_critic.build_portfolio_allocation",
        lambda **kwargs: expected,
    )

    response, role = portfolio_critic.run(settings=_settings(tmp_path), source="screener")

    assert response is expected
    assert role.name == "portfolio_allocator"
    assert role.focus_symbols == ("RELIANCE.NS",)
    assert "portfolio_critic=enabled" in role.notes


def test_research_planner_run_returns_response_and_role(tmp_path: Path, monkeypatch):
    expected = TrainingResearchPlanResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        selection_policy="diversified",
        refresh_target="rl",
        discovery_recommended_refresh_target="ml",
        effective_refresh_target="all",
        ml_symbols=("RELIANCE.NS",),
        rl_symbols=("SBIN.NS",),
        rows=(
            TrainingResearchRow(symbol="RELIANCE.NS", target="ml", selection_rank=1),
            TrainingResearchRow(symbol="SBIN.NS", target="rl", selection_rank=2),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.agent_roles.research_planner.build_training_research_plan",
        lambda **kwargs: expected,
    )

    response, role = research_planner.run(settings=_settings(tmp_path), source="screener")

    assert response is expected
    assert role.name == "research_planner"
    assert role.focus_symbols == ("SBIN.NS", "RELIANCE.NS")
    assert "selection_policy=diversified" in role.notes


def test_operations_monitor_run_returns_response_and_role(tmp_path: Path, monkeypatch):
    expected = NightlyAlignmentStatus(
        report_count=2,
        enabled_reports=2,
        aligned_reports=1,
        latest_target_mix="both=1, rl=1",
        recommended_discovery_action="Rebuild the market universe from Screener liquidity inputs.",
        recent_rows=(
            NightlyAlignmentRow(
                path="reports/nightly/20260603_0100.json",
                overall_status="ok",
                basket_size=2,
                alignment_enabled=True,
                target_mix="both=1, rl=1",
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.agent_roles.operations_monitor.build_nightly_alignment_status",
        lambda settings: expected,
    )

    response, role = operations_monitor.run(settings=_settings(tmp_path))

    assert response is expected
    assert role.name == "operations_monitor"
    assert role.ok is True
    assert "alignment ratio=0.50" in role.headline
    assert "discovery follow-up suggested" in role.headline
