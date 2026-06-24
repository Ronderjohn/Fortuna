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
    TrainingCandidate,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
    TrainingResearchRow,
)
from fortuna.app.multi_agent_team import build_multi_agent_workflow, replace_workflow_universe
from fortuna.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )


def test_build_multi_agent_workflow_composes_explicit_roles(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(
            report_count=2,
            enabled_reports=2,
            aligned_reports=1,
            latest_target_mix="both=1, rl=1",
            latest_refreshed_target_mix="both=1",
            recommended_action="Refresh the full ML/RL training research plan.",
            recommended_refresh_target="all",
            recommended_force_refresh=True,
            recommended_discovery_action=(
                "Rebuild the market universe from Screener liquidity inputs."
            ),
            recommended_discovery_cli_command=(
                "uv run python scripts/build_market_universe.py --source screener"
            ),
            recent_rows=(
                NightlyAlignmentRow(
                    path="reports/nightly/20260603_0100.json",
                    overall_status="ok",
                    basket_size=2,
                    alignment_enabled=True,
                    target_mix="both=1, rl=1",
                    refreshed_target_mix="both=1",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_market_universe",
        lambda **kwargs: MarketUniverseResponse(
            ok=True,
            source="screener",
            timeframe="1d",
            lookback_days=30,
            nightly_alignment_summary="recent=1/1 over 1 run(s); target=all force_refresh=1",
            nightly_alignment_target="all",
            nightly_alignment_force_refresh=True,
            candidates=(
                MarketUniverseCandidate(
                    symbol="RELIANCE.NS",
                    display_name="RELIANCE",
                    source="screener",
                    liquidity_score=12.4,
                    notes=("adaptive_boost=+0.10",),
                ),
                MarketUniverseCandidate(
                    symbol="SBIN.NS",
                    display_name="SBIN",
                    source="screener",
                    liquidity_score=11.6,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.analyze_market_shortlist",
        lambda **kwargs: ShortlistAnalysisResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            items=(
                ShortlistAnalysisItem(
                    symbol="RELIANCE.NS",
                    selection_rank=1,
                ),
                ShortlistAnalysisItem(
                    symbol="SBIN.NS",
                    selection_rank=2,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_shortlist_briefing",
        lambda **kwargs: ShortlistBriefingResponse(
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
                BriefingItem(
                    symbol="SBIN.NS",
                    action="SELL",
                    confidence=0.78,
                    verdict="candidate",
                    summary="balancing sell",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_training_candidates",
        lambda **kwargs: TrainingCandidateResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            candidates=(
                TrainingCandidate(
                    symbol="RELIANCE.NS",
                    shortlist_rank=1,
                    selection_rank=1,
                    decision_action="BUY",
                    critique_verdict="candidate",
                    ml_candidate=True,
                    rl_candidate=True,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_portfolio_allocation",
        lambda **kwargs: PortfolioAllocationResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            headline="1 selected, 1 skipped under portfolio limits",
            max_positions=1,
            selected_symbols=("RELIANCE.NS",),
            notes=("portfolio_critic=enabled",),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            selection_policy="diversified",
            refresh_target="rl",
            discovery_posture="ml",
            strategy_posture="rl",
            effective_posture="all",
            discovery_recommended_refresh_target="ml",
            effective_refresh_target="all",
            nightly_alignment_summary="recent=1/1 over 1 run(s); target=rl force_refresh=1",
            nightly_alignment_target="rl",
            nightly_alignment_force_refresh=True,
            selected_target_mix="ml=1, rl=1",
            selected_regime_mix="ranging=1, trending=1",
            ml_symbols=("RELIANCE.NS",),
            rl_symbols=("SBIN.NS",),
            rows=(
                TrainingResearchRow(
                    symbol="RELIANCE.NS",
                    target="ml",
                    selection_rank=1,
                ),
                TrainingResearchRow(
                    symbol="SBIN.NS",
                    target="rl",
                    selection_rank=2,
                ),
            ),
        ),
    )

    response = build_multi_agent_workflow(
        settings=_settings(tmp_path),
        timeframe="5m",
        days=30,
        source="screener",
    )

    assert response.ok is True
    assert response.universe is not None
    assert response.candidates is not None
    assert response.allocation is not None
    assert response.research is not None
    assert response.nightly_alignment is not None
    assert response.headline.startswith("Multi-agent flow prepared 2 universe names")
    assert "basket/research aligned 1/1" in response.headline
    roles = {row.name: row for row in response.roles}
    assert roles["universe_scout"].focus_symbols == ("RELIANCE.NS", "SBIN.NS")
    assert "adaptive_boost=+0.10" in roles["universe_scout"].notes
    assert "blend=screener_liquidity+ohlcv_activity" in roles["universe_scout"].notes
    assert "recent=1/1 over 1 run(s); target=all force_refresh=1" in roles["universe_scout"].notes
    assert "nightly_target=all" in roles["universe_scout"].notes
    assert "nightly_force_refresh=1" in roles["universe_scout"].notes
    assert (
        roles["liquidity_scout"].headline
        == "2 liquidity-ranked names prepared from screener seeds"
    )
    assert roles["liquidity_scout"].focus_symbols == ("RELIANCE.NS", "SBIN.NS")
    assert "top_liquidity=RELIANCE.NS:12.40" in roles["liquidity_scout"].notes
    assert "seed_source=screener" in roles["liquidity_scout"].notes
    assert roles["activity_scout"].headline == "Activity scout awaiting OHLCV enrichment"
    assert roles["activity_scout"].ok is False
    assert roles["instrument_critic"].headline == "2 analyzed setups with deterministic critique"
    assert roles["briefing_agent"].headline == "2 candidate setups ready for operator briefing"
    assert roles["portfolio_allocator"].focus_symbols == ("RELIANCE.NS",)
    assert roles["research_planner"].focus_symbols == ("SBIN.NS", "RELIANCE.NS")
    assert "selection_policy=diversified" in roles["research_planner"].notes
    assert "basket/research aligned 1/1" in roles["research_planner"].notes
    assert "basket_research_mix=ml=1" in roles["research_planner"].notes
    assert "discovery_target=ml" in roles["research_planner"].notes
    assert "effective_target=all" in roles["research_planner"].notes
    assert "target_mismatch=requested=rl discovery=ml" in roles["research_planner"].notes
    assert "recent=1/1 over 1 run(s); target=rl force_refresh=1" in roles["research_planner"].notes
    assert "nightly_target=rl" in roles["research_planner"].notes
    assert (
        roles["training_data_curator"].headline
        == "2 training rows curated for mixed model refresh"
    )
    assert roles["training_data_curator"].focus_symbols == ("RELIANCE.NS", "SBIN.NS")
    assert "discovery_posture=ml" in roles["training_data_curator"].notes
    assert "setup_posture=rl" in roles["training_data_curator"].notes
    assert "effective_posture=all" in roles["training_data_curator"].notes
    assert "target_mix=ml=1, rl=1" in roles["training_data_curator"].notes
    assert "regime_mix=ranging=1, trending=1" in roles["training_data_curator"].notes
    assert (
        roles["operations_monitor"].headline
        == "2 nightly reports tracked; alignment ratio=0.50; discovery follow-up suggested"
    )
    assert "recent=1/1" in roles["operations_monitor"].notes
    assert "latest_run=ok:2" in roles["operations_monitor"].notes
    assert "refresh_target=all" in roles["operations_monitor"].notes
    assert "force_refresh=1" in roles["operations_monitor"].notes


def test_replace_workflow_universe_recomputes_discovery_roles(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_market_universe",
        lambda **kwargs: MarketUniverseResponse(
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
                ),
                MarketUniverseCandidate(
                    symbol="SBIN.NS",
                    display_name="SBIN",
                    source="screener",
                    liquidity_score=11.8,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.analyze_market_shortlist",
        lambda **kwargs: ShortlistAnalysisResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            items=(),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_shortlist_briefing",
        lambda **kwargs: ShortlistBriefingResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            headline="0 candidate setups ready for operator briefing",
            items=(),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_portfolio_allocation",
        lambda **kwargs: PortfolioAllocationResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            headline="0 selected, 0 skipped under portfolio limits",
            max_positions=3,
            items=(),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_training_candidates",
        lambda **kwargs: TrainingCandidateResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            candidates=(),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=True,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            rows=(),
        ),
    )

    response = build_multi_agent_workflow(settings=_settings(tmp_path), source="screener")
    refreshed = replace_workflow_universe(
        response,
        universe=MarketUniverseResponse(
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
                    activity_score=1.8,
                    trend_pct=2.4,
                    volume_ratio=1.7,
                    regime="TRENDING",
                ),
                MarketUniverseCandidate(
                    symbol="SBIN.NS",
                    display_name="SBIN",
                    source="screener",
                    liquidity_score=11.8,
                    activity_score=1.6,
                    trend_pct=1.9,
                    volume_ratio=1.5,
                    regime="TRENDING",
                ),
            ),
        ),
    )

    roles = {row.name: row for row in refreshed.roles}
    assert "discovery aligned 2/2" in refreshed.headline
    assert "discovery aligned 2/2" in roles["universe_scout"].notes
    assert roles["activity_scout"].ok is True
    assert roles["activity_scout"].headline.startswith("2 activity/trend names")


def test_build_multi_agent_workflow_activity_scout_uses_ohlcv_context(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_market_universe",
        lambda **kwargs: MarketUniverseResponse(
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
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.analyze_market_shortlist",
        lambda **kwargs: ShortlistAnalysisResponse(
            ok=False,
            source="screener",
            timeframe="5m",
            lookback_days=30,
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_shortlist_briefing",
        lambda **kwargs: ShortlistBriefingResponse(
            ok=False,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            headline="Briefing unavailable",
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_training_candidates",
        lambda **kwargs: TrainingCandidateResponse(
            ok=False,
            source="screener",
            timeframe="5m",
            lookback_days=30,
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_portfolio_allocation",
        lambda **kwargs: PortfolioAllocationResponse(
            ok=False,
            source="screener",
            timeframe="5m",
            lookback_days=30,
            headline="Allocation unavailable",
            max_positions=3,
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=False,
            source="screener",
            timeframe="5m",
            lookback_days=30,
        ),
    )

    response = build_multi_agent_workflow(
        settings=_settings(tmp_path),
        timeframe="5m",
        days=30,
        source="screener",
    )

    roles = {row.name: row for row in response.roles}
    assert roles["activity_scout"].ok is True
    assert (
        roles["activity_scout"].headline
        == "2 activity/trend names enriched from local OHLCV context"
    )
    assert "discovery aligned 2/2" in response.headline
    assert roles["activity_scout"].focus_symbols == ("RELIANCE.NS", "SBIN.NS")
    assert "discovery aligned 2/2" in roles["universe_scout"].notes
    assert "discovery_overlap=RELIANCE.NS, SBIN.NS" in roles["universe_scout"].notes
    assert "top_regime=RELIANCE.NS:trending" in roles["activity_scout"].notes
    assert "top_volume_ratio=RELIANCE.NS:1.35" in roles["activity_scout"].notes
    assert "top_trend=RELIANCE.NS:+2.80%" in roles["activity_scout"].notes
    assert "coverage=2/2" in roles["activity_scout"].notes


def test_build_multi_agent_workflow_returns_error_when_all_roles_fail(
    tmp_path: Path,
    monkeypatch,
):
    universe = MarketUniverseResponse(
        ok=False,
        source="registry",
        timeframe="1d",
        lookback_days=30,
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_market_universe",
        lambda **kwargs: universe,
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.analyze_market_shortlist",
        lambda **kwargs: ShortlistAnalysisResponse(
            ok=False,
            source="registry",
            timeframe="5m",
            lookback_days=30,
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_shortlist_briefing",
        lambda **kwargs: ShortlistBriefingResponse(
            ok=False,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            headline="unavailable",
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_training_candidates",
        lambda **kwargs: TrainingCandidateResponse(
            ok=False,
            source="registry",
            timeframe="5m",
            lookback_days=30,
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_portfolio_allocation",
        lambda **kwargs: PortfolioAllocationResponse(
            ok=False,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            headline="unavailable",
            max_positions=3,
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=False,
            source="registry",
            timeframe="5m",
            lookback_days=30,
        ),
    )

    response = build_multi_agent_workflow(settings=_settings(tmp_path))

    assert response.ok is False
    assert response.headline == "Multi-agent workflow unavailable"
