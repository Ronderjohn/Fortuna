from __future__ import annotations

import json

from fortuna.agentic.contracts import (
    AgentRoleStatus,
    AllocationDecision,
    BriefingItem,
    DecisionSummary,
    MarketUniverseCandidate,
    MarketUniverseResponse,
    MultiAgentWorkflowResponse,
    NightlyAlignmentRow,
    NightlyAlignmentStatus,
    PortfolioAllocationResponse,
    PortfolioCritique,
    PortfolioExposureNote,
    RiskCritique,
    ShortlistAnalysisItem,
    ShortlistAnalysisResponse,
    ShortlistBriefingResponse,
    TrainingCandidate,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
    TrainingResearchRow,
)
from fortuna.app.operator_workflow import (
    WorkflowSnapshot,
    WorkflowSummary,
    annotate_artifact_recovery_posture,
    annotate_discovery_refresh,
    build_workflow_snapshot,
    export_workflow_snapshot,
    replace_workflow_snapshot_sections,
    summarize_market_universe,
    summarize_multi_agent_workflow,
    summarize_portfolio_allocation,
    summarize_shortlist_analysis,
    summarize_shortlist_briefing,
    summarize_training_candidates,
    summarize_training_research,
)
from fortuna.config.settings import Settings


def _settings(tmp_path) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )


def test_summarize_market_universe_builds_rows_and_metrics():
    response = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        nightly_alignment_summary="recent=1/2 over 2 run(s); target=all force_refresh=1",
        nightly_alignment_target="all",
        nightly_alignment_force_refresh=True,
        nightly_recent_window=2,
        nightly_recent_enabled=2,
        nightly_recent_aligned=1,
        nightly_recent_latest_status="ok",
        nightly_recent_latest_basket_size=3,
        scout_liquidity_symbols=("RELIANCE.NS",),
        scout_activity_symbols=("RELIANCE.NS",),
        scout_volume_dense_symbols=("RELIANCE.NS",),
        scout_overlap_symbols=("RELIANCE.NS",),
        scout_summary=(
            "liquidity=RELIANCE.NS | activity=RELIANCE.NS | "
            "volume_dense=RELIANCE.NS | overlap=RELIANCE.NS"
        ),
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                liquidity_score=12.345,
                trend_pct=3.456,
                regime="TRENDING",
                adaptive_score_adjustment=0.12,
                adaptive_row_count=3,
                avg_turnover=1234567.8,
                notes=("adaptive_boost=+0.12 avg_pnl=+1.80% rows=3",),
            ),
        ),
    )
    summary = summarize_market_universe(response)
    assert summary.metrics["count"] == 1
    assert summary.metrics["source"] == "screener"
    assert summary.metrics["adaptive_count"] == 1
    assert summary.metrics["scout_liquidity_symbols"] == "RELIANCE.NS"
    assert summary.metrics["scout_activity_symbols"] == "RELIANCE.NS"
    assert summary.metrics["scout_volume_dense_symbols"] == "RELIANCE.NS"
    assert summary.metrics["scout_overlap_symbols"] == "RELIANCE.NS"
    assert summary.metrics["nightly_alignment_target"] == "all"
    assert summary.metrics["nightly_alignment_force_refresh"] is True
    assert summary.metrics["nightly_recent_window"] == 2
    assert summary.metrics["nightly_recent_aligned"] == 1
    assert summary.rows[0]["symbol"] == "RELIANCE.NS"
    assert summary.rows[0]["liquidity_score"] == 12.35
    assert summary.rows[0]["adaptive_score_adjustment"] == 0.12
    assert summary.rows[0]["adaptive_note"] == "adaptive_boost=+0.12 avg_pnl=+1.80% rows=3"
    assert summary.notes[0].startswith("liquidity=RELIANCE.NS")
    assert summary.notes[-1] == "recent=1/2 over 2 run(s); target=all force_refresh=1"


def test_summarize_shortlist_briefing_tracks_candidates_and_notes():
    response = ShortlistBriefingResponse(
        ok=True,
        source="auto",
        timeframe="5m",
        lookback_days=20,
        headline="2 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.82,
                verdict="candidate",
                summary="Trend aligned",
                selection_rank=1,
                priority_score=1.31,
                exposure_penalty=0.0,
                winning_strategy="ORB",
            ),
            BriefingItem(
                symbol="ITC.NS",
                action="DO_NOT_ENTER",
                confidence=0.61,
                verdict="watch",
                summary="Not enough edge",
                selection_rank=2,
                priority_score=0.63,
                exposure_penalty=0.12,
            ),
        ),
        portfolio=PortfolioCritique(
            summary="Directional crowding risk",
            notes=(PortfolioExposureNote(level="warn", message="3 symbols lean BUY"),),
        ),
    )
    summary = summarize_shortlist_briefing(response)
    assert summary.metrics["count"] == 2
    assert summary.metrics["candidates"] == 1
    assert "3 symbols lean BUY" in summary.notes
    assert summary.rows[0]["confidence_pct"] == 82.0
    assert summary.rows[0]["winning_strategy"] == "ORB"
    assert summary.rows[1]["exposure_penalty"] == 0.12


def test_summarize_shortlist_analysis_tracks_priority_and_penalty():
    response = ShortlistAnalysisResponse(
        ok=True,
        source="auto",
        timeframe="5m",
        lookback_days=20,
        items=(
            ShortlistAnalysisItem(
                symbol="SBIN.NS",
                selection_rank=1,
                priority_score=1.22,
                exposure_penalty=0.0,
                liquidity_score=12.0,
                decision=DecisionSummary(action="BUY", confidence=0.82, summary="BUY won"),
                critique=RiskCritique(
                    severity="low",
                    summary="clean",
                    verdict="candidate",
                ),
            ),
        ),
    )
    summary = summarize_shortlist_analysis(response)
    assert summary.metrics["count"] == 1
    assert summary.rows[0]["selection_rank"] == 1
    assert summary.rows[0]["priority_score"] == 1.22


def test_summarize_training_candidates_counts_ml_and_rl():
    response = TrainingCandidateResponse(
        ok=True,
        source="auto",
        timeframe="15m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                symbol="SBIN.NS",
                shortlist_rank=1,
                selection_rank=1,
                decision_action="BUY",
                decision_confidence=0.78,
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                priority_score=1.28,
                remediation_target="rl",
                remediation_pressure=0.05,
                exposure_penalty=0.0,
            ),
            TrainingCandidate(
                symbol="ITC.NS",
                shortlist_rank=2,
                selection_rank=2,
                decision_action="DO_NOT_ENTER",
                decision_confidence=0.55,
                critique_verdict="watch",
                ml_candidate=True,
                rl_candidate=False,
                priority_score=0.66,
                exposure_penalty=0.12,
            ),
        ),
    )
    summary = summarize_training_candidates(response)
    assert summary.metrics["ml_count"] == 2
    assert summary.metrics["rl_count"] == 1
    assert summary.rows[0]["remediation_target"] == "rl"
    assert summary.rows[0]["remediation_pressure"] == 0.05
    assert summary.rows[0]["setup_family_reinforcement"] is None
    assert summary.rows[1]["rl_candidate"] is False
    assert summary.rows[1]["exposure_penalty"] == 0.12


def test_summarize_training_candidates_exposes_setup_family_reinforcement():
    response = TrainingCandidateResponse(
        ok=True,
        source="auto",
        timeframe="5m",
        lookback_days=20,
        candidates=(
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=1,
                selection_rank=1,
                winning_strategy="ORB",
                decision_action="BUY",
                decision_confidence=0.82,
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                priority_score=1.34,
                setup_family_reinforcement=0.05,
                setup_family_verdict="winner",
            ),
        ),
    )
    summary = summarize_training_candidates(response)
    assert summary.rows[0]["setup_family_reinforcement"] == 0.05
    assert summary.rows[0]["setup_family_verdict"] == "winner"


def test_summarize_training_research_counts_ml_and_rl_targets():
    response = TrainingResearchPlanResponse(
        ok=True,
        source="auto",
        timeframe="15m",
        lookback_days=30,
        selection_policy="diversified",
        refresh_requested=True,
        refresh_target="rl",
        discovery_posture="rl",
        strategy_posture="rl",
        candidate_posture="ml",
        nightly_posture="rl",
        effective_posture="rl",
        discovery_recommended_refresh_target="rl",
        effective_refresh_target="rl",
        refresh_urgency="medium",
        refresh_urgency_target="rl",
        refresh_urgency_score=0.05,
        refresh_urgency_rows=1,
        refresh_urgency_summary="medium RL remediation pressure on 1 row(s); max=+0.05 total=+0.05",
        follow_up_action=(
            "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s)."
        ),
        discovery_follow_up_target="rl",
        discovery_follow_up_summary=(
            "Discovery scouts currently lean RL-focused from rl universe posture."
        ),
        discovery_follow_up_action=(
            "Refresh market universe and shortlist review with RL-focused discovery focus "
            "before the next training cycle."
        ),
        research_follow_up_target="rl",
        research_follow_up_summary=(
            "medium RL remediation pressure on 1 row(s); max=+0.05 total=+0.05"
        ),
        research_follow_up_action=(
            "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s)."
        ),
        execution_follow_up_target="rl",
        execution_follow_up_summary=(
            "Nightly execution drift currently leans RL-focused from rl execution posture."
        ),
        execution_follow_up_action=(
            "Review nightly execution path and retarget execution candidate selection "
            "toward RL-focused before the next promotion or nightly cycle."
        ),
        nightly_alignment_summary="recent=1/2 over 2 run(s); target=rl force_refresh=1",
        nightly_alignment_target="rl",
        nightly_alignment_force_refresh=True,
        nightly_recent_window=2,
        nightly_recent_enabled=2,
        nightly_recent_aligned=1,
        nightly_recent_latest_status="ok",
        nightly_recent_latest_basket_size=3,
        discovery_preferred_symbols=("SBIN.NS",),
        discovery_volume_dense_symbols=("SBIN.NS", "ITC.NS"),
        ml_symbols=("SBIN.NS", "ITC.NS"),
        rl_symbols=("SBIN.NS",),
        selected_target_mix="both=1, ml=1",
        selected_regime_mix="trending=1",
        rows=(
            TrainingResearchRow(
                symbol="SBIN.NS",
                target="both",
                universe_rank=1,
                shortlist_rank=1,
                selection_rank=1,
                decision_action="BUY",
                critique_verdict="candidate",
                priority_score=1.28,
                remediation_target="rl",
                remediation_pressure=0.05,
                setup_family_reinforcement=0.05,
                setup_family_verdict="winner",
                refreshed=True,
            ),
            TrainingResearchRow(
                symbol="ITC.NS",
                target="ml",
                universe_rank=2,
                shortlist_rank=2,
                selection_rank=2,
                decision_action="DO_NOT_ENTER",
                critique_verdict="watch",
                priority_score=0.66,
            ),
        ),
    )
    summary = summarize_training_research(response)
    assert summary.metrics["count"] == 2
    assert summary.metrics["refresh_urgency"] == "medium"
    assert summary.metrics["refresh_urgency_target"] == "rl"
    assert summary.metrics["refresh_urgency_score"] == 0.05
    assert summary.metrics["refresh_urgency_rows"] == 1
    assert (
        summary.metrics["refresh_urgency_summary"]
        == "medium RL remediation pressure on 1 row(s); max=+0.05 total=+0.05"
    )
    assert (
        summary.metrics["follow_up_action"]
        == "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s)."
    )
    assert summary.metrics["discovery_follow_up_target"] == "rl"
    assert summary.metrics["discovery_follow_up_summary"] == (
        "Discovery scouts currently lean RL-focused from rl universe posture."
    )
    assert summary.metrics["discovery_follow_up_action"] == (
        "Refresh market universe and shortlist review with RL-focused discovery focus "
        "before the next training cycle."
    )
    assert summary.metrics["research_follow_up_target"] == "rl"
    assert summary.metrics["research_follow_up_summary"] == (
        "medium RL remediation pressure on 1 row(s); max=+0.05 total=+0.05"
    )
    assert summary.metrics["research_follow_up_action"] == (
        "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s)."
    )
    assert summary.metrics["execution_follow_up_target"] == "rl"
    assert summary.metrics["execution_follow_up_summary"] == (
        "Nightly execution drift currently leans RL-focused from rl execution posture."
    )
    assert summary.metrics["execution_follow_up_action"] == (
        "Review nightly execution path and retarget execution candidate selection "
        "toward RL-focused before the next promotion or nightly cycle."
    )
    assert summary.notes == (
        "recent=1/2 over 2 run(s); target=rl force_refresh=1",
        "medium RL remediation pressure on 1 row(s); max=+0.05 total=+0.05",
        "supported=2, target=ml, overlap=1, volume_dense=2, symbols=SBIN.NS,ITC.NS",
        "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s).",
        "Discovery scouts currently lean RL-focused from rl universe posture.",
        "medium RL remediation pressure on 1 row(s); max=+0.05 total=+0.05",
        "Nightly execution drift currently leans RL-focused from rl execution posture.",
    )
    assert summary.metrics["ml_count"] == 2
    assert summary.metrics["rl_count"] == 1
    assert summary.metrics["selection_policy"] == "diversified"
    assert summary.metrics["discovery_recommended_refresh_target"] == "rl"
    assert summary.metrics["discovery_posture"] == "rl"
    assert summary.metrics["strategy_posture"] == "rl"
    assert summary.metrics["candidate_posture"] == "ml"
    assert summary.metrics["nightly_posture"] == "rl"
    assert summary.metrics["effective_posture"] == "rl"
    assert summary.metrics["effective_refresh_target"] == "rl"
    assert summary.metrics["target_mismatch"] is False
    assert summary.metrics["refresh_requested"] is True
    assert summary.metrics["refreshed_count"] == 1
    assert summary.metrics["selected_target_mix"] == "both=1, ml=1"
    assert summary.metrics["selected_regime_mix"] == "trending=1"
    assert summary.metrics["scout_support_summary"] == (
        "supported=2, target=ml, overlap=1, volume_dense=2, symbols=SBIN.NS,ITC.NS"
    )
    assert summary.metrics["scout_supported_symbols"] == "SBIN.NS,ITC.NS"
    assert summary.metrics["scout_overlap_selected_count"] == 1
    assert summary.metrics["scout_volume_dense_selected_count"] == 2
    assert summary.metrics["nightly_alignment_target"] == "rl"
    assert summary.metrics["nightly_alignment_force_refresh"] is True
    assert summary.metrics["nightly_recent_window"] == 2
    assert summary.metrics["nightly_recent_aligned"] == 1
    assert summary.notes == (
        "recent=1/2 over 2 run(s); target=rl force_refresh=1",
        "medium RL remediation pressure on 1 row(s); max=+0.05 total=+0.05",
        "supported=2, target=ml, overlap=1, volume_dense=2, symbols=SBIN.NS,ITC.NS",
        "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s).",
        "Discovery scouts currently lean RL-focused from rl universe posture.",
        "medium RL remediation pressure on 1 row(s); max=+0.05 total=+0.05",
        "Nightly execution drift currently leans RL-focused from rl execution posture.",
    )
    assert summary.rows[0]["target"] == "both"
    assert summary.rows[0]["remediation_target"] == "rl"
    assert summary.rows[0]["remediation_pressure"] == 0.05
    assert summary.rows[0]["setup_family_reinforcement"] == 0.05
    assert summary.rows[0]["setup_family_verdict"] == "winner"
    assert summary.rows[1]["target"] == "ml"


def test_summarize_multi_agent_workflow_surfaces_team_research_posture():
    response = MultiAgentWorkflowResponse(
        ok=True,
        source="screener",
        timeframe="15m",
        lookback_days=30,
        headline="multi-agent flow ready",
        roles=(
            AgentRoleStatus(
                name="universe_scout",
                ok=True,
                headline="2 universe names",
                notes=("discovery partial 1/2", "discovery_overlap=SBIN.NS"),
            ),
            AgentRoleStatus(
                name="research_planner",
                ok=True,
                headline="3 ML/RL research rows prepared",
                focus_symbols=("SBIN.NS", "RELIANCE.NS"),
            ),
        ),
        research=TrainingResearchPlanResponse(
            ok=True,
            source="screener",
            timeframe="15m",
            lookback_days=30,
            selection_policy="diversified",
            refresh_requested=True,
            refresh_target="rl",
            discovery_posture="ml",
            strategy_posture="rl",
            candidate_posture="ml",
            nightly_posture="rl",
            effective_posture="all",
            discovery_recommended_refresh_target="ml",
            effective_refresh_target="all",
            discovery_preferred_symbols=("SBIN.NS",),
            discovery_volume_dense_symbols=("SBIN.NS", "RELIANCE.NS"),
            selected_target_mix="both=1, ml=2",
            selected_regime_mix="ranging=1, trending=2",
            ml_symbols=("SBIN.NS", "RELIANCE.NS"),
            rl_symbols=("SBIN.NS",),
            rows=(
                TrainingResearchRow(symbol="SBIN.NS", target="both", refreshed=True),
                TrainingResearchRow(symbol="RELIANCE.NS", target="ml", refreshed=True),
                TrainingResearchRow(symbol="ITC.NS", target="ml"),
            ),
        ),
        allocation=PortfolioAllocationResponse(
            ok=True,
            source="screener",
            timeframe="15m",
            lookback_days=30,
            headline="1 selected",
            max_positions=1,
            selected_symbols=("SBIN.NS",),
        ),
        universe=MarketUniverseResponse(
            ok=True,
            source="screener",
            timeframe="1d",
            lookback_days=30,
            candidates=(
                MarketUniverseCandidate(
                    symbol="SBIN.NS",
                    display_name="SBIN",
                    source="screener",
                    liquidity_score=11.8,
                    activity_score=1.5,
                    trend_pct=2.4,
                    volume_ratio=1.3,
                    regime="TRENDING",
                ),
                MarketUniverseCandidate(
                    symbol="RELIANCE.NS",
                    display_name="RELIANCE",
                    source="screener",
                    liquidity_score=12.1,
                    activity_score=1.1,
                    trend_pct=-1.0,
                    volume_ratio=0.9,
                    regime="RANGING",
                ),
                MarketUniverseCandidate(
                    symbol="ITC.NS",
                    display_name="ITC",
                    source="screener",
                    liquidity_score=10.4,
                    activity_score=1.6,
                    trend_pct=3.1,
                    volume_ratio=1.4,
                    regime="TRENDING",
                ),
            ),
        ),
    )

    summary = summarize_multi_agent_workflow(response)

    assert summary.team.metrics["research_headline"] == "3 ML/RL research rows prepared"
    assert summary.team.metrics["research_selection_policy"] == "diversified"
    assert summary.team.metrics["research_refresh_target"] == "rl"
    assert summary.team.metrics["research_discovery_recommended_target"] == "ml"
    assert summary.team.metrics["research_discovery_posture"] == "ml"
    assert summary.team.metrics["research_strategy_posture"] == "rl"
    assert summary.team.metrics["research_candidate_posture"] == "ml"
    assert summary.team.metrics["research_nightly_posture"] == "rl"
    assert summary.team.metrics["research_effective_posture"] == "all"
    assert summary.team.metrics["research_effective_target"] == "all"
    assert summary.team.metrics["research_target_mismatch"] is True
    assert summary.team.metrics["research_refresh_requested"] is True
    assert summary.team.metrics["research_refreshed_count"] == 2
    assert summary.team.metrics["research_ml_count"] == 2
    assert summary.team.metrics["research_rl_count"] == 1
    assert summary.team.metrics["research_selected_target_mix"] == "both=1, ml=2"
    assert summary.team.metrics["research_selected_regime_mix"] == "ranging=1, trending=2"
    assert summary.team.metrics["research_scout_support_summary"] == (
        "supported=2, target=ml, overlap=1, volume_dense=2, symbols=SBIN.NS,RELIANCE.NS"
    )
    assert summary.team.metrics["research_scout_supported_symbols"] == "SBIN.NS,RELIANCE.NS"
    assert summary.team.metrics["research_scout_overlap_selected_count"] == 1
    assert summary.team.metrics["research_scout_volume_dense_selected_count"] == 2
    assert summary.team.metrics["research_alignment_summary"] == "basket/research aligned 1/1"
    assert summary.team.metrics["research_alignment_target_mix"] == "both=1"
    assert summary.team.metrics["discovery_alignment_summary"] == "discovery aligned 3/3"
    assert summary.team.metrics["discovery_overlap_symbols"] == "RELIANCE.NS, SBIN.NS, ITC.NS"
    assert summary.team.metrics["discovery_alignment_overlap_count"] == 3
    assert summary.team.metrics["discovery_alignment_compare_count"] == 3
    assert "research_planner: 3 ML/RL research rows prepared" in summary.team.notes
    assert "discovery aligned 3/3" in summary.team.notes
    assert (
        "research_posture: policy=diversified target=rl discovery_target=ml "
        "effective_target=all ml=2 rl=1 refreshed=2"
    ) in summary.team.notes
    assert (
        "research_posture_mix: discovery=ml setup=rl candidate=ml nightly=rl effective=all"
        in summary.team.notes
    )
    assert (
        "research_cohorts: targets=both=1, ml=2 regimes=ranging=1, trending=2"
        in summary.team.notes
    )
    assert (
        "research_scouts: supported=2, target=ml, overlap=1, volume_dense=2, "
        "symbols=SBIN.NS,RELIANCE.NS"
        in summary.team.notes
    )
    assert (
        "research_target_mismatch: requested=rl discovery=ml effective=all"
    ) in summary.team.notes


def test_annotate_discovery_refresh_records_audit_metrics():
    response = MultiAgentWorkflowResponse(
        ok=True,
        source="auto",
        timeframe="5m",
        lookback_days=20,
        headline="workflow ready",
        roles=(),
        universe=MarketUniverseResponse(
            ok=True,
            source="auto",
            timeframe="1d",
            lookback_days=20,
            candidates=(
                MarketUniverseCandidate(
                    symbol="RELIANCE.NS",
                    display_name="Reliance",
                    source="auto",
                    liquidity_score=2.1,
                ),
                MarketUniverseCandidate(
                    symbol="SBIN.NS",
                    display_name="SBI",
                    source="auto",
                    liquidity_score=1.9,
                ),
            ),
        ),
        shortlist=ShortlistAnalysisResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            items=(),
        ),
        briefing=ShortlistBriefingResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            headline="0 candidate setups ready for operator briefing",
            items=(),
        ),
        candidates=TrainingCandidateResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            candidates=(),
        ),
        research=TrainingResearchPlanResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            selection_policy="diversified",
            refresh_target="all",
            rows=(),
        ),
        allocation=PortfolioAllocationResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            headline="0 selected, 0 skipped under portfolio limits",
            max_positions=3,
            items=(),
        ),
    )

    summary = annotate_discovery_refresh(
        summarize_multi_agent_workflow(response),
        source="screener",
        timeframe="1d",
        days=30,
    )

    assert summary.team.metrics["discovery_refresh_source"] == "screener"
    assert summary.team.metrics["discovery_refresh_timeframe"] == "1d"
    assert summary.team.metrics["discovery_refresh_days"] == 30
    assert summary.team.metrics["discovery_refresh_requested"] is True
    assert summary.team.metrics["discovery_refreshed_count"] == 2
    assert (
        "discovery_refresh: source=screener timeframe=1d days=30 refreshed=2"
        in summary.team.notes
    )


def test_annotate_artifact_recovery_posture_records_follow_up_metrics():
    summary = annotate_artifact_recovery_posture(
        WorkflowSnapshot(
            source="auto",
            timeframe="5m",
            lookback_days=20,
            team=WorkflowSummary(metrics={"count": 1, "ok_count": 1}, rows=()),
            universe=WorkflowSummary(metrics={"count": 2, "source": "auto"}, rows=()),
            shortlist=WorkflowSummary(metrics={"count": 1, "source": "auto"}, rows=()),
            briefing=WorkflowSummary(metrics={"candidates": 1, "source": "auto"}, rows=()),
            candidates=WorkflowSummary(metrics={"ml_count": 1, "rl_count": 1}, rows=()),
            research=WorkflowSummary(metrics={"count": 1, "source": "auto"}, rows=()),
            allocation=WorkflowSummary(metrics={"selected_count": 1, "source": "auto"}, rows=()),
        ),
        action_label="Refresh training research",
        target="rl",
        posture_text="Review drift currently favors research recovery target=rl.",
    )

    assert summary.team.metrics["artifact_follow_up_action"] == "Refresh training research"
    assert summary.team.metrics["artifact_follow_up_target"] == "rl"
    assert (
        summary.team.metrics["artifact_recovery_posture"]
        == "Review drift currently favors research recovery target=rl."
    )
    assert (
        "artifact_recovery: action=Refresh training research "
        "target=rl posture=Review drift currently favors research recovery target=rl."
    ) in summary.team.notes


def test_replace_workflow_snapshot_sections_updates_visible_stage_summary():
    response = MultiAgentWorkflowResponse(
        ok=True,
        source="auto",
        timeframe="5m",
        lookback_days=20,
        headline="workflow ready",
        roles=(),
        universe=MarketUniverseResponse(
            ok=True,
            source="auto",
            timeframe="1d",
            lookback_days=20,
            candidates=(),
        ),
        shortlist=ShortlistAnalysisResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            items=(),
        ),
        briefing=ShortlistBriefingResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            headline="0 candidate setups ready for operator briefing",
            items=(),
        ),
        candidates=TrainingCandidateResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            candidates=(),
        ),
        research=TrainingResearchPlanResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            selection_policy="ranked",
            refresh_target="rl",
            rows=(),
        ),
        allocation=PortfolioAllocationResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            headline="0 selected, 0 skipped under portfolio limits",
            max_positions=3,
            items=(),
        ),
    )

    snapshot = summarize_multi_agent_workflow(response)
    refreshed_research = WorkflowSummary(
        metrics={
            "count": 2,
            "source": "auto",
            "ml_count": 1,
            "rl_count": 1,
            "selection_policy": "diversified",
            "refresh_target": "all",
            "refresh_requested": True,
            "refreshed_count": 2,
        },
        rows=(
            {"symbol": "RELIANCE.NS", "target": "ml"},
            {"symbol": "SBIN.NS", "target": "rl"},
        ),
    )

    updated = replace_workflow_snapshot_sections(snapshot, research=refreshed_research)

    assert updated.research.metrics["count"] == 2
    assert updated.research.metrics["refresh_requested"] is True
    assert updated.research.metrics["refreshed_count"] == 2
    assert updated.universe.metrics["count"] == snapshot.universe.metrics["count"]


def test_summarize_portfolio_allocation_counts_selected_and_skipped():
    response = PortfolioAllocationResponse(
        ok=True,
        source="auto",
        timeframe="5m",
        lookback_days=20,
        headline="1 selected, 1 skipped under portfolio limits",
        max_positions=1,
        items=(
            AllocationDecision(
                symbol="SBIN.NS",
                action="BUY",
                verdict="candidate",
                selected=True,
                reason="selected under current portfolio constraints",
                selection_rank=1,
                allocation_rank=1,
                allocation_weight=0.65,
                priority_score=1.3,
                allocation_score=1.36,
                exposure_key="SBIN",
                winning_strategy="ORB",
                regime="TRENDING",
                risk_bucket="low",
                sizing_hint="overweight",
                research_target="both",
                research_priority_score=1.28,
                research_refreshed=True,
                research_nightly_reports=2,
                research_nightly_refreshed_reports=1,
                research_nightly_promoted_reports=1,
                research_nightly_score=0.74,
                critic_penalty=0.0,
                critic_note="",
            ),
            AllocationDecision(
                symbol="SBIN.FUT",
                action="BUY",
                verdict="candidate",
                selected=False,
                reason="exposure limit reached for SBIN",
                selection_rank=2,
                priority_score=1.1,
                exposure_penalty=0.25,
                exposure_key="SBIN",
                critic_penalty=0.08,
                critic_note=(
                    "portfolio critic: same-side crowding (1), "
                    "same-regime concentration (TRENDING)"
                ),
            ),
        ),
        notes=(
            "max_positions=1",
            "allocation_freshness_mix: refreshed=1/1",
            "allocation_target_balance_summary: concentrated=both",
        ),
    )
    summary = summarize_portfolio_allocation(response)
    assert summary.metrics["selected_count"] == 1
    assert summary.metrics["skipped_count"] == 1
    assert summary.metrics["critic_enabled"] is False
    assert summary.metrics["freshness_mix"] == "refreshed=1/1"
    assert summary.metrics["target_balance_summary"] == "concentrated=both"
    assert summary.metrics["regime_mix"] == "TRENDING=1"
    assert summary.metrics["strategy_mix"] == "ORB=1"
    assert summary.metrics["risk_mix"] == "low=1"
    assert summary.rows[0]["allocation_weight"] == 0.65
    assert summary.rows[0]["allocation_score"] == 1.36
    assert summary.rows[0]["winning_strategy"] == "ORB"
    assert summary.rows[0]["regime"] == "TRENDING"
    assert summary.rows[0]["research_target"] == "both"
    assert summary.rows[0]["research_priority_score"] == 1.28
    assert summary.rows[0]["research_refreshed"] is True
    assert summary.rows[0]["research_nightly_reports"] == 2
    assert summary.rows[0]["research_nightly_refreshed_reports"] == 1
    assert summary.rows[0]["research_nightly_promoted_reports"] == 1
    assert summary.rows[0]["research_nightly_score"] == 0.74
    assert summary.rows[0]["critic_penalty"] == 0.0
    assert summary.rows[0]["critic_note"] is None
    assert summary.rows[1]["selected"] is False
    assert summary.rows[1]["critic_penalty"] == 0.08
    assert "same-regime concentration" in summary.rows[1]["critic_note"]
    assert "max_positions=1" in summary.notes


def test_summarize_portfolio_allocation_normalizes_mix_casing():
    response = PortfolioAllocationResponse(
        ok=True,
        source="unit",
        timeframe="5m",
        lookback_days=20,
        headline="2 reviewed",
        max_positions=2,
        items=(
            AllocationDecision(
                symbol="RELIANCE.NS",
                action="BUY",
                verdict="candidate",
                selected=True,
                reason="selected",
                selection_rank=1,
                allocation_rank=1,
                winning_strategy="orb",
                regime="trending",
                risk_bucket="LOW",
            ),
            AllocationDecision(
                symbol="SBIN.NS",
                action="BUY",
                verdict="candidate",
                selected=True,
                reason="selected",
                selection_rank=2,
                allocation_rank=2,
                winning_strategy="ORB",
                regime="TRENDING",
                risk_bucket="low",
            ),
        ),
        selected_symbols=("RELIANCE.NS", "SBIN.NS"),
        skipped_symbols=(),
        notes=("portfolio_critic=enabled",),
    )

    summary = summarize_portfolio_allocation(response)

    assert summary.metrics["regime_mix"] == "TRENDING=2"
    assert summary.metrics["strategy_mix"] == "ORB=2"
    assert summary.metrics["risk_mix"] == "low=2"


def test_summarize_multi_agent_workflow_preserves_stage_summaries():
    response = MultiAgentWorkflowResponse(
        ok=True,
        source="auto",
        timeframe="5m",
        lookback_days=20,
        headline="multi-agent ready",
        roles=(
            AgentRoleStatus(
                name="universe_scout",
                ok=True,
                headline="2 liquid/trend candidates from auto",
            ),
        ),
        universe=MarketUniverseResponse(
            ok=True,
            source="auto",
            timeframe="1d",
            lookback_days=20,
            candidates=(
                MarketUniverseCandidate(
                    symbol="RELIANCE.NS",
                    display_name="RELIANCE",
                    source="auto",
                    liquidity_score=12.5,
                ),
            ),
        ),
        shortlist=ShortlistAnalysisResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            items=(
                ShortlistAnalysisItem(
                    symbol="RELIANCE.NS",
                    selection_rank=1,
                    decision=DecisionSummary(action="BUY", confidence=0.82, summary="BUY won"),
                    critique=RiskCritique(severity="low", summary="clean", verdict="candidate"),
                ),
            ),
        ),
        briefing=ShortlistBriefingResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            headline="1 candidate setups, 1 reviewed",
            items=(
                BriefingItem(
                    symbol="RELIANCE.NS",
                    action="BUY",
                    confidence=0.82,
                    verdict="candidate",
                    summary="Trend aligned",
                ),
            ),
        ),
        candidates=TrainingCandidateResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
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
        research=TrainingResearchPlanResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            rows=(
                TrainingResearchRow(
                    symbol="RELIANCE.NS",
                    target="both",
                    selection_rank=1,
                    refreshed=True,
                ),
            ),
            ml_symbols=("RELIANCE.NS",),
            rl_symbols=("RELIANCE.NS",),
        ),
        allocation=PortfolioAllocationResponse(
            ok=True,
            source="auto",
            timeframe="5m",
            lookback_days=20,
            headline="1 selected, 0 skipped under portfolio limits",
            max_positions=1,
            items=(
                AllocationDecision(
                    symbol="RELIANCE.NS",
                    action="BUY",
                    verdict="candidate",
                    selected=True,
                    reason="selected",
                    selection_rank=1,
                    allocation_rank=1,
                    allocation_weight=1.0,
                    exposure_key="RELIANCE",
                ),
            ),
        ),
        nightly_alignment=NightlyAlignmentStatus(
            report_count=2,
            enabled_reports=2,
            aligned_reports=1,
            latest_execution_target="ml",
            latest_execution_model_family="ml",
            latest_execution_selection_source="training_research",
            latest_target_mix="both=1, rl=1",
            latest_refreshed_target_mix="both=1",
            nightly_posture="rl",
            latest_promotion_review_model_kind="mixed",
            latest_promotion_review_model_kinds=("ml_scorer", "rl_policy"),
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
                    basket_size=1,
                    alignment_enabled=True,
                    execution_target="ml",
                    execution_model_family="ml",
                    execution_selection_source="training_research",
                    target_mix="both=1, rl=1",
                    refreshed_target_mix="both=1",
                    promotion_review_model_kind="mixed",
                    promotion_review_model_kinds=("ml_scorer", "rl_policy"),
                ),
            ),
        ),
    )

    snapshot = summarize_multi_agent_workflow(response)
    assert snapshot.source == "auto"
    assert snapshot.team.metrics["count"] == 1
    assert snapshot.team.metrics["nightly_report_count"] == 2
    assert snapshot.team.metrics["nightly_recommended_target"] == "all"
    assert snapshot.team.metrics["nightly_posture"] == "rl"
    assert snapshot.team.metrics["nightly_recommended_force_refresh"] is True
    assert snapshot.team.metrics["nightly_latest_execution_target"] == "ml"
    assert snapshot.team.metrics["nightly_latest_execution_model_family"] == "ml"
    assert snapshot.team.metrics["nightly_latest_execution_selection_source"] == "training_research"
    assert snapshot.team.metrics["nightly_latest_promotion_review_model_kind"] == "mixed"
    assert (
        snapshot.team.metrics["nightly_latest_promotion_review_model_kinds"]
        == "ml_scorer, rl_policy"
    )
    assert snapshot.team.metrics["nightly_recent_window"] == 1
    assert snapshot.team.metrics["nightly_recent_enabled"] == 1
    assert snapshot.team.metrics["nightly_recent_aligned"] == 1
    assert snapshot.team.metrics["nightly_recent_latest_status"] == "ok"
    assert snapshot.team.metrics["nightly_recent_latest_basket_size"] == 1
    assert (
        snapshot.team.metrics["nightly_recommended_discovery_action"]
        == "Rebuild the market universe from Screener liquidity inputs."
    )
    assert "nightly_alignment_force_refresh=1" in snapshot.team.notes
    assert "nightly_alignment_posture: rl" in snapshot.team.notes
    assert (
        "nightly_alignment_execution: target=ml family=ml source=training_research"
        in snapshot.team.notes
    )
    assert "nightly_alignment_review_kind=mixed" in snapshot.team.notes
    assert "nightly_alignment_recent: 1/1" in snapshot.team.notes
    assert "nightly_alignment_discovery_follow_up=1" in snapshot.team.notes
    assert snapshot.universe.metrics["count"] == 1
    assert snapshot.shortlist.metrics["count"] == 1
    assert snapshot.briefing.metrics["candidates"] == 1
    assert snapshot.candidates.metrics["ml_count"] == 1
    assert snapshot.research.metrics["count"] == 1
    assert snapshot.allocation.metrics["selected_count"] == 1


def test_build_and_export_workflow_snapshot(tmp_path, monkeypatch):
    universe = MarketUniverseResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="RELIANCE",
                source="registry",
                liquidity_score=12.5,
                trend_pct=2.3,
            ),
        ),
    )
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                selection_rank=1,
                priority_score=1.2,
                decision=DecisionSummary(action="BUY", confidence=0.82, summary="BUY won"),
                critique=RiskCritique(severity="low", summary="clean", verdict="candidate"),
            ),
        ),
    )
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="1 candidate setups, 1 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.82,
                verdict="candidate",
                summary="Trend aligned",
                selection_rank=1,
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                selection_rank=1,
                decision_action="BUY",
                decision_confidence=0.82,
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
            ),
        ),
    )
    research = TrainingResearchPlanResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        selection_policy="diversified",
        refresh_requested=False,
        refresh_target="rl",
        ml_symbols=("RELIANCE.NS",),
        rl_symbols=("RELIANCE.NS",),
        rows=(
            TrainingResearchRow(
                symbol="RELIANCE.NS",
                target="both",
                universe_rank=1,
                shortlist_rank=1,
                selection_rank=1,
                decision_action="BUY",
                critique_verdict="candidate",
            ),
        ),
    )
    allocation = PortfolioAllocationResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="1 selected, 0 skipped under portfolio limits",
        max_positions=1,
        items=(
            AllocationDecision(
                symbol="RELIANCE.NS",
                action="BUY",
                verdict="candidate",
                selected=True,
                reason="selected under current portfolio constraints",
                selection_rank=1,
                allocation_rank=1,
                allocation_weight=1.0,
                exposure_key="RELIANCE",
            ),
        ),
        notes=("max_positions=1",),
    )

    monkeypatch.setattr(
        "fortuna.app.multi_agent_team.build_multi_agent_workflow",
        lambda **kwargs: MultiAgentWorkflowResponse(
            ok=True,
            source="registry",
            timeframe="5m",
            lookback_days=20,
            headline="multi-agent ready",
            roles=(
                AgentRoleStatus(
                    name="universe_scout",
                    ok=True,
                    headline="1 liquid/trend candidates from registry",
                    focus_symbols=("RELIANCE.NS",),
                ),
                AgentRoleStatus(
                    name="operations_monitor",
                    ok=True,
                    headline="2 nightly reports tracked; alignment ratio=0.50",
                    notes=("refresh_target=all", "force_refresh=1", "discovery_follow_up=1"),
                ),
            ),
            universe=universe,
            shortlist=shortlist,
            briefing=briefing,
            candidates=candidates,
            research=research,
            allocation=allocation,
            nightly_alignment=NightlyAlignmentStatus(
                report_count=2,
                enabled_reports=2,
                aligned_reports=1,
                latest_target_mix="both=1, rl=1",
                latest_refreshed_target_mix="both=1",
                recommended_refresh_target="all",
                recommended_force_refresh=True,
                recommended_discovery_action=(
                    "Rebuild the market universe from Screener liquidity inputs."
                ),
                recommended_discovery_cli_command=(
                    "uv run python scripts/build_market_universe.py --source screener"
                ),
            ),
        ),
    )

    snapshot = build_workflow_snapshot(
        settings=_settings(tmp_path),
        timeframe="5m",
        days=20,
        source="registry",
    )
    assert snapshot.universe.metrics["count"] == 1
    assert snapshot.team.metrics["count"] == 2
    assert snapshot.team.metrics["nightly_report_count"] == 2
    assert snapshot.shortlist.metrics["count"] == 1
    assert snapshot.briefing.metrics["candidates"] == 1
    assert snapshot.candidates.metrics["ml_count"] == 1
    assert snapshot.research.metrics["rl_count"] == 1
    assert snapshot.allocation.metrics["selected_count"] == 1

    out = export_workflow_snapshot(snapshot, tmp_path / "reports" / "workflow_snapshot.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"] == "registry"
    assert payload["team"]["metrics"]["count"] == 2
    assert payload["team"]["metrics"]["nightly_report_count"] == 2
    assert payload["team"]["metrics"]["nightly_recommended_target"] == "all"
    assert payload["team"]["metrics"]["nightly_recommended_force_refresh"] is True
    assert payload["briefing"]["metrics"]["candidates"] == 1
    assert payload["research"]["metrics"]["selection_policy"] == "diversified"
    assert payload["allocation"]["metrics"]["selected_count"] == 1


def test_export_workflow_snapshot_preserves_discovery_refresh_audit(tmp_path):
    snapshot = annotate_discovery_refresh(
        WorkflowSnapshot(
            source="auto",
            timeframe="5m",
            lookback_days=20,
            team=WorkflowSummary(metrics={"count": 1, "ok_count": 1}, rows=()),
            universe=WorkflowSummary(metrics={"count": 3, "source": "auto"}, rows=()),
            shortlist=WorkflowSummary(metrics={"count": 1, "source": "auto"}, rows=()),
            briefing=WorkflowSummary(metrics={"candidates": 1, "source": "auto"}, rows=()),
            candidates=WorkflowSummary(metrics={"ml_count": 1, "rl_count": 1}, rows=()),
            research=WorkflowSummary(metrics={"count": 1, "source": "auto"}, rows=()),
            allocation=WorkflowSummary(metrics={"selected_count": 1, "source": "auto"}, rows=()),
        ),
        source="screener",
        timeframe="1d",
        days=30,
    )

    out = export_workflow_snapshot(
        snapshot,
        tmp_path / "reports" / "workflow_snapshot_discovery.json",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["team"]["metrics"]["discovery_refresh_source"] == "screener"
    assert payload["team"]["metrics"]["discovery_refresh_timeframe"] == "1d"
    assert payload["team"]["metrics"]["discovery_refresh_days"] == 30
    assert payload["team"]["metrics"]["discovery_refresh_requested"] is True
    assert payload["team"]["metrics"]["discovery_refreshed_count"] == 3


def test_export_workflow_snapshot_preserves_artifact_recovery_posture(tmp_path):
    snapshot = annotate_artifact_recovery_posture(
        WorkflowSnapshot(
            source="auto",
            timeframe="5m",
            lookback_days=20,
            team=WorkflowSummary(metrics={"count": 1, "ok_count": 1}, rows=()),
            universe=WorkflowSummary(metrics={"count": 3, "source": "auto"}, rows=()),
            shortlist=WorkflowSummary(metrics={"count": 1, "source": "auto"}, rows=()),
            briefing=WorkflowSummary(metrics={"candidates": 1, "source": "auto"}, rows=()),
            candidates=WorkflowSummary(metrics={"ml_count": 1, "rl_count": 1}, rows=()),
            research=WorkflowSummary(metrics={"count": 1, "source": "auto"}, rows=()),
            allocation=WorkflowSummary(metrics={"selected_count": 1, "source": "auto"}, rows=()),
        ),
        action_label="Refresh market universe",
        target=None,
        posture_text=(
            "Review drift currently favors discovery recovery before another research pass."
        ),
    )

    out = export_workflow_snapshot(
        snapshot,
        tmp_path / "reports" / "workflow_snapshot_artifact_recovery.json",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["team"]["metrics"]["artifact_follow_up_action"] == "Refresh market universe"
    assert payload["team"]["metrics"]["artifact_follow_up_target"] is None
    assert (
        payload["team"]["metrics"]["artifact_recovery_posture"]
        == "Review drift currently favors discovery recovery before another research pass."
    )
