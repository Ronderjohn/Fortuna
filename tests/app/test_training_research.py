from __future__ import annotations

import json
from pathlib import Path

from fortuna.agentic.contracts import (
    MarketUniverseCandidate,
    MarketUniverseResponse,
    NightlyAlignmentStatus,
    TrainingCandidate,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
)
from fortuna.app.training_research import (
    _research_rationale,
    _strategy_posture_votes,
    build_training_research_plan,
    export_training_research_plan,
)
from fortuna.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )


def test_build_training_research_plan_combines_ml_rl_targets(tmp_path: Path, monkeypatch):
    universe = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        scout_volume_dense_symbols=("SBIN.NS", "TCS.NS"),
        scout_overlap_symbols=("TCS.NS", "SBIN.NS"),
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                liquidity_score=10.4,
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="screener",
                liquidity_score=12.1,
                activity_score=1.2,
                trend_pct=1.4,
                volume_ratio=1.1,
                regime="TRENDING",
            ),
            MarketUniverseCandidate(
                symbol="SBIN.NS",
                display_name="SBIN",
                source="screener",
                liquidity_score=11.8,
                activity_score=1.5,
                trend_pct=2.3,
                volume_ratio=1.3,
                regime="TRENDING",
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                selection_rank=1,
                winning_strategy="ORB",
                decision_action="BUY",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.1,
                remediation_target="rl",
                remediation_pressure=0.04,
                rationale=("leader",),
            ),
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=2,
                selection_rank=2,
                winning_strategy="MMTS",
                decision_action="BUY",
                critique_verdict="watch",
                ml_candidate=True,
                rl_candidate=False,
                market_regime="RANGING",
                priority_score=0.8,
                rationale=("ml only",),
            ),
            TrainingCandidate(
                symbol="SBIN.NS",
                shortlist_rank=3,
                selection_rank=3,
                winning_strategy="VWAP",
                decision_action="SELL",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.0,
                rationale=("rl hedge",),
            ),
        ),
    )

    monkeypatch.setattr("fortuna.app.training_research.build_market_universe", lambda **_: universe)
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_candidates",
        lambda **_: candidates,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(
            report_count=2,
            enabled_reports=2,
            aligned_reports=1,
            recommended_refresh_target="rl",
            recommended_force_refresh=True,
            recent_trend_window=2,
            recent_trend_enabled=2,
            recent_trend_aligned=1,
            recent_trend_latest_status="ok",
            recent_trend_latest_basket_size=3,
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.backfill_training_candidates",
        lambda *args, **kwargs: ["RELIANCE.NS", "SBIN.NS"],
    )

    response = build_training_research_plan(
        settings=_settings(tmp_path),
        source="screener",
        ml_top_n=2,
        rl_top_n=2,
        selection_policy="diversified",
        refresh_data=True,
        refresh_target="rl",
    )

    assert response.ok is True
    assert response.source == "screener"
    assert response.selection_policy == "diversified"
    assert response.universe_symbols == ("RELIANCE.NS", "TCS.NS", "SBIN.NS")
    assert response.discovery_preferred_symbols == ("TCS.NS", "SBIN.NS")
    assert response.discovery_liquidity_symbols == ("TCS.NS", "SBIN.NS", "RELIANCE.NS")
    assert response.discovery_activity_symbols == ("SBIN.NS", "TCS.NS")
    assert response.discovery_volume_dense_symbols == ("SBIN.NS", "TCS.NS")
    assert response.discovery_regime_mix == "trending=2"
    assert response.discovery_posture == "rl"
    assert response.strategy_posture == "rl"
    assert response.effective_posture == "rl"
    assert response.discovery_summary == (
        "preferred=2 | liquidity=TCS.NS,SBIN.NS | "
        "activity=SBIN.NS,TCS.NS | regimes=trending=2 | setup_posture=rl | posture=rl"
    )
    assert response.discovery_recommended_refresh_target == "rl"
    assert response.effective_refresh_target == "rl"
    assert response.refresh_urgency == "medium"
    assert response.refresh_urgency_target == "rl"
    assert response.refresh_urgency_score == 0.04
    assert response.refresh_urgency_rows == 1
    assert response.refresh_urgency_summary == (
        "medium RL remediation pressure on 1 row(s); max=+0.04 total=+0.04"
    )
    assert response.follow_up_action == (
        "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s)."
    )
    assert response.discovery_follow_up_target == "rl"
    assert response.discovery_follow_up_summary == (
        "Discovery scouts currently lean RL-focused from rl universe posture."
    )
    assert response.discovery_follow_up_action == (
        "Refresh market universe and shortlist review with RL-focused discovery focus "
        "before the next training cycle."
    )
    assert response.research_follow_up_target == "rl"
    assert response.research_follow_up_summary == (
        "medium RL remediation pressure on 1 row(s); max=+0.04 total=+0.04"
    )
    assert response.research_follow_up_action == (
        "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s)."
    )
    assert response.execution_follow_up_target == "rl"
    assert response.execution_follow_up_summary == (
        "Nightly execution drift currently leans RL-focused from mixed execution posture."
    )
    assert response.execution_follow_up_action == (
        "Review nightly execution path and retarget execution candidate selection "
        "toward RL-focused before the next promotion or nightly cycle."
    )
    assert response.ml_symbols == ("TCS.NS", "SBIN.NS")
    assert response.rl_symbols == ("SBIN.NS", "RELIANCE.NS")
    assert [row.symbol for row in response.rows] == ["RELIANCE.NS", "SBIN.NS"]
    assert response.selected_target_mix == "both=1, rl=1"
    assert response.selected_regime_mix == "trending=2"
    assert response.rows[0].target == "rl"
    assert response.rows[1].target == "both"
    assert response.rows[0].remediation_target == "rl"
    assert response.rows[0].remediation_pressure == 0.04
    assert response.rows[0].universe_rank == 1
    assert response.rows[0].refreshed is True
    assert (
        "Discovery overlap favored by both liquidity and activity scouts"
        not in response.rows[0].rationale
    )
    assert (
        "Discovery overlap favored by both liquidity and activity scouts"
        in response.rows[1].rationale
    )
    assert (
        "Discovery volume-dense cohort reinforced this symbol for retraining priority"
        in response.rows[1].rationale
    )
    assert any(
        "Setup-family posture reinforced this symbol" in text
        for text in response.rows[0].rationale
    )
    assert response.nightly_alignment_target == "rl"
    assert response.nightly_alignment_force_refresh is True
    assert response.nightly_recent_window == 2
    assert response.nightly_recent_enabled == 2
    assert response.nightly_recent_aligned == 1
    assert response.nightly_recent_latest_status == "ok"
    assert response.nightly_recent_latest_basket_size == 3
    assert (
        response.nightly_alignment_summary
        == "recent=1/2 over 2 run(s); target=rl force_refresh=1"
    )


def test_export_training_research_plan_writes_json(tmp_path: Path):
    payload = export_training_research_plan(
        TrainingResearchPlanResponse(
            ok=True,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            nightly_alignment_summary="recent=1/2 over 2 run(s); target=rl force_refresh=1",
            nightly_alignment_target="rl",
            nightly_alignment_force_refresh=True,
            nightly_recent_window=2,
            nightly_recent_enabled=2,
            nightly_recent_aligned=1,
            nightly_recent_latest_status="ok",
            nightly_recent_latest_basket_size=3,
        ),
        tmp_path / "reports" / "training_research_plan.json",
    )
    parsed = json.loads(payload.read_text(encoding="utf-8"))
    assert parsed["ok"] is True
    assert parsed["source"] == "registry"
    assert (
        parsed["nightly_alignment_summary"]
        == "recent=1/2 over 2 run(s); target=rl force_refresh=1"
    )
    assert parsed["nightly_alignment_target"] == "rl"
    assert parsed["nightly_alignment_force_refresh"] is True
    assert parsed["nightly_recent_window"] == 2
    assert parsed["discovery_preferred_symbols"] == []
    assert parsed["discovery_recommended_refresh_target"] is None
    assert parsed["effective_refresh_target"] is None
    assert parsed["selected_target_mix"] is None


def test_build_training_research_plan_prefers_ml_when_discovery_is_ranging(
    tmp_path: Path,
    monkeypatch,
):
    universe = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                liquidity_score=12.4,
                activity_score=1.0,
                trend_pct=0.3,
                volume_ratio=1.0,
                regime="RANGING",
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="screener",
                liquidity_score=12.1,
                activity_score=0.9,
                trend_pct=0.2,
                volume_ratio=1.0,
                regime="RANGING",
            ),
            MarketUniverseCandidate(
                symbol="SBIN.NS",
                display_name="SBIN",
                source="screener",
                liquidity_score=8.0,
                activity_score=0.2,
                trend_pct=0.1,
                volume_ratio=0.9,
                regime="TRENDING",
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(),
    )

    monkeypatch.setattr("fortuna.app.training_research.build_market_universe", lambda **_: universe)
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_candidates",
        lambda **_: candidates,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(),
    )

    response = build_training_research_plan(
        settings=_settings(tmp_path),
        source="screener",
        ml_top_n=1,
        rl_top_n=1,
        selection_policy="diversified",
    )

    assert response.ok is True
    assert response.discovery_preferred_symbols == ("RELIANCE.NS", "TCS.NS", "SBIN.NS")
    assert response.discovery_regime_mix == "ranging=2, trending=1"
    assert response.discovery_posture == "ml"
    assert response.strategy_posture is None
    assert response.effective_posture == "ml"
    assert response.discovery_recommended_refresh_target == "ml"
    assert response.effective_refresh_target == "ml"
    assert response.discovery_summary == (
        "preferred=3 | liquidity=RELIANCE.NS,TCS.NS | "
        "activity=RELIANCE.NS,TCS.NS | regimes=ranging=2, trending=1 | posture=ml"
    )


def test_build_training_research_plan_prefers_rl_from_volume_dense_activity_cohort(
    tmp_path: Path,
    monkeypatch,
):
    universe = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="screener",
                liquidity_score=12.8,
                activity_score=1.1,
                trend_pct=1.4,
                volume_ratio=1.16,
                regime="TRENDING",
            ),
            MarketUniverseCandidate(
                symbol="SBIN.NS",
                display_name="SBIN",
                source="screener",
                liquidity_score=12.5,
                activity_score=1.7,
                trend_pct=1.8,
                volume_ratio=1.35,
                regime="VOLATILE",
            ),
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                liquidity_score=12.1,
                activity_score=0.8,
                trend_pct=0.2,
                volume_ratio=1.01,
                regime="RANGING",
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(),
    )

    monkeypatch.setattr("fortuna.app.training_research.build_market_universe", lambda **_: universe)
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_candidates",
        lambda **_: candidates,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(),
    )

    response = build_training_research_plan(
        settings=_settings(tmp_path),
        source="screener",
        ml_top_n=1,
        rl_top_n=1,
        selection_policy="diversified",
    )

    assert response.ok is True
    assert response.discovery_preferred_symbols == ("TCS.NS", "SBIN.NS", "RELIANCE.NS")
    assert response.discovery_regime_mix == "ranging=1, trending=1, volatile=1"
    assert response.discovery_posture == "rl"
    assert response.strategy_posture is None
    assert response.effective_posture == "rl"
    assert response.discovery_recommended_refresh_target == "rl"
    assert response.effective_refresh_target == "rl"
    assert response.discovery_summary == (
        "preferred=3 | liquidity=TCS.NS,SBIN.NS | "
        "activity=SBIN.NS,TCS.NS | regimes=ranging=1, trending=1, volatile=1 | posture=rl"
    )


def test_build_training_research_plan_uses_strategy_posture_when_discovery_is_inconclusive(
    tmp_path: Path,
    monkeypatch,
):
    universe = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                liquidity_score=12.4,
                activity_score=1.0,
                trend_pct=0.9,
                volume_ratio=1.04,
                regime="TRENDING",
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="screener",
                liquidity_score=12.1,
                activity_score=1.0,
                trend_pct=0.8,
                volume_ratio=1.03,
                regime="RANGING",
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                selection_rank=1,
                winning_strategy="ORB",
                decision_action="BUY",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.05,
                rationale=("lead rl setup",),
            ),
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=2,
                selection_rank=2,
                winning_strategy="VWAP",
                decision_action="SELL",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="RANGING",
                priority_score=0.96,
                rationale=("support rl setup",),
            ),
        ),
    )

    monkeypatch.setattr("fortuna.app.training_research.build_market_universe", lambda **_: universe)
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_candidates",
        lambda **_: candidates,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(),
    )

    response = build_training_research_plan(
        settings=_settings(tmp_path),
        source="screener",
        ml_top_n=1,
        rl_top_n=1,
        selection_policy="diversified",
    )

    assert response.ok is True
    assert response.discovery_posture == "all"
    assert response.strategy_posture == "rl"
    assert response.effective_posture == "rl"
    assert response.discovery_recommended_refresh_target == "rl"
    assert response.effective_refresh_target == "rl"
    assert response.discovery_summary == (
        "preferred=2 | liquidity=RELIANCE.NS,TCS.NS | "
        "activity=RELIANCE.NS,TCS.NS | regimes=ranging=1, trending=1 | "
        "setup_posture=rl | posture=rl"
    )
    assert any(
        "Setup-family posture reinforced this symbol" in text
        for text in response.rows[0].rationale
    )


def test_build_training_research_plan_uses_candidate_posture_when_other_signals_are_mixed(
    tmp_path: Path,
    monkeypatch,
):
    universe = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                avg_turnover=100.0,
                avg_volume=10.0,
                liquidity_score=9.5,
                regime="TRENDING",
                volume_ratio=1.05,
                activity_score=0.6,
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="screener",
                avg_turnover=98.0,
                avg_volume=9.8,
                liquidity_score=9.2,
                regime="RANGING",
                volume_ratio=1.04,
                activity_score=0.58,
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                selection_rank=1,
                winning_strategy="UNKNOWN_A",
                decision_action="BUY",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.08,
                adaptive_score_adjustment=0.02,
                rationale=("balanced candidate",),
            ),
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=2,
                selection_rank=2,
                winning_strategy="UNKNOWN_B",
                decision_action="SELL",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=False,
                market_regime="RANGING",
                priority_score=1.12,
                adaptive_score_adjustment=0.06,
                rationale=("mean-reversion candidate",),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_market_universe",
        lambda **kwargs: universe,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_candidates",
        lambda **kwargs: candidates,
    )

    response = build_training_research_plan(
        settings=_settings(tmp_path),
        source="screener",
        ml_top_n=2,
        rl_top_n=1,
        selection_policy="diversified",
    )

    assert response.ok is True
    assert response.discovery_posture == "all"
    assert response.strategy_posture is None
    assert response.candidate_posture == "ml"
    assert response.effective_posture == "ml"
    assert response.discovery_recommended_refresh_target == "ml"
    assert response.discovery_summary == (
        "preferred=2 | liquidity=RELIANCE.NS,TCS.NS | "
        "activity=RELIANCE.NS,TCS.NS | regimes=ranging=1, trending=1 | "
        "candidate_posture=ml | posture=ml"
    )
    assert any(
        "Candidate-cohort posture reinforced this symbol" in text
        for text in response.rows[0].rationale + response.rows[1].rationale
    )


def test_build_training_research_plan_uses_nightly_posture_when_other_signals_stay_mixed(
    tmp_path: Path,
    monkeypatch,
):
    universe = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                avg_turnover=100.0,
                avg_volume=10.0,
                liquidity_score=9.5,
                regime="TRENDING",
                volume_ratio=1.05,
                activity_score=0.6,
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="screener",
                avg_turnover=98.0,
                avg_volume=9.8,
                liquidity_score=9.2,
                regime="RANGING",
                volume_ratio=1.04,
                activity_score=0.58,
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                selection_rank=1,
                winning_strategy="UNKNOWN_A",
                decision_action="BUY",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.05,
                rationale=("balanced candidate",),
            ),
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=2,
                selection_rank=2,
                winning_strategy="UNKNOWN_B",
                decision_action="SELL",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="RANGING",
                priority_score=1.02,
                rationale=("balanced hedge",),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_market_universe",
        lambda **kwargs: universe,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_candidates",
        lambda **kwargs: candidates,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(
            report_count=3,
            enabled_reports=3,
            aligned_reports=2,
            nightly_posture="rl",
            recommended_refresh_target="rl",
            recent_trend_window=3,
            recent_trend_enabled=3,
            recent_trend_aligned=2,
        ),
    )

    response = build_training_research_plan(
        settings=_settings(tmp_path),
        source="screener",
        ml_top_n=1,
        rl_top_n=1,
        selection_policy="diversified",
    )

    assert response.ok is True
    assert response.discovery_posture == "all"
    assert response.strategy_posture is None
    assert response.candidate_posture == "all"
    assert response.nightly_posture == "rl"
    assert response.effective_posture == "rl"
    assert response.discovery_recommended_refresh_target == "rl"
    assert response.discovery_summary == (
        "preferred=2 | liquidity=RELIANCE.NS,TCS.NS | "
        "activity=RELIANCE.NS,TCS.NS | regimes=ranging=1, trending=1 | "
        "nightly_posture=rl | posture=rl"
    )


def test_build_training_research_plan_uses_nightly_target_to_break_refresh_ties(
    tmp_path: Path,
    monkeypatch,
):
    universe = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                avg_turnover=100.0,
                avg_volume=10.0,
                liquidity_score=9.5,
                regime="TRENDING",
                volume_ratio=1.04,
                activity_score=0.6,
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="screener",
                avg_turnover=98.0,
                avg_volume=9.8,
                liquidity_score=9.2,
                regime="RANGING",
                volume_ratio=1.03,
                activity_score=0.58,
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                selection_rank=1,
                winning_strategy="UNKNOWN_A",
                decision_action="BUY",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.0,
                rationale=("balanced candidate",),
            ),
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=2,
                selection_rank=2,
                winning_strategy="UNKNOWN_B",
                decision_action="SELL",
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="RANGING",
                priority_score=0.98,
                rationale=("balanced hedge",),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_market_universe",
        lambda **kwargs: universe,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_candidates",
        lambda **kwargs: candidates,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(
            report_count=3,
            enabled_reports=3,
            aligned_reports=1,
            recommended_refresh_target="ml",
            recommended_force_refresh=False,
            recent_trend_window=3,
            recent_trend_enabled=3,
            recent_trend_aligned=1,
            recent_trend_latest_status="warn",
            recent_trend_latest_basket_size=2,
        ),
    )

    response = build_training_research_plan(
        settings=_settings(tmp_path),
        source="screener",
        ml_top_n=2,
        rl_top_n=1,
        selection_policy="diversified",
    )

    assert response.ok is True
    assert response.discovery_posture == "all"
    assert response.strategy_posture is None
    assert response.candidate_posture == "all"
    assert response.discovery_recommended_refresh_target == "all"
    assert response.nightly_alignment_target == "ml"
    assert response.effective_refresh_target == "ml"


def test_build_training_research_plan_uses_scout_support_target_to_break_discovery_nightly_tie(
    tmp_path: Path,
    monkeypatch,
):
    universe = MarketUniverseResponse(
        ok=True,
        source="screener",
        timeframe="1d",
        lookback_days=30,
        scout_overlap_symbols=("RELIANCE.NS",),
        scout_volume_dense_symbols=("RELIANCE.NS", "TCS.NS"),
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="Reliance",
                source="screener",
                liquidity_score=12.0,
                regime="RANGING",
                volume_ratio=1.0,
                activity_score=0.7,
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="screener",
                liquidity_score=11.4,
                regime="RANGING",
                volume_ratio=1.02,
                activity_score=0.65,
            ),
            MarketUniverseCandidate(
                symbol="SBIN.NS",
                display_name="SBIN",
                source="screener",
                liquidity_score=10.8,
                regime="TRENDING",
                volume_ratio=1.18,
                activity_score=1.1,
            ),
        ),
    )
    candidates = TrainingCandidateResponse(
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
                critique_verdict="watch",
                ml_candidate=True,
                rl_candidate=False,
                market_regime="RANGING",
                priority_score=1.08,
                rationale=("ml scout",),
            ),
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=2,
                selection_rank=2,
                decision_action="BUY",
                critique_verdict="watch",
                ml_candidate=True,
                rl_candidate=False,
                market_regime="RANGING",
                priority_score=1.02,
                rationale=("ml scout 2",),
            ),
            TrainingCandidate(
                symbol="SBIN.NS",
                shortlist_rank=3,
                selection_rank=3,
                decision_action="SELL",
                critique_verdict="candidate",
                ml_candidate=False,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=0.96,
                rationale=("rl hedge",),
            ),
        ),
    )

    monkeypatch.setattr("fortuna.app.training_research.build_market_universe", lambda **_: universe)
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_candidates",
        lambda **_: candidates,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(
            recommended_refresh_target="rl",
            nightly_posture="rl",
        ),
    )

    response = build_training_research_plan(
        settings=_settings(tmp_path),
        source="screener",
        ml_top_n=2,
        rl_top_n=1,
        selection_policy="diversified",
    )

    assert response.ok is True
    assert response.discovery_recommended_refresh_target == "ml"
    assert response.nightly_alignment_target == "rl"
    assert response.scout_support_target == "ml"
    assert response.effective_refresh_target == "ml"


def test_research_rationale_includes_durable_setup_family_evidence():
    rationale = _research_rationale(
        ("base rationale",),
        symbol="TCS.NS",
        preferred_symbols=(),
        winning_strategy="ORB",
        strategy_posture="rl",
        ml_candidate=True,
        rl_candidate=True,
        candidate_posture=None,
        volume_dense_symbols=(),
        setup_family_verdict="winner",
    )
    assert any(
        "Durable setup-family evidence reinforced ORB as recurring winner" in line
        for line in rationale
    )


def test_research_rationale_includes_volume_dense_cohort_support():
    rationale = _research_rationale(
        ("base rationale",),
        symbol="TCS.NS",
        preferred_symbols=(),
        winning_strategy="MMTS",
        strategy_posture=None,
        ml_candidate=True,
        rl_candidate=False,
        candidate_posture=None,
        volume_dense_symbols=("TCS.NS",),
        setup_family_verdict=None,
    )

    assert (
        "Discovery volume-dense cohort reinforced this symbol for retraining priority"
        in rationale
    )


def test_strategy_posture_votes_apply_durable_setup_family_tiebreak():
    rows = (
        TrainingCandidate(
            symbol="RELIANCE.NS",
            shortlist_rank=1,
            selection_rank=1,
            winning_strategy="ORB",
            decision_action="BUY",
            critique_verdict="candidate",
            ml_candidate=True,
            rl_candidate=True,
            priority_score=1.0,
            setup_family_reinforcement=0.05,
            setup_family_verdict="winner",
        ),
        TrainingCandidate(
            symbol="TCS.NS",
            shortlist_rank=2,
            selection_rank=2,
            winning_strategy="VWAP",
            decision_action="BUY",
            critique_verdict="candidate",
            ml_candidate=True,
            rl_candidate=True,
            priority_score=1.0,
            setup_family_reinforcement=0.05,
            setup_family_verdict="winner",
        ),
    )
    base_rl, _base_ml = _strategy_posture_votes(rows, use_family_tiebreak=False)
    tie_rl, _tie_ml = _strategy_posture_votes(rows, use_family_tiebreak=True)
    assert tie_rl > base_rl
