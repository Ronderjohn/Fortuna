from __future__ import annotations

import json
from pathlib import Path

from fortuna.app.workflow_snapshot import (
    format_workflow_snapshot_summary,
    load_workflow_snapshot_summary,
)


def test_load_and_format_workflow_snapshot_summary(tmp_path: Path):
    path = tmp_path / "workflow_snapshot.json"
    path.write_text(
        json.dumps(
            {
                "source": "auto",
                "timeframe": "15m",
                "lookback_days": 20,
                "team": {
                    "metrics": {
                        "count": 6,
                        "ok_count": 5,
                        "headline": "multi-agent ready",
                        "nightly_report_count": 2,
                        "nightly_enabled_reports": 2,
                        "nightly_aligned_reports": 1,
                        "nightly_latest_target_mix": "both=1, rl=1",
                        "nightly_latest_execution_target": "ml",
                        "nightly_latest_execution_model_family": "ml",
                        "nightly_latest_execution_selection_source": "training_research",
                        "nightly_latest_promotion_review_model_kind": "ml_scorer",
                        "nightly_latest_promotion_review_model_kinds": "ml_scorer, rl_policy",
                        "nightly_latest_refreshed_target_mix": "both=1",
                        "nightly_posture": "rl",
                        "nightly_recommended_target": "all",
                        "nightly_recommended_force_refresh": True,
                        "nightly_recent_window": 2,
                        "nightly_recent_enabled": 2,
                        "nightly_recent_aligned": 1,
                        "nightly_recent_latest_status": "ok",
                        "nightly_recent_latest_basket_size": 3,
                        "research_headline": "4 ML/RL research rows prepared",
                        "research_selection_policy": "diversified",
                        "research_refresh_target": "rl",
                        "research_discovery_summary": (
                            "preferred=2 | regimes=trending=2 | posture=rl"
                        ),
                        "research_discovery_recommended_target": "ml",
                        "research_discovery_posture": "ml",
                        "research_strategy_posture": "rl",
                        "research_candidate_posture": "ml",
                        "research_nightly_posture": "rl",
                        "research_effective_posture": "all",
                        "research_effective_target": "all",
                        "research_refresh_urgency": "high",
                        "research_refresh_urgency_target": "rl",
                        "research_refresh_urgency_score": 0.08,
                        "research_refresh_urgency_rows": 2,
                        "research_refresh_urgency_summary": (
                            "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
                        ),
                        "research_follow_up_action": (
                            "Prioritize RL refresh follow-up from remediation pressure "
                            "across 2 selected row(s)."
                        ),
                        "research_discovery_follow_up_target": "ml",
                        "research_discovery_follow_up_summary": (
                            "Discovery scouts currently lean ML-focused from ml universe posture."
                        ),
                        "research_discovery_follow_up_action": (
                            "Refresh market universe and shortlist review with ML-focused "
                            "discovery focus before the next training cycle."
                        ),
                        "research_research_follow_up_target": "rl",
                        "research_research_follow_up_summary": (
                            "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
                        ),
                        "research_research_follow_up_action": (
                            "Prioritize RL refresh follow-up from remediation pressure "
                            "across 2 selected row(s)."
                        ),
                        "research_execution_follow_up_target": "rl",
                        "research_execution_follow_up_summary": (
                            "Nightly execution drift currently leans RL-focused from rl "
                            "execution posture."
                        ),
                        "research_execution_follow_up_action": (
                            "Review nightly execution path and retarget execution "
                            "candidate selection toward RL-focused before the next "
                            "promotion or nightly cycle."
                        ),
                        "research_target_mismatch": True,
                        "research_refresh_requested": True,
                        "research_refreshed_count": 2,
                        "research_ml_count": 3,
                        "research_rl_count": 2,
                        "research_selected_target_mix": "both=1, ml=1, rl=2",
                        "research_selected_regime_mix": "ranging=1, trending=2, volatile=1",
                        "research_scout_support_summary": (
                            "supported=2, overlap=1, volume_dense=2, "
                            "symbols=RELIANCE.NS,SBIN.NS"
                        ),
                        "research_scout_supported_symbols": "RELIANCE.NS,SBIN.NS",
                        "research_scout_overlap_selected_count": 1,
                        "research_scout_volume_dense_selected_count": 2,
                        "research_alignment_summary": "basket/research aligned 3/3",
                        "research_alignment_target_mix": "both=1, ml=1, rl=1",
                        "research_alignment_overlap_count": 3,
                        "research_alignment_selected_count": 3,
                        "discovery_refresh_source": "auto",
                        "discovery_refresh_timeframe": "1d",
                        "discovery_refresh_days": 20,
                        "discovery_refresh_requested": True,
                        "discovery_refreshed_count": 14,
                        "discovery_alignment_summary": "discovery partial 2/3",
                        "discovery_overlap_symbols": "RELIANCE.NS, SBIN.NS",
                        "discovery_alignment_overlap_count": 2,
                        "discovery_alignment_compare_count": 3,
                        "artifact_follow_up_action": "Refresh training research",
                        "artifact_follow_up_target": "rl",
                        "artifact_recovery_posture": (
                            "Review drift currently favors research recovery target=rl."
                        ),
                    }
                },
                "universe": {
                    "metrics": {
                        "count": 14,
                        "adaptive_count": 1,
                        "scout_summary": (
                            "liquidity=RELIANCE.NS,SBIN.NS | activity=SBIN.NS,TCS.NS | "
                            "volume_dense=SBIN.NS,TCS.NS | overlap=RELIANCE.NS,SBIN.NS"
                        ),
                        "scout_liquidity_symbols": "RELIANCE.NS,SBIN.NS",
                        "scout_activity_symbols": "SBIN.NS,TCS.NS",
                        "scout_volume_dense_symbols": "SBIN.NS,TCS.NS",
                        "scout_overlap_symbols": "RELIANCE.NS,SBIN.NS",
                    },
                    "notes": ["adaptive_boost=+0.10 avg_pnl=+1.25% rows=2"],
                },
                "shortlist": {"metrics": {"count": 7}},
                "briefing": {"metrics": {"candidates": 4}},
                "allocation": {
                    "metrics": {
                        "selected_count": 3,
                        "skipped_count": 2,
                        "max_positions": 3,
                        "critic_enabled": True,
                        "regime_mix": "TRENDING=2, RANGING=1",
                        "strategy_mix": "MMTS=1, ORB=2",
                        "risk_mix": "low=2, medium=1",
                    },
                    "notes": [
                        "portfolio_critic=enabled",
                        "portfolio_research_alignment=enabled",
                        "allocation_research_targets: both=1, rl=1",
                        "allocation_refreshed_research_targets: both=1",
                        "selected_regimes: TRENDING=2, RANGING=1",
                        "allocation_strategy_mix: MMTS=1, ORB=2",
                        "allocation_risk_mix: low=2, medium=1",
                        "allocation_freshness_mix: refreshed=2/3",
                        "allocation_target_balance_summary: balanced",
                    ],
                },
                "candidates": {"metrics": {"ml_count": 5, "rl_count": 3}},
                "research": {
                    "metrics": {
                        "count": 4,
                        "ml_count": 3,
                        "rl_count": 2,
                        "selection_policy": "diversified",
                        "refresh_target": "rl",
                        "discovery_summary": "preferred=2 | regimes=trending=2 | posture=rl",
                        "discovery_posture": "ml",
                        "discovery_volume_dense_symbols": "SBIN.NS,TCS.NS",
                        "strategy_posture": "rl",
                        "candidate_posture": "ml",
                        "nightly_posture": "rl",
                        "effective_posture": "all",
                        "discovery_recommended_refresh_target": "ml",
                        "effective_refresh_target": "all",
                        "refresh_urgency": "high",
                        "refresh_urgency_target": "rl",
                        "refresh_urgency_score": 0.08,
                        "refresh_urgency_rows": 2,
                        "refresh_urgency_summary": (
                            "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
                        ),
                        "follow_up_action": (
                            "Prioritize RL refresh follow-up from remediation pressure "
                            "across 2 selected row(s)."
                        ),
                        "discovery_follow_up_target": "ml",
                        "discovery_follow_up_summary": (
                            "Discovery scouts currently lean ML-focused from ml universe posture."
                        ),
                        "discovery_follow_up_action": (
                            "Refresh market universe and shortlist review with ML-focused "
                            "discovery focus before the next training cycle."
                        ),
                        "research_follow_up_target": "rl",
                        "research_follow_up_summary": (
                            "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
                        ),
                        "research_follow_up_action": (
                            "Prioritize RL refresh follow-up from remediation pressure "
                            "across 2 selected row(s)."
                        ),
                        "execution_follow_up_target": "rl",
                        "execution_follow_up_summary": (
                            "Nightly execution drift currently leans RL-focused from rl "
                            "execution posture."
                        ),
                        "execution_follow_up_action": (
                            "Review nightly execution path and retarget execution "
                            "candidate selection toward RL-focused before the next "
                            "promotion or nightly cycle."
                        ),
                        "target_mismatch": True,
                        "refresh_requested": True,
                        "refreshed_count": 2,
                        "selected_target_mix": "both=1, ml=1, rl=2",
                        "selected_regime_mix": "ranging=1, trending=2, volatile=1",
                        "scout_support_summary": (
                            "supported=2, overlap=1, volume_dense=2, "
                            "symbols=RELIANCE.NS,SBIN.NS"
                        ),
                        "scout_supported_symbols": "RELIANCE.NS,SBIN.NS",
                        "scout_overlap_selected_count": 1,
                        "scout_volume_dense_selected_count": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary = load_workflow_snapshot_summary(str(path))

    assert summary is not None
    assert summary.team_role_count == 6
    assert summary.team_ok_role_count == 5
    assert summary.team_nightly_report_count == 2
    assert summary.team_nightly_aligned_reports == 1
    assert summary.team_nightly_enabled_reports == 2
    assert summary.team_nightly_latest_target_mix == "both=1, rl=1"
    assert summary.team_nightly_latest_execution_target == "ml"
    assert summary.team_nightly_latest_execution_model_family == "ml"
    assert summary.team_nightly_latest_execution_selection_source == "training_research"
    assert summary.team_nightly_latest_promotion_review_model_kind == "ml_scorer"
    assert summary.team_nightly_latest_promotion_review_model_kinds == "ml_scorer, rl_policy"
    assert summary.team_nightly_latest_refreshed_target_mix == "both=1"
    assert summary.team_nightly_posture == "rl"
    assert summary.team_nightly_recommended_target == "all"
    assert summary.team_nightly_recommended_force_refresh is True
    assert summary.team_nightly_recent_window == 2
    assert summary.team_nightly_recent_enabled == 2
    assert summary.team_nightly_recent_aligned == 1
    assert summary.team_nightly_recent_latest_status == "ok"
    assert summary.team_nightly_recent_latest_basket_size == 3
    assert summary.team_research_headline == "4 ML/RL research rows prepared"
    assert summary.team_research_selection_policy == "diversified"
    assert summary.team_research_refresh_target == "rl"
    assert (
        summary.team_research_discovery_summary
        == "preferred=2 | regimes=trending=2 | posture=rl"
    )
    assert summary.team_research_discovery_recommended_target == "ml"
    assert summary.team_research_discovery_posture == "ml"
    assert summary.team_research_strategy_posture == "rl"
    assert summary.team_research_candidate_posture == "ml"
    assert summary.team_research_nightly_posture == "rl"
    assert summary.team_research_effective_posture == "all"
    assert summary.team_research_effective_target == "all"
    assert summary.team_research_refresh_urgency == "high"
    assert summary.team_research_refresh_urgency_target == "rl"
    assert summary.team_research_refresh_urgency_score == 0.08
    assert summary.team_research_refresh_urgency_rows == 2
    assert summary.team_research_refresh_urgency_summary == (
        "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
    )
    assert summary.team_research_follow_up_action == (
        "Prioritize RL refresh follow-up from remediation pressure across 2 selected row(s)."
    )
    assert summary.team_research_discovery_follow_up_target == "ml"
    assert summary.team_research_discovery_follow_up_summary == (
        "Discovery scouts currently lean ML-focused from ml universe posture."
    )
    assert summary.team_research_research_follow_up_target == "rl"
    assert summary.team_research_execution_follow_up_target == "rl"
    assert summary.team_research_target_mismatch is True
    assert summary.team_research_refresh_requested is True
    assert summary.team_research_refreshed_count == 2
    assert summary.team_research_ml_count == 3
    assert summary.team_research_rl_count == 2
    assert summary.team_research_selected_target_mix == "both=1, ml=1, rl=2"
    assert summary.team_research_selected_regime_mix == "ranging=1, trending=2, volatile=1"
    assert (
        summary.team_research_scout_support_summary
        == "supported=2, overlap=1, volume_dense=2, symbols=RELIANCE.NS,SBIN.NS"
    )
    assert summary.team_research_scout_supported_symbols == "RELIANCE.NS,SBIN.NS"
    assert summary.team_research_scout_overlap_selected_count == 1
    assert summary.team_research_scout_volume_dense_selected_count == 2
    assert summary.research_discovery_summary == "preferred=2 | regimes=trending=2 | posture=rl"
    assert summary.team_research_alignment_summary == "basket/research aligned 3/3"
    assert summary.team_research_alignment_target_mix == "both=1, ml=1, rl=1"
    assert summary.team_research_alignment_overlap_count == 3
    assert summary.team_research_alignment_selected_count == 3
    assert summary.team_discovery_refresh_source == "auto"
    assert summary.team_discovery_refresh_timeframe == "1d"
    assert summary.team_discovery_refresh_days == 20
    assert summary.team_discovery_refresh_requested is True
    assert summary.team_discovery_refreshed_count == 14
    assert summary.team_discovery_alignment_summary == "discovery partial 2/3"
    assert summary.team_discovery_overlap_symbols == "RELIANCE.NS, SBIN.NS"
    assert summary.team_discovery_alignment_overlap_count == 2
    assert summary.team_discovery_alignment_compare_count == 3
    assert summary.team_artifact_follow_up_action == "Refresh training research"
    assert summary.team_artifact_follow_up_target == "rl"
    assert (
        summary.team_artifact_recovery_posture
        == "Review drift currently favors research recovery target=rl."
    )
    assert summary.universe_count == 14
    assert summary.universe_adaptive_count == 1
    assert summary.universe_scout_summary is not None
    assert summary.universe_volume_dense_symbols == "SBIN.NS,TCS.NS"
    assert summary.shortlist_count == 7
    assert summary.allocation_selected == 3
    assert summary.allocation_skipped == 2
    assert summary.allocation_research_alignment_enabled is True
    assert summary.allocation_research_target_mix == "both=1, rl=1"
    assert summary.allocation_refreshed_research_target_mix == "both=1"
    assert summary.allocation_critic_enabled is True
    assert summary.allocation_regime_mix == "TRENDING=2, RANGING=1"
    assert summary.allocation_strategy_mix == "MMTS=1, ORB=2"
    assert summary.allocation_risk_mix == "low=2, medium=1"
    assert summary.allocation_freshness_mix == "refreshed=2/3"
    assert summary.allocation_target_balance_summary == "balanced"
    assert summary.research_count == 4
    assert summary.research_ml_count == 3
    assert summary.research_rl_count == 2
    assert summary.research_discovery_posture == "ml"
    assert summary.research_volume_dense_symbols == "SBIN.NS,TCS.NS"
    assert summary.research_strategy_posture == "rl"
    assert summary.research_candidate_posture == "ml"
    assert summary.research_nightly_posture == "rl"
    assert summary.research_effective_posture == "all"
    assert summary.research_discovery_recommended_target == "ml"
    assert summary.research_effective_target == "all"
    assert summary.research_refresh_urgency == "high"
    assert summary.research_refresh_urgency_target == "rl"
    assert summary.research_refresh_urgency_score == 0.08
    assert summary.research_refresh_urgency_rows == 2
    assert summary.research_refresh_urgency_summary == (
        "high RL remediation pressure on 2 row(s); max=+0.08 total=+0.13"
    )
    assert summary.research_follow_up_action == (
        "Prioritize RL refresh follow-up from remediation pressure across 2 selected row(s)."
    )
    assert summary.research_discovery_follow_up_target == "ml"
    assert summary.research_discovery_follow_up_summary == (
        "Discovery scouts currently lean ML-focused from ml universe posture."
    )
    assert summary.research_research_follow_up_target == "rl"
    assert summary.research_execution_follow_up_target == "rl"
    assert summary.research_target_mismatch is True
    assert summary.research_refresh_requested is True
    assert summary.research_refreshed_count == 2
    assert summary.research_selected_target_mix == "both=1, ml=1, rl=2"
    assert summary.research_selected_regime_mix == "ranging=1, trending=2, volatile=1"
    assert (
        summary.research_scout_support_summary
        == "supported=2, overlap=1, volume_dense=2, symbols=RELIANCE.NS,SBIN.NS"
    )
    assert summary.research_scout_supported_symbols == "RELIANCE.NS,SBIN.NS"
    assert summary.research_scout_overlap_selected_count == 1
    assert summary.research_scout_volume_dense_selected_count == 2
    text = format_workflow_snapshot_summary(summary)
    assert "source=auto" in text
    assert "timeframe=15m" in text
    assert "team=5/6" in text
    assert "universe=14" in text
    assert "allocation=3/3" in text
    assert "training_research=4(3/2) refreshed=2" in text
    assert "nightly=1/2" in text
    assert "nightly_recent=1/2" in text
    assert "nightly_posture=rl" in text
    assert "nightly_target=all" in text
    assert "discovery_follow_up=ml" in text
    assert "research_follow_up=rl" in text
    assert "execution_follow_up=rl" in text
    assert "nightly_execution=ml/ml source=training_research" in text
    assert (
        "team_research=3/2 policy=diversified target=rl "
        "discovery=preferred=2 | regimes=trending=2 | posture=rl "
        "discovery_target=ml candidate=ml nightly=rl posture=all effective_target=all "
        "urgency=high/rl pressure=+0.08 rows=2 "
        "targets=both=1, ml=1, rl=2 "
        "scouts=supported=2, overlap=1, volume_dense=2, symbols=RELIANCE.NS,SBIN.NS "
        "regimes=ranging=1, trending=2, volatile=1 "
        "refreshed=2 alignment=basket/research aligned 3/3 mismatch=1 "
        "follow_up=Prioritize RL refresh follow-up from remediation pressure "
        "across 2 selected row(s)."
    ) in text
    assert "discovery_refresh=14 source=auto timeframe=1d days=20" in text
    assert "discovery=discovery partial 2/3" in text
    assert "artifact_follow_up=Refresh training research" in text
    assert "artifact_target=rl" in text
    assert "artifact_recovery=Review drift currently favors research recovery target=rl." in text
    assert "overlap=RELIANCE.NS, SBIN.NS" in text
    assert "nightly_force_refresh=1" in text
    assert "research=both=1, rl=1" in text
    assert "refreshed_research=both=1" in text
    assert "allocation_regimes=TRENDING=2, RANGING=1" in text
    assert "allocation_strategies=MMTS=1, ORB=2" in text
    assert "allocation_risk=low=2, medium=1" in text
    assert "allocation_critic=enabled" in text
    assert "adaptive=adaptive_boost=+0.10 avg_pnl=+1.25% rows=2" in text
    assert "rl=3" in text


def test_load_workflow_snapshot_summary_falls_back_to_allocation_notes(tmp_path: Path):
    path = tmp_path / "workflow_snapshot_notes_only.json"
    path.write_text(
        json.dumps(
            {
                "allocation": {
                    "metrics": {"selected_count": 1, "skipped_count": 0, "max_positions": 2},
                    "notes": [
                        "portfolio_critic=enabled",
                        "selected_regimes: TRENDING=1",
                        "allocation_strategy_mix: ORB=1",
                        "allocation_risk_mix: low=1",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    summary = load_workflow_snapshot_summary(str(path))

    assert summary is not None
    assert summary.allocation_critic_enabled is True
    assert summary.allocation_regime_mix == "TRENDING=1"
    assert summary.allocation_strategy_mix == "ORB=1"
    assert summary.allocation_risk_mix == "low=1"
