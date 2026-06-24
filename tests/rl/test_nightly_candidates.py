from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from fortuna.agentic.contracts import TrainingCandidate, TrainingCandidateResponse
from fortuna.app import nightly_basket


def _base_args() -> Namespace:
    return Namespace(
        candidate_manifest="",
        use_training_candidates=False,
        candidate_target="all",
        candidate_top_n=0,
        candidate_selection_policy="ranked",
        candidate_source="auto",
        candidate_universe_limit=15,
        candidate_analysis_limit=8,
        training_candidate_out="",
        training_research_out="",
        include_futures=True,
        futures_symbols=None,
        max_symbols=None,
        symbols=["RELIANCE.NS", "TCS.NS"],
        timeframe="5m",
        days=30,
    )


def test_resolve_symbol_basket_from_manifest(tmp_path, monkeypatch):
    manifest = TrainingCandidateResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate("RELIANCE.NS", 1, ml_candidate=True, rl_candidate=True),
            TrainingCandidate("TCS.NS", 2, ml_candidate=True, rl_candidate=False),
        ),
    )
    args = _base_args()
    args.candidate_manifest = str(tmp_path / "training_candidates.json")
    args.candidate_target = "rl"

    monkeypatch.setattr(nightly_basket, "load_training_candidates_manifest", lambda path: manifest)
    symbols, meta = nightly_basket.resolve_symbol_basket(args)
    assert symbols == ["RELIANCE.NS", "RELIANCE.FUT"]
    assert meta["mode"] == "candidate_manifest"


def test_resolve_symbol_basket_from_derived_candidates(monkeypatch):
    response = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate("RELIANCE.NS", 1, ml_candidate=True, rl_candidate=True),
            TrainingCandidate("INFY.NS", 2, ml_candidate=True, rl_candidate=True),
        ),
    )
    args = _base_args()
    args.use_training_candidates = True
    args.include_futures = False
    args.candidate_top_n = 1

    monkeypatch.setattr(nightly_basket, "build_training_candidates", lambda **kwargs: response)
    symbols, meta = nightly_basket.resolve_symbol_basket(args)
    assert symbols == ["RELIANCE.NS"]
    assert meta["mode"] == "derived_training_candidates"


def test_resolve_symbol_basket_auto_narrows_target_from_discovery(monkeypatch):
    response = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate("RELIANCE.NS", 1, ml_candidate=True, rl_candidate=True),
            TrainingCandidate("TCS.NS", 2, ml_candidate=True, rl_candidate=False),
            TrainingCandidate("SBIN.NS", 3, ml_candidate=False, rl_candidate=True),
        ),
    )
    args = _base_args()
    args.use_training_candidates = True
    args.include_futures = False
    args.candidate_top_n = 3

    class ProbeResponse:
        ok = True
        discovery_recommended_refresh_target = "ml"

    monkeypatch.setattr(nightly_basket, "build_training_candidates", lambda **kwargs: response)
    monkeypatch.setattr(
        nightly_basket,
        "build_training_research_plan",
        lambda **kwargs: ProbeResponse(),
    )
    monkeypatch.setattr(nightly_basket, "get_settings", lambda: object())

    symbols, meta = nightly_basket.resolve_symbol_basket(args)
    assert symbols == ["RELIANCE.NS", "TCS.NS"]
    assert meta["mode"] == "derived_training_candidates"
    assert meta["requested_target"] == "all"
    assert meta["target"] == "ml"
    assert meta["target_auto_narrowed"] is True
    assert meta["selection_source"] == "training_candidates"
    assert meta["discovery_recommended_refresh_target"] == "ml"


def test_resolve_symbol_basket_auto_narrows_target_from_effective_refresh_target(monkeypatch):
    response = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate("RELIANCE.NS", 1, ml_candidate=True, rl_candidate=True),
            TrainingCandidate("TCS.NS", 2, ml_candidate=True, rl_candidate=False),
        ),
    )
    args = _base_args()
    args.use_training_candidates = True
    args.include_futures = False
    args.candidate_top_n = 3

    class ProbeResponse:
        ok = True
        discovery_recommended_refresh_target = "all"
        effective_refresh_target = "ml"

    monkeypatch.setattr(nightly_basket, "build_training_candidates", lambda **kwargs: response)
    monkeypatch.setattr(
        nightly_basket,
        "build_training_research_plan",
        lambda **kwargs: ProbeResponse(),
    )
    monkeypatch.setattr(nightly_basket, "get_settings", lambda: object())

    symbols, meta = nightly_basket.resolve_symbol_basket(args)
    assert symbols == ["RELIANCE.NS", "TCS.NS"]
    assert meta["requested_target"] == "all"
    assert meta["target"] == "ml"
    assert meta["target_auto_narrowed"] is True


def test_resolve_symbol_basket_prefers_training_research_rows_when_available(monkeypatch):
    response = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                "RELIANCE.NS",
                1,
                selection_rank=1,
                ml_candidate=True,
                rl_candidate=True,
            ),
            TrainingCandidate(
                "TCS.NS",
                2,
                selection_rank=2,
                ml_candidate=True,
                rl_candidate=False,
            ),
            TrainingCandidate(
                "SBIN.NS",
                3,
                selection_rank=3,
                ml_candidate=False,
                rl_candidate=True,
            ),
        ),
    )
    args = _base_args()
    args.use_training_candidates = True
    args.include_futures = False
    args.candidate_top_n = 2

    class ProbeResponse:
        ok = True
        discovery_recommended_refresh_target = "rl"
        discovery_summary = "preferred=2 | posture=rl"
        rows = (
            SimpleNamespace(
                symbol="SBIN.NS",
                target="rl",
                selection_rank=1,
                shortlist_rank=3,
                universe_rank=2,
            ),
            SimpleNamespace(
                symbol="RELIANCE.NS",
                target="both",
                selection_rank=2,
                shortlist_rank=1,
                universe_rank=1,
            ),
            SimpleNamespace(
                symbol="TCS.NS",
                target="ml",
                selection_rank=1,
                shortlist_rank=2,
                universe_rank=3,
            ),
        )

    monkeypatch.setattr(nightly_basket, "build_training_candidates", lambda **kwargs: response)
    monkeypatch.setattr(
        nightly_basket,
        "build_training_research_plan",
        lambda **kwargs: ProbeResponse(),
    )
    monkeypatch.setattr(nightly_basket, "get_settings", lambda: object())

    symbols, meta = nightly_basket.resolve_symbol_basket(args)
    assert symbols == ["SBIN.NS", "RELIANCE.NS"]
    assert meta["target"] == "rl"
    assert meta["selection_source"] == "training_research"
    assert meta["discovery_summary"] == "preferred=2 | posture=rl"


def test_resolve_symbol_basket_from_manifest_with_diversified_selection(tmp_path, monkeypatch):
    manifest = TrainingCandidateResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                "RELIANCE.NS",
                1,
                selection_rank=1,
                decision_action="BUY",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
            ),
            TrainingCandidate(
                "TCS.NS",
                2,
                selection_rank=2,
                decision_action="BUY",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
            ),
            TrainingCandidate(
                "SBIN.NS",
                3,
                selection_rank=3,
                decision_action="SELL",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="VOLATILE",
            ),
        ),
    )
    args = _base_args()
    args.candidate_manifest = str(tmp_path / "training_candidates.json")
    args.include_futures = False
    args.candidate_target = "rl"
    args.candidate_top_n = 2
    args.candidate_selection_policy = "diversified"

    monkeypatch.setattr(nightly_basket, "load_training_candidates_manifest", lambda path: manifest)
    symbols, meta = nightly_basket.resolve_symbol_basket(args)
    assert symbols == ["RELIANCE.NS", "SBIN.NS"]
    assert meta["selection_policy"] == "diversified"


def test_emit_workflow_snapshot_for_candidate_mode(tmp_path, monkeypatch):
    args = _base_args()
    args.use_training_candidates = True
    args.workflow_snapshot_out = ""

    class DummySnapshot:
        source = "screener"
        team = type(
            "S",
            (),
            {
                "metrics": {
                    "count": 6,
                    "ok_count": 5,
                    "headline": "multi-agent ready",
                    "nightly_report_count": 2,
                    "nightly_enabled_reports": 2,
                    "nightly_aligned_reports": 1,
                    "nightly_latest_target_mix": "both=1, rl=1",
                    "nightly_latest_refreshed_target_mix": "both=1",
                    "nightly_recommended_target": "all",
                    "nightly_recommended_force_refresh": True,
                    "research_refresh_urgency_summary": "medium RL remediation pressure",
                    "research_refresh_urgency_target": "rl",
                    "research_follow_up_action": "Prioritize RL refresh follow-up.",
                    "research_discovery_follow_up_target": "ml",
                    "research_discovery_follow_up_summary": "Discovery scouts lean ML.",
                    "research_discovery_follow_up_action": "Refresh discovery with ML focus.",
                    "research_research_follow_up_target": "rl",
                    "research_research_follow_up_summary": "Research remediation leans RL.",
                    "research_research_follow_up_action": "Refresh research with RL focus.",
                    "research_execution_follow_up_target": "rl",
                    "research_execution_follow_up_summary": "Execution drift leans RL.",
                    "research_execution_follow_up_action": "Retarget execution toward RL.",
                    "research_scout_support_target": "ml",
                    "research_scout_support_summary": "supported=2, target=ml",
                    "research_scout_supported_symbols": "RELIANCE.NS,SBIN.NS",
                    "research_scout_overlap_selected_count": 1,
                    "research_scout_volume_dense_selected_count": 2,
                    "discovery_alignment_summary": "discovery partial 2/3",
                    "discovery_overlap_symbols": "RELIANCE.NS, SBIN.NS",
                    "discovery_alignment_overlap_count": 2,
                    "discovery_alignment_compare_count": 3,
                }
            },
        )()
        universe = type("S", (), {"metrics": {"count": 10}})()
        shortlist = type("S", (), {"metrics": {"count": 5}})()
        briefing = type("S", (), {"metrics": {"candidates": 3}})()
        allocation = type(
            "S",
            (),
            {
                "notes": (
                    "portfolio_research_alignment=enabled",
                    "allocation_research_targets: both=1, rl=1",
                    "allocation_refreshed_research_targets: both=1",
                )
            },
        )()
        candidates = type("S", (), {"metrics": {"ml_count": 4, "rl_count": 2}})()
        research = type(
            "S",
            (),
            {
                "metrics": {
                    "count": 3,
                    "ml_count": 2,
                    "rl_count": 2,
                    "selection_policy": "diversified",
                    "refresh_target": "rl",
                    "refresh_urgency_summary": "medium RL remediation pressure",
                    "refresh_urgency_target": "rl",
                    "follow_up_action": "Prioritize RL refresh follow-up.",
                    "discovery_follow_up_target": "ml",
                    "discovery_follow_up_summary": "Discovery scouts lean ML.",
                    "discovery_follow_up_action": "Refresh discovery with ML focus.",
                    "research_follow_up_target": "rl",
                    "research_follow_up_summary": "Research remediation leans RL.",
                    "research_follow_up_action": "Refresh research with RL focus.",
                    "execution_follow_up_target": "rl",
                    "execution_follow_up_summary": "Execution drift leans RL.",
                    "execution_follow_up_action": "Retarget execution toward RL.",
                    "scout_support_target": "ml",
                    "scout_support_summary": "supported=2, target=ml",
                    "scout_supported_symbols": "RELIANCE.NS,SBIN.NS",
                    "scout_overlap_selected_count": 1,
                    "scout_volume_dense_selected_count": 2,
                    "refresh_requested": True,
                    "refreshed_count": 1,
                }
            },
        )()

    monkeypatch.setattr(nightly_basket, "get_settings", lambda: object())
    monkeypatch.setattr(nightly_basket, "build_workflow_snapshot", lambda **kwargs: DummySnapshot())
    monkeypatch.setattr(
        nightly_basket,
        "export_workflow_snapshot",
        lambda snapshot, out_path: tmp_path / "workflow_snapshot.json",
    )

    meta = nightly_basket.emit_workflow_snapshot(args, report_dir=tmp_path)
    assert meta is not None
    assert meta["source"] == "screener"
    assert meta["team_role_count"] == 6
    assert meta["team_ok_role_count"] == 5
    assert meta["team_nightly_report_count"] == 2
    assert meta["team_nightly_recommended_target"] == "all"
    assert meta["team_nightly_recommended_force_refresh"] is True
    assert meta["team_research_refresh_urgency_target"] == "rl"
    assert meta["team_research_follow_up_action"] == "Prioritize RL refresh follow-up."
    assert meta["team_research_discovery_follow_up_target"] == "ml"
    assert meta["team_research_research_follow_up_target"] == "rl"
    assert meta["team_research_execution_follow_up_target"] == "rl"
    assert meta["team_research_scout_support_summary"] == "supported=2, target=ml"
    assert meta["team_research_scout_supported_symbols"] == "RELIANCE.NS,SBIN.NS"
    assert meta["team_research_scout_overlap_selected_count"] == 1
    assert meta["team_research_scout_volume_dense_selected_count"] == 2
    assert meta["briefing_candidates"] == 3
    assert meta["allocation_research_alignment_enabled"] is True
    assert meta["allocation_research_target_mix"] == "both=1, rl=1"
    assert meta["allocation_refreshed_research_target_mix"] == "both=1"
    assert meta["research_refreshed_count"] == 1
    assert meta["research_refresh_urgency_target"] == "rl"
    assert meta["research_follow_up_action"] == "Prioritize RL refresh follow-up."
    assert meta["research_discovery_follow_up_target"] == "ml"
    assert meta["research_research_follow_up_target"] == "rl"
    assert meta["research_execution_follow_up_target"] == "rl"
    assert meta["research_scout_support_summary"] == "supported=2, target=ml"
    assert meta["research_scout_supported_symbols"] == "RELIANCE.NS,SBIN.NS"
    assert meta["research_scout_overlap_selected_count"] == 1
    assert meta["research_scout_volume_dense_selected_count"] == 2
    assert meta["rl_count"] == 2
    assert meta["team_discovery_alignment_summary"] == "discovery partial 2/3"
    assert meta["team_discovery_overlap_symbols"] == "RELIANCE.NS, SBIN.NS"


def test_emit_training_candidate_manifest_for_candidate_mode(tmp_path, monkeypatch):
    args = _base_args()
    args.use_training_candidates = True

    response = TrainingCandidateResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                "RELIANCE.NS",
                1,
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                winning_strategy="ORB",
                adaptive_score_adjustment=0.12,
            ),
        ),
    )

    monkeypatch.setattr(nightly_basket, "get_settings", lambda: object())
    monkeypatch.setattr(
        nightly_basket,
        "export_training_candidates",
        lambda manifest, out_path: tmp_path / "training_candidates.json",
    )

    meta = nightly_basket.emit_training_candidate_manifest(
        args,
        report_dir=tmp_path,
        candidate_response=response,
    )
    assert meta is not None
    assert meta["source"] == "screener"
    assert meta["count"] == 1
    assert meta["rl_count"] == 1


def test_emit_training_research_plan_for_candidate_mode(tmp_path, monkeypatch):
    args = _base_args()
    args.use_training_candidates = True

    class DummyResponse:
        ok = True
        source = "screener"
        selection_policy = "diversified"
        refresh_requested = False
        refresh_target = "all"
        ml_symbols = ("RELIANCE.NS", "TCS.NS")
        rl_symbols = ("RELIANCE.NS",)
        rows = (
            SimpleNamespace(refreshed=False),
            SimpleNamespace(refreshed=True),
        )

    monkeypatch.setattr(nightly_basket, "get_settings", lambda: object())
    monkeypatch.setattr(
        nightly_basket,
        "build_training_research_plan",
        lambda **kwargs: DummyResponse(),
    )
    monkeypatch.setattr(
        nightly_basket,
        "export_training_research_plan",
        lambda response, out_path: tmp_path / "training_research_plan.json",
    )

    meta = nightly_basket.emit_training_research_plan(args, report_dir=tmp_path)
    assert meta is not None
    assert meta["source"] == "screener"
    assert meta["count"] == 2
    assert meta["ml_count"] == 2
    assert meta["rl_count"] == 1
    assert meta["refreshed_count"] == 1


def test_emit_training_research_plan_auto_narrows_refresh_target(tmp_path, monkeypatch):
    args = _base_args()
    args.use_training_candidates = True

    class ProbeResponse:
        ok = True
        discovery_recommended_refresh_target = "ml"

    class FinalResponse:
        ok = True
        source = "screener"
        selection_policy = "diversified"
        refresh_requested = False
        refresh_target = "ml"
        effective_refresh_target = "ml"
        discovery_posture = "ml"
        strategy_posture = "ml"
        effective_posture = "ml"
        selected_target_mix = "ml=2"
        selected_regime_mix = "ranging=2"
        discovery_recommended_refresh_target = "ml"
        discovery_preferred_symbols = ("RELIANCE.NS",)
        discovery_regime_mix = "ranging=2"
        discovery_summary = "preferred=1 | regimes=ranging=2"
        ml_symbols = ("RELIANCE.NS", "TCS.NS")
        rl_symbols = ("RELIANCE.NS",)
        rows = (
            SimpleNamespace(refreshed=False),
            SimpleNamespace(refreshed=True),
        )

    calls: list[str] = []

    def _build(**kwargs):
        calls.append(str(kwargs.get("refresh_target", "")))
        if kwargs.get("refresh_target") == "all":
            return ProbeResponse()
        return FinalResponse()

    monkeypatch.setattr(nightly_basket, "get_settings", lambda: object())
    monkeypatch.setattr(nightly_basket, "build_training_research_plan", _build)
    monkeypatch.setattr(
        nightly_basket,
        "export_training_research_plan",
        lambda response, out_path: tmp_path / "training_research_plan.json",
    )

    meta = nightly_basket.emit_training_research_plan(args, report_dir=tmp_path)
    assert meta is not None
    assert calls == ["all", "ml"]
    assert meta["requested_refresh_target"] == "all"
    assert meta["refresh_target"] == "ml"
    assert meta["effective_refresh_target"] == "ml"
    assert meta["target_auto_narrowed"] is True
    assert meta["discovery_posture"] == "ml"
    assert meta["strategy_posture"] == "ml"
    assert meta["effective_posture"] == "ml"
    assert meta["selected_target_mix"] == "ml=2"
    assert meta["selected_regime_mix"] == "ranging=2"
    assert meta["discovery_recommended_refresh_target"] == "ml"
