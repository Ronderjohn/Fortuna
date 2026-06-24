from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fortuna.agentic.contracts import (
    AdvisoryError,
    AdvisoryErrorCode,
    DecisionSummary,
    InstrumentSnapshot,
    NightlyAlignmentStatus,
    RiskCritique,
    ShortlistAnalysisItem,
    ShortlistAnalysisResponse,
    TrainingCandidate,
    TrainingCandidateResponse,
)
from fortuna.agentic.learning import LearningExample, LearningOutcome
from fortuna.agentic.store import AgenticLearningStore
from fortuna.app.training_candidates import (
    backfill_training_candidates,
    build_training_candidates,
    export_training_candidates,
    load_training_candidates_manifest,
    select_training_symbols,
)
from fortuna.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )


def test_build_training_candidates_maps_shortlist_items(tmp_path: Path, monkeypatch):
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                instrument=InstrumentSnapshot(
                    requested_symbol="RELIANCE.NS",
                    resolved_symbol="RELIANCE.NS",
                    segment_label="EQUITY",
                    timeframe="5m",
                    lookback_days=30,
                ),
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY won",
                    reasons=("orb fired",),
                ),
                liquidity_score=12.0,
                regime="TRENDING",
                volume_ratio=1.24,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                decision=DecisionSummary(
                    action="DO_NOT_ENTER",
                    confidence=0.62,
                    summary="DO_NOT_ENTER won",
                    reasons=("weak setup",),
                ),
                liquidity_score=9.5,
                critique=RiskCritique(
                    severity="high",
                    summary="DO_NOT_ENTER is not an active entry candidate",
                    concerns=("Current setup explicitly not attractive enough",),
                    verdict="avoid",
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    response = build_training_candidates(settings=_settings(tmp_path))
    assert response.ok is True
    assert len(response.candidates) == 2
    assert response.candidates[0].symbol == "RELIANCE.NS"
    assert response.candidates[0].ml_candidate is True
    assert response.candidates[0].rl_candidate is True
    assert response.candidates[0].selection_rank == 1
    assert response.candidates[0].market_regime == "TRENDING"
    assert response.candidates[0].volume_ratio == 1.24
    assert response.candidates[0].priority_score is not None
    assert response.candidates[0].winning_strategy is None
    assert any("Market regime TRENDING" in text for text in response.candidates[0].rationale)


def test_build_training_candidates_applies_scout_cohort_priority_hints(
    tmp_path: Path,
    monkeypatch,
):
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        scout_volume_dense_symbols=("TCS.NS",),
        scout_overlap_symbols=("TCS.NS",),
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.72,
                    summary="BUY won",
                    reasons=("clean setup",),
                ),
                liquidity_score=11.5,
                regime="TRENDING",
                volume_ratio=1.15,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.72,
                    summary="BUY won",
                    reasons=("clean setup",),
                ),
                liquidity_score=11.5,
                regime="TRENDING",
                volume_ratio=1.15,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.load_adaptive_outcome_policy",
        lambda *args, **kwargs: SimpleNamespace(
            symbol_biases={},
            action_biases={},
            regime_biases={},
            durable_setup_family_biases={},
            signal_biases={},
            global_bias=None,
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates._load_recent_nightly_feedback",
        lambda settings: ({}, 0),
    )

    response = build_training_candidates(settings=_settings(tmp_path))

    assert response.ok is True
    ranked = {row.symbol: row for row in response.candidates}
    assert ranked["TCS.NS"].priority_score > ranked["RELIANCE.NS"].priority_score
    assert any(
        "Discovery overlap cohort reinforced this symbol" in text
        for text in ranked["TCS.NS"].rationale
    )
    assert any(
        "Volume-dense scout cohort kept this symbol in rotation" in text
        for text in ranked["TCS.NS"].rationale
    )


def test_build_training_candidates_boosts_activity_rich_setups(tmp_path: Path, monkeypatch):
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.70,
                    summary="BUY won",
                    reasons=("steady setup",),
                ),
                liquidity_score=12.2,
                regime="RANGING",
                volume_ratio=1.0,
                critique=RiskCritique(
                    severity="low",
                    summary="Decent but quiet",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="SBIN.NS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.69,
                    summary="BUY won",
                    reasons=("breakout pressure",),
                ),
                liquidity_score=11.7,
                regime="TRENDING",
                volume_ratio=1.45,
                critique=RiskCritique(
                    severity="low",
                    summary="Active tape with better follow-through",
                    verdict="candidate",
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    response = build_training_candidates(settings=_settings(tmp_path))

    assert response.ok is True
    assert [row.symbol for row in response.candidates[:2]] == ["SBIN.NS", "RELIANCE.NS"]
    assert any(
        "Discovery activity context volume_ratio=1.45 regime=TRENDING priority_adj=+0.11"
        in text
        for text in response.candidates[0].rationale
    )
    assert any(
        "Discovery posture RL favored TRENDING volume_ratio=1.45 priority_adj=+0.04" in text
        for text in response.candidates[0].rationale
    )


def test_build_training_candidates_prefers_calm_ranging_setups_when_posture_is_ml(
    tmp_path: Path,
    monkeypatch,
):
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.70,
                    summary="BUY won",
                    reasons=("quiet mean reversion",),
                ),
                liquidity_score=11.9,
                regime="RANGING",
                volume_ratio=1.01,
                critique=RiskCritique(
                    severity="low",
                    summary="Orderly tape with cleaner calibration target",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="SBIN.NS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.66,
                    summary="BUY won",
                    reasons=("late breakout",),
                ),
                liquidity_score=11.4,
                regime="TRENDING",
                volume_ratio=1.04,
                critique=RiskCritique(
                    severity="low",
                    summary="Tradeable, but less representative of the current calm cohort",
                    verdict="candidate",
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    response = build_training_candidates(settings=_settings(tmp_path))

    assert response.ok is True
    assert [row.symbol for row in response.candidates[:2]] == ["RELIANCE.NS", "SBIN.NS"]
    assert any(
        "Discovery posture ML favored RANGING" in text
        for text in response.candidates[0].rationale
    )


def test_build_training_candidates_propagates_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: ShortlistAnalysisResponse(
            ok=False,
            source="auto",
            timeframe="5m",
            lookback_days=30,
            error=AdvisoryError(
                code=AdvisoryErrorCode.LOAD_FAILED,
                message="shortlist failed",
            ),
        ),
    )
    response = build_training_candidates(settings=_settings(tmp_path))
    assert response.ok is False
    assert response.error is not None


def test_export_training_candidates_writes_json(tmp_path: Path, monkeypatch):
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY won",
                    reasons=("orb fired",),
                ),
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    response = build_training_candidates(settings=_settings(tmp_path))
    out = tmp_path / "reports" / "training_candidates.json"
    export_training_candidates(response, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["candidates"][0]["symbol"] == "RELIANCE.NS"


def test_load_and_select_training_candidates_manifest(tmp_path: Path):
    payload = {
        "ok": True,
        "source": "registry",
        "timeframe": "5m",
        "lookback_days": 30,
        "candidates": [
            {
                "symbol": "RELIANCE.NS",
                "shortlist_rank": 1,
                "selection_rank": 1,
                "ml_candidate": True,
                "rl_candidate": True,
                "market_regime": "TRENDING",
                "volume_ratio": 1.2,
                "winning_strategy": "ORB",
                "critique_verdict": "candidate",
                "priority_score": 1.22,
                "adaptive_score_adjustment": 0.14,
                "exposure_penalty": 0.0,
                "rationale": ["orb fired"],
            },
            {
                "symbol": "TCS.NS",
                "shortlist_rank": 2,
                "selection_rank": 2,
                "ml_candidate": True,
                "rl_candidate": False,
                "market_regime": "RANGING",
                "volume_ratio": 0.98,
                "winning_strategy": "MMTS",
                "critique_verdict": "watch",
                "priority_score": 0.74,
                "exposure_penalty": 0.12,
                "rationale": ["watch only"],
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_training_candidates_manifest(path)
    assert manifest.ok is True
    assert select_training_symbols(manifest, target="ml") == ["RELIANCE.NS", "TCS.NS"]
    assert select_training_symbols(manifest, target="rl") == ["RELIANCE.NS"]
    assert manifest.candidates[0].market_regime == "TRENDING"
    assert manifest.candidates[0].winning_strategy == "ORB"
    assert manifest.candidates[0].adaptive_score_adjustment == 0.14
    assert manifest.candidates[1].exposure_penalty == 0.12


def test_select_training_symbols_diversifies_action_and_regime():
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
            TrainingCandidate(
                "INFY.NS",
                4,
                selection_rank=4,
                decision_action="BUY",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="RANGING",
            ),
        ),
    )

    assert select_training_symbols(
        manifest,
        target="rl",
        top_n=3,
        selection_policy="diversified",
    ) == ["RELIANCE.NS", "SBIN.NS", "INFY.NS"]


def test_select_training_symbols_prefers_discovery_overlap_symbols():
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
                ml_candidate=True,
                rl_candidate=True,
            ),
        ),
    )

    assert select_training_symbols(
        manifest,
        target="ml",
        top_n=2,
        preferred_symbols=("SBIN.NS", "TCS.NS"),
    ) == ["TCS.NS", "SBIN.NS"]


def test_build_training_candidates_penalizes_overlapping_exposure(tmp_path: Path, monkeypatch):
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.82,
                    summary="BUY won",
                    reasons=("orb fired",),
                ),
                liquidity_score=12.0,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="RELIANCE.FUT",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.81,
                    summary="BUY won",
                    reasons=("trend intact",),
                ),
                liquidity_score=11.0,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    response = build_training_candidates(settings=_settings(tmp_path))
    assert response.candidates[0].symbol == "RELIANCE.NS"
    assert response.candidates[0].exposure_penalty == 0.0
    assert response.candidates[1].symbol == "RELIANCE.FUT"
    assert response.candidates[1].exposure_penalty > 0.0
    assert any("Exposure penalty" in text for text in response.candidates[1].rationale)


def test_build_training_candidates_applies_recent_outcome_bias(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path).model_copy(
        update={
            "agentic_log_dir": Path("logs/agentic"),
            "training_candidate_adaptive_weighting_enabled": True,
        }
    )
    store = AgenticLearningStore(settings.resolve_path(settings.agentic_log_dir))
    store.upsert(
        LearningExample(
            decision_hash="g1",
            bar_time="2026-01-10T10:00:00",
            bar_idx=1,
            symbol="TCS.NS",
            timeframe="5m",
            action="BUY",
            confidence=0.7,
            current_side=None,
            bar_close=200.0,
            metadata={"primary_signal": "ORB", "regime": "TRENDING"},
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=1.9,
            ),
        )
    )
    store.upsert(
        LearningExample(
            decision_hash="g2",
            bar_time="2026-01-10T10:05:00",
            bar_idx=2,
            symbol="RELIANCE.NS",
            timeframe="5m",
            action="BUY",
            confidence=0.7,
            current_side=None,
            bar_close=100.0,
            metadata={"primary_signal": "MMTS", "regime": "RANGING"},
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=-2.1,
            ),
        )
    )

    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                winning_strategy="MMTS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY won",
                    reasons=("orb fired",),
                ),
                liquidity_score=12.0,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                winning_strategy="ORB",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY won",
                    reasons=("trend fired",),
                ),
                liquidity_score=12.0,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )

    response = build_training_candidates(settings=settings)
    ranked = {row.symbol: row for row in response.candidates}
    assert ranked["TCS.NS"].priority_score > ranked["RELIANCE.NS"].priority_score
    assert ranked["TCS.NS"].adaptive_score_adjustment is not None
    assert any("Recent paper outcomes imply a boost" in text for text in ranked["TCS.NS"].rationale)
    assert any("Strategy context ORB" in text for text in ranked["TCS.NS"].rationale)
    assert any(
        "Recent paper outcomes imply a penalty" in text
        for text in ranked["RELIANCE.NS"].rationale
    )


def test_build_training_candidates_applies_recent_nightly_feedback(
    tmp_path: Path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "training_candidate_nightly_feedback_enabled": True,
            "training_candidate_nightly_feedback_max_boost": 0.08,
            "training_candidate_nightly_feedback_lookback_reports": 4,
            "training_candidate_nightly_report_dir": Path("reports/nightly"),
        }
    )
    nightly_dir = settings.resolve_path(settings.training_candidate_nightly_report_dir)
    nightly_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = nightly_dir / "training_candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "ok": True,
                "source": "registry",
                "timeframe": "5m",
                "lookback_days": 30,
                "candidates": [
                    {
                        "symbol": "TCS.NS",
                        "shortlist_rank": 1,
                        "selection_rank": 1,
                        "ml_candidate": True,
                        "rl_candidate": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (nightly_dir / "20260603_0100.json").write_text(
        json.dumps(
            {
                "overall_status": "ok",
                "steps": [
                    {
                        "name": "basket_resolution",
                        "status": "ok",
                        "detail": {"mode": "derived_training_candidates"},
                    },
                    {
                        "name": "training_candidates",
                        "status": "ok",
                        "detail": {
                            "path": "reports/nightly/training_candidates.json",
                            "count": 1,
                            "ml_count": 1,
                            "rl_count": 1,
                            "selection_policy": "ranked",
                        },
                    },
                    {
                        "name": "training_execution_target",
                        "status": "ok",
                        "detail": {
                            "target": "rl",
                            "selection_source": "training_candidates",
                        },
                    },
                    {
                        "name": "promote_best",
                        "status": "ok",
                        "detail": {"count": 1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                winning_strategy="MMTS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY candidate",
                    reasons=("setup fired",),
                ),
                liquidity_score=10.0,
                regime="TRENDING",
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                winning_strategy="ORB",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY candidate",
                    reasons=("setup fired",),
                ),
                liquidity_score=10.0,
                regime="TRENDING",
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )

    response = build_training_candidates(settings=settings)
    ranked = {row.symbol: row for row in response.candidates}

    assert ranked["TCS.NS"].priority_score > ranked["RELIANCE.NS"].priority_score
    assert ranked["TCS.NS"].adaptive_score_adjustment is not None
    assert any(
        "Recent nightly evidence kept this symbol" in text
        for text in ranked["TCS.NS"].rationale
    )
    assert any("executed=1" in text for text in ranked["TCS.NS"].rationale)


def test_build_training_candidates_applies_remediation_pressure_from_learning_and_nightly_drift(
    tmp_path: Path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "agentic_log_dir": Path("logs/agentic"),
            "training_candidate_adaptive_weighting_enabled": True,
            "training_candidate_remediation_pressure_enabled": True,
            "training_candidate_remediation_max_boost": 0.06,
        }
    )
    store = AgenticLearningStore(settings.resolve_path(settings.agentic_log_dir))
    store.upsert(
        LearningExample(
            decision_hash="r1",
            bar_time="2026-01-10T10:00:00",
            bar_idx=1,
            symbol="RELIANCE.NS",
            timeframe="5m",
            action="BUY",
            confidence=0.7,
            current_side=None,
            bar_close=100.0,
            metadata={"primary_signal": "MMTS", "regime": "RANGING"},
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=-2.4,
            ),
        )
    )
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                winning_strategy="MMTS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.70,
                    summary="BUY won",
                    reasons=("quiet mean reversion",),
                ),
                liquidity_score=11.5,
                regime="RANGING",
                volume_ratio=1.01,
                critique=RiskCritique(
                    severity="low",
                    summary="Orderly tape with cleaner retraining target",
                    verdict="watch",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                winning_strategy="ORB",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.71,
                    summary="BUY won",
                    reasons=("trend continuation",),
                ),
                liquidity_score=11.5,
                regime="TRENDING",
                volume_ratio=1.02,
                critique=RiskCritique(
                    severity="low",
                    summary="Tradeable but less urgent for the current retraining drift",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.build_nightly_alignment_status",
        lambda settings: NightlyAlignmentStatus(
            report_count=3,
            enabled_reports=3,
            aligned_reports=1,
            nightly_posture="ml",
            recommended_action="Prepare an ML-oriented training research plan.",
            recommended_refresh_target="ml",
            effective_refresh_target="ml",
            recommended_force_refresh=True,
            workflow_target_mismatch=True,
        ),
    )

    response = build_training_candidates(settings=settings)
    ranked = {row.symbol: row for row in response.candidates}

    assert ranked["RELIANCE.NS"].remediation_target == "ml"
    assert ranked["RELIANCE.NS"].remediation_pressure is not None
    assert ranked["RELIANCE.NS"].remediation_pressure > 0.0
    assert ranked["TCS.NS"].remediation_pressure is not None
    assert ranked["RELIANCE.NS"].remediation_pressure > ranked["TCS.NS"].remediation_pressure
    assert any("Remediation pressure target=ML" in text for text in ranked["RELIANCE.NS"].rationale)


def test_build_training_candidates_applies_research_plan_nightly_feedback(
    tmp_path: Path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "training_candidate_nightly_feedback_enabled": True,
            "training_candidate_nightly_feedback_max_boost": 0.08,
            "training_candidate_nightly_feedback_lookback_reports": 4,
            "training_candidate_nightly_report_dir": Path("reports/nightly"),
        }
    )
    nightly_dir = settings.resolve_path(settings.training_candidate_nightly_report_dir)
    nightly_dir.mkdir(parents=True, exist_ok=True)
    research_path = nightly_dir / "training_research_plan.json"
    research_path.write_text(
        json.dumps(
            {
                "ok": True,
                "source": "registry",
                "timeframe": "5m",
                "lookback_days": 30,
                "selection_policy": "diversified",
                "refresh_target": "all",
                "ml_symbols": ["TCS.NS"],
                "rl_symbols": ["TCS.NS"],
                "rows": [
                    {"symbol": "TCS.NS", "target": "rl"},
                    {"symbol": "RELIANCE.NS", "target": "ml"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (nightly_dir / "20260603_0100.json").write_text(
        json.dumps(
            {
                "overall_status": "ok",
                "steps": [
                    {
                        "name": "basket_resolution",
                        "status": "ok",
                        "detail": {"mode": "derived_training_candidates"},
                    },
                    {
                        "name": "training_research",
                        "status": "ok",
                        "detail": {
                            "path": "reports/nightly/training_research_plan.json",
                            "count": 1,
                            "ml_count": 1,
                            "rl_count": 1,
                            "selection_policy": "diversified",
                            "refresh_target": "all",
                        },
                    },
                    {
                        "name": "training_execution_target",
                        "status": "ok",
                        "detail": {
                            "target": "rl",
                            "selection_source": "training_research",
                        },
                    },
                    {
                        "name": "promote_best",
                        "status": "ok",
                        "detail": {"count": 1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                winning_strategy="MMTS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY candidate",
                    reasons=("setup fired",),
                ),
                liquidity_score=10.0,
                regime="TRENDING",
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                winning_strategy="ORB",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY candidate",
                    reasons=("setup fired",),
                ),
                liquidity_score=10.0,
                regime="TRENDING",
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )

    response = build_training_candidates(settings=settings)
    ranked = {row.symbol: row for row in response.candidates}

    assert ranked["TCS.NS"].priority_score > ranked["RELIANCE.NS"].priority_score
    assert any("research plans" in text for text in ranked["TCS.NS"].rationale)
    assert any("executed=1" in text for text in ranked["TCS.NS"].rationale)


def test_build_training_candidates_weights_refreshed_research_evidence_more(
    tmp_path: Path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "training_candidate_nightly_feedback_enabled": True,
            "training_candidate_nightly_feedback_max_boost": 0.08,
            "training_candidate_nightly_feedback_lookback_reports": 4,
            "training_candidate_nightly_refresh_bonus_weight": 0.75,
            "training_candidate_nightly_report_dir": Path("reports/nightly"),
        }
    )
    nightly_dir = settings.resolve_path(settings.training_candidate_nightly_report_dir)
    nightly_dir.mkdir(parents=True, exist_ok=True)
    research_path = nightly_dir / "training_research_plan.json"
    nightly_report = nightly_dir / "20260603_0100.json"
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                winning_strategy="MMTS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY candidate",
                    reasons=("setup fired",),
                ),
                liquidity_score=10.0,
                regime="TRENDING",
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                winning_strategy="ORB",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY candidate",
                    reasons=("setup fired",),
                ),
                liquidity_score=10.0,
                regime="TRENDING",
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )

    def _write_fixture(*, refreshed: bool) -> None:
        research_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "source": "registry",
                    "timeframe": "5m",
                    "lookback_days": 30,
                    "selection_policy": "diversified",
                    "refresh_target": "rl",
                    "ml_symbols": ["TCS.NS"],
                    "rl_symbols": ["TCS.NS"],
                    "rows": [{"symbol": "TCS.NS", "target": "both", "refreshed": refreshed}],
                }
            ),
            encoding="utf-8",
        )
        nightly_report.write_text(
            json.dumps(
                {
                    "overall_status": "ok",
                    "steps": [
                        {
                            "name": "basket_resolution",
                            "status": "ok",
                            "detail": {"mode": "derived_training_candidates"},
                        },
                        {
                            "name": "training_research",
                            "status": "ok",
                            "detail": {
                                "path": "reports/nightly/training_research_plan.json",
                                "count": 1,
                                "ml_count": 1,
                                "rl_count": 1,
                                "selection_policy": "diversified",
                                "refresh_target": "rl",
                                "refresh_requested": refreshed,
                                "refreshed_count": 1 if refreshed else 0,
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    _write_fixture(refreshed=False)
    planned_only = build_training_candidates(settings=settings)
    _write_fixture(refreshed=True)
    refreshed = build_training_candidates(settings=settings)

    planned_ranked = {row.symbol: row for row in planned_only.candidates}
    refreshed_ranked = {row.symbol: row for row in refreshed.candidates}
    assert refreshed_ranked["TCS.NS"].priority_score > planned_ranked["TCS.NS"].priority_score
    assert any("refreshed=1" in text for text in refreshed_ranked["TCS.NS"].rationale)


def test_backfill_training_candidates_filters_to_selected_symbols(tmp_path: Path, monkeypatch):
    manifest = TrainingCandidateResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                selection_rank=1,
                ml_candidate=True,
                rl_candidate=True,
            ),
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=2,
                selection_rank=2,
                ml_candidate=True,
                rl_candidate=False,
            ),
            TrainingCandidate(
                symbol="SBIN.NS",
                shortlist_rank=3,
                selection_rank=3,
                ml_candidate=True,
                rl_candidate=True,
            ),
        ),
    )
    refreshed: list[str] = []

    class _FakeMarketDataManager:
        def __init__(self, settings, data_source):
            del settings, data_source

        def get_ohlcv(self, symbol, timeframe, days, force_refresh=False):
            del timeframe, days, force_refresh
            refreshed.append(symbol)
            return None

    monkeypatch.setattr("fortuna.app.training_candidates.MarketDataManager", _FakeMarketDataManager)

    result = backfill_training_candidates(
        _settings(tmp_path),
        manifest,
        timeframe="5m",
        days=30,
        target="rl",
        top_n=1,
        selection_policy="ranked",
    )

    assert result == ["RELIANCE.NS"]
    assert refreshed == ["RELIANCE.NS"]


def test_build_training_candidates_applies_durable_setup_family_reinforcement(
    tmp_path: Path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "agentic_log_dir": Path("logs/agentic"),
            "training_candidate_adaptive_weighting_enabled": True,
            "training_candidate_setup_family_reinforcement_enabled": True,
            "training_candidate_setup_family_min_durable_rows": 2,
            "training_candidate_setup_family_max_boost": 0.05,
        }
    )
    store = AgenticLearningStore(settings.resolve_path(settings.agentic_log_dir))
    for idx, (symbol, strategy, pnl) in enumerate(
        (
            ("TCS.NS", "ORB", 1.9),
            ("SBIN.NS", "ORB", 2.1),
            ("RELIANCE.NS", "MMTS", -2.1),
            ("INFY.NS", "MMTS", -2.4),
        ),
        start=1,
    ):
        store.upsert(
            LearningExample(
                decision_hash=f"durable-{idx}",
                bar_time=f"2026-01-10T10:0{idx}:00",
                bar_idx=idx,
                symbol=symbol,
                timeframe="5m",
                action="BUY",
                confidence=0.7,
                current_side=None,
                bar_close=100.0 + idx,
                metadata={"primary_signal": strategy, "regime": "TRENDING"},
                outcome=LearningOutcome(
                    status="resolved",
                    paper_closed=True,
                    realized_pnl_pct=pnl,
                ),
            )
        )

    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                winning_strategy="MMTS",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY won",
                    reasons=("mmts fired",),
                ),
                liquidity_score=12.0,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                winning_strategy="ORB",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY won",
                    reasons=("orb fired",),
                ),
                liquidity_score=12.0,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )

    response = build_training_candidates(settings=settings)
    ranked = {row.symbol: row for row in response.candidates}
    assert ranked["TCS.NS"].priority_score > ranked["RELIANCE.NS"].priority_score
    assert ranked["TCS.NS"].setup_family_reinforcement is not None
    assert ranked["TCS.NS"].setup_family_reinforcement > 0.0
    assert ranked["TCS.NS"].setup_family_verdict == "winner"
    assert ranked["RELIANCE.NS"].setup_family_verdict == "loser"
    assert any(
        "Setup-family reinforcement ORB: recurring winner" in text
        for text in ranked["TCS.NS"].rationale
    )
    assert not any("Strategy context ORB" in text for text in ranked["TCS.NS"].rationale)


def test_build_training_candidates_skips_durable_setup_family_when_disabled(
    tmp_path: Path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "agentic_log_dir": Path("logs/agentic"),
            "training_candidate_setup_family_reinforcement_enabled": False,
        }
    )
    store = AgenticLearningStore(settings.resolve_path(settings.agentic_log_dir))
    for idx in (1, 2):
        store.upsert(
            LearningExample(
                decision_hash=f"disabled-{idx}",
                bar_time=f"2026-01-10T10:0{idx}:00",
                bar_idx=idx,
                symbol="TCS.NS",
                timeframe="5m",
                action="BUY",
                confidence=0.7,
                current_side=None,
                bar_close=200.0,
                metadata={"primary_signal": "ORB"},
                outcome=LearningOutcome(
                    status="resolved",
                    paper_closed=True,
                    realized_pnl_pct=1.9,
                ),
            )
        )

    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="TCS.NS",
                winning_strategy="ORB",
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.76,
                    summary="BUY won",
                    reasons=("orb fired",),
                ),
                liquidity_score=12.0,
                critique=RiskCritique(
                    severity="low",
                    summary="BUY candidate with relatively clean support",
                    verdict="candidate",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.training_candidates.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )

    response = build_training_candidates(settings=settings)
    row = response.candidates[0]
    assert row.setup_family_reinforcement is None
    assert row.setup_family_verdict is None
    assert not any("Setup-family reinforcement" in text for text in row.rationale)
