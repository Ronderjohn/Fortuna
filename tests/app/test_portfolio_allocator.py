from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fortuna.agentic.contracts import (
    BriefingItem,
    PortfolioCritique,
    PortfolioExposureNote,
    ShortlistBriefingResponse,
    TrainingResearchPlanResponse,
    TrainingResearchRow,
)
from fortuna.agentic.learning import LearningExample, LearningOutcome
from fortuna.app.portfolio_allocator import build_portfolio_allocation
from fortuna.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
    )


def test_build_portfolio_allocation_respects_exposure_and_side_limits(tmp_path: Path, monkeypatch):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 4 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.82,
                verdict="candidate",
                summary="clean buy",
                selection_rank=1,
                priority_score=1.20,
                regime="TRENDING",
                volume_ratio=1.25,
                activity_score=0.8,
            ),
            BriefingItem(
                symbol="RELIANCE.FUT",
                action="BUY",
                confidence=0.80,
                verdict="candidate",
                summary="overlapping buy",
                selection_rank=2,
                priority_score=1.15,
                regime="TRENDING",
                volume_ratio=1.18,
                activity_score=0.7,
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="BUY",
                confidence=0.75,
                verdict="candidate",
                summary="second buy",
                selection_rank=3,
                priority_score=1.00,
                regime="RANGING",
                volume_ratio=1.02,
                activity_score=0.35,
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="SELL",
                confidence=0.72,
                verdict="watch",
                summary="watch sell",
                selection_rank=4,
                priority_score=0.85,
                regime="VOLATILE",
                volume_ratio=1.35,
                activity_score=0.9,
            ),
        ),
        portfolio=PortfolioCritique(
            summary="directional crowding risk",
            notes=(PortfolioExposureNote(level="warn", message="3 symbols lean BUY"),),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )

    response = build_portfolio_allocation(
        settings=_settings(tmp_path),
        max_positions=2,
        max_per_exposure=1,
        max_same_side=1,
    )

    assert response.ok is True
    assert response.selected_symbols == ("RELIANCE.NS", "SBIN.NS")
    assert "RELIANCE.FUT" in response.skipped_symbols
    assert "TCS.NS" in response.skipped_symbols
    selected = [row for row in response.items if row.selected]
    assert selected[0].allocation_weight is not None
    assert selected[0].regime == "TRENDING"
    assert selected[0].allocation_score is not None
    assert selected[0].sizing_hint in {"overweight", "core", "starter"}
    skipped = {row.symbol: row.reason for row in response.items if not row.selected}
    assert skipped["RELIANCE.FUT"] == "exposure limit reached for RELIANCE"
    assert skipped["TCS.NS"] == "same-side limit reached for BUY"


def test_build_portfolio_allocation_uses_open_positions_and_recent_learning(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.83,
                verdict="candidate",
                summary="clean buy",
                selection_rank=1,
                priority_score=1.20,
                regime="TRENDING",
                volume_ratio=1.1,
                activity_score=0.5,
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.80,
                verdict="candidate",
                summary="second buy",
                selection_rank=2,
                priority_score=1.05,
                regime="TRENDING",
                volume_ratio=1.3,
                activity_score=0.95,
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.78,
                verdict="candidate",
                summary="clean sell",
                selection_rank=3,
                priority_score=1.00,
                regime="RANGING",
                volume_ratio=1.05,
                activity_score=0.4,
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )

    class FakeStore:
        def read_all(self, limit=None):
            return [
                LearningExample(
                    decision_hash="l1",
                    bar_time="2026-06-02T10:00:00",
                    bar_idx=1,
                    symbol="SBIN.NS",
                    timeframe="5m",
                    action="BUY",
                    confidence=0.7,
                    current_side=None,
                    bar_close=100.0,
                    outcome=LearningOutcome(
                        status="resolved",
                        paper_closed=True,
                        realized_pnl_pct=1.8,
                    ),
                ),
                LearningExample(
                    decision_hash="l2",
                    bar_time="2026-06-02T10:05:00",
                    bar_idx=2,
                    symbol="RELIANCE.NS",
                    timeframe="5m",
                    action="BUY",
                    confidence=0.7,
                    current_side=None,
                    bar_close=200.0,
                    outcome=LearningOutcome(
                        status="resolved",
                        paper_closed=True,
                        realized_pnl_pct=-1.6,
                    ),
                ),
            ]

    class FakeAccount:
        def __init__(self):
            self.positions = {
                "RELIANCE.NS": SimpleNamespace(
                    side=SimpleNamespace(value="BUY"),
                )
            }

    engine = SimpleNamespace(
        live_account=FakeAccount(),
        _agentic_learning_store=FakeStore(),
    )

    response = build_portfolio_allocation(
        settings=_settings(tmp_path),
        engine_factory=lambda: engine,
        max_positions=2,
        max_per_exposure=1,
        max_same_side=2,
    )

    assert response.ok is True
    assert response.selected_symbols == ("SBIN.NS", "TCS.NS")
    skipped = {row.symbol: row.reason for row in response.items if not row.selected}
    assert skipped["RELIANCE.NS"].startswith("exposure limit reached for RELIANCE")
    assert "recent paper outcomes imply a penalty" in skipped["RELIANCE.NS"]
    assert "existing_open_positions=1" in response.notes
    assert any("RELIANCE" in note and "avg -1.60%" in note for note in response.notes)
    assert any("selected_regimes" in note for note in response.notes)


def test_build_portfolio_allocation_applies_context_weighting_and_weight_cap(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.87,
                verdict="candidate",
                summary="strong trend buy",
                selection_rank=1,
                priority_score=1.25,
                regime="TRENDING",
                volume_ratio=1.4,
                activity_score=1.1,
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.81,
                verdict="candidate",
                summary="second trend buy",
                selection_rank=2,
                priority_score=1.08,
                regime="TRENDING",
                volume_ratio=1.2,
                activity_score=0.8,
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.76,
                verdict="candidate",
                summary="ranging sell",
                selection_rank=3,
                priority_score=0.98,
                regime="RANGING",
                volume_ratio=1.01,
                activity_score=0.3,
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    settings = _settings(tmp_path).model_copy(
        update={"portfolio_allocator_max_single_weight": 0.55}
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=1,
        max_same_side=3,
    )

    selected = [row for row in response.items if row.selected]
    assert response.ok is True
    assert selected[0].symbol == "RELIANCE.NS"
    assert selected[0].allocation_score > selected[2].allocation_score
    assert max(float(row.allocation_weight or 0.0) for row in selected) <= 0.55
    assert selected[0].risk_bucket == "low"
    assert any(row.sizing_hint for row in selected)


def test_build_portfolio_allocation_caps_high_risk_and_regime_concentration(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="4 candidate setups, 4 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.93,
                verdict="candidate",
                summary="Trending leader",
                selection_rank=1,
                priority_score=1.4,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="INFY.NS",
                action="BUY",
                confidence=0.9,
                verdict="candidate",
                summary="Second trending leader",
                selection_rank=2,
                priority_score=1.3,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="BUY",
                confidence=0.74,
                verdict="candidate",
                summary="Volatile long",
                selection_rank=3,
                priority_score=1.15,
                regime="VOLATILE",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="SELL",
                confidence=0.71,
                verdict="candidate",
                summary="Second volatile setup",
                selection_rank=4,
                priority_score=1.1,
                regime="VOLATILE",
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_max_per_regime": 1,
            "portfolio_allocator_max_high_risk_positions": 1,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=4,
        max_per_exposure=1,
        max_same_side=4,
    )

    assert response.ok is True
    assert response.selected_symbols == ("RELIANCE.NS", "TCS.NS")
    skipped = {row.symbol: row.reason for row in response.items if not row.selected}
    assert skipped["INFY.NS"] == "regime limit reached for TRENDING"
    assert skipped["SBIN.NS"] == "high-risk limit reached (1)"
    assert "max_high_risk_positions=1" in response.notes
    assert "max_per_regime=1" in response.notes


def test_build_portfolio_allocation_portfolio_critic_blocks_redundant_weak_setup(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.86,
                verdict="candidate",
                summary="Primary trend buy",
                selection_rank=1,
                priority_score=1.20,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.68,
                verdict="watch",
                summary="Redundant weaker trend buy",
                selection_rank=2,
                priority_score=1.05,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.8,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=3,
                priority_score=1.0,
                regime="RANGING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_max_per_regime": 3,
            "portfolio_allocator_max_high_risk_positions": 3,
            "portfolio_allocator_critic_block_threshold": 0.12,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    assert response.selected_symbols == ("RELIANCE.NS", "TCS.NS")
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert skipped["SBIN.NS"].critic_penalty >= 0.12
    assert skipped["SBIN.NS"].critic_note.startswith("portfolio critic:")
    assert "same-side crowding" in skipped["SBIN.NS"].reason
    assert "portfolio_critic=enabled" in response.notes


def test_build_portfolio_allocation_uses_training_research_alignment(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.86,
                verdict="candidate",
                summary="Strong buy",
                selection_rank=1,
                priority_score=1.24,
                regime="TRENDING",
                volume_ratio=1.3,
                activity_score=0.9,
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.84,
                verdict="candidate",
                summary="Secondary buy",
                selection_rank=2,
                priority_score=1.02,
                regime="VOLATILE",
                volume_ratio=1.08,
                activity_score=0.35,
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.79,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=3,
                priority_score=0.98,
                regime="RANGING",
                volume_ratio=1.02,
                activity_score=0.25,
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=True,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            selection_policy="diversified",
            refresh_requested=False,
            refresh_target="all",
            refresh_timeframe="5m",
            refresh_lookback_days=30,
            universe_symbols=("RELIANCE.NS", "SBIN.NS", "TCS.NS"),
            ml_symbols=("RELIANCE.NS",),
            rl_symbols=("RELIANCE.NS", "TCS.NS"),
            rows=(
                TrainingResearchRow(
                    symbol="RELIANCE.NS",
                    target="both",
                    universe_rank=1,
                    shortlist_rank=1,
                    selection_rank=1,
                    liquidity_score=13.0,
                    decision_action="BUY",
                    critique_verdict="candidate",
                    winning_strategy="trend_follow",
                    market_regime="TRENDING",
                    volume_ratio=1.3,
                    priority_score=1.31,
                    adaptive_score_adjustment=0.05,
                    refreshed=True,
                    rationale=("high liquidity",),
                ),
                TrainingResearchRow(
                    symbol="TCS.NS",
                    target="rl",
                    universe_rank=2,
                    shortlist_rank=3,
                    selection_rank=2,
                    liquidity_score=10.1,
                    decision_action="SELL",
                    critique_verdict="candidate",
                    winning_strategy="mean_reversion",
                    market_regime="RANGING",
                    volume_ratio=1.02,
                    priority_score=1.02,
                    adaptive_score_adjustment=0.0,
                    refreshed=False,
                    rationale=("balanced hedge",),
                ),
            ),
        ),
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_research_alignment_enabled": True,
            "portfolio_allocator_critic_block_threshold": 0.12,
            "portfolio_allocator_critic_missing_research_penalty": 0.05,
            "portfolio_allocator_critic_weaker_research_penalty": 0.03,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=2,
        max_per_exposure=2,
        max_same_side=2,
    )

    assert response.ok is True
    assert response.selected_symbols == ("RELIANCE.NS", "TCS.NS")
    selected = {row.symbol: row for row in response.items if row.selected}
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert selected["RELIANCE.NS"].research_target == "both"
    assert selected["RELIANCE.NS"].research_priority_score == 1.31
    assert selected["RELIANCE.NS"].research_refreshed is True
    assert skipped["SBIN.NS"].research_target == ""
    assert skipped["SBIN.NS"].critic_penalty >= 0.15
    assert "no current ML/RL research support" in skipped["SBIN.NS"].critic_note
    assert "portfolio_research_alignment=enabled" in response.notes
    assert any("research_plan_targets:" in note for note in response.notes)
    assert any("research_plan_refreshed_rows=1" in note for note in response.notes)
    assert any("allocation_refreshed_research_targets:" in note for note in response.notes)


def test_build_portfolio_allocation_penalizes_plan_only_support_against_refreshed_side(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.87,
                verdict="candidate",
                summary="Lead buy",
                selection_rank=1,
                priority_score=1.28,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.85,
                verdict="candidate",
                summary="Follow-on buy",
                selection_rank=2,
                priority_score=1.1,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.8,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=3,
                priority_score=1.02,
                regime="RANGING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=True,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            selection_policy="diversified",
            refresh_requested=True,
            refresh_target="all",
            refresh_timeframe="5m",
            refresh_lookback_days=30,
            universe_symbols=("RELIANCE.NS", "SBIN.NS", "TCS.NS"),
            ml_symbols=("RELIANCE.NS",),
            rl_symbols=("RELIANCE.NS", "SBIN.NS", "TCS.NS"),
            rows=(
                TrainingResearchRow(
                    symbol="RELIANCE.NS",
                    target="both",
                    universe_rank=1,
                    shortlist_rank=1,
                    selection_rank=1,
                    priority_score=1.3,
                    refreshed=True,
                ),
                TrainingResearchRow(
                    symbol="SBIN.NS",
                    target="both",
                    universe_rank=2,
                    shortlist_rank=2,
                    selection_rank=2,
                    priority_score=1.12,
                    refreshed=False,
                ),
                TrainingResearchRow(
                    symbol="TCS.NS",
                    target="rl",
                    universe_rank=3,
                    shortlist_rank=3,
                    selection_rank=3,
                    priority_score=1.0,
                    refreshed=True,
                ),
            ),
        ),
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_research_alignment_enabled": True,
            "portfolio_allocator_critic_same_side_penalty": 0.05,
            "portfolio_allocator_critic_same_regime_penalty": 0.04,
            "portfolio_allocator_critic_unrefreshed_research_penalty": 0.025,
            "portfolio_allocator_critic_block_threshold": 0.2,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    selected = {row.symbol: row for row in response.items if row.selected}
    assert selected["RELIANCE.NS"].research_refreshed is True
    assert selected["SBIN.NS"].research_refreshed is False
    assert selected["SBIN.NS"].critic_penalty >= 0.075
    assert "plan-only research support versus refreshed current BUY basket" in (
        selected["SBIN.NS"].critic_note
    )


def test_build_portfolio_allocation_uses_effective_research_target_focus(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.87,
                verdict="candidate",
                summary="Lead buy",
                selection_rank=1,
                priority_score=1.28,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.85,
                verdict="candidate",
                summary="Follow-on buy",
                selection_rank=2,
                priority_score=1.1,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.8,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=3,
                priority_score=1.02,
                regime="RANGING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=True,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            selection_policy="diversified",
            refresh_requested=True,
            refresh_target="all",
            discovery_recommended_refresh_target="ml",
            refresh_timeframe="5m",
            refresh_lookback_days=30,
            universe_symbols=("RELIANCE.NS", "SBIN.NS", "TCS.NS"),
            ml_symbols=("RELIANCE.NS", "TCS.NS"),
            rl_symbols=("RELIANCE.NS", "SBIN.NS"),
            rows=(
                TrainingResearchRow(
                    symbol="RELIANCE.NS",
                    target="both",
                    universe_rank=1,
                    shortlist_rank=1,
                    selection_rank=1,
                    priority_score=1.3,
                    refreshed=True,
                ),
                TrainingResearchRow(
                    symbol="SBIN.NS",
                    target="rl",
                    universe_rank=2,
                    shortlist_rank=2,
                    selection_rank=2,
                    priority_score=1.12,
                    refreshed=False,
                ),
                TrainingResearchRow(
                    symbol="TCS.NS",
                    target="ml",
                    universe_rank=3,
                    shortlist_rank=3,
                    selection_rank=3,
                    priority_score=1.0,
                    refreshed=True,
                ),
            ),
        ),
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_research_alignment_enabled": True,
            "portfolio_allocator_critic_missing_research_penalty": 0.05,
            "portfolio_allocator_critic_block_threshold": 0.12,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=2,
        max_per_exposure=2,
        max_same_side=2,
    )

    assert response.ok is True
    selected = {row.symbol: row for row in response.items if row.selected}
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert selected["RELIANCE.NS"].research_target == "ml"
    assert selected["TCS.NS"].research_target == "ml"
    assert skipped["SBIN.NS"].research_target == ""
    assert "no current ML/RL research support" in skipped["SBIN.NS"].critic_note
    assert "research_plan_discovery_target=ml" in response.notes
    assert "research_plan_effective_target=ml" in response.notes
    assert any("allocation_research_targets: ml=2" in note for note in response.notes)


def test_build_portfolio_allocation_uses_recent_nightly_research_evidence(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.88,
                verdict="candidate",
                summary="Lead buy",
                selection_rank=1,
                priority_score=1.26,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.86,
                verdict="candidate",
                summary="Follow-on buy",
                selection_rank=2,
                priority_score=1.14,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.81,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=3,
                priority_score=1.02,
                regime="RANGING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=True,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            rows=(
                TrainingResearchRow(
                    symbol="RELIANCE.NS",
                    target="both",
                    universe_rank=1,
                    shortlist_rank=1,
                    selection_rank=1,
                    priority_score=1.32,
                    refreshed=True,
                ),
                TrainingResearchRow(
                    symbol="SBIN.NS",
                    target="both",
                    universe_rank=2,
                    shortlist_rank=2,
                    selection_rank=2,
                    priority_score=1.18,
                    refreshed=True,
                ),
                TrainingResearchRow(
                    symbol="TCS.NS",
                    target="rl",
                    universe_rank=3,
                    shortlist_rank=3,
                    selection_rank=3,
                    priority_score=1.01,
                    refreshed=False,
                ),
            ),
        ),
    )

    nightly_dir = tmp_path / "reports" / "nightly"
    nightly_dir.mkdir(parents=True, exist_ok=True)
    training_research_path = nightly_dir / "training_research_plan.json"
    training_research_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"symbol": "RELIANCE.NS", "target": "both", "refreshed": True},
                    {"symbol": "TCS.NS", "target": "rl", "refreshed": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    (nightly_dir / "20260603_0100.json").write_text(
        json.dumps(
            {
                "overall_status": "ok",
                "basket": ["RELIANCE.NS", "TCS.NS"],
                "steps": [
                    {
                        "name": "training_research",
                        "detail": {"path": "training_research_plan.json", "count": 2},
                    },
                    {
                        "name": "training_execution_target",
                        "detail": {
                            "target": "rl",
                            "selection_source": "training_research",
                        },
                    },
                    {"name": "promote_best", "detail": {"count": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_research_alignment_enabled": True,
            "portfolio_allocator_nightly_feedback_enabled": True,
            "portfolio_allocator_nightly_report_dir": nightly_dir,
            "portfolio_allocator_critic_same_side_penalty": 0.05,
            "portfolio_allocator_critic_same_regime_penalty": 0.04,
            "portfolio_allocator_critic_weaker_nightly_research_penalty": 0.04,
            "portfolio_allocator_critic_block_threshold": 0.13,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    selected = {row.symbol: row for row in response.items if row.selected}
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert response.selected_symbols == ("RELIANCE.NS", "TCS.NS")
    assert selected["RELIANCE.NS"].research_nightly_reports == 1
    assert selected["RELIANCE.NS"].research_nightly_refreshed_reports == 1
    assert selected["RELIANCE.NS"].research_nightly_promoted_reports == 1
    assert selected["RELIANCE.NS"].research_nightly_score is not None
    assert skipped["SBIN.NS"].critic_penalty >= 0.13
    assert "weaker recurring nightly research evidence than current BUY basket" in (
        skipped["SBIN.NS"].critic_note
    )
    assert any("research_nightly_reports=1" in note for note in response.notes)
    assert any("research_nightly_executed_symbols=" in note for note in response.notes)
    assert any("research_nightly_executed_targets:" in note for note in response.notes)
    assert any("allocation_nightly_research_targets:" in note for note in response.notes)
    assert any("allocation_nightly_promoted_research_targets:" in note for note in response.notes)


def test_build_portfolio_allocation_penalizes_weaker_discovery_alignment(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.87,
                verdict="candidate",
                summary="Lead buy",
                selection_rank=1,
                priority_score=1.28,
                liquidity_score=12.0,
                regime="TRENDING",
                volume_ratio=1.30,
                activity_score=1.10,
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.85,
                verdict="candidate",
                summary="Follow-on buy",
                selection_rank=2,
                priority_score=1.12,
                liquidity_score=11.8,
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.8,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=3,
                priority_score=1.01,
                liquidity_score=10.5,
                regime="RANGING",
                volume_ratio=1.05,
                activity_score=0.45,
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_discovery_alignment_enabled": True,
            "portfolio_allocator_critic_same_side_penalty": 0.05,
            "portfolio_allocator_critic_weaker_discovery_penalty": 0.03,
            "portfolio_allocator_critic_block_threshold": 0.08,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    selected = {row.symbol: row for row in response.items if row.selected}
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert response.selected_symbols == ("RELIANCE.NS", "TCS.NS")
    assert skipped["SBIN.NS"].critic_penalty >= 0.08
    assert "weaker discovery alignment than current BUY basket" in skipped["SBIN.NS"].critic_note
    assert "portfolio_discovery_alignment=enabled" in response.notes
    assert any("allocation_discovery_overlap_symbols:" in note for note in response.notes)
    assert selected["RELIANCE.NS"].critic_penalty == 0.0


def test_build_portfolio_allocation_penalizes_duplicate_same_side_strategy_family(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.88,
                verdict="candidate",
                summary="Primary ORB candidate",
                selection_rank=1,
                priority_score=1.34,
                winning_strategy="ORB",
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="BUY",
                confidence=0.84,
                verdict="candidate",
                summary="Diversifying MMTS candidate",
                selection_rank=2,
                priority_score=1.27,
                winning_strategy="MMTS",
                regime="MOMENTUM",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.8,
                verdict="candidate",
                summary="Weaker follow-on ORB candidate",
                selection_rank=3,
                priority_score=1.14,
                winning_strategy="ORB",
                regime="BREAKOUT",
            ),
        ),
        portfolio=PortfolioCritique(summary="Balanced enough", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_critic_same_side_penalty": 0.0,
            "portfolio_allocator_critic_same_regime_penalty": 0.0,
            "portfolio_allocator_critic_same_strategy_penalty": 0.04,
            "portfolio_allocator_critic_watch_penalty": 0.0,
            "portfolio_allocator_critic_high_risk_penalty": 0.0,
            "portfolio_allocator_critic_block_threshold": 0.03,
            "portfolio_allocator_discovery_alignment_enabled": False,
            "portfolio_allocator_research_alignment_enabled": False,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    assert response.selected_symbols == ("RELIANCE.NS", "TCS.NS")
    selected = {row.symbol: row for row in response.items if row.selected}
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert selected["RELIANCE.NS"].winning_strategy == "ORB"
    assert selected["TCS.NS"].winning_strategy == "MMTS"
    assert skipped["SBIN.NS"].winning_strategy == "ORB"
    assert skipped["SBIN.NS"].critic_penalty >= 0.04
    assert "duplicate setup family `ORB`" in skipped["SBIN.NS"].critic_note
    assert any("allocation_strategy_mix: MMTS=1, ORB=1" in note for note in response.notes)


def test_build_portfolio_allocation_penalizes_duplicate_setup_family_case_insensitively(
    tmp_path: Path,
):
    settings = _settings(tmp_path)
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="unit",
        timeframe="5m",
        lookback_days=20,
        headline="2 candidates",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.82,
                verdict="candidate",
                summary="Lead orb setup",
                selection_rank=1,
                priority_score=1.34,
                winning_strategy="orb",
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.71,
                verdict="candidate",
                summary="Weaker ORB follow-on",
                selection_rank=2,
                priority_score=1.12,
                winning_strategy="ORB",
                regime="BREAKOUT",
            ),
        ),
        portfolio=PortfolioCritique(summary="Balanced enough", notes=()),
    )

    with patch(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        return_value=briefing,
    ):
        response = build_portfolio_allocation(
            settings=settings,
            engine_factory=None,
            registry=None,
            max_positions=2,
            max_same_side=2,
        )

    selected = {row.symbol: row for row in response.items if row.selected}
    skipped = {row.symbol: row for row in response.items if not row.selected}

    assert selected["RELIANCE.NS"].winning_strategy == "orb"
    assert skipped["SBIN.NS"].critic_penalty >= 0.04
    assert "duplicate setup family `ORB`" in skipped["SBIN.NS"].critic_note
    assert any("allocation_strategy_mix: ORB=1" in note for note in response.notes)


def test_build_portfolio_allocation_penalizes_same_side_open_exposure_before_limit(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="2 candidate setups, 2 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.FUT",
                action="BUY",
                confidence=0.82,
                verdict="candidate",
                summary="Futures add-on buy",
                selection_rank=1,
                priority_score=1.18,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.78,
                verdict="candidate",
                summary="Offsetting sell candidate",
                selection_rank=2,
                priority_score=1.06,
                regime="RANGING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )

    class FakeAccount:
        def __init__(self):
            self.positions = {
                "RELIANCE.NS": SimpleNamespace(
                    side=SimpleNamespace(value="BUY"),
                )
            }

    engine = SimpleNamespace(live_account=FakeAccount(), _agentic_learning_store=None)
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_critic_same_side_penalty": 0.0,
            "portfolio_allocator_critic_same_regime_penalty": 0.0,
            "portfolio_allocator_critic_same_strategy_penalty": 0.0,
            "portfolio_allocator_critic_watch_penalty": 0.0,
            "portfolio_allocator_critic_high_risk_penalty": 0.0,
            "portfolio_allocator_critic_open_same_exposure_penalty": 0.04,
            "portfolio_allocator_critic_block_threshold": 0.03,
            "portfolio_allocator_discovery_alignment_enabled": False,
            "portfolio_allocator_research_alignment_enabled": False,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        engine_factory=lambda: engine,
        max_positions=2,
        max_per_exposure=2,
        max_same_side=2,
    )

    assert response.selected_symbols == ("TCS.NS",)
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert skipped["RELIANCE.FUT"].critic_penalty >= 0.04
    assert "same-side live exposure in RELIANCE" in skipped["RELIANCE.FUT"].critic_note
    assert any("open_side_exposures: BUY:RELIANCE=1" in note for note in response.notes)


def test_build_portfolio_allocation_penalizes_weaker_same_side_conviction(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.89,
                verdict="candidate",
                summary="Lead buy",
                selection_rank=1,
                priority_score=1.32,
                winning_strategy="ORB",
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.83,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=2,
                priority_score=1.2,
                winning_strategy="MMTS",
                regime="RANGING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.8,
                verdict="candidate",
                summary="Weaker follow-on buy",
                selection_rank=3,
                priority_score=1.1,
                winning_strategy="VWAP",
                regime="TRENDING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_critic_enabled": True,
            "portfolio_allocator_critic_same_side_penalty": 0.05,
            "portfolio_allocator_critic_weaker_same_side_penalty": 0.09,
            "portfolio_allocator_critic_block_threshold": 0.12,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    assert response.selected_symbols == ("RELIANCE.NS", "TCS.NS")
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert skipped["SBIN.NS"].critic_penalty >= 0.14
    assert "weaker conviction than current BUY basket" in skipped["SBIN.NS"].critic_note


def test_build_portfolio_allocation_critic_disabled_skips_penalties_and_blocks(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="2 candidate setups, 2 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.88,
                verdict="candidate",
                summary="Lead buy",
                selection_rank=1,
                priority_score=1.32,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.68,
                verdict="watch",
                summary="Weaker follow-on buy",
                selection_rank=2,
                priority_score=1.05,
                regime="TRENDING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    settings = _settings(tmp_path).model_copy(
        update={"portfolio_allocator_critic_enabled": False}
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=2,
        max_per_exposure=2,
        max_same_side=2,
    )

    assert response.selected_symbols == ("RELIANCE.NS", "SBIN.NS")
    rows = {row.symbol: row for row in response.items}
    assert rows["RELIANCE.NS"].critic_penalty == 0.0
    assert rows["SBIN.NS"].critic_penalty == 0.0
    assert rows["SBIN.NS"].critic_note == ""
    assert "portfolio_critic=enabled" not in response.notes


def test_build_portfolio_allocation_penalizes_open_same_regime_live_exposure(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="2 candidate setups, 2 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.84,
                verdict="candidate",
                summary="Existing open position",
                selection_rank=1,
                priority_score=1.2,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.82,
                verdict="candidate",
                summary="Follow-on same-regime buy",
                selection_rank=2,
                priority_score=1.15,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.8,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=3,
                priority_score=1.05,
                regime="RANGING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )

    class FakeAccount:
        def __init__(self):
            self.positions = {
                "RELIANCE.NS": SimpleNamespace(
                    side=SimpleNamespace(value="BUY"),
                )
            }

    engine = SimpleNamespace(live_account=FakeAccount(), _agentic_learning_store=None)
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_critic_same_side_penalty": 0.0,
            "portfolio_allocator_critic_same_strategy_penalty": 0.0,
            "portfolio_allocator_critic_watch_penalty": 0.0,
            "portfolio_allocator_critic_high_risk_penalty": 0.0,
            "portfolio_allocator_critic_same_regime_penalty": 0.05,
            "portfolio_allocator_critic_block_threshold": 0.04,
            "portfolio_allocator_discovery_alignment_enabled": False,
            "portfolio_allocator_research_alignment_enabled": False,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        engine_factory=lambda: engine,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    assert response.selected_symbols == ("TCS.NS",)
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert skipped["SBIN.NS"].critic_penalty >= 0.05
    assert "same-regime live exposure (TRENDING)" in skipped["SBIN.NS"].critic_note
    assert any("open_regime_side_exposures: BUY:TRENDING=1" in note for note in response.notes)


def test_build_portfolio_allocation_matches_open_regime_exposure_by_underlying_symbol(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="2 candidate setups, 2 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.FUT",
                action="BUY",
                confidence=0.84,
                verdict="candidate",
                summary="Existing open futures position",
                selection_rank=1,
                priority_score=1.2,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.82,
                verdict="candidate",
                summary="Follow-on same-regime buy",
                selection_rank=2,
                priority_score=1.15,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.8,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=3,
                priority_score=1.05,
                regime="RANGING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )

    class FakeAccount:
        def __init__(self):
            self.positions = {
                "RELIANCE.NS": SimpleNamespace(
                    side=SimpleNamespace(value="BUY"),
                )
            }

    engine = SimpleNamespace(live_account=FakeAccount(), _agentic_learning_store=None)
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_critic_same_side_penalty": 0.0,
            "portfolio_allocator_critic_same_strategy_penalty": 0.0,
            "portfolio_allocator_critic_watch_penalty": 0.0,
            "portfolio_allocator_critic_high_risk_penalty": 0.0,
            "portfolio_allocator_critic_same_regime_penalty": 0.05,
            "portfolio_allocator_critic_block_threshold": 0.04,
            "portfolio_allocator_discovery_alignment_enabled": False,
            "portfolio_allocator_research_alignment_enabled": False,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        engine_factory=lambda: engine,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    assert response.selected_symbols == ("TCS.NS",)
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert skipped["SBIN.NS"].critic_penalty >= 0.05
    assert "same-regime live exposure (TRENDING)" in skipped["SBIN.NS"].critic_note
    assert any("open_regime_side_exposures: BUY:TRENDING=1" in note for note in response.notes)


def test_build_portfolio_allocation_blocks_combined_same_side_and_same_regime_weaker_setup(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=20,
        headline="3 candidate setups, 3 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.9,
                verdict="candidate",
                summary="Lead trending buy",
                selection_rank=1,
                priority_score=1.4,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="SELL",
                confidence=0.83,
                verdict="candidate",
                summary="Balancing sell",
                selection_rank=2,
                priority_score=1.2,
                regime="RANGING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.78,
                verdict="candidate",
                summary="Weaker same-regime follow-on",
                selection_rank=3,
                priority_score=1.08,
                regime="TRENDING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_critic_same_side_penalty": 0.05,
            "portfolio_allocator_critic_same_regime_penalty": 0.04,
            "portfolio_allocator_critic_weaker_same_side_penalty": 0.03,
            "portfolio_allocator_critic_block_threshold": 0.11,
            "portfolio_allocator_discovery_alignment_enabled": False,
            "portfolio_allocator_research_alignment_enabled": False,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    assert response.selected_symbols == ("RELIANCE.NS", "TCS.NS")
    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert skipped["SBIN.NS"].critic_penalty >= 0.11
    assert "same-side crowding" in skipped["SBIN.NS"].critic_note
    assert "same-regime concentration (TRENDING)" in skipped["SBIN.NS"].critic_note
    assert "weaker conviction than current BUY basket" in skipped["SBIN.NS"].critic_note


def test_build_portfolio_allocation_freshness_preference_reorders_selection(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="2 candidate setups",
        items=(
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.84,
                verdict="candidate",
                summary="Higher base score",
                selection_rank=1,
                priority_score=1.2,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.86,
                verdict="candidate",
                summary="Refreshed research support",
                selection_rank=2,
                priority_score=1.15,
                regime="TRENDING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=True,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            selection_policy="diversified",
            refresh_requested=True,
            refresh_target="all",
            refresh_timeframe="5m",
            refresh_lookback_days=30,
            universe_symbols=("SBIN.NS", "RELIANCE.NS"),
            ml_symbols=("RELIANCE.NS",),
            rl_symbols=("SBIN.NS", "RELIANCE.NS"),
            rows=(
                TrainingResearchRow(
                    symbol="SBIN.NS",
                    target="rl",
                    universe_rank=1,
                    shortlist_rank=1,
                    selection_rank=1,
                    priority_score=1.2,
                    refreshed=False,
                ),
                TrainingResearchRow(
                    symbol="RELIANCE.NS",
                    target="both",
                    universe_rank=2,
                    shortlist_rank=2,
                    selection_rank=2,
                    priority_score=1.15,
                    refreshed=True,
                ),
            ),
        ),
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_research_alignment_enabled": True,
            "portfolio_allocator_freshness_preference_enabled": True,
            "portfolio_allocator_freshness_preference_weight": 0.06,
            "portfolio_allocator_critic_enabled": False,
        }
    )

    without_freshness = build_portfolio_allocation(
        settings=settings.model_copy(
            update={"portfolio_allocator_freshness_preference_enabled": False}
        ),
        max_positions=1,
        max_per_exposure=2,
        max_same_side=2,
    )
    with_freshness = build_portfolio_allocation(
        settings=settings,
        max_positions=1,
        max_per_exposure=2,
        max_same_side=2,
    )

    assert without_freshness.selected_symbols == ("SBIN.NS",)
    assert with_freshness.selected_symbols == ("RELIANCE.NS",)


def test_build_portfolio_allocation_same_target_balance_penalizes_stale_crowding(
    tmp_path: Path,
    monkeypatch,
):
    briefing = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="3 candidate setups",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.9,
                verdict="candidate",
                summary="Lead rl buy",
                selection_rank=1,
                priority_score=1.3,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="SBIN.NS",
                action="BUY",
                confidence=0.88,
                verdict="candidate",
                summary="Second rl buy",
                selection_rank=2,
                priority_score=1.22,
                regime="TRENDING",
            ),
            BriefingItem(
                symbol="TCS.NS",
                action="BUY",
                confidence=0.8,
                verdict="candidate",
                summary="Stale rl crowding",
                selection_rank=3,
                priority_score=1.18,
                regime="TRENDING",
            ),
        ),
        portfolio=PortfolioCritique(summary="balanced", notes=()),
    )
    monkeypatch.setattr(
        "fortuna.app.portfolio_allocator.build_shortlist_briefing",
        lambda **kwargs: briefing,
    )
    monkeypatch.setattr(
        "fortuna.app.training_research.build_training_research_plan",
        lambda **kwargs: TrainingResearchPlanResponse(
            ok=True,
            source="registry",
            timeframe="5m",
            lookback_days=30,
            selection_policy="diversified",
            refresh_requested=True,
            refresh_target="rl",
            refresh_timeframe="5m",
            refresh_lookback_days=30,
            universe_symbols=("RELIANCE.NS", "SBIN.NS", "TCS.NS"),
            ml_symbols=(),
            rl_symbols=("RELIANCE.NS", "SBIN.NS", "TCS.NS"),
            rows=(
                TrainingResearchRow(
                    symbol="RELIANCE.NS",
                    target="rl",
                    universe_rank=1,
                    shortlist_rank=1,
                    selection_rank=1,
                    priority_score=1.3,
                    refreshed=True,
                ),
                TrainingResearchRow(
                    symbol="SBIN.NS",
                    target="rl",
                    universe_rank=2,
                    shortlist_rank=2,
                    selection_rank=2,
                    priority_score=1.22,
                    refreshed=True,
                ),
                TrainingResearchRow(
                    symbol="TCS.NS",
                    target="rl",
                    universe_rank=3,
                    shortlist_rank=3,
                    selection_rank=3,
                    priority_score=1.18,
                    refreshed=False,
                ),
            ),
        ),
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "portfolio_allocator_research_alignment_enabled": True,
            "portfolio_allocator_same_target_balance_enabled": True,
            "portfolio_allocator_same_target_balance_penalty": 0.05,
            "portfolio_allocator_critic_same_side_penalty": 0.0,
            "portfolio_allocator_critic_same_regime_penalty": 0.0,
            "portfolio_allocator_critic_block_threshold": 0.04,
        }
    )

    response = build_portfolio_allocation(
        settings=settings,
        max_positions=3,
        max_per_exposure=3,
        max_same_side=3,
    )

    skipped = {row.symbol: row for row in response.items if not row.selected}
    assert response.selected_symbols == ("RELIANCE.NS", "SBIN.NS")
    assert skipped["TCS.NS"].critic_penalty >= 0.04
    assert "same RL research target crowding" in skipped["TCS.NS"].critic_note
    assert any(
        note.startswith("allocation_freshness_mix:")
        for note in response.notes
    )
    assert any(
        note.startswith("allocation_target_balance_summary: concentrated=rl")
        for note in response.notes
    )
