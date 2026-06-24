from __future__ import annotations

from pathlib import Path

from fortuna.agentic.contracts import (
    DecisionSummary,
    RiskCritique,
    ShortlistAnalysisItem,
    ShortlistAnalysisResponse,
)
from fortuna.app.shortlist_briefing import build_shortlist_briefing, critique_portfolio
from fortuna.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )


def test_critique_portfolio_warns_on_directional_crowding():
    items = [
        ShortlistAnalysisItem(
            symbol="RELIANCE.NS",
            decision=DecisionSummary("BUY", 0.8, "BUY won"),
            critique=RiskCritique("low", "clean", verdict="candidate"),
        ),
        ShortlistAnalysisItem(
            symbol="TCS.NS",
            decision=DecisionSummary("BUY", 0.7, "BUY won"),
            critique=RiskCritique("low", "clean", verdict="candidate"),
        ),
        ShortlistAnalysisItem(
            symbol="INFY.NS",
            decision=DecisionSummary("BUY", 0.68, "BUY won"),
            critique=RiskCritique("medium", "watch", verdict="watch"),
        ),
    ]
    portfolio = critique_portfolio(items)
    assert portfolio.notes
    assert "directional crowding" in portfolio.notes[0].message


def test_critique_portfolio_warns_on_overlapping_underlying_exposure():
    items = [
        ShortlistAnalysisItem(
            symbol="RELIANCE.NS",
            decision=DecisionSummary("BUY", 0.8, "BUY won"),
            critique=RiskCritique("low", "clean", verdict="candidate"),
        ),
        ShortlistAnalysisItem(
            symbol="RELIANCE.FUT",
            decision=DecisionSummary("BUY", 0.74, "BUY won"),
            critique=RiskCritique("low", "clean", verdict="candidate"),
        ),
    ]
    portfolio = critique_portfolio(items)
    assert any("overlapping underlying exposure" in note.message for note in portfolio.notes)


def test_critique_portfolio_warns_on_overlap_with_open_positions():
    items = [
        ShortlistAnalysisItem(
            symbol="SBIN.NS",
            decision=DecisionSummary("BUY", 0.8, "BUY won"),
            critique=RiskCritique("low", "clean", verdict="candidate"),
        ),
    ]
    portfolio = critique_portfolio(
        items,
        account_summary={
            "gross_exposure": 25000.0,
            "open_positions": 1,
            "open_symbols": ("SBIN.FUT",),
            "open_position_sides": {"SBIN.FUT": "BUY"},
        },
    )
    assert any("overlaps existing same-side exposure" in note.message for note in portfolio.notes)


def test_build_shortlist_briefing_builds_headline(tmp_path: Path, monkeypatch):
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
                    confidence=0.78,
                    summary="BUY won",
                    reasons=("orb fired", "rl agreed"),
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
                decision=DecisionSummary(
                    action="DO_NOT_ENTER",
                    confidence=0.62,
                    summary="DO_NOT_ENTER won",
                    reasons=("weak setup",),
                ),
                liquidity_score=10.0,
                critique=RiskCritique(
                    severity="high",
                    summary="DO_NOT_ENTER is not an active entry candidate",
                    verdict="avoid",
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "fortuna.app.shortlist_briefing.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    response = build_shortlist_briefing(settings=_settings(tmp_path))
    assert response.ok is True
    assert "candidate setups" in response.headline
    assert response.items[0].symbol == "RELIANCE.NS"
    assert response.items[0].selection_rank == 1
    assert response.items[0].priority_score is not None


def test_build_shortlist_briefing_applies_overlap_penalty_upstream(tmp_path: Path, monkeypatch):
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
        "fortuna.app.shortlist_briefing.analyze_market_shortlist",
        lambda **kwargs: shortlist,
    )
    response = build_shortlist_briefing(settings=_settings(tmp_path))
    assert response.items[0].symbol == "RELIANCE.NS"
    assert response.items[0].exposure_penalty == 0.0
    assert response.items[1].symbol == "RELIANCE.FUT"
    assert response.items[1].exposure_penalty > 0.0
