from __future__ import annotations

from pathlib import Path

from fortuna.agentic.contracts import (
    DecisionSummary,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSnapshot,
    MarketUniverseCandidate,
    MarketUniverseResponse,
)
from fortuna.app.shortlist_analysis import analyze_market_shortlist, critique_analysis
from fortuna.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )


def test_critique_analysis_flags_low_confidence_buy():
    response = InstrumentAnalysisResponse(
        ok=True,
        request=InstrumentAnalysisRequest(symbol="RELIANCE.NS"),
        instrument=InstrumentSnapshot(
            requested_symbol="RELIANCE.NS",
            resolved_symbol="RELIANCE.NS",
            segment_label="EQUITY",
            timeframe="5m",
            lookback_days=30,
        ),
        decision=DecisionSummary(
            action="BUY",
            confidence=0.52,
            summary="BUY won",
            reasons=("orb fired",),
            risk_notes=("ML support missing",),
        ),
        winning_strategy="orb",
    )
    critique = critique_analysis(response)
    assert critique.severity == "medium"
    assert critique.verdict == "watch"
    assert "Low confidence advisory" in critique.concerns


def test_critique_analysis_marks_clean_entry_candidate():
    response = InstrumentAnalysisResponse(
        ok=True,
        request=InstrumentAnalysisRequest(symbol="RELIANCE.NS"),
        instrument=InstrumentSnapshot(
            requested_symbol="RELIANCE.NS",
            resolved_symbol="RELIANCE.NS",
            segment_label="EQUITY",
            timeframe="5m",
            lookback_days=30,
        ),
        decision=DecisionSummary(
            action="BUY",
            confidence=0.78,
            summary="BUY won",
            reasons=("orb fired", "rl agreed"),
            risk_notes=(),
        ),
        winning_strategy="orb",
    )
    critique = critique_analysis(response)
    assert critique.severity == "low"
    assert critique.verdict == "candidate"


def test_analyze_market_shortlist_builds_ranked_items(tmp_path: Path, monkeypatch):
    universe = MarketUniverseResponse(
        ok=True,
        source="registry",
        timeframe="1d",
        lookback_days=30,
        scout_liquidity_symbols=("RELIANCE.NS", "TCS.NS"),
        scout_activity_symbols=("RELIANCE.NS",),
        scout_volume_dense_symbols=("RELIANCE.NS",),
        scout_overlap_symbols=("RELIANCE.NS",),
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="RELIANCE",
                source="registry",
                liquidity_score=12.0,
                trend_pct=2.0,
                activity_score=1.4,
                volume_ratio=1.2,
                regime="TRENDING",
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="registry",
                liquidity_score=10.0,
                trend_pct=-1.0,
                activity_score=0.8,
                volume_ratio=0.9,
                regime="RANGING",
            ),
        ),
    )

    def _fake_universe(**kwargs):
        return universe

    def _fake_analyze(*, request, settings, engine_factory=None, registry=None):
        action = "BUY" if request.symbol == "RELIANCE.NS" else "DO_NOT_ENTER"
        confidence = 0.76 if request.symbol == "RELIANCE.NS" else 0.6
        return InstrumentAnalysisResponse(
            ok=True,
            request=request,
            instrument=InstrumentSnapshot(
                requested_symbol=request.symbol,
                resolved_symbol=request.symbol,
                segment_label="EQUITY",
                timeframe=request.timeframe,
                lookback_days=request.days,
            ),
            decision=DecisionSummary(
                action=action,
                confidence=confidence,
                summary=f"{action} won",
                reasons=("orb fired",),
                risk_notes=(),
            ),
            winning_strategy="orb",
        )

    monkeypatch.setattr("fortuna.app.shortlist_analysis.build_market_universe", _fake_universe)
    monkeypatch.setattr("fortuna.app.shortlist_analysis.analyze_instrument", _fake_analyze)
    response = analyze_market_shortlist(
        settings=_settings(tmp_path),
        universe_limit=5,
        analysis_limit=2,
        timeframe="5m",
        days=20,
    )
    assert response.ok is True
    assert len(response.items) == 2
    assert response.scout_liquidity_symbols == ("RELIANCE.NS", "TCS.NS")
    assert response.scout_activity_symbols == ("RELIANCE.NS",)
    assert response.scout_volume_dense_symbols == ("RELIANCE.NS",)
    assert response.scout_overlap_symbols == ("RELIANCE.NS",)
    assert response.items[0].symbol == "RELIANCE.NS"
    assert response.items[0].critique is not None
    assert response.items[0].critique.verdict == "candidate"
    assert response.items[0].selection_rank == 1
    assert response.items[0].priority_score is not None
    assert (
        "Discovery overlap backed by liquidity and activity scouts"
        in response.items[0].critique.supports
    )


def test_analyze_market_shortlist_applies_discovery_alignment_context(tmp_path: Path, monkeypatch):
    universe = MarketUniverseResponse(
        ok=True,
        source="registry",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="RELIANCE",
                source="registry",
                liquidity_score=12.0,
                trend_pct=2.1,
                activity_score=1.5,
                volume_ratio=1.3,
                regime="TRENDING",
            ),
            MarketUniverseCandidate(
                symbol="TCS.NS",
                display_name="TCS",
                source="registry",
                liquidity_score=11.9,
            ),
        ),
    )

    def _fake_universe(**kwargs):
        return universe

    def _fake_analyze(*, request, settings, engine_factory=None, registry=None):
        return InstrumentAnalysisResponse(
            ok=True,
            request=request,
            instrument=InstrumentSnapshot(
                requested_symbol=request.symbol,
                resolved_symbol=request.symbol,
                segment_label="EQUITY",
                timeframe=request.timeframe,
                lookback_days=request.days,
            ),
            decision=DecisionSummary(
                action="BUY",
                confidence=0.74,
                summary="BUY won",
                reasons=("orb fired",),
                risk_notes=(),
            ),
            winning_strategy="orb",
        )

    monkeypatch.setattr("fortuna.app.shortlist_analysis.build_market_universe", _fake_universe)
    monkeypatch.setattr("fortuna.app.shortlist_analysis.analyze_instrument", _fake_analyze)
    response = analyze_market_shortlist(
        settings=_settings(tmp_path),
        universe_limit=5,
        analysis_limit=2,
        timeframe="5m",
        days=20,
    )

    ranked = {row.symbol: row for row in response.items}
    assert ranked["RELIANCE.NS"].selection_rank == 1
    assert ranked["RELIANCE.NS"].priority_score > ranked["TCS.NS"].priority_score
    assert (
        "Discovery overlap backed by liquidity and activity scouts"
        in ranked["RELIANCE.NS"].critique.supports
    )
    assert (
        "Discovery support is liquidity-led without activity confirmation"
        in ranked["TCS.NS"].critique.concerns
    )


def test_analyze_market_shortlist_applies_overlap_penalty(tmp_path: Path, monkeypatch):
    universe = MarketUniverseResponse(
        ok=True,
        source="registry",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="RELIANCE",
                source="registry",
                liquidity_score=12.0,
                trend_pct=2.0,
            ),
            MarketUniverseCandidate(
                symbol="RELIANCE.FUT",
                display_name="RELIANCE FUT",
                source="registry",
                liquidity_score=11.0,
                trend_pct=1.5,
            ),
        ),
    )

    def _fake_universe(**kwargs):
        return universe

    def _fake_analyze(*, request, settings, engine_factory=None, registry=None):
        return InstrumentAnalysisResponse(
            ok=True,
            request=request,
            instrument=InstrumentSnapshot(
                requested_symbol=request.symbol,
                resolved_symbol=request.symbol,
                segment_label="EQUITY" if request.symbol.endswith(".NS") else "FUTURE",
                timeframe=request.timeframe,
                lookback_days=request.days,
            ),
            decision=DecisionSummary(
                action="BUY",
                confidence=0.82 if request.symbol.endswith(".NS") else 0.81,
                summary="BUY won",
                reasons=("orb fired",),
                risk_notes=(),
            ),
            winning_strategy="orb",
        )

    monkeypatch.setattr("fortuna.app.shortlist_analysis.build_market_universe", _fake_universe)
    monkeypatch.setattr("fortuna.app.shortlist_analysis.analyze_instrument", _fake_analyze)
    response = analyze_market_shortlist(
        settings=_settings(tmp_path),
        universe_limit=5,
        analysis_limit=2,
        timeframe="5m",
        days=20,
    )
    assert response.items[0].symbol == "RELIANCE.NS"
    assert response.items[0].exposure_penalty == 0.0
    assert response.items[1].symbol == "RELIANCE.FUT"
    assert response.items[1].exposure_penalty > 0.0


def test_analyze_market_shortlist_passes_overlay_metadata(tmp_path: Path, monkeypatch):
    universe = MarketUniverseResponse(
        ok=True,
        source="registry",
        timeframe="1d",
        lookback_days=30,
        fundamentals_overlay_summary="loaded=1 skipped=0",
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="RELIANCE",
                source="registry",
                liquidity_score=12.0,
                sector="Energy",
                market_cap_bucket="large",
                operator_quality_score=0.85,
                fundamentals_overlay_adjustment=0.035,
            ),
        ),
    )

    def _fake_universe(**kwargs):
        return universe

    def _fake_analyze(*, request, settings, engine_factory=None, registry=None):
        return InstrumentAnalysisResponse(
            ok=True,
            request=request,
            instrument=InstrumentSnapshot(
                requested_symbol=request.symbol,
                resolved_symbol=request.symbol,
                segment_label="EQUITY",
                timeframe=request.timeframe,
                lookback_days=request.days,
            ),
            decision=DecisionSummary(
                action="BUY",
                confidence=0.76,
                summary="BUY won",
                reasons=("orb fired",),
                risk_notes=(),
            ),
            winning_strategy="orb",
        )

    monkeypatch.setattr("fortuna.app.shortlist_analysis.build_market_universe", _fake_universe)
    monkeypatch.setattr("fortuna.app.shortlist_analysis.analyze_instrument", _fake_analyze)
    response = analyze_market_shortlist(
        settings=_settings(tmp_path),
        universe_limit=1,
        analysis_limit=1,
        timeframe="5m",
        days=20,
    )
    assert response.ok is True
    assert response.fundamentals_overlay_summary == "loaded=1 skipped=0"
    item = response.items[0]
    assert item.sector == "Energy"
    assert item.market_cap_bucket == "large"
    assert item.operator_quality_score == 0.85
    assert item.fundamentals_overlay_adjustment == 0.035
