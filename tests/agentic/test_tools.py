from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from fortuna.agentic import AgenticLearningStore, LearningExample, LearningOutcome
from fortuna.agentic.contracts import (
    AdvisoryErrorCode,
    BriefingItem,
    MarketUniverseCandidate,
    MarketUniverseResponse,
    MultiAgentWorkflowResponse,
    ShortlistAnalysisItem,
    ShortlistAnalysisResponse,
    ShortlistBriefingResponse,
    TrainingCandidate,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
)
from fortuna.agentic.models import ActionRecommendation, AgentDecision, DecisionRationale
from fortuna.agentic.tools import FortunaAdvisoryTools
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRef, SymbolSearchHit


@dataclass
class _Winner:
    strategy_name: str


@dataclass
class _Batch:
    winner: _Winner


@dataclass
class _State:
    ohlcv: pd.DataFrame
    last_bar_time: pd.Timestamp
    batch: _Batch
    load_error: str | None = None


class _FakeEngine:
    def __init__(self, *, learning_store=None):
        idx = pd.date_range("2026-01-06 09:15", periods=5, freq="5min")
        self.loaded_symbol = ""
        self._state = _State(
            ohlcv=pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=idx),
            last_bar_time=idx[-1],
            batch=_Batch(winner=_Winner(strategy_name="orb")),
        )
        self._decisions = {
            "RELIANCE.NS": AgentDecision(
                symbol="RELIANCE.NS",
                action=ActionRecommendation.BUY,
                confidence=0.78,
                bar_time=idx[-1].to_pydatetime(),
                bar_close=104.0,
                rationale=DecisionRationale(summary="BUY won"),
            ),
        }
        self._agentic_learning_store = learning_store
        self.settings = None
        self._agentic_store = None
        self.rl_generator = None
        self.state = SimpleNamespace(symbol="RELIANCE.NS")
        self._agentic_orchestrator = None

    def load_symbol(self, symbol, timeframe="5m", days=30, force_refresh=False):
        self.loaded_symbol = symbol
        return self._state

    def agent_decisions(self):
        return dict(self._decisions)

    def live_signals(self):
        return {}


class _FakeRegistry:
    def ensure_loaded(self, force_refresh=False):
        return None

    def resolve(self, symbol: str):
        sym = symbol.upper()
        if sym in {"RELIANCE.NS", "RELIANCE"}:
            return InstrumentRef("RELIANCE", "RELIANCE-EQ", "2885", "NSE")
        if sym == "RELIANCE.FUT":
            return InstrumentRef(
                "RELIANCE.FUT",
                "RELIANCE28MAY26FUT",
                "5001",
                "NFO",
                instrumenttype="FUTSTK",
                name="RELIANCE",
            )
        if sym == "NIFTY28MAY26C25000":
            return InstrumentRef(
                "NIFTY.OPT",
                "NIFTY28MAY26C25000",
                "7001",
                "NFO",
                instrumenttype="OPTIDX",
                expiry=None,
                name="NIFTY",
                option_type="CE",
                strike=25000.0,
            )
        raise KeyError(symbol)

    def search_options(self, query, limit=4):
        if "NIFTY" in query.upper():
            return [
                SymbolSearchHit(
                    display="NIFTY CE 25000",
                    symbol="NIFTY.OPT.CE.25000.28MAY2026",
                    tradingsymbol="NIFTY28MAY26C25000",
                    segment="OPTIONS",
                )
            ]
        return []


class _FakeCatalog:
    def ensure_loaded(self, force_refresh=False):
        return None

    def search(self, query, limit=8):
        return [
            SymbolSearchHit(
                display="RELIANCE — RELIANCE-EQ (NSE)",
                symbol="RELIANCE.NS",
                tradingsymbol="RELIANCE-EQ",
                segment="EQUITY",
            )
        ]


def _tools(tmp_path, *, engine=None, registry=None, catalog=None) -> FortunaAdvisoryTools:
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )
    eng = engine or _FakeEngine()
    return FortunaAdvisoryTools(
        settings=settings,
        engine_factory=lambda: eng,
        registry=registry or _FakeRegistry(),
        catalog=catalog or _FakeCatalog(),
    )


def test_search_instruments_invalid_query(tmp_path):
    tools = _tools(tmp_path)
    response = tools.search_instruments("")
    assert response.ok is False
    assert response.error is not None
    assert response.error.code == AdvisoryErrorCode.INVALID_REQUEST


def test_search_instruments_equity_and_option_hits(tmp_path):
    tools = _tools(tmp_path)
    response = tools.search_instruments("RELIANCE")
    assert response.ok is True
    assert len(response.hits) >= 1
    assert response.hits[0].symbol == "RELIANCE.NS"

    option_response = tools.search_instruments("NIFTY CE")
    assert option_response.ok is True
    assert any("OPT" in hit.symbol for hit in option_response.hits)


def test_search_instruments_normalizes_generic_option_alias(tmp_path):
    class GenericOptionRegistry(_FakeRegistry):
        def search_options(self, query, limit=4):
            return [
                SymbolSearchHit(
                    display="NIFTY CE 25000 28-May-2026 — NIFTY28MAY26C25000 (NFO)",
                    symbol="NIFTY.OPT",
                    tradingsymbol="NIFTY28MAY26C25000",
                    segment="OPTIONS",
                )
            ]

        def resolve(self, symbol: str):
            sym = symbol.upper()
            if sym == "NIFTY28MAY26C25000":
                from datetime import date

                return InstrumentRef(
                    "NIFTY.OPT",
                    "NIFTY28MAY26C25000",
                    "7001",
                    "NFO",
                    instrumenttype="OPTIDX",
                    expiry=date(2026, 5, 28),
                    name="NIFTY",
                    option_type="CE",
                    strike=25000.0,
                )
            return super().resolve(symbol)

    tools = _tools(tmp_path, registry=GenericOptionRegistry())
    response = tools.search_instruments("NIFTY")
    assert response.ok is True
    assert any(hit.symbol == "NIFTY.OPT.CE.25000.28MAY2026" for hit in response.hits)


def test_analyze_instrument_delegates_to_builder(tmp_path):
    engine = _FakeEngine()
    tools = _tools(tmp_path, engine=engine)
    response = tools.analyze_instrument("RELIANCE")
    assert response.ok is True
    assert engine.loaded_symbol == "RELIANCE.NS"
    assert response.decision is not None
    assert response.decision.action == "BUY"


def test_get_model_health_returns_typed_status(tmp_path):
    engine = _FakeEngine()
    engine.settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        agentic_ml_scorer_enabled=False,
    )
    tools = _tools(tmp_path, engine=engine)
    status = tools.get_model_health()
    assert status.ml.load_error == "not_loaded"
    assert status.rl.load_error == "no_generator"


def test_get_recent_learning_summary_empty_when_store_missing(tmp_path):
    tools = _tools(tmp_path, engine=_FakeEngine(learning_store=None))
    summary = tools.get_recent_learning_summary()
    assert summary.total_rows == 0


def test_get_recent_learning_summary_filters_by_symbol(tmp_path):
    store = AgenticLearningStore(tmp_path / "agentic")
    store.upsert(
        LearningExample(
            decision_hash="d1",
            bar_time="2026-01-06T09:30:00",
            bar_idx=3,
            symbol="RELIANCE.NS",
            timeframe="5m",
            action="BUY",
            confidence=0.8,
            current_side=None,
            bar_close=100.0,
            outcome=LearningOutcome(status="resolved", paper_closed=True),
        )
    )
    store.upsert(
        LearningExample(
            decision_hash="d2",
            bar_time="2026-01-06T10:30:00",
            bar_idx=4,
            symbol="TCS.NS",
            timeframe="5m",
            action="SELL",
            confidence=0.6,
            current_side=None,
            bar_close=200.0,
            outcome=LearningOutcome(status="resolved", paper_closed=False),
        )
    )
    engine = _FakeEngine(learning_store=store)
    tools = _tools(tmp_path, engine=engine)
    summary = tools.get_recent_learning_summary("RELIANCE.NS")
    assert summary.total_rows == 1
    assert summary.recent_rows[0].symbol == "RELIANCE.NS"


def test_get_market_universe_returns_typed_response(tmp_path, monkeypatch):
    expected = MarketUniverseResponse(
        ok=True,
        source="registry",
        timeframe="1d",
        lookback_days=30,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="RELIANCE",
                source="registry",
                avg_turnover=1_000_000.0,
                avg_volume=10000.0,
                trend_pct=2.4,
                last_close=1500.0,
                liquidity_score=12.3,
            ),
        ),
    )

    def _fake_builder(**kwargs):
        assert kwargs["timeframe"] == "1d"
        return expected

    monkeypatch.setattr("fortuna.agentic.tools.build_market_universe", _fake_builder)
    tools = _tools(tmp_path)
    response = tools.get_market_universe()
    assert response.ok is True
    assert response.candidates[0].symbol == "RELIANCE.NS"


def test_analyze_market_shortlist_returns_typed_response(tmp_path, monkeypatch):
    expected = ShortlistAnalysisResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        items=(
            ShortlistAnalysisItem(
                symbol="RELIANCE.NS",
                liquidity_score=12.3,
                source="registry",
            ),
        ),
    )

    def _fake_shortlist(**kwargs):
        assert kwargs["analysis_limit"] == 5
        return expected

    monkeypatch.setattr("fortuna.agentic.tools.analyze_market_shortlist", _fake_shortlist)
    tools = _tools(tmp_path)
    response = tools.analyze_market_shortlist()
    assert response.ok is True
    assert response.items[0].symbol == "RELIANCE.NS"


def test_get_training_candidates_returns_typed_response(tmp_path, monkeypatch):
    expected = TrainingCandidateResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                ml_candidate=True,
                rl_candidate=True,
            ),
        ),
    )

    def _fake_candidates(**kwargs):
        assert kwargs["analysis_limit"] == 8
        return expected

    monkeypatch.setattr("fortuna.agentic.tools.build_training_candidates", _fake_candidates)
    tools = _tools(tmp_path)
    response = tools.get_training_candidates()
    assert response.ok is True
    assert response.candidates[0].symbol == "RELIANCE.NS"


def test_get_training_research_plan_returns_typed_response(tmp_path, monkeypatch):
    expected = TrainingResearchPlanResponse(
        ok=True,
        source="screener",
        timeframe="5m",
        lookback_days=30,
        ml_symbols=("RELIANCE.NS",),
        rl_symbols=("RELIANCE.NS",),
    )

    def _fake_plan(**kwargs):
        assert kwargs["selection_policy"] == "diversified"
        return expected

    monkeypatch.setattr("fortuna.agentic.tools.build_training_research_plan", _fake_plan)
    tools = _tools(tmp_path)
    response = tools.get_training_research_plan()
    assert response.ok is True
    assert response.ml_symbols == ("RELIANCE.NS",)


def test_get_multi_agent_workflow_returns_typed_response(tmp_path, monkeypatch):
    expected = MultiAgentWorkflowResponse(
        ok=True,
        source="auto",
        timeframe="5m",
        lookback_days=30,
        headline="multi-agent ready",
    )

    def _fake_workflow(**kwargs):
        assert kwargs["timeframe"] == "5m"
        assert kwargs["analysis_limit"] == 8
        return expected

    monkeypatch.setattr("fortuna.agentic.tools.build_multi_agent_workflow", _fake_workflow)
    tools = _tools(tmp_path)
    response = tools.get_multi_agent_workflow()
    assert response.ok is True
    assert response.headline == "multi-agent ready"


def test_get_shortlist_briefing_returns_typed_response(tmp_path, monkeypatch):
    expected = ShortlistBriefingResponse(
        ok=True,
        source="registry",
        timeframe="5m",
        lookback_days=30,
        headline="1 candidate setups, 1 reviewed",
        items=(
            BriefingItem(
                symbol="RELIANCE.NS",
                action="BUY",
                confidence=0.8,
                verdict="candidate",
                summary="BUY candidate with relatively clean support",
            ),
        ),
    )

    def _fake_briefing(**kwargs):
        assert kwargs["analysis_limit"] == 5
        return expected

    monkeypatch.setattr("fortuna.agentic.tools.build_shortlist_briefing", _fake_briefing)
    tools = _tools(tmp_path)
    response = tools.get_shortlist_briefing()
    assert response.ok is True
    assert response.items[0].symbol == "RELIANCE.NS"
