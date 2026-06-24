from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fortuna.agentic.contracts import NightlyAlignmentStatus
from fortuna.agentic.learning import LearningExample, LearningOutcome
from fortuna.agentic.store import AgenticLearningStore
from fortuna.app.market_universe import build_market_universe
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRef, SymbolSearchHit


class _FakeRegistry:
    def ensure_loaded(self, force_refresh: bool = False):
        return None

    def resolve(self, symbol: str):
        raw = symbol.upper().strip()
        base = raw.replace(".NS", "").replace("-EQ", "")
        if base in {"RELIANCE", "TCS", "INFY"}:
            return InstrumentRef(base, f"{base}-EQ", base, "NSE")
        raise KeyError(symbol)

    @staticmethod
    def _equity_hits() -> list[SymbolSearchHit]:
        return [
            SymbolSearchHit(
                display="RELIANCE - RELIANCE-EQ (NSE)",
                symbol="RELIANCE.NS",
                tradingsymbol="RELIANCE-EQ",
                segment="EQUITY",
            ),
            SymbolSearchHit(
                display="TCS - TCS-EQ (NSE)",
                symbol="TCS.NS",
                tradingsymbol="TCS-EQ",
                segment="EQUITY",
            ),
            SymbolSearchHit(
                display="INFY - INFY-EQ (NSE)",
                symbol="INFY.NS",
                tradingsymbol="INFY-EQ",
                segment="EQUITY",
            ),
        ]

    def catalog(self):
        return self._equity_hits()

    def search(self, query: str, limit: int = 20):
        return self.catalog()[:limit]


class _EmptyCatalogRegistry(_FakeRegistry):
    def catalog(self):
        return []

    def search(self, query: str, limit: int = 20):
        return self._equity_hits()[:limit]


def _settings(tmp_path: Path, screener_rel: str = "data/universe/screener_export.csv") -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        market_universe_screener_csv=Path(screener_rel),
        market_universe_default_limit=5,
        agentic_log_dir=Path("logs/agentic"),
    )


def _provider(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=10, freq="1D")
    if symbol == "RELIANCE.NS":
        close = [100, 101, 103, 104, 105, 106, 108, 109, 110, 112]
        volume = [1000, 1100, 1200, 1300, 1400, 1400, 1500, 1600, 1700, 1800]
    elif symbol == "TCS.NS":
        close = [200, 199, 198, 197, 198, 197, 196, 195, 194, 193]
        volume = [500, 550, 560, 570, 580, 600, 610, 620, 630, 640]
    else:
        close = [50, 50.5, 50.2, 50.6, 50.7, 50.8, 51, 50.9, 51.2, 51.1]
        volume = [400, 420, 410, 430, 440, 450, 460, 470, 480, 490]
    return pd.DataFrame({"close": close, "volume": volume}, index=idx)


def test_build_market_universe_prefers_screener_candidates(tmp_path: Path):
    settings = _settings(tmp_path)
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    screener_path.parent.mkdir(parents=True, exist_ok=True)
    screener_path.write_text(
        "Symbol,Company Name,Current Price,Volume\n"
        "RELIANCE,Reliance Industries,112,500000\n"
        "TCS,Tata Consultancy Services,193,250000\n",
        encoding="utf-8",
    )

    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=2,
        timeframe="1d",
        days=10,
        source="auto",
    )

    assert response.ok is True
    assert response.source == "screener"
    assert len(response.candidates) == 2
    assert response.candidates[0].symbol == "RELIANCE.NS"
    assert response.candidates[0].avg_turnover is not None
    assert response.candidates[0].regime == "TRENDING"
    assert response.candidates[0].volume_ratio is not None
    assert response.candidates[0].activity_score is not None
    assert "ohlcv_ranked" in response.candidates[0].notes
    assert response.scout_liquidity_symbols
    assert response.scout_activity_symbols
    assert response.scout_volume_dense_symbols
    assert response.scout_summary is not None


def test_build_market_universe_falls_back_to_registry_when_no_screener(tmp_path: Path):
    settings = _settings(tmp_path)
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )

    assert response.ok is True
    assert response.source == "registry"
    assert len(response.candidates) == 3
    assert {row.symbol for row in response.candidates} == {"RELIANCE.NS", "TCS.NS", "INFY.NS"}


def test_build_market_universe_seed_only_when_data_unavailable(tmp_path: Path):
    settings = _settings(tmp_path)
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    screener_path.parent.mkdir(parents=True, exist_ok=True)
    screener_path.write_text(
        "Symbol,Company Name,Current Price,Volume\n"
        "RELIANCE,Reliance Industries,112,500000\n",
        encoding="utf-8",
    )

    def _missing(symbol: str, timeframe: str, days: int):
        return pd.DataFrame()

    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_missing,
        limit=1,
        source="screener",
    )

    assert response.ok is True
    assert response.candidates[0].symbol == "RELIANCE.NS"
    assert response.candidates[0].trend_pct is None
    assert "screener_snapshot_only" in response.candidates[0].notes


def test_build_market_universe_blends_liquidity_with_activity_context(tmp_path: Path):
    settings = _settings(tmp_path)
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    screener_path.parent.mkdir(parents=True, exist_ok=True)
    screener_path.write_text(
        "Symbol,Company Name,Current Price,Volume\n"
        "RELIANCE,Reliance Industries,100,2200\n"
        "TCS,Tata Consultancy Services,110,1800\n",
        encoding="utf-8",
    )

    def _activity_provider(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=10, freq="1D")
        if symbol == "RELIANCE.NS":
            return pd.DataFrame(
                {
                    "close": [100.0] * 10,
                    "volume": [2200, 2200, 2200, 2200, 2200, 2200, 2200, 2200, 2200, 2200],
                },
                index=idx,
            )
        return pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 110],
                "volume": [900, 900, 920, 940, 960, 1400, 1500, 1600, 1700, 1800],
            },
            index=idx,
        )

    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_activity_provider,
        limit=2,
        timeframe="1d",
        days=10,
        source="screener",
    )

    assert response.ok is True
    assert len(response.candidates) == 2
    assert response.candidates[0].symbol == "TCS.NS"
    assert response.candidates[0].activity_score is not None
    assert response.candidates[0].volume_ratio is not None
    assert response.candidates[0].liquidity_score > response.candidates[1].liquidity_score


def test_build_market_universe_applies_adaptive_outcome_weighting(tmp_path: Path):
    settings = _settings(tmp_path)
    store = AgenticLearningStore(settings.resolve_path(settings.agentic_log_dir))
    store.upsert(
        LearningExample(
            decision_hash="g1",
            bar_time="2026-01-10T10:00:00",
            bar_idx=1,
            symbol="TCS.NS",
            timeframe="1d",
            action="BUY",
            confidence=0.7,
            current_side=None,
            bar_close=200.0,
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=2.0,
            ),
        )
    )
    store.upsert(
        LearningExample(
            decision_hash="g2",
            bar_time="2026-01-10T10:05:00",
            bar_idx=2,
            symbol="RELIANCE.NS",
            timeframe="1d",
            action="BUY",
            confidence=0.7,
            current_side=None,
            bar_close=100.0,
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=-2.2,
            ),
        )
    )

    baseline = build_market_universe(
        settings=settings.model_copy(update={"market_universe_adaptive_weighting_enabled": False}),
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )

    assert baseline.ok is True
    assert response.ok is True
    baseline_ranked = {row.symbol: row for row in baseline.candidates}
    ranked = {row.symbol: row for row in response.candidates}
    assert any("adaptive_penalty" in note for note in ranked["RELIANCE.NS"].notes)
    assert any("adaptive_boost" in note for note in ranked["TCS.NS"].notes)
    assert ranked["RELIANCE.NS"].adaptive_score_adjustment is not None
    assert ranked["TCS.NS"].adaptive_score_adjustment is not None
    assert ranked["RELIANCE.NS"].liquidity_score < baseline_ranked["RELIANCE.NS"].liquidity_score
    assert ranked["TCS.NS"].liquidity_score > baseline_ranked["TCS.NS"].liquidity_score


def test_build_market_universe_applies_action_and_global_adaptive_policy(tmp_path: Path):
    settings = _settings(tmp_path)
    store = AgenticLearningStore(settings.resolve_path(settings.agentic_log_dir))
    store.upsert(
        LearningExample(
            decision_hash="g1",
            bar_time="2026-01-10T10:00:00",
            bar_idx=1,
            symbol="INFY.NS",
            timeframe="1d",
            action="BUY",
            confidence=0.7,
            current_side=None,
            bar_close=50.0,
            metadata={"regime": "TRENDING"},
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=2.4,
            ),
        )
    )
    store.upsert(
        LearningExample(
            decision_hash="g2",
            bar_time="2026-01-10T10:05:00",
            bar_idx=2,
            symbol="TCS.NS",
            timeframe="1d",
            action="SELL",
            confidence=0.7,
            current_side=None,
            bar_close=200.0,
            metadata={"regime": "TRENDING"},
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=1.9,
            ),
        )
    )

    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )

    ranked = {row.symbol: row for row in response.candidates}
    note = next(
        note
        for note in ranked["RELIANCE.NS"].notes
        if str(note).startswith("adaptive_")
    )
    assert "action=" in note
    assert "regime=" in note
    assert "global=" in note
    assert ranked["RELIANCE.NS"].adaptive_row_count >= 2


def test_build_market_universe_applies_signal_family_adaptive_policy(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_action_bias_weight": 0.0,
            "market_universe_regime_bias_weight": 0.0,
            "market_universe_global_bias_weight": 0.0,
            "market_universe_signal_bias_weight": 0.4,
        }
    )
    store = AgenticLearningStore(settings.resolve_path(settings.agentic_log_dir))
    store.upsert(
        LearningExample(
            decision_hash="s1",
            bar_time="2026-01-10T10:00:00",
            bar_idx=1,
            symbol="TCS.NS",
            timeframe="1d",
            action="BUY",
            confidence=0.7,
            current_side=None,
            bar_close=200.0,
            metadata={"primary_signal": "ORB"},
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=2.2,
            ),
        )
    )
    store.upsert(
        LearningExample(
            decision_hash="s2",
            bar_time="2026-01-10T10:05:00",
            bar_idx=2,
            symbol="INFY.NS",
            timeframe="1d",
            action="BUY",
            confidence=0.7,
            current_side=None,
            bar_close=50.0,
            metadata={"primary_signal": "MMTS"},
            outcome=LearningOutcome(
                status="resolved",
                paper_closed=True,
                realized_pnl_pct=-2.1,
            ),
        )
    )

    baseline = build_market_universe(
        settings=settings.model_copy(update={"market_universe_adaptive_weighting_enabled": False}),
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )

    baseline_ranked = {row.symbol: row for row in baseline.candidates}
    ranked = {row.symbol: row for row in response.candidates}
    assert ranked["RELIANCE.NS"].liquidity_score > baseline_ranked["RELIANCE.NS"].liquidity_score
    assert ranked["INFY.NS"].liquidity_score < baseline_ranked["INFY.NS"].liquidity_score
    assert any("signal[ORB,VWAP]=" in note for note in ranked["RELIANCE.NS"].notes)
    assert any("signal[MMTS]=" in note for note in ranked["INFY.NS"].notes)


def test_build_market_universe_applies_recent_nightly_feedback(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_nightly_feedback_enabled": True,
            "market_universe_nightly_feedback_max_boost": 0.1,
            "market_universe_nightly_feedback_lookback_reports": 4,
            "market_universe_nightly_report_dir": Path("reports/nightly"),
            "market_universe_adaptive_weighting_enabled": False,
        }
    )
    nightly_dir = settings.resolve_path(settings.market_universe_nightly_report_dir)
    nightly_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = nightly_dir / "training_candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "ok": True,
                "source": "registry",
                "timeframe": "1d",
                "lookback_days": 10,
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

    baseline = build_market_universe(
        settings=settings.model_copy(update={"market_universe_nightly_feedback_enabled": False}),
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )

    baseline_ranked = {row.symbol: row for row in baseline.candidates}
    ranked = {row.symbol: row for row in response.candidates}
    assert ranked["TCS.NS"].liquidity_score > baseline_ranked["TCS.NS"].liquidity_score
    assert any("nightly_feedback=" in note for note in ranked["TCS.NS"].notes)
    assert any("executed=1" in note for note in ranked["TCS.NS"].notes)
    assert ranked["TCS.NS"].adaptive_score_adjustment is not None


def test_build_market_universe_carries_nightly_posture_context(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_nightly_feedback_enabled": False,
            "market_universe_adaptive_weighting_enabled": False,
        }
    )
    monkeypatch.setattr(
        "fortuna.app.market_universe.build_nightly_alignment_status",
        lambda runtime_settings: NightlyAlignmentStatus(
            report_count=2,
            enabled_reports=2,
            aligned_reports=1,
            recommended_refresh_target="all",
            recommended_force_refresh=True,
            recent_trend_window=2,
            recent_trend_enabled=2,
            recent_trend_aligned=1,
            recent_trend_latest_status="ok",
            recent_trend_latest_basket_size=3,
        ),
    )

    response = build_market_universe(
        settings=settings,
        limit=3,
        source="registry",
        data_provider=lambda symbol, timeframe, days: None,
    )

    assert response.ok is True
    assert response.nightly_alignment_target == "all"
    assert response.nightly_alignment_force_refresh is True
    assert response.nightly_recent_window == 2
    assert response.nightly_recent_enabled == 2
    assert response.nightly_recent_aligned == 1
    assert response.nightly_recent_latest_status == "ok"
    assert response.nightly_recent_latest_basket_size == 3
    assert (
        response.nightly_alignment_summary
        == "recent=1/2 over 2 run(s); target=all force_refresh=1"
    )
    assert response.scout_liquidity_symbols == ()
    assert response.scout_activity_symbols == ()
    assert response.scout_summary is None


def test_build_market_universe_applies_research_plan_nightly_feedback(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_nightly_feedback_enabled": True,
            "market_universe_nightly_feedback_max_boost": 0.1,
            "market_universe_nightly_feedback_lookback_reports": 4,
            "market_universe_nightly_report_dir": Path("reports/nightly"),
            "market_universe_adaptive_weighting_enabled": False,
        }
    )
    nightly_dir = settings.resolve_path(settings.market_universe_nightly_report_dir)
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
                "rows": [{"symbol": "TCS.NS", "target": "rl"}],
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

    baseline = build_market_universe(
        settings=settings.model_copy(update={"market_universe_nightly_feedback_enabled": False}),
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )

    baseline_ranked = {row.symbol: row for row in baseline.candidates}
    ranked = {row.symbol: row for row in response.candidates}
    assert ranked["TCS.NS"].liquidity_score > baseline_ranked["TCS.NS"].liquidity_score
    assert any("research=1" in note for note in ranked["TCS.NS"].notes)
    assert any("executed=1" in note for note in ranked["TCS.NS"].notes)


def test_build_market_universe_applies_executed_lane_long_horizon_feedback(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_nightly_feedback_enabled": True,
            "market_universe_nightly_feedback_max_boost": 0.1,
            "market_universe_nightly_feedback_lookback_reports": 4,
            "market_universe_nightly_report_dir": Path("reports/nightly"),
            "market_universe_adaptive_weighting_enabled": False,
            "market_universe_long_horizon_feedback_enabled": True,
            "market_universe_executed_lane_boost_weight": 0.06,
            "market_universe_long_horizon_lookback_reports": 6,
        }
    )
    nightly_dir = settings.resolve_path(settings.market_universe_nightly_report_dir)
    nightly_dir.mkdir(parents=True, exist_ok=True)
    research_path = nightly_dir / "training_research_plan.json"
    research_path.write_text(
        json.dumps(
            {
                "ok": True,
                "rows": [{"symbol": "TCS.NS", "target": "rl"}],
            }
        ),
        encoding="utf-8",
    )
    for stamp in ("20260603_0100", "20260603_0200", "20260603_0300"):
        (nightly_dir / f"{stamp}.json").write_text(
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
                            "detail": {"path": "reports/nightly/training_research_plan.json"},
                        },
                        {
                            "name": "training_execution_target",
                            "status": "ok",
                            "detail": {
                                "target": "rl",
                                "selection_source": "training_research",
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    without_lane = build_market_universe(
        settings=settings.model_copy(
            update={"market_universe_long_horizon_feedback_enabled": False}
        ),
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )
    with_lane = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )

    baseline = {row.symbol: row for row in without_lane.candidates}
    ranked = {row.symbol: row for row in with_lane.candidates}
    assert ranked["TCS.NS"].liquidity_score > baseline["TCS.NS"].liquidity_score
    assert any("executed_lane=rl" in note for note in ranked["TCS.NS"].notes)
    assert any("reports=3" in note for note in ranked["TCS.NS"].notes)


def test_build_market_universe_weights_refreshed_research_evidence_more(tmp_path: Path):
    base_settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_nightly_feedback_enabled": True,
            "market_universe_nightly_feedback_max_boost": 0.1,
            "market_universe_nightly_feedback_lookback_reports": 4,
            "market_universe_nightly_refresh_bonus_weight": 0.7,
            "market_universe_nightly_report_dir": Path("reports/nightly"),
            "market_universe_adaptive_weighting_enabled": False,
        }
    )
    nightly_dir = base_settings.resolve_path(base_settings.market_universe_nightly_report_dir)
    nightly_dir.mkdir(parents=True, exist_ok=True)
    research_path = nightly_dir / "training_research_plan.json"
    nightly_report = nightly_dir / "20260603_0100.json"

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
    planned_only = build_market_universe(
        settings=base_settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )
    _write_fixture(refreshed=True)
    refreshed = build_market_universe(
        settings=base_settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        timeframe="1d",
        days=10,
        source="registry",
    )

    planned_ranked = {row.symbol: row for row in planned_only.candidates}
    refreshed_ranked = {row.symbol: row for row in refreshed.candidates}
    assert refreshed_ranked["TCS.NS"].liquidity_score > planned_ranked["TCS.NS"].liquidity_score
    assert any("refreshed=1" in note for note in refreshed_ranked["TCS.NS"].notes)


def test_build_market_universe_smartapi_source_ranks_registry_catalog(tmp_path: Path):
    settings = _settings(tmp_path, screener_rel="data/universe/missing_screener.csv")
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=2,
        timeframe="1d",
        days=10,
        source="smartapi",
    )
    assert response.ok is True
    assert response.source == "smartapi"
    assert response.seed_source == "smartapi"
    assert response.scoring_source == "smartapi_ohlcv"
    assert response.provider_summary is not None
    assert "seeds=smartapi" in response.provider_summary
    assert response.candidates[0].symbol == "RELIANCE.NS"
    assert response.candidates[0].source == "smartapi"


def test_build_market_universe_smartapi_uses_tradable_universe_csv(tmp_path: Path):
    settings = _settings(tmp_path, screener_rel="data/universe/missing_screener.csv")
    tradable_path = settings.resolve_path(settings.market_universe_tradable_universe_csv)
    tradable_path.parent.mkdir(parents=True, exist_ok=True)
    tradable_path.write_text("symbol\nTCS\n", encoding="utf-8")
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=2,
        source="smartapi",
    )
    assert response.ok is True
    assert response.source == "smartapi"
    assert {row.symbol for row in response.candidates} == {"TCS.NS"}


def test_build_market_universe_auto_prefers_screener_when_csv_present(tmp_path: Path):
    settings = _settings(tmp_path)
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    screener_path.parent.mkdir(parents=True, exist_ok=True)
    screener_path.write_text("symbol,volume,cmp\nRELIANCE,1000,100\n", encoding="utf-8")
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=2,
        source="auto",
    )
    assert response.source == "screener"
    assert response.seed_source == "screener"


def test_build_market_universe_auto_uses_smartapi_without_screener_csv(tmp_path: Path):
    settings = _settings(tmp_path, screener_rel="data/universe/missing_screener.csv")
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=2,
        source="auto",
    )
    assert response.source == "smartapi"
    assert response.seed_source == "smartapi"


def test_build_market_universe_auto_prefer_smartapi_over_screener(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(update={"market_universe_auto_prefer_smartapi": True})
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    screener_path.parent.mkdir(parents=True, exist_ok=True)
    screener_path.write_text("symbol,volume,cmp\nRELIANCE,1000,100\n", encoding="utf-8")
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=2,
        source="auto",
    )
    assert response.source == "smartapi"
    assert response.seed_source == "smartapi"


def test_build_market_universe_smartapi_falls_back_to_registry_when_catalog_empty(
    tmp_path: Path,
):
    settings = _settings(tmp_path, screener_rel="data/universe/missing_screener.csv")
    response = build_market_universe(
        settings=settings,
        registry=_EmptyCatalogRegistry(),
        data_provider=_provider,
        limit=2,
        source="smartapi",
    )
    assert response.ok is True
    assert response.source == "registry"
    assert response.fallback_from == "smartapi"
    assert "fallback=smartapi->registry" in (response.provider_summary or "")


def _write_screener_seeds(settings: Settings) -> None:
    screener_path = settings.resolve_path(settings.market_universe_screener_csv)
    screener_path.parent.mkdir(parents=True, exist_ok=True)
    screener_path.write_text(
        "Symbol,Company Name,Current Price,Volume\n"
        "RELIANCE,Reliance Industries,112,500000\n"
        "TCS,Tata Consultancy Services,193,250000\n"
        "INFY,Infosys,51,250000\n",
        encoding="utf-8",
    )


def _write_overlay(settings: Settings, content: str) -> None:
    overlay_path = settings.resolve_path(settings.market_universe_fundamentals_overlay_csv)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(content, encoding="utf-8")


def test_build_market_universe_overlay_disabled_unchanged(tmp_path: Path):
    settings = _settings(tmp_path)
    _write_screener_seeds(settings)
    baseline = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        source="screener",
    )
    enabled_missing = settings.model_copy(
        update={
            "market_universe_fundamentals_overlay_enabled": True,
            "market_universe_fundamentals_overlay_csv": Path(
                "data/universe/missing_overlay.csv"
            ),
        }
    )
    missing = build_market_universe(
        settings=enabled_missing,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        source="screener",
    )
    assert baseline.ok is True
    assert missing.ok is True
    assert missing.fundamentals_overlay_enabled is True
    assert missing.fundamentals_overlay_diagnostics
    assert baseline.candidates[0].liquidity_score == missing.candidates[0].liquidity_score


def test_build_market_universe_overlay_boosts_high_quality(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_fundamentals_overlay_enabled": True,
            "market_universe_fundamentals_overlay_max_boost": 0.05,
        }
    )
    _write_screener_seeds(settings)
    baseline = build_market_universe(
        settings=settings.model_copy(
            update={"market_universe_fundamentals_overlay_enabled": False}
        ),
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=2,
        source="screener",
    )
    _write_overlay(
        settings,
        "symbol,sector,market_cap_bucket,operator_quality_score,note\n"
        "RELIANCE,Energy,large,0.9,core holding\n"
        "TCS,IT,large,0.5,\n",
    )
    enriched = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=2,
        source="screener",
    )
    assert enriched.ok is True
    assert enriched.fundamentals_overlay_summary.startswith("loaded=2")
    reliance = next(row for row in enriched.candidates if row.symbol == "RELIANCE.NS")
    base_reliance = next(row for row in baseline.candidates if row.symbol == "RELIANCE.NS")
    assert reliance.sector == "Energy"
    assert reliance.market_cap_bucket == "large"
    assert reliance.operator_quality_score == 0.9
    assert reliance.fundamentals_overlay_adjustment is not None
    assert reliance.fundamentals_overlay_adjustment > 0
    assert reliance.liquidity_score > base_reliance.liquidity_score
    assert any("fundamentals_overlay=" in note for note in reliance.notes)


def test_build_market_universe_overlay_excludes_symbol(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(
        update={"market_universe_fundamentals_overlay_enabled": True}
    )
    _write_screener_seeds(settings)
    _write_overlay(
        settings,
        "symbol,operator_exclude\nINFY,true\n",
    )
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=5,
        source="screener",
    )
    assert response.ok is True
    symbols = {row.symbol for row in response.candidates}
    assert "INFY.NS" not in symbols
    assert "excluded=1" in (response.fundamentals_overlay_summary or "")


def test_build_market_universe_overlay_malformed_rows_fail_soft(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(
        update={"market_universe_fundamentals_overlay_enabled": True}
    )
    _write_screener_seeds(settings)
    _write_overlay(
        settings,
        "symbol,operator_quality_score\n"
        ",0.8\n"
        "UNKNOWN,0.7\n"
        "RELIANCE,not-a-number\n"
        "TCS,0.6\n",
    )
    response = build_market_universe(
        settings=settings,
        registry=_FakeRegistry(),
        data_provider=_provider,
        limit=3,
        source="screener",
    )
    assert response.ok is True
    assert response.fundamentals_overlay_diagnostics
    assert response.fundamentals_overlay_summary.startswith("loaded=1")
    tcs = next((row for row in response.candidates if row.symbol == "TCS.NS"), None)
    assert tcs is not None
    assert tcs.operator_quality_score == 0.6


def test_build_market_universe_overlay_bounded_quality_delta(tmp_path: Path):
    settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_fundamentals_overlay_enabled": True,
            "market_universe_fundamentals_overlay_max_boost": 0.05,
        }
    )
    _write_screener_seeds(settings)

    def _score_for_quality(quality: float) -> float:
        _write_overlay(
            settings,
            f"symbol,operator_quality_score\nRELIANCE,{quality}\n",
        )
        response = build_market_universe(
            settings=settings,
            registry=_FakeRegistry(),
            data_provider=_provider,
            limit=1,
            source="screener",
        )
        return float(response.candidates[0].liquidity_score or 0.0)

    high = _score_for_quality(1.0)
    low = _score_for_quality(0.0)
    assert high - low <= 0.10 + 1e-6
    assert high > low
