from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fortuna.agentic.contracts import (
    AdvisoryErrorCode,
    InstrumentAnalysisRequest,
)
from fortuna.agentic.models import ActionRecommendation, AgentDecision, DecisionRationale
from fortuna.app.advisory_service import analyze_instrument
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRef


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
    def __init__(self):
        idx = pd.date_range("2026-01-06 09:15", periods=5, freq="5min")
        self.loaded_symbol = ""
        self._state = _State(
            ohlcv=pd.DataFrame(
                {
                    "open": [100, 101, 102, 103, 104],
                    "high": [101, 102, 103, 104, 105],
                    "low": [99, 100, 101, 102, 103],
                    "close": [100, 101, 102, 103, 104],
                    "volume": [1000] * 5,
                },
                index=idx,
            ),
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
                rationale=DecisionRationale(
                    summary="BUY won",
                    reasons=["deterministic breakout"],
                ),
            ),
        }

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
        raise KeyError(symbol)


def test_analyze_instrument_success(tmp_path):
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )
    engine = _FakeEngine()
    response = analyze_instrument(
        request=InstrumentAnalysisRequest(symbol="RELIANCE"),
        settings=settings,
        engine_factory=lambda: engine,
        registry=_FakeRegistry(),
    )
    assert response.ok is True
    assert response.instrument is not None
    assert response.instrument.resolved_symbol == "RELIANCE.NS"
    assert response.decision is not None
    assert response.decision.action == "BUY"
    assert response.winning_strategy == "orb"
    payload = response.to_dict()
    assert "ohlcv" not in payload
    assert "batch" not in payload


def test_analyze_instrument_resolve_failure(tmp_path):
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )
    response = analyze_instrument(
        request=InstrumentAnalysisRequest(symbol="UNKNOWN"),
        settings=settings,
        engine_factory=lambda: _FakeEngine(),
        registry=_FakeRegistry(),
    )
    assert response.ok is False
    assert response.error is not None
    assert response.error.code == AdvisoryErrorCode.RESOLVE_FAILED


def test_analyze_instrument_invalid_request(tmp_path):
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )
    response = analyze_instrument(
        request=InstrumentAnalysisRequest(symbol=""),
        settings=settings,
        engine_factory=lambda: _FakeEngine(),
        registry=_FakeRegistry(),
    )
    assert response.ok is False
    assert response.error is not None
    assert response.error.code == AdvisoryErrorCode.INVALID_REQUEST


def test_analyze_instrument_load_failure(tmp_path):
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )
    engine = _FakeEngine()
    engine._state.load_error = "no data"
    response = analyze_instrument(
        request=InstrumentAnalysisRequest(symbol="RELIANCE"),
        settings=settings,
        engine_factory=lambda: engine,
        registry=_FakeRegistry(),
    )
    assert response.ok is False
    assert response.error is not None
    assert response.error.code == AdvisoryErrorCode.LOAD_FAILED
