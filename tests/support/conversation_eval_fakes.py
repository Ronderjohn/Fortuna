"""Deterministic fakes for conversation evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from fortuna.agentic.models import ActionRecommendation, AgentDecision, DecisionRationale
from fortuna.agentic.tools import FortunaAdvisoryTools
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRef, SymbolSearchHit
from fortuna.telegram.assistant import TelegramAnalysisAssistant


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


class FakeEvalEngine:
    def __init__(self, *, learning_store=None):
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
                rationale=DecisionRationale(summary="BUY won"),
            ),
            "RELIANCE.FUT": AgentDecision(
                symbol="RELIANCE.FUT",
                action=ActionRecommendation.SELL,
                confidence=0.64,
                bar_time=idx[-1].to_pydatetime(),
                bar_close=104.0,
                rationale=DecisionRationale(summary="SELL won"),
            ),
            "NIFTY.OPT.CE.25000.28MAY2026": AgentDecision(
                symbol="NIFTY.OPT.CE.25000.28MAY2026",
                action=ActionRecommendation.DO_NOT_ENTER,
                confidence=0.71,
                bar_time=idx[-1].to_pydatetime(),
                bar_close=104.0,
                rationale=DecisionRationale(summary="DO_NOT_ENTER won"),
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


class FakeEvalRegistry:
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
        if sym == "NIFTY.OPT.CE.25000.28MAY2026":
            return InstrumentRef(
                "NIFTY.OPT",
                "NIFTY28MAY26C25000",
                "7001",
                "NFO",
                instrumenttype="OPTIDX",
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


class FakeEvalCatalog:
    def ensure_loaded(self, force_refresh=False):
        return None

    def search(self, query, limit=8):
        upper = str(query).upper()
        if "RELIANCE" not in upper and "NIFTY" not in upper:
            return []
        return [
            SymbolSearchHit(
                display="RELIANCE — RELIANCE-EQ (NSE)",
                symbol="RELIANCE.NS",
                tradingsymbol="RELIANCE-EQ",
                segment="EQUITY",
            )
        ]


def build_eval_settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        conversational_adapter_enabled=True,
    )


def build_eval_tools(
    tmp_path: Path,
    *,
    engine: FakeEvalEngine | None = None,
) -> FortunaAdvisoryTools:
    settings = build_eval_settings(tmp_path)
    eng = engine or FakeEvalEngine()
    return FortunaAdvisoryTools(
        settings=settings,
        engine_factory=lambda: eng,
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )


def build_eval_assistant(
    tmp_path: Path,
    *,
    engine: FakeEvalEngine | None = None,
) -> TelegramAnalysisAssistant:
    settings = build_eval_settings(tmp_path)
    eng = engine or FakeEvalEngine()
    return TelegramAnalysisAssistant(
        settings,
        engine_factory=lambda: eng,
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
