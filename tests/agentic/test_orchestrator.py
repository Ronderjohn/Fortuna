from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from fortuna.agentic import (
    ActionRecommendation,
    AgenticOrchestrator,
    AgentRunContext,
)
from fortuna.ml.features import ML_FEATURE_NAMES
from fortuna.ml.signal_scorer import SignalScorer


class Sig:
    def __init__(self, action, *, rl_confidence=None, rl_action=None, regime=None):
        self.action = action
        self.rl_confidence = rl_confidence
        self.rl_action = rl_action
        self.regime = regime


def _trained_scorer() -> SignalScorer:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, len(ML_FEATURE_NAMES)))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    scorer = SignalScorer(random_state=42)
    scorer.fit(X, y)
    return scorer


def _enriched_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 09:15", periods=30, freq="5min")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 129, 30),
            "high": np.linspace(100.5, 129.5, 30),
            "low": np.linspace(99.5, 128.5, 30),
            "close": np.linspace(100, 129, 30),
            "volume": [1000] * 30,
            "atr": [1.0] * 30,
        },
        index=idx,
    )


def test_flat_symbol_buys_when_strategy_and_rl_agree():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={
            "orb": Sig("BUY", rl_confidence="agree", rl_action="BUY", regime="TRENDING"),
            "RL:abc123": Sig("BUY"),
        },
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=2000.0,
    )

    decision = AgenticOrchestrator(paper_tracking=True).decide(ctx)

    assert decision.action is ActionRecommendation.BUY
    assert decision.confidence > 0.8
    assert decision.paper_tracking is True
    assert decision.notification is not None
    assert "BUY RELIANCE.NS" in decision.notification.message
    assert "Advisory only" in decision.notification.message


def test_open_long_requires_exit_confirmation():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("BUY")},
        current_side="LONG",
        bar_time=datetime(2026, 1, 6, 9, 35),
        bar_close=2010.0,
    )

    decision = AgenticOrchestrator().decide(ctx)

    assert decision.action is ActionRecommendation.HOLD
    assert "Hold LONG" in decision.rationale.summary


def test_open_long_exits_on_exit_vote():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("EXIT_LONG")},
        current_side="LONG",
        bar_time=datetime(2026, 1, 6, 9, 40),
        bar_close=1990.0,
    )

    decision = AgenticOrchestrator().decide(ctx)

    assert decision.action is ActionRecommendation.EXIT_LONG
    assert decision.notification is not None


def test_market_context_vote_present():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("HOLD", regime="TRENDING")},
        bar_time=datetime(2026, 1, 6, 10, 0),
        bar_close=2000.0,
    )

    decision = AgenticOrchestrator().decide(ctx)

    market_votes = [v for v in decision.votes if v.agent == "market_context"]
    assert len(market_votes) == 1
    assert "TRENDING" in market_votes[0].reason
    assert market_votes[0].metadata.get("regime") == "TRENDING"


def test_ml_unavailable_no_crash():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={
            "orb": Sig("BUY", rl_confidence="agree", rl_action="BUY", regime="TRENDING"),
            "RL:abc123": Sig("BUY"),
        },
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=2000.0,
    )

    decision = AgenticOrchestrator(ml_scorer=None).decide(ctx)

    assert decision.action is ActionRecommendation.BUY
    assert not any(v.agent == "ml_signal_scorer" for v in decision.votes)


def test_ml_available_adds_vote():
    enriched = _enriched_frame()
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("BUY", regime="TRENDING")},
        bar_time=datetime(2026, 1, 6, 10, 0),
        bar_close=120.0,
        metadata={"enriched": enriched, "bar_idx": 25},
    )

    decision = AgenticOrchestrator(ml_scorer=_trained_scorer()).decide(ctx)

    ml_votes = [v for v in decision.votes if v.agent == "ml_signal_scorer"]
    assert len(ml_votes) == 1
    assert ml_votes[0].source == "orb"


def test_transient_ml_context_not_persisted_in_decision_metadata():
    enriched = _enriched_frame()
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("BUY", regime="TRENDING")},
        bar_time=datetime(2026, 1, 6, 10, 0),
        bar_close=120.0,
        metadata={"enriched": enriched, "bar_idx": 25, "timeframe": "5m"},
    )

    decision = AgenticOrchestrator(ml_scorer=_trained_scorer()).decide(ctx)

    assert decision.metadata["bar_idx"] == 25
    assert decision.metadata["timeframe"] == "5m"
    assert "enriched" not in decision.metadata


def test_rl_disagree_reduces_confidence():
    agree_ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={
            "orb": Sig("BUY", rl_confidence="agree", regime="TRENDING"),
            "momentum": Sig("BUY"),
            "RL:abc123": Sig("BUY"),
        },
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=2000.0,
    )
    disagree_ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={
            "orb": Sig("BUY", rl_confidence="agree", regime="TRENDING"),
            "momentum": Sig("BUY"),
            "RL:abc123": Sig("SELL"),
        },
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=2000.0,
    )

    agree = AgenticOrchestrator().decide(agree_ctx)
    disagree = AgenticOrchestrator().decide(disagree_ctx)

    assert agree.action is ActionRecommendation.BUY
    assert disagree.action is ActionRecommendation.BUY
    assert disagree.confidence < agree.confidence
    assert disagree.rationale.dissent


def test_flat_buy_beats_sell_consensus():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={
            "orb": Sig("BUY", rl_confidence="agree", regime="TRENDING"),
            "momentum": Sig("BUY"),
            "vwap": Sig("SELL"),
        },
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=2000.0,
    )

    decision = AgenticOrchestrator().decide(ctx)

    assert decision.action is ActionRecommendation.BUY


def test_weak_entry_downgraded():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("BUY", regime="TRENDING")},
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=2000.0,
    )

    decision = AgenticOrchestrator().decide(ctx)

    assert decision.action is ActionRecommendation.DO_NOT_ENTER
    assert decision.confidence < 0.6 or decision.action is ActionRecommendation.DO_NOT_ENTER


def test_notification_skipped_for_hold():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("HOLD", regime="TRENDING")},
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=2000.0,
    )

    decision = AgenticOrchestrator().decide(ctx)

    assert decision.action is ActionRecommendation.HOLD
    assert decision.notification is None
