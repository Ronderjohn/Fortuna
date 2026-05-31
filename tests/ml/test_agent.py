"""Tests for MLSignalScorerAgent adapter."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from fortuna.agentic.models import ActionRecommendation, AgentRunContext
from fortuna.ml.agent import MLSignalScorerAgent
from fortuna.ml.features import ML_FEATURE_NAMES
from fortuna.ml.signal_scorer import SignalScorer


class Sig:
    def __init__(self, action: str):
        self.action = action


def _enriched(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 09:15", periods=n, freq="5min")
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000] * n,
            "atr": [1.0] * n,
        },
        index=idx,
    )


def _trained_scorer() -> SignalScorer:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, len(ML_FEATURE_NAMES)))
    y = (X[:, 0] + X[:, 2] > 0).astype(int)
    scorer = SignalScorer(random_state=42)
    scorer.fit(X, y)
    return scorer


def test_unavailable_scorer_returns_none():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("BUY")},
        bar_time=datetime(2026, 1, 6, 9, 30),
    )
    agent = MLSignalScorerAgent(scorer=None)
    vote = agent.vote_for_strategy(ctx, strategy_name="orb", enriched=_enriched(), bar_idx=2)
    assert vote is None


def test_hold_signal_returns_none():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("HOLD")},
    )
    agent = MLSignalScorerAgent(scorer=_trained_scorer())
    vote = agent.vote_for_strategy(ctx, strategy_name="orb", enriched=_enriched(), bar_idx=2)
    assert vote is None


def test_scores_named_strategy_only():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("BUY"), "vwap": Sig("SELL")},
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=102.0,
    )
    agent = MLSignalScorerAgent(scorer=_trained_scorer())
    vote = agent.vote_for_strategy(ctx, strategy_name="orb", enriched=_enriched(), bar_idx=2)
    assert vote is not None
    assert vote.agent == "ml_signal_scorer"
    assert vote.action is ActionRecommendation.BUY
    assert vote.source == "orb"
    assert 0.0 <= vote.confidence <= 1.0
    assert "ML scorer" in vote.reason


def test_missing_strategy_returns_none():
    ctx = AgentRunContext(
        symbol="RELIANCE.NS",
        timeframe="5m",
        signals={"orb": Sig("BUY")},
    )
    agent = MLSignalScorerAgent(scorer=_trained_scorer())
    vote = agent.vote_for_strategy(ctx, strategy_name="missing", enriched=_enriched(), bar_idx=2)
    assert vote is None
