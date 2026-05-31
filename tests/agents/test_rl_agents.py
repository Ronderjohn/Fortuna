"""Contract tests for Phase 2 agent implementations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Pre-warm import cycle.
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401

from fortuna.agents import (
    AdaptiveOptimizerAgent,
    AgentProposal,
    BacktestCriticAgent,
    CriticAgent,
    MarketContext,
    OptimizerAgent,
    ResearchAgent,
    RLResearchAgent,
)
from fortuna.app.live_signals import SignalType
from fortuna.features.position import PositionState
from fortuna.rl.env.reward import RewardConfig
from fortuna.rl.inference import RLSignalGenerator


def _make_ohlcv(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-01 09:15", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_research_agent_isinstance():
    gen = RLSignalGenerator(None)
    agent = RLResearchAgent(gen)
    assert isinstance(agent, ResearchAgent)


def test_critic_agent_isinstance():
    gen = RLSignalGenerator(None)
    agent = BacktestCriticAgent(gen)
    assert isinstance(agent, CriticAgent)


def test_optimizer_agent_isinstance(tmp_path):
    agent = AdaptiveOptimizerAgent(tmp_path)
    assert isinstance(agent, OptimizerAgent)


def test_research_agent_returns_hold_when_unavailable():
    gen = RLSignalGenerator(None)
    agent = RLResearchAgent(gen)
    ctx = MarketContext(
        symbol="ICICIBANK.NS",
        timeframe="5m",
        timestamp=pd.Timestamp("2026-04-01 10:00"),
        latest_indicator_row=pd.Series({"close": 100.0}),
        position_state=PositionState(),
    )
    proposal = agent.propose(ctx)
    assert isinstance(proposal, AgentProposal)
    assert proposal.signal == SignalType.HOLD
    assert proposal.source == "rl_unavailable"


def test_critic_agent_unavailable_returns_rejection():
    gen = RLSignalGenerator(None)
    agent = BacktestCriticAgent(gen)
    result = agent.critique_policy(_make_ohlcv())
    assert result.approved is False
    assert any("No RL policy loaded" in r for r in result.issues)


def test_critic_agent_phase1_contract_returns_critique():
    """The legacy ``critique(strategy, metrics)`` contract still works."""
    from fortuna.backtesting.metrics import BacktestMetrics

    gen = RLSignalGenerator(None)
    agent = BacktestCriticAgent(gen)
    fake_metrics = BacktestMetrics(
        total_return=0.1,
        sharpe_ratio=1.2,
        max_drawdown=0.05,
        win_rate=0.55,
        expectancy=0.5,
        profit_factor=1.5,
        total_trades=20,
        final_value=110_000.0,
    )
    result = agent.critique(None, fake_metrics)  # type: ignore[arg-type]
    assert result.approved is True
    assert result.score_adjustment == pytest.approx(1.2)


def test_optimizer_suggest_perturbs_with_no_history(tmp_path):
    agent = AdaptiveOptimizerAgent(tmp_path)
    base = RewardConfig()
    out = agent.suggest(base, strategy_stem="never_trained")
    assert isinstance(out, dict)
    assert out["holding_penalty"] != base.holding_penalty


def test_optimizer_lowers_holding_penalty_when_few_trades(tmp_path):
    from fortuna.paper.learner import FoldRecord, StrategyLearningState

    agent = AdaptiveOptimizerAgent(tmp_path)
    state = StrategyLearningState(
        strategy_stem="rl_policy",
        fold_history=[
            FoldRecord(i, 0.1, 50.0, 1, f"c{i}", "") for i in range(3)
        ],
    )
    agent._learner.save(state)

    base = RewardConfig()
    out = agent.suggest(base, strategy_stem="rl_policy")
    assert out["holding_penalty"] < base.holding_penalty
