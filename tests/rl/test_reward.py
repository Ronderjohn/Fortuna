"""Tests for ``rl/env/reward.py``."""

from __future__ import annotations

from fortuna.backtesting.standard.config import MarketCostModel
from fortuna.backtesting.standard.filters import FilterVerdict
from fortuna.reporting.strategy_tester.metrics import StrategyMetrics
from fortuna.rl.env.reward import RewardConfig, RewardFunction


def test_step_reward_subtracts_costs_only_on_close():
    fn = RewardFunction(RewardConfig(), MarketCostModel())
    cost_pct = MarketCostModel().total_cost_pct()

    r_open = fn.step_reward(realized_pnl_pct=0.0, bars_held=0, n_trades=0, trade_closed=False)
    r_close = fn.step_reward(realized_pnl_pct=0.0, bars_held=0, n_trades=1, trade_closed=True)

    assert r_open == 0.0
    assert r_close == -cost_pct


def test_holding_penalty_grows_with_bars():
    cfg = RewardConfig(holding_penalty=0.01, step_pnl_weight=0.0, overtrading_penalty=0.0)
    fn = RewardFunction(cfg, MarketCostModel())
    r10 = fn.step_reward(realized_pnl_pct=0.0, bars_held=10, n_trades=0, trade_closed=False)
    r20 = fn.step_reward(realized_pnl_pct=0.0, bars_held=20, n_trades=0, trade_closed=False)
    assert r20 < r10 < 0


def test_overtrading_penalty_applies_beyond_threshold():
    cfg = RewardConfig(
        holding_penalty=0.0,
        step_pnl_weight=0.0,
        overtrading_penalty=0.1,
        max_trades_per_episode=10,
    )
    fn = RewardFunction(cfg, MarketCostModel())
    assert fn.step_reward(0, 0, 5, False) == 0.0
    assert fn.step_reward(0, 0, 15, False) == -0.5  # 5 excess trades * 0.1


def test_terminal_reward_penalizes_filter_fail():
    fn = RewardFunction(RewardConfig(), MarketCostModel())
    zero = StrategyMetrics.zero()
    verdict_fail = FilterVerdict(passed=False, reasons=["x"])
    verdict_pass = FilterVerdict(passed=True)

    r_fail = fn.terminal_reward(zero, verdict_fail)
    r_pass = fn.terminal_reward(zero, verdict_pass)

    assert r_pass > r_fail
    assert r_fail < 0  # filter_fail_penalty active and zero metrics give negative bonus
