"""Adaptive learner tests."""

from pathlib import Path

from fortuna.paper.learner import AdaptiveLearner
from fortuna.reporting.strategy_report import StrategyReport
from fortuna.strategy.schema import StrategyDefinition


def test_learner_narrows_grid(tmp_path: Path) -> None:
    learner = AdaptiveLearner(tmp_path)
    grid = {"indicators.ema_fast.params.window": [8, 12, 16]}
    strat = StrategyDefinition.model_validate(
        {
            "name": "ema",
            "indicators": [
                {"id": "ema_fast", "type": "ema", "params": {"window": 12}},
            ],
            "rules": {},
        }
    )
    report = StrategyReport(
        strategy_name="ema",
        candidate_id="c0",
        symbol="X",
        timeframe="5m",
        profit_pct=5.0,
        win_ratio_pct=55.0,
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        max_drawdown_pct=3.0,
        profit_factor=1.5,
        sharpe_ratio=1.0,
        avg_trade_pct=0.5,
        gross_profit_pct=8.0,
        gross_loss_pct=3.0,
        composite_score=60.0,
        passed_validation=True,
        params_summary="ema_fast=12",
    )
    state = learner.record_fold("ema", 0, report, strat, grid)
    refined = state.param_grid
    assert "indicators.ema_fast.params.window" in refined
    assert 12 in refined["indicators.ema_fast.params.window"]
