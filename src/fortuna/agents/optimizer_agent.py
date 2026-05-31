"""``AdaptiveOptimizerAgent`` — suggests next RewardConfig hyperparams."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fortuna.agents.base import OptimizerAgent
from fortuna.paper.learner import AdaptiveLearner
from fortuna.rl.env.reward import RewardConfig
from fortuna.strategy.schema import StrategyDefinition


class AdaptiveOptimizerAgent(OptimizerAgent):
    """Reads Phase 1 ``AdaptiveLearner`` state and proposes next-iteration
    ``RewardConfig`` perturbations.

    Phase 2.1 implementation: simple parameter perturbation around the current
    best (no Bayesian / meta-RL yet). The output is a ``dict`` ready to feed
    into ``TrainerConfig.reward_config = RewardConfig(**suggested)`` for the
    next training run.
    """

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        self._learner = AdaptiveLearner(state_dir or Path("logs/learner"))

    # Phase 1 contract — unchanged behaviour.
    def suggest_params(
        self,
        strategy: StrategyDefinition,
        history: list[dict[str, Any]],
    ) -> StrategyDefinition:
        return strategy

    # Phase 2 entry point.
    def suggest(
        self,
        current: RewardConfig,
        *,
        strategy_stem: str = "rl_policy",
    ) -> dict[str, Any]:
        """Suggest next ``RewardConfig`` based on adaptive-learner fold history."""
        state = self._learner.load(strategy_stem)
        base = asdict(current)

        if state is None or not state.fold_history:
            # No history → take a small exploratory step.
            return self._perturb(base, factor=1.1)

        recent = state.fold_history[-3:]
        avg_profit = sum(f.profit_pct for f in recent) / max(len(recent), 1)
        avg_trades = sum(f.total_trades for f in recent) / max(len(recent), 1)

        if avg_profit < 0:
            # Losing — push exploration: less holding penalty, more terminal Sharpe weight.
            base["holding_penalty"] = max(0.0, base["holding_penalty"] * 0.5)
            base["sharpe_weight"] = base["sharpe_weight"] * 1.3
            base["filter_fail_penalty"] = base["filter_fail_penalty"] * 1.2
        elif avg_trades < 5:
            # Hold-locked — strongly reduce holding penalty.
            base["holding_penalty"] = max(0.0, base["holding_penalty"] * 0.25)
            base["overtrading_penalty"] = base["overtrading_penalty"] * 0.5
        elif avg_trades > 100:
            # Over-trading — tighten penalties.
            base["overtrading_penalty"] = base["overtrading_penalty"] * 1.5
            base["max_trades_per_episode"] = max(
                10, int(base["max_trades_per_episode"] * 0.8)
            )
        else:
            # Healthy regime — small exploratory perturbation.
            base = self._perturb(base, factor=1.05)

        return base

    @staticmethod
    def _perturb(base: dict[str, Any], factor: float) -> dict[str, Any]:
        out = dict(base)
        for k in ("holding_penalty", "sharpe_weight", "drawdown_penalty"):
            if k in out and isinstance(out[k], (int, float)):
                out[k] = float(out[k]) * factor
        return out
