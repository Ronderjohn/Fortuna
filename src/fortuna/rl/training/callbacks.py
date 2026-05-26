"""SB3 callbacks: per-episode metrics + best-deterministic comparison."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class TensorboardCallback(BaseCallback):
    """Logs Fortuna-specific per-episode metrics from the env's ``info`` dict.

    SB3 already logs reward sums and lengths. This callback adds the
    Fortuna-domain metrics: n_trades, win_rate, profit_factor, Sharpe, and
    whether the FilterVerdict passed.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buffer: dict[str, list[float]] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key in (
                "episode_n_trades",
                "episode_sharpe",
                "episode_profit_factor",
                "episode_win_rate",
            ):
                if key in info:
                    self._buffer.setdefault(key, []).append(float(info[key]))
            if "episode_verdict_passed" in info:
                self._buffer.setdefault("episode_verdict_passed", []).append(
                    1.0 if info["episode_verdict_passed"] else 0.0
                )

        if self.num_timesteps % 1000 == 0 and self._buffer:
            for key, values in self._buffer.items():
                if not values:
                    continue
                self.logger.record(f"fortuna/{key}", float(np.mean(values[-20:])))
        return True


class FortunaEvalCallback(EvalCallback):
    """SB3 EvalCallback with OOS comparison vs a baseline Sharpe.

    Tracks the best deterministic Sharpe (provided at construction) and emits an
    info line whenever the trained policy crosses it. Early-stop after
    ``patience`` non-improving evaluations.
    """

    def __init__(
        self,
        *args,
        baseline_sharpe: Optional[float] = None,
        patience: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.baseline_sharpe = baseline_sharpe
        self.patience = int(patience)
        self._no_improvement = 0
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        result = super()._on_step()

        mean_reward = self.last_mean_reward if hasattr(self, "last_mean_reward") else None
        if mean_reward is not None and self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if mean_reward > self._best_mean_reward + 1e-6:
                self._best_mean_reward = mean_reward
                self._no_improvement = 0
            else:
                self._no_improvement += 1

            self.logger.record("fortuna/eval_mean_reward", float(mean_reward))
            self.logger.record(
                "fortuna/eval_best_mean_reward", float(self._best_mean_reward)
            )
            self.logger.record(
                "fortuna/eval_patience_used", int(self._no_improvement)
            )
            if self.baseline_sharpe is not None:
                self.logger.record(
                    "fortuna/baseline_sharpe", float(self.baseline_sharpe)
                )

            if self._no_improvement >= self.patience:
                logger.info(
                    "[RL] Early stopping after %d non-improving evals "
                    "(best mean reward=%.4f)",
                    self._no_improvement,
                    self._best_mean_reward,
                )
                return False
        return result
