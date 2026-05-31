"""RL evaluation helpers — baseline comparison and advisory readiness."""

from fortuna.rl.evaluation.baseline import baseline_sharpe_for_symbol
from fortuna.rl.training.checkpoint import (
    DEFAULT_MIN_ADVISORY_TRADES,
    build_failure_modes,
    compute_advisory_ready,
)

__all__ = [
    "DEFAULT_MIN_ADVISORY_TRADES",
    "baseline_sharpe_for_symbol",
    "build_failure_modes",
    "compute_advisory_ready",
]
