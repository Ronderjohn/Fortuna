"""PPO training pipeline + callbacks + checkpoint serialization."""

from __future__ import annotations

from fortuna.rl.training.checkpoint import OOSMetricsSummary, PolicyCheckpoint

__all__ = [
    "PolicyCheckpoint",
    "OOSMetricsSummary",
    "FortunaRLTrainer",
    "TrainerConfig",
]


def __getattr__(name: str):
    if name == "FortunaRLTrainer":
        from fortuna.rl.training.trainer import FortunaRLTrainer

        return FortunaRLTrainer
    if name == "TrainerConfig":
        from fortuna.rl.training.trainer import TrainerConfig

        return TrainerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
