"""``PolicyCheckpoint`` — metadata that ships alongside every policy.zip."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fortuna.features.registry import feature_registry_hash
from fortuna.reporting.strategy_tester.metrics import StrategyMetrics


@dataclass
class OOSMetricsSummary:
    """Compact, JSON-safe slice of a ``StrategyMetrics`` for the dashboard."""

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_drawdown_pct: float = 0.0
    total_net_profit: float = 0.0
    total_trades: int = 0
    win_rate_pct: float = 0.0

    @classmethod
    def from_strategy_metrics(cls, metrics: StrategyMetrics) -> "OOSMetricsSummary":
        return cls(
            sharpe_ratio=float(metrics.risk.sharpe_ratio),
            sortino_ratio=float(metrics.risk.sortino_ratio),
            calmar_ratio=float(metrics.risk.calmar_ratio),
            profit_factor=float(metrics.performance.profit_factor),
            expectancy=float(metrics.risk.expectancy),
            max_drawdown_pct=float(metrics.risk.max_drawdown_pct),
            total_net_profit=float(metrics.performance.total_net_profit),
            total_trades=int(metrics.performance.total_trades),
            win_rate_pct=float(metrics.performance.win_rate_pct),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OOSMetricsSummary":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class PolicyCheckpoint:
    """All training metadata for one trained policy.

    Stored as ``metadata.json`` next to ``policy.zip`` and ``normalizer.json``.
    The dashboard reads this directly for the leaderboard row + strategy detail.
    """

    run_id: str
    symbol: str
    timeframe: str
    obs_shape: list[int]
    policy_type: str
    total_timesteps: int
    n_folds: int
    oos_metrics: OOSMetricsSummary
    verdict_passed: bool
    verdict_reasons: list[str] = field(default_factory=list)
    verdict_score: float = 0.0
    created_at: str = ""
    feature_registry_hash: str = ""
    reward_config: dict[str, float] = field(default_factory=dict)
    train_bars: int = 0
    test_bars: int = 0
    step_bars: int = 0
    n_envs: int = 1
    n_bars: int = 20
    seed: Optional[int] = None
    sb3_version: str = ""
    torch_version: str = ""
    notes: str = ""
    baseline_sharpe: Optional[float] = None
    beats_baseline: Optional[bool] = None
    oos_fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    advisory_ready: bool = False

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")
        if not self.feature_registry_hash:
            self.feature_registry_hash = feature_registry_hash()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["oos_metrics"] = self.oos_metrics.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyCheckpoint":
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if isinstance(kwargs.get("oos_metrics"), dict):
            kwargs["oos_metrics"] = OOSMetricsSummary.from_dict(kwargs["oos_metrics"])
        if "advisory_ready" not in d:
            kwargs["advisory_ready"] = bool(d.get("verdict_passed", False))
        return cls(**kwargs)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "PolicyCheckpoint":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_MIN_ADVISORY_TRADES = 1


def compute_advisory_ready(
    cp: PolicyCheckpoint,
    *,
    min_trades: int = DEFAULT_MIN_ADVISORY_TRADES,
) -> tuple[bool, list[str]]:
    """Return whether a checkpoint is safe for live advisory RL overlay."""
    failures: list[str] = []
    if not cp.verdict_passed:
        failures.append("verdict_failed")
    if cp.oos_metrics.total_trades < min_trades:
        failures.append("hold_locked")
    if cp.beats_baseline is False:
        failures.append("below_baseline")
    return len(failures) == 0, failures


def build_failure_modes(
    cp: PolicyCheckpoint,
    *,
    min_trades: int = DEFAULT_MIN_ADVISORY_TRADES,
) -> list[str]:
    """Derive failure mode tags from checkpoint evaluation results."""
    modes: list[str] = []
    if not cp.verdict_passed:
        modes.append("verdict_failed")
    if cp.oos_metrics.total_trades < min_trades:
        modes.append("hold_locked")
    if cp.beats_baseline is False:
        modes.append("below_baseline")
    return modes
