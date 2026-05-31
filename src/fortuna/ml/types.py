"""Typed contracts for the ML signal-quality scorer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from fortuna.backtesting.standard.config import MarketCostModel


@dataclass(frozen=True)
class LabelConfig:
    horizon_bars: int = 3
    min_return_pct: float = 0.0
    include_hold: bool = False
    hold_sample_rate: float = 0.1
    cost_model: MarketCostModel = field(default_factory=MarketCostModel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_bars": self.horizon_bars,
            "min_return_pct": self.min_return_pct,
            "include_hold": self.include_hold,
            "hold_sample_rate": self.hold_sample_rate,
            "cost_model": {
                "brokerage_rate": self.cost_model.brokerage_rate,
                "exchange_fee_rate": self.cost_model.exchange_fee_rate,
                "slippage_rate": self.cost_model.slippage_rate,
                "spread_rate": self.cost_model.spread_rate,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelConfig":
        cost_raw = d.get("cost_model") or {}
        cost = MarketCostModel(
            brokerage_rate=float(cost_raw.get("brokerage_rate", 0.0003)),
            exchange_fee_rate=float(cost_raw.get("exchange_fee_rate", 0.0000345)),
            slippage_rate=float(cost_raw.get("slippage_rate", 0.0003)),
            spread_rate=float(cost_raw.get("spread_rate", 0.0001)),
        )
        return cls(
            horizon_bars=int(d.get("horizon_bars", 3)),
            min_return_pct=float(d.get("min_return_pct", 0.0)),
            include_hold=bool(d.get("include_hold", False)),
            hold_sample_rate=float(d.get("hold_sample_rate", 0.1)),
            cost_model=cost,
        )


@dataclass(frozen=True)
class SignalExample:
    """Labeled signal event. Features are built separately — not stored here."""

    bar_idx: int
    bar_time: Any  # pd.Timestamp
    symbol: str
    strategy_name: str
    action: str
    label: int
    forward_return_pct: float
    cost_adjusted_return_pct: float


@dataclass(frozen=True)
class SignalScoreResult:
    available: bool
    probability: float
    predicted_class: int
    model_id: Optional[str]
    reason: str
    feature_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unavailable(cls, reason: str = "ML scorer artifact unavailable") -> "SignalScoreResult":
        return cls(
            available=False,
            probability=0.5,
            predicted_class=-1,
            model_id=None,
            reason=reason,
        )


@dataclass(frozen=True)
class SplitRange:
    name: str
    start: str
    end: str
    bar_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplitRange":
        return cls(
            name=str(d["name"]),
            start=str(d["start"]),
            end=str(d["end"]),
            bar_count=int(d["bar_count"]),
        )


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: float = 0.0
    brier_score: float = 0.0
    n_samples: int = 0
    positive_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClassificationMetrics":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class ScorerMetadata:
    run_id: str
    symbol: str
    timeframe: str
    strategy_name: str = ""
    model_kind: str = "logistic"
    label_config: LabelConfig = field(default_factory=LabelConfig)
    feature_names: tuple[str, ...] = ()
    feature_schema_hash: str = ""
    split_ranges: list[SplitRange] = field(default_factory=list)
    train_metrics: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    oos_metrics: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    random_state: int = 42
    sklearn_version: str = ""
    created_at: str = ""
    verdict_passed: bool = False
    verdict_reasons: list[str] = field(default_factory=list)
    advisory_ready: bool = False

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_name": self.strategy_name,
            "model_kind": self.model_kind,
            "label_config": self.label_config.to_dict(),
            "feature_names": list(self.feature_names),
            "feature_schema_hash": self.feature_schema_hash,
            "split_ranges": [r.to_dict() for r in self.split_ranges],
            "train_metrics": self.train_metrics.to_dict(),
            "oos_metrics": self.oos_metrics.to_dict(),
            "random_state": self.random_state,
            "sklearn_version": self.sklearn_version,
            "created_at": self.created_at,
            "verdict_passed": self.verdict_passed,
            "verdict_reasons": list(self.verdict_reasons),
            "advisory_ready": self.advisory_ready,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScorerMetadata":
        split_ranges = [SplitRange.from_dict(r) for r in d.get("split_ranges", [])]
        return cls(
            run_id=str(d["run_id"]),
            symbol=str(d.get("symbol", "")),
            timeframe=str(d.get("timeframe", "")),
            strategy_name=str(d.get("strategy_name", "")),
            model_kind=str(d.get("model_kind", "logistic")),
            label_config=LabelConfig.from_dict(d.get("label_config") or {}),
            feature_names=tuple(d.get("feature_names") or ()),
            feature_schema_hash=str(d.get("feature_schema_hash", "")),
            split_ranges=split_ranges,
            train_metrics=ClassificationMetrics.from_dict(d.get("train_metrics") or {}),
            oos_metrics=ClassificationMetrics.from_dict(d.get("oos_metrics") or {}),
            random_state=int(d.get("random_state", 42)),
            sklearn_version=str(d.get("sklearn_version", "")),
            created_at=str(d.get("created_at", "")),
            verdict_passed=bool(d.get("verdict_passed", False)),
            verdict_reasons=list(d.get("verdict_reasons") or []),
            advisory_ready=bool(
                d.get("advisory_ready", d.get("verdict_passed", False))
            ),
        )
