"""Pydantic models for the strategy JSON DSL."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class MarketType(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"


class TradeSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class IndicatorType(str, Enum):
    EMA = "ema"
    SMA = "sma"
    RSI = "rsi"
    ATR = "atr"
    VWAP = "vwap"
    MACD = "macd"
    BOLLINGER = "bollinger"
    VOLUME_SMA = "volume_sma"
    ROLLING_HIGH = "rolling_high"
    ROLLING_LOW = "rolling_low"


class PriceSource(str, Enum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    HLC3 = "hlc3"
    OHLC4 = "ohlc4"
    VOLUME = "volume"


class StopLossType(str, Enum):
    PERCENT = "percent"
    ATR = "atr"


class PositionSizingType(str, Enum):
    FIXED_FRACTION = "fixed_fraction"
    FIXED_UNITS = "fixed_units"


class IndicatorSpec(BaseModel):
    """Technical indicator configuration."""

    id: str
    type: IndicatorType
    params: dict[str, Any] = Field(default_factory=dict)
    source: PriceSource = PriceSource.CLOSE


class CompareCondition(BaseModel):
    """Compare two series or a series to a constant."""

    type: Literal["compare"] = "compare"
    left: str
    operator: Literal["gt", "gte", "lt", "lte", "eq", "neq"]
    right: Union[str, float, int]


class CrossoverCondition(BaseModel):
    """Left crosses above right."""

    type: Literal["crossover"] = "crossover"
    left: str
    right: str


class CrossunderCondition(BaseModel):
    """Left crosses below right."""

    type: Literal["crossunder"] = "crossunder"
    left: str
    right: str


class AndCondition(BaseModel):
    type: Literal["and"] = "and"
    conditions: list["Condition"]


class OrCondition(BaseModel):
    type: Literal["or"] = "or"
    conditions: list["Condition"]


class NotCondition(BaseModel):
    type: Literal["not"] = "not"
    condition: "Condition"


Condition = Annotated[
    Union[
        CompareCondition,
        CrossoverCondition,
        CrossunderCondition,
        AndCondition,
        OrCondition,
        NotCondition,
    ],
    Field(discriminator="type"),
]

AndCondition.model_rebuild()
OrCondition.model_rebuild()
NotCondition.model_rebuild()


class RuleBlock(BaseModel):
    """Entry, exit, and filter rules."""

    entry_conditions: list[Condition] = Field(default_factory=list)
    exit_conditions: list[Condition] = Field(default_factory=list)
    filters: list[Condition] = Field(default_factory=list)


class StopLoss(BaseModel):
    type: StopLossType = StopLossType.PERCENT
    value: float = 0.02
    atr_indicator_id: Optional[str] = None
    atr_multiplier: float = 2.0


class TakeProfit(BaseModel):
    type: Literal["percent"] = "percent"
    value: float = 0.04


class PositionSizing(BaseModel):
    type: PositionSizingType = PositionSizingType.FIXED_FRACTION
    value: float = 0.95


class RiskManagement(BaseModel):
    """Risk and position sizing parameters."""

    stop_loss: Optional[StopLoss] = None
    take_profit: Optional[TakeProfit] = None
    position_sizing: PositionSizing = Field(default_factory=PositionSizing)


class StrategyMetadata(BaseModel):
    """Strategy metadata for tracking and agent context."""

    author: str = "system"
    version: str = "1.0"
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    parameter_count: int = 0
    engine: Optional[str] = None
    """Builtin engine id, e.g. ``mmts`` for Pine-parity strategies."""


class StrategyDefinition(BaseModel):
    """Complete strategy definition loaded from JSON."""

    name: str
    market: MarketType = MarketType.EQUITY
    symbol: str = "AAPL"
    timeframe: str = "1d"
    side: TradeSide = TradeSide.LONG
    indicators: list[IndicatorSpec] = Field(default_factory=list)
    rules: RuleBlock = Field(default_factory=RuleBlock)
    risk: RiskManagement = Field(default_factory=RiskManagement)
    metadata: StrategyMetadata = Field(default_factory=StrategyMetadata)
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)
