"""Institutional 5m intraday backtesting configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ExecutionTiming(str, Enum):
    """When signals become fills (no lookahead)."""

    ON_CLOSE = "on_close"
    NEXT_OPEN = "next_open"


class PositionSizingMode(str, Enum):
    FIXED_QTY = "fixed_qty"
    FIXED_CAPITAL = "fixed_capital"
    RISK_PCT = "risk_pct"
    ATR_RISK = "atr_risk"


@dataclass(frozen=True)
class MarketCostModel:
    """Realistic Indian equity intraday costs."""

    brokerage_rate: float = 0.0003
    exchange_fee_rate: float = 0.0000345
    slippage_rate: float = 0.0003
    spread_rate: float = 0.0001

    @property
    def round_trip_rate(self) -> float:
        """Approximate all-in friction per round trip (both sides)."""
        per_side = self.brokerage_rate + self.exchange_fee_rate + self.slippage_rate + self.spread_rate
        return 2 * per_side


@dataclass(frozen=True)
class SessionRules:
    """NSE cash session (IST)."""

    market_open: str = "09:15"
    market_close: str = "15:30"
    square_off_time: str = "15:15"
    allow_overnight: bool = False


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


@dataclass(frozen=True)
class FilterThresholds:
    """Auto-reject weak strategies (OOS test period)."""

    min_profit_factor: float = 1.2
    max_drawdown_pct: float = 25.0
    min_sharpe: float = 0.5
    min_trades: int = 100
    min_expectancy: float = 0.0
    min_win_rate_pct: float = 0.0
    max_win_rate_std: float = 30.0


@dataclass(frozen=True)
class BenchmarkTargets:
    """Healthy 5m intraday reference levels."""

    profit_factor: float = 1.5
    sharpe: float = 1.2
    max_drawdown_pct: float = 15.0
    risk_reward: float = 1.5


@dataclass
class InstitutionalBacktestConfig:
    """
    Standardized framework for comparable strategy evaluation.

    Primary: 5m NSE intraday, 6+ months history, train/val/test split.
    """

    symbol: str = "ICICIBANK.NS"
    timeframe: str = "5m"
    min_history_days: int = 20
    preferred_history_days: int = 30
    days: Optional[int] = None

    initial_capital: float = 100_000.0
    splits: SplitRatios = field(default_factory=SplitRatios)
    costs: MarketCostModel = field(default_factory=MarketCostModel)
    session: SessionRules = field(default_factory=SessionRules)
    execution: ExecutionTiming = ExecutionTiming.ON_CLOSE
    sizing_mode: PositionSizingMode = PositionSizingMode.RISK_PCT
    risk_per_trade_pct: float = 0.01
    fixed_capital_per_trade: float = 10_000.0
    fixed_qty: float = 1.0

    filters: FilterThresholds = field(default_factory=FilterThresholds)
    benchmarks: BenchmarkTargets = field(default_factory=BenchmarkTargets)

    strategy_dirs: list[Path] = field(
        default_factory=lambda: [
            Path("strategies/intraday"),
            Path("strategies/generated"),
            Path("strategies/builtin"),
        ]
    )
    param_grid_path: Optional[Path] = Path("strategies/grids/ema_grid.json")
    max_candidates: int = 27
    time_budget_sec: float = 120.0

    walk_forward_train_bars: int = 1560
    walk_forward_test_bars: int = 390
    walk_forward_step_bars: int = 390
    monte_carlo_iterations: int = 500

    output_dir: Path = Path("logs/institutional")
    data_source: Optional[str] = None
    rank_by: str = "composite"
