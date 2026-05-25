"""Paper trading league configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PaperLeagueConfig:
    """
    Walk-forward paper league: train (learn params) -> test (black-box paper trade).

    Analogous to calibrating a model on in-sample data and validating out-of-sample
    (e.g. Black-Scholes implied vol fit on history, PnL tested forward).
    """

    symbol: str = "RELIANCE.NS"
    timeframe: str = "5m"
    days: Optional[int] = 60
    data_source: Optional[str] = None

    strategy_dir: Path = Path("strategies/generated")
    param_grid_path: Optional[Path] = Path("strategies/grids/ema_grid.json")

    train_bars: int = 156
    """~2 sessions of 5m bars for parameter learning."""
    test_bars: int = 78
    """Held-out window for paper trading (never used in param search)."""
    fold_step_bars: int = 78
    """Advance between walk-forward folds."""

    max_candidates_per_strategy: int = 36
    max_workers: int = 2
    time_budget_sec: float = 45.0

    init_cash: float = 100_000.0
    rank_by: str = "profit_pct"
    enable_learning: bool = True
    """Narrow param_grid toward winners after each OOS fold."""

    output_dir: Path = Path("logs/paper_league")
    min_folds: int = 2
