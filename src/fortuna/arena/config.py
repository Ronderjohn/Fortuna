"""Arena run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ArenaConfig:
    """Multi-strategy competition on streamed DuckDB-backed OHLCV."""

    symbol: str = "RELIANCE.NS"
    timeframe: str = "5m"
    days: Optional[int] = 30
    data_source: Optional[str] = None

    strategy_dir: Path = Path("strategies/generated")
    param_grid_path: Optional[Path] = None
    max_candidates_per_strategy: int = 50
    max_workers: int = 2
    chunk_size: int = 10
    time_budget_sec: float = 60.0

    window_bars: int = 78
    warmup_bars: int = 78
    stream_step: int = 1
    """1 = evaluate every bar; increase to sample fewer windows on long histories."""

    use_streaming: bool = True
    """If False, run one full-window param search only (faster smoke tests)."""

    eval_full_series: bool = True
    """Backtest on entire loaded OHLCV (TradingView-style), not only trailing window."""

    rank_by: str = "profit_pct"
    """Leaderboard sort: profit_pct | win_ratio_pct | composite_score."""

    output_dir: Path = Path("logs/arena")
    parallel_strategies: bool = True
