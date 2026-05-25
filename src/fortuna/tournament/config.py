"""Tournament configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TournamentConfig:
    symbol: str = "RELIANCE.NS"
    timeframe: str = "5m"
    window_bars: int = 78
    warmup_bars: int = 78
    time_budget_sec: float = 30.0
    max_workers: int = 2
    chunk_size: int = 10
    max_candidates: int = 200
    tier: int = 1
    parallel: bool = False
    vectorbt_confirm_top_k: int = 0
    data_source: Optional[str] = None
    days: int = 30
    bar_step: int = 1
    """Evaluate every Nth bar (use 5–15 on 90d histories to save time)."""
    strategy_paths: list[Path] = field(default_factory=list)
    param_grid_path: Optional[Path] = None
    output_dir: Path = Path("logs/tournament")
