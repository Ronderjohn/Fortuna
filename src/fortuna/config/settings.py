"""Centralized settings with YAML and environment variable support."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ScoringWeights(BaseSettings):
    sharpe: float = 0.35
    total_return: float = 0.15
    drawdown_penalty: float = 0.25
    profit_factor: float = 0.15
    consistency: float = 0.10


class Settings(BaseSettings):
    """Application settings merged from YAML and environment."""

    model_config = SettingsConfigDict(
        env_prefix="FORTUNA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "development"
    log_level: str = "INFO"

    data_cache_dir: Path = Field(default=Path("data/market_cache"))
    duckdb_path: Path = Field(default=Path("data/fortuna.duckdb"))
    default_symbol: str = "AAPL"
    default_timeframe: str = "1d"
    default_days: int = 30
    data_source: str = "yfinance"
    cache_max_age_days: int = 1
    cache_max_age_hours: Optional[int] = None

    init_cash: float = 100_000.0
    fees: float = 0.001
    slippage: float = 0.0005

    validation_score_threshold: float = 50.0
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    max_drawdown_threshold: float = 0.25
    min_trades: int = 10
    min_profit_factor: float = 1.0

    strategies_generated: Path = Field(default=Path("strategies/generated"))
    strategies_validated: Path = Field(default=Path("strategies/validated"))
    strategies_rejected: Path = Field(default=Path("strategies/rejected"))
    logs_dir: Path = Field(default=Path("logs"))

    project_root: Path = Field(default=_PROJECT_ROOT)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "Settings":
        """Build settings from default.yaml merged with env."""
        load_dotenv(_PROJECT_ROOT / ".env")
        yaml_path = yaml_path or (_PROJECT_ROOT / "configs" / "default.yaml")
        raw = _load_yaml_config(yaml_path)

        flat: dict[str, Any] = {}
        if "env" in raw:
            flat["env"] = raw["env"]
        if "log_level" in raw:
            flat["log_level"] = raw["log_level"]

        data = raw.get("data", {})
        if data:
            if "cache_dir" in data:
                flat["data_cache_dir"] = Path(data["cache_dir"])
            if "duckdb_path" in data:
                flat["duckdb_path"] = Path(data["duckdb_path"])
            if "default_symbol" in data:
                flat["default_symbol"] = data["default_symbol"]
            if "default_timeframe" in data:
                flat["default_timeframe"] = data["default_timeframe"]
            if "default_days" in data:
                flat["default_days"] = int(data["default_days"])
            if "cache_max_age_days" in data:
                flat["cache_max_age_days"] = data["cache_max_age_days"]
            if "cache_max_age_hours" in data:
                flat["cache_max_age_hours"] = data["cache_max_age_hours"]
            if "data_source" in data:
                flat["data_source"] = data["data_source"]

        backtest = raw.get("backtest", {})
        if backtest:
            flat["init_cash"] = backtest.get("init_cash", 100_000.0)
            flat["fees"] = backtest.get("fees", 0.001)
            flat["slippage"] = backtest.get("slippage", 0.0005)

        scoring = raw.get("scoring", {})
        if scoring:
            flat["validation_score_threshold"] = scoring.get("validation_threshold", 50.0)
            flat["max_drawdown_threshold"] = scoring.get("max_drawdown_threshold", 0.25)
            flat["min_trades"] = scoring.get("min_trades", 10)
            flat["min_profit_factor"] = scoring.get("min_profit_factor", 1.0)
            weights = scoring.get("weights", {})
            if weights:
                flat["scoring_weights"] = ScoringWeights(**weights)

        paths = raw.get("paths", {})
        if paths:
            if "strategies_generated" in paths:
                flat["strategies_generated"] = Path(paths["strategies_generated"])
            if "strategies_validated" in paths:
                flat["strategies_validated"] = Path(paths["strategies_validated"])
            if "strategies_rejected" in paths:
                flat["strategies_rejected"] = Path(paths["strategies_rejected"])
            if "logs" in paths:
                flat["logs_dir"] = Path(paths["logs"])

        flat["project_root"] = _PROJECT_ROOT
        return cls(**flat)

    def resolve_path(self, path: Path) -> Path:
        """Resolve relative paths against project root."""
        if path.is_absolute():
            return path
        return self.project_root / path


@lru_cache
def load_settings(yaml_path: Optional[Path] = None) -> Settings:
    """Load and cache settings (``FORTUNA_CONFIG`` env → path under project root)."""
    import os

    if yaml_path is None:
        cfg = os.environ.get("FORTUNA_CONFIG", "").strip()
        if cfg:
            yaml_path = _PROJECT_ROOT / cfg
    return Settings.from_yaml(yaml_path)


def get_settings() -> Settings:
    """Return cached settings instance."""
    return load_settings()
