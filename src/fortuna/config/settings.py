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

    # ─── Live execution (Tier 1) ────────────────────────────────────────
    # Default OFF: existing CLI / notebook users see no behaviour change.
    # Turn on via FORTUNA_EXECUTION_ENABLED=1 in .env or settings YAML to
    # have the dashboard route live signals through the paper broker.
    execution_enabled: bool = False
    execution_config_path: Path = Field(default=Path("config/execution.yaml"))
    execution_broker: str = "paper"  # one of: "paper", "smartapi" (stub)

    # ─── Portfolio-aware paper simulation (Phase 1) ─────────────────────
    # Read-only sync of cash + positions + holdings from SmartAPI into the
    # paper LiveAccount. NEVER places real orders — those go through the
    # broker, which still requires execution_broker="smartapi" (stub).
    # Default OFF so users opt in explicitly.
    execution_account_sync: bool = False
    # Re-pull the account snapshot every N minutes during the session
    # (0 = sync once at start only).
    execution_account_refresh_minutes: int = 0

    # ─── Agentic advisory layer ──────────────────────────────────────────
    # Default OFF: the existing dashboard/signal workflow remains unchanged
    # unless explicitly enabled.
    agentic_enabled: bool = False
    agentic_paper_learning_enabled: bool = False
    agentic_log_dir: Path = Field(default=Path("logs/agentic"))
    agentic_ml_scorer_enabled: bool = False
    ml_scorer_artifact_dir: Path = Field(default=Path("models/ml_signal_scorer/validated"))

    model_registry_enabled: bool = False
    model_promotion_required: bool = True

    # RL inference: when False (default), only per-symbol checkpoints are used.
    rl_allow_global_policy: bool = False

    # Telegram notifications (bot token + chat id).
    telegram_enabled: bool = False
    telegram_provider: str = "telegram"
    telegram_dedupe_memory: int = 500
    telegram_max_per_symbol_per_session: int = 10
    telegram_min_interval_seconds: int = 300
    telegram_quiet_hours_enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_request_audit_enabled: bool = True
    conversational_adapter_enabled: bool = False
    conversational_adapter_mode: str = "heuristic"
    conversational_max_history: int = 12

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

        execution = raw.get("execution", {})
        if execution:
            if "enabled" in execution:
                flat["execution_enabled"] = bool(execution["enabled"])
            if "config_path" in execution:
                flat["execution_config_path"] = Path(execution["config_path"])
            if "broker" in execution:
                flat["execution_broker"] = str(execution["broker"])
            if "account_sync" in execution:
                flat["execution_account_sync"] = bool(execution["account_sync"])
            if "account_refresh_minutes" in execution:
                flat["execution_account_refresh_minutes"] = int(
                    execution["account_refresh_minutes"]
                )

        agentic = raw.get("agentic", {})
        if agentic:
            if "enabled" in agentic:
                flat["agentic_enabled"] = bool(agentic["enabled"])
            if "paper_learning_enabled" in agentic:
                flat["agentic_paper_learning_enabled"] = bool(agentic["paper_learning_enabled"])
            if "log_dir" in agentic:
                flat["agentic_log_dir"] = Path(agentic["log_dir"])
            if "ml_scorer_enabled" in agentic:
                flat["agentic_ml_scorer_enabled"] = bool(agentic["ml_scorer_enabled"])
            if "ml_scorer_artifact_dir" in agentic:
                flat["ml_scorer_artifact_dir"] = Path(agentic["ml_scorer_artifact_dir"])
            telegram = agentic.get("telegram", {}) or {}
            if telegram:
                if "enabled" in telegram:
                    flat["telegram_enabled"] = bool(telegram["enabled"])
                if "provider" in telegram:
                    flat["telegram_provider"] = str(telegram["provider"])
                if "dedupe_memory" in telegram:
                    flat["telegram_dedupe_memory"] = int(telegram["dedupe_memory"])
                if "max_per_symbol_per_session" in telegram:
                    flat["telegram_max_per_symbol_per_session"] = int(
                        telegram["max_per_symbol_per_session"]
                    )
                if "min_interval_seconds" in telegram:
                    flat["telegram_min_interval_seconds"] = int(
                        telegram["min_interval_seconds"]
                    )
                if "quiet_hours_enabled" in telegram:
                    flat["telegram_quiet_hours_enabled"] = bool(
                        telegram["quiet_hours_enabled"]
                    )
                if "bot_token" in telegram:
                    flat["telegram_bot_token"] = str(telegram["bot_token"])
                if "chat_id" in telegram:
                    flat["telegram_chat_id"] = str(telegram["chat_id"])
                if "request_audit_enabled" in telegram:
                    flat["telegram_request_audit_enabled"] = bool(
                        telegram["request_audit_enabled"]
                    )
                if "conversational_adapter_enabled" in telegram:
                    flat["conversational_adapter_enabled"] = bool(
                        telegram["conversational_adapter_enabled"]
                    )
                if "conversational_adapter_mode" in telegram:
                    flat["conversational_adapter_mode"] = str(
                        telegram["conversational_adapter_mode"]
                    )
                if "conversational_max_history" in telegram:
                    flat["conversational_max_history"] = int(
                        telegram["conversational_max_history"]
                    )

        models = raw.get("models", {})
        if models:
            if "registry_enabled" in models:
                flat["model_registry_enabled"] = bool(models["registry_enabled"])
            if "promotion_required" in models:
                flat["model_promotion_required"] = bool(models["promotion_required"])

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
