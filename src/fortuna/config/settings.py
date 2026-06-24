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


def _split_csv_tokens(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        tokens = [str(item).strip() for item in value if str(item).strip()]
        return tuple(tokens)
    text = str(value).strip()
    if not text:
        return ()
    return tuple(token.strip() for token in text.split(",") if token.strip())


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
    market_universe_screener_csv: Path = Field(
        default=Path("data/universe/screener_export.csv")
    )
    market_universe_tradable_universe_csv: Path = Field(
        default=Path("data/universe/tradable_universe.csv")
    )
    market_universe_smartapi_seed_limit: int = 75
    market_universe_auto_prefer_smartapi: bool = False
    market_universe_default_limit: int = 15
    market_universe_adaptive_weighting_enabled: bool = True
    market_universe_adaptive_max_boost: float = 0.2
    market_universe_turnover_weight: float = 0.46
    market_universe_volume_weight: float = 0.18
    market_universe_trend_weight: float = 0.08
    market_universe_volume_ratio_weight: float = 0.10
    market_universe_activity_weight: float = 0.18
    market_universe_action_bias_weight: float = 0.45
    market_universe_regime_bias_weight: float = 0.35
    market_universe_signal_bias_weight: float = 0.25
    market_universe_global_bias_weight: float = 0.2
    market_universe_adaptive_recency_halflife: float = 18.0
    market_universe_nightly_feedback_enabled: bool = True
    market_universe_nightly_feedback_max_boost: float = 0.1
    market_universe_nightly_feedback_lookback_reports: int = 8
    market_universe_nightly_refresh_bonus_weight: float = 0.65
    market_universe_nightly_report_dir: Path = Field(default=Path("reports/nightly"))
    market_universe_long_horizon_feedback_enabled: bool = False
    market_universe_long_horizon_lookback_reports: int = 12
    market_universe_executed_lane_boost_weight: float = 0.04
    market_universe_long_horizon_halflife: float = 48.0
    market_universe_fundamentals_overlay_enabled: bool = False
    market_universe_fundamentals_overlay_csv: Path = Field(
        default=Path("data/universe/fundamentals_overlay.csv")
    )
    market_universe_fundamentals_overlay_max_boost: float = 0.05
    market_universe_fundamentals_quality_weight: float = 1.0
    scaling_basket_small_max: int = 8
    scaling_basket_medium_max: int = 15
    scaling_enforce_basket_cap: bool = False
    nightly_report_retention_enabled: bool = False
    nightly_report_retention_count: int = 30
    observability_enabled: bool = False
    observability_log_dir: Path = Field(default=Path("logs/observability"))
    observability_exporter: str = "local"
    observability_otlp_endpoint: str = ""
    nightly_automation_enabled: bool = False
    nightly_fail_fast_global_blockers: bool = True
    nightly_lane_gating_enabled: bool = True
    training_candidate_adaptive_weighting_enabled: bool = True
    training_candidate_adaptive_max_boost: float = 0.18
    training_candidate_volume_ratio_bonus_weight: float = 0.08
    training_candidate_trending_bonus_weight: float = 0.04
    training_candidate_scout_overlap_boost: float = 0.04
    training_candidate_scout_volume_dense_boost: float = 0.03
    training_candidate_nightly_feedback_enabled: bool = True
    training_candidate_nightly_feedback_max_boost: float = 0.08
    training_candidate_nightly_feedback_lookback_reports: int = 8
    training_candidate_nightly_refresh_bonus_weight: float = 0.7
    training_candidate_nightly_report_dir: Path = Field(default=Path("reports/nightly"))
    training_candidate_remediation_pressure_enabled: bool = True
    training_candidate_remediation_max_boost: float = 0.06
    training_candidate_setup_family_reinforcement_enabled: bool = True
    training_candidate_setup_family_min_durable_rows: int = 2
    training_candidate_setup_family_max_boost: float = 0.05
    shortlist_discovery_alignment_enabled: bool = True
    shortlist_discovery_alignment_boost: float = 0.05
    shortlist_discovery_alignment_penalty: float = 0.03
    portfolio_allocator_context_weighting_enabled: bool = True
    portfolio_allocator_regime_weight: float = 0.06
    portfolio_allocator_activity_weight: float = 0.04
    portfolio_allocator_max_single_weight: float = 0.6
    portfolio_allocator_max_high_risk_positions: int = 1
    portfolio_allocator_max_per_regime: int = 2
    portfolio_allocator_critic_enabled: bool = True
    portfolio_allocator_critic_same_side_penalty: float = 0.05
    portfolio_allocator_critic_same_regime_penalty: float = 0.04
    portfolio_allocator_critic_same_strategy_penalty: float = 0.035
    portfolio_allocator_critic_open_same_exposure_penalty: float = 0.03
    portfolio_allocator_critic_weaker_same_side_penalty: float = 0.025
    portfolio_allocator_critic_watch_penalty: float = 0.04
    portfolio_allocator_critic_high_risk_penalty: float = 0.06
    portfolio_allocator_discovery_alignment_enabled: bool = True
    portfolio_allocator_critic_weaker_discovery_penalty: float = 0.025
    portfolio_allocator_research_alignment_enabled: bool = False
    portfolio_allocator_critic_missing_research_penalty: float = 0.04
    portfolio_allocator_critic_weaker_research_penalty: float = 0.03
    portfolio_allocator_critic_unrefreshed_research_penalty: float = 0.02
    portfolio_allocator_critic_weaker_nightly_research_penalty: float = 0.025
    portfolio_allocator_critic_block_threshold: float = 0.14
    portfolio_allocator_nightly_feedback_enabled: bool = True
    portfolio_allocator_nightly_feedback_lookback_reports: int = 8
    portfolio_allocator_nightly_refresh_bonus_weight: float = 0.65
    portfolio_allocator_nightly_promote_bonus_weight: float = 0.3
    portfolio_allocator_nightly_report_dir: Path = Path("reports/nightly")
    portfolio_allocator_freshness_preference_enabled: bool = True
    portfolio_allocator_freshness_preference_weight: float = 0.03
    portfolio_allocator_same_target_balance_enabled: bool = True
    portfolio_allocator_same_target_balance_penalty: float = 0.035
    model_status_nightly_alignment_enabled: bool = True
    model_status_nightly_alignment_lookback_reports: int = 8
    model_status_nightly_report_dir: Path = Path("reports/nightly")
    model_status_nightly_alignment_recommendation_ratio: float = 0.75
    acceptance_bundle_nightly_alignment_min_enabled_reports: int = 2
    acceptance_bundle_nightly_alignment_warn_ratio: float = 0.6
    acceptance_bundle_nightly_refreshed_alignment_warn_ratio: float = 0.35
    acceptance_bundle_team_research_alignment_warn_ratio: float = 0.6
    acceptance_bundle_team_discovery_alignment_warn_ratio: float = 0.6

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
    telegram_allowed_chat_ids: str = ""
    telegram_admin_chat_ids: str = ""
    telegram_request_audit_enabled: bool = True
    conversational_adapter_enabled: bool = False
    conversational_adapter_mode: str = "heuristic"
    conversational_max_history: int = 12
    signal_max_concurrent_requests: int = 4
    signal_max_concurrent_per_user: int = 1
    signal_max_pending_per_user: int = 1
    signal_timeout_seconds: int = 18
    signal_slow_command_timeout_seconds: int = 45
    signal_result_ttl_seconds: int = 20
    signal_cache_enabled: bool = True
    signal_intent_cache_ttl_seconds: int = 120
    signal_image_cache_ttl_seconds: int = 300
    signal_rl_mode: str = "warm"
    signal_ml_enabled_in_request_path: bool = True
    signal_expert_commands_visible: bool = False
    signal_reply_style: str = "compact"
    signal_image_input_enabled: bool = False
    signal_image_max_bytes: int = 3_000_000
    signal_image_temp_dir: Path = Field(default=Path("logs/telegram/tmp_images"))
    signal_image_temp_retention_minutes: int = 10
    signal_agent_timeout_seconds: int = 12
    signal_request_audit_retention_days: int = 7
    signal_request_audit_retention_count: int = 5_000
    signal_session_retention_days: int = 30
    signal_secret_redaction_enabled: bool = True
    signal_openai_required_for_images: bool = True
    signal_forecast_enabled: bool = True
    signal_forecast_trigger_confidence_threshold: float = 0.62
    signal_forecast_cache_ttl_seconds: int = 60
    signal_forecast_max_bars: int = 240
    signal_forecast_max_tasks_per_request: int = 2
    signal_replay_enabled: bool = True
    signal_replay_cache_ttl_seconds: int = 60
    signal_replay_max_bars: int = 180
    signal_replay_hold_bars: int = 12
    signal_futures_oi_enabled: bool = True
    signal_code_execution_enabled: bool = True
    signal_code_execution_timeout_seconds: int = 4
    signal_code_execution_max_output_chars: int = 1200
    signal_code_execution_temp_dir: Path = Field(default=Path("logs/telegram/tmp_code"))
    signal_code_execution_retention_minutes: int = 10
    signal_code_execution_allow_live_generation: bool = True
    signal_code_execution_fallback_only: bool = False
    signal_forecast_audit_enabled: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: int = 30
    openai_max_output_tokens: int = 250

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
            if "market_universe_screener_csv" in agentic:
                flat["market_universe_screener_csv"] = Path(
                    agentic["market_universe_screener_csv"]
                )
            if "market_universe_tradable_universe_csv" in agentic:
                flat["market_universe_tradable_universe_csv"] = Path(
                    agentic["market_universe_tradable_universe_csv"]
                )
            if "market_universe_smartapi_seed_limit" in agentic:
                flat["market_universe_smartapi_seed_limit"] = int(
                    agentic["market_universe_smartapi_seed_limit"]
                )
            if "market_universe_auto_prefer_smartapi" in agentic:
                flat["market_universe_auto_prefer_smartapi"] = bool(
                    agentic["market_universe_auto_prefer_smartapi"]
                )
            if "market_universe_default_limit" in agentic:
                flat["market_universe_default_limit"] = int(
                    agentic["market_universe_default_limit"]
                )
            if "market_universe_adaptive_weighting_enabled" in agentic:
                flat["market_universe_adaptive_weighting_enabled"] = bool(
                    agentic["market_universe_adaptive_weighting_enabled"]
                )
            if "market_universe_adaptive_max_boost" in agentic:
                flat["market_universe_adaptive_max_boost"] = float(
                    agentic["market_universe_adaptive_max_boost"]
                )
            if "market_universe_turnover_weight" in agentic:
                flat["market_universe_turnover_weight"] = float(
                    agentic["market_universe_turnover_weight"]
                )
            if "market_universe_volume_weight" in agentic:
                flat["market_universe_volume_weight"] = float(
                    agentic["market_universe_volume_weight"]
                )
            if "market_universe_trend_weight" in agentic:
                flat["market_universe_trend_weight"] = float(
                    agentic["market_universe_trend_weight"]
                )
            if "market_universe_volume_ratio_weight" in agentic:
                flat["market_universe_volume_ratio_weight"] = float(
                    agentic["market_universe_volume_ratio_weight"]
                )
            if "market_universe_activity_weight" in agentic:
                flat["market_universe_activity_weight"] = float(
                    agentic["market_universe_activity_weight"]
                )
            if "market_universe_action_bias_weight" in agentic:
                flat["market_universe_action_bias_weight"] = float(
                    agentic["market_universe_action_bias_weight"]
                )
            if "market_universe_regime_bias_weight" in agentic:
                flat["market_universe_regime_bias_weight"] = float(
                    agentic["market_universe_regime_bias_weight"]
                )
            if "market_universe_signal_bias_weight" in agentic:
                flat["market_universe_signal_bias_weight"] = float(
                    agentic["market_universe_signal_bias_weight"]
                )
            if "market_universe_global_bias_weight" in agentic:
                flat["market_universe_global_bias_weight"] = float(
                    agentic["market_universe_global_bias_weight"]
                )
            if "market_universe_adaptive_recency_halflife" in agentic:
                flat["market_universe_adaptive_recency_halflife"] = float(
                    agentic["market_universe_adaptive_recency_halflife"]
                )
            if "market_universe_nightly_feedback_enabled" in agentic:
                flat["market_universe_nightly_feedback_enabled"] = bool(
                    agentic["market_universe_nightly_feedback_enabled"]
                )
            if "market_universe_nightly_feedback_max_boost" in agentic:
                flat["market_universe_nightly_feedback_max_boost"] = float(
                    agentic["market_universe_nightly_feedback_max_boost"]
                )
            if "market_universe_nightly_feedback_lookback_reports" in agentic:
                flat["market_universe_nightly_feedback_lookback_reports"] = int(
                    agentic["market_universe_nightly_feedback_lookback_reports"]
                )
            if "market_universe_nightly_refresh_bonus_weight" in agentic:
                flat["market_universe_nightly_refresh_bonus_weight"] = float(
                    agentic["market_universe_nightly_refresh_bonus_weight"]
                )
            if "market_universe_nightly_report_dir" in agentic:
                flat["market_universe_nightly_report_dir"] = Path(
                    agentic["market_universe_nightly_report_dir"]
                )
            if "market_universe_long_horizon_feedback_enabled" in agentic:
                flat["market_universe_long_horizon_feedback_enabled"] = bool(
                    agentic["market_universe_long_horizon_feedback_enabled"]
                )
            if "market_universe_long_horizon_lookback_reports" in agentic:
                flat["market_universe_long_horizon_lookback_reports"] = int(
                    agentic["market_universe_long_horizon_lookback_reports"]
                )
            if "market_universe_executed_lane_boost_weight" in agentic:
                flat["market_universe_executed_lane_boost_weight"] = float(
                    agentic["market_universe_executed_lane_boost_weight"]
                )
            if "market_universe_long_horizon_halflife" in agentic:
                flat["market_universe_long_horizon_halflife"] = float(
                    agentic["market_universe_long_horizon_halflife"]
                )
            if "market_universe_fundamentals_overlay_enabled" in agentic:
                flat["market_universe_fundamentals_overlay_enabled"] = bool(
                    agentic["market_universe_fundamentals_overlay_enabled"]
                )
            if "market_universe_fundamentals_overlay_csv" in agentic:
                flat["market_universe_fundamentals_overlay_csv"] = Path(
                    agentic["market_universe_fundamentals_overlay_csv"]
                )
            if "market_universe_fundamentals_overlay_max_boost" in agentic:
                flat["market_universe_fundamentals_overlay_max_boost"] = float(
                    agentic["market_universe_fundamentals_overlay_max_boost"]
                )
            if "market_universe_fundamentals_quality_weight" in agentic:
                flat["market_universe_fundamentals_quality_weight"] = float(
                    agentic["market_universe_fundamentals_quality_weight"]
                )
            if "scaling_basket_small_max" in agentic:
                flat["scaling_basket_small_max"] = int(agentic["scaling_basket_small_max"])
            if "scaling_basket_medium_max" in agentic:
                flat["scaling_basket_medium_max"] = int(agentic["scaling_basket_medium_max"])
            if "scaling_enforce_basket_cap" in agentic:
                flat["scaling_enforce_basket_cap"] = bool(agentic["scaling_enforce_basket_cap"])
            if "nightly_report_retention_enabled" in agentic:
                flat["nightly_report_retention_enabled"] = bool(
                    agentic["nightly_report_retention_enabled"]
                )
            if "nightly_report_retention_count" in agentic:
                flat["nightly_report_retention_count"] = int(
                    agentic["nightly_report_retention_count"]
                )
            if "observability_enabled" in agentic:
                flat["observability_enabled"] = bool(agentic["observability_enabled"])
            if "observability_log_dir" in agentic:
                flat["observability_log_dir"] = Path(agentic["observability_log_dir"])
            if "observability_exporter" in agentic:
                flat["observability_exporter"] = str(agentic["observability_exporter"])
            if "observability_otlp_endpoint" in agentic:
                flat["observability_otlp_endpoint"] = str(agentic["observability_otlp_endpoint"])
            if "nightly_automation_enabled" in agentic:
                flat["nightly_automation_enabled"] = bool(
                    agentic["nightly_automation_enabled"]
                )
            if "nightly_fail_fast_global_blockers" in agentic:
                flat["nightly_fail_fast_global_blockers"] = bool(
                    agentic["nightly_fail_fast_global_blockers"]
                )
            if "nightly_lane_gating_enabled" in agentic:
                flat["nightly_lane_gating_enabled"] = bool(
                    agentic["nightly_lane_gating_enabled"]
                )
            if "training_candidate_adaptive_weighting_enabled" in agentic:
                flat["training_candidate_adaptive_weighting_enabled"] = bool(
                    agentic["training_candidate_adaptive_weighting_enabled"]
                )
            if "training_candidate_adaptive_max_boost" in agentic:
                flat["training_candidate_adaptive_max_boost"] = float(
                    agentic["training_candidate_adaptive_max_boost"]
                )
            if "training_candidate_volume_ratio_bonus_weight" in agentic:
                flat["training_candidate_volume_ratio_bonus_weight"] = float(
                    agentic["training_candidate_volume_ratio_bonus_weight"]
                )
            if "training_candidate_trending_bonus_weight" in agentic:
                flat["training_candidate_trending_bonus_weight"] = float(
                    agentic["training_candidate_trending_bonus_weight"]
                )
            if "training_candidate_scout_overlap_boost" in agentic:
                flat["training_candidate_scout_overlap_boost"] = float(
                    agentic["training_candidate_scout_overlap_boost"]
                )
            if "training_candidate_scout_volume_dense_boost" in agentic:
                flat["training_candidate_scout_volume_dense_boost"] = float(
                    agentic["training_candidate_scout_volume_dense_boost"]
                )
            if "training_candidate_nightly_feedback_enabled" in agentic:
                flat["training_candidate_nightly_feedback_enabled"] = bool(
                    agentic["training_candidate_nightly_feedback_enabled"]
                )
            if "training_candidate_nightly_feedback_max_boost" in agentic:
                flat["training_candidate_nightly_feedback_max_boost"] = float(
                    agentic["training_candidate_nightly_feedback_max_boost"]
                )
            if "training_candidate_nightly_feedback_lookback_reports" in agentic:
                flat["training_candidate_nightly_feedback_lookback_reports"] = int(
                    agentic["training_candidate_nightly_feedback_lookback_reports"]
                )
            if "training_candidate_nightly_refresh_bonus_weight" in agentic:
                flat["training_candidate_nightly_refresh_bonus_weight"] = float(
                    agentic["training_candidate_nightly_refresh_bonus_weight"]
                )
            if "training_candidate_nightly_report_dir" in agentic:
                flat["training_candidate_nightly_report_dir"] = Path(
                    agentic["training_candidate_nightly_report_dir"]
                )
            if "training_candidate_remediation_pressure_enabled" in agentic:
                flat["training_candidate_remediation_pressure_enabled"] = bool(
                    agentic["training_candidate_remediation_pressure_enabled"]
                )
            if "training_candidate_remediation_max_boost" in agentic:
                flat["training_candidate_remediation_max_boost"] = float(
                    agentic["training_candidate_remediation_max_boost"]
                )
            if "training_candidate_setup_family_reinforcement_enabled" in agentic:
                flat["training_candidate_setup_family_reinforcement_enabled"] = bool(
                    agentic["training_candidate_setup_family_reinforcement_enabled"]
                )
            if "training_candidate_setup_family_min_durable_rows" in agentic:
                flat["training_candidate_setup_family_min_durable_rows"] = int(
                    agentic["training_candidate_setup_family_min_durable_rows"]
                )
            if "training_candidate_setup_family_max_boost" in agentic:
                flat["training_candidate_setup_family_max_boost"] = float(
                    agentic["training_candidate_setup_family_max_boost"]
                )
            if "shortlist_discovery_alignment_enabled" in agentic:
                flat["shortlist_discovery_alignment_enabled"] = bool(
                    agentic["shortlist_discovery_alignment_enabled"]
                )
            if "shortlist_discovery_alignment_boost" in agentic:
                flat["shortlist_discovery_alignment_boost"] = float(
                    agentic["shortlist_discovery_alignment_boost"]
                )
            if "shortlist_discovery_alignment_penalty" in agentic:
                flat["shortlist_discovery_alignment_penalty"] = float(
                    agentic["shortlist_discovery_alignment_penalty"]
                )
            if "portfolio_allocator_context_weighting_enabled" in agentic:
                flat["portfolio_allocator_context_weighting_enabled"] = bool(
                    agentic["portfolio_allocator_context_weighting_enabled"]
                )
            if "portfolio_allocator_regime_weight" in agentic:
                flat["portfolio_allocator_regime_weight"] = float(
                    agentic["portfolio_allocator_regime_weight"]
                )
            if "portfolio_allocator_activity_weight" in agentic:
                flat["portfolio_allocator_activity_weight"] = float(
                    agentic["portfolio_allocator_activity_weight"]
                )
            if "portfolio_allocator_max_single_weight" in agentic:
                flat["portfolio_allocator_max_single_weight"] = float(
                    agentic["portfolio_allocator_max_single_weight"]
                )
            if "portfolio_allocator_max_high_risk_positions" in agentic:
                flat["portfolio_allocator_max_high_risk_positions"] = int(
                    agentic["portfolio_allocator_max_high_risk_positions"]
                )
            if "portfolio_allocator_max_per_regime" in agentic:
                flat["portfolio_allocator_max_per_regime"] = int(
                    agentic["portfolio_allocator_max_per_regime"]
                )
            if "portfolio_allocator_critic_enabled" in agentic:
                flat["portfolio_allocator_critic_enabled"] = bool(
                    agentic["portfolio_allocator_critic_enabled"]
                )
            if "portfolio_allocator_critic_same_side_penalty" in agentic:
                flat["portfolio_allocator_critic_same_side_penalty"] = float(
                    agentic["portfolio_allocator_critic_same_side_penalty"]
                )
            if "portfolio_allocator_critic_same_regime_penalty" in agentic:
                flat["portfolio_allocator_critic_same_regime_penalty"] = float(
                    agentic["portfolio_allocator_critic_same_regime_penalty"]
                )
            if "portfolio_allocator_critic_same_strategy_penalty" in agentic:
                flat["portfolio_allocator_critic_same_strategy_penalty"] = float(
                    agentic["portfolio_allocator_critic_same_strategy_penalty"]
                )
            if "portfolio_allocator_critic_open_same_exposure_penalty" in agentic:
                flat["portfolio_allocator_critic_open_same_exposure_penalty"] = float(
                    agentic["portfolio_allocator_critic_open_same_exposure_penalty"]
                )
            if "portfolio_allocator_critic_weaker_same_side_penalty" in agentic:
                flat["portfolio_allocator_critic_weaker_same_side_penalty"] = float(
                    agentic["portfolio_allocator_critic_weaker_same_side_penalty"]
                )
            if "portfolio_allocator_critic_watch_penalty" in agentic:
                flat["portfolio_allocator_critic_watch_penalty"] = float(
                    agentic["portfolio_allocator_critic_watch_penalty"]
                )
            if "portfolio_allocator_critic_high_risk_penalty" in agentic:
                flat["portfolio_allocator_critic_high_risk_penalty"] = float(
                    agentic["portfolio_allocator_critic_high_risk_penalty"]
                )
            if "portfolio_allocator_discovery_alignment_enabled" in agentic:
                flat["portfolio_allocator_discovery_alignment_enabled"] = bool(
                    agentic["portfolio_allocator_discovery_alignment_enabled"]
                )
            if "portfolio_allocator_critic_weaker_discovery_penalty" in agentic:
                flat["portfolio_allocator_critic_weaker_discovery_penalty"] = float(
                    agentic["portfolio_allocator_critic_weaker_discovery_penalty"]
                )
            if "portfolio_allocator_research_alignment_enabled" in agentic:
                flat["portfolio_allocator_research_alignment_enabled"] = bool(
                    agentic["portfolio_allocator_research_alignment_enabled"]
                )
            if "portfolio_allocator_critic_missing_research_penalty" in agentic:
                flat["portfolio_allocator_critic_missing_research_penalty"] = float(
                    agentic["portfolio_allocator_critic_missing_research_penalty"]
                )
            if "portfolio_allocator_critic_weaker_research_penalty" in agentic:
                flat["portfolio_allocator_critic_weaker_research_penalty"] = float(
                    agentic["portfolio_allocator_critic_weaker_research_penalty"]
                )
            if "portfolio_allocator_critic_unrefreshed_research_penalty" in agentic:
                flat["portfolio_allocator_critic_unrefreshed_research_penalty"] = float(
                    agentic["portfolio_allocator_critic_unrefreshed_research_penalty"]
                )
            if "portfolio_allocator_critic_weaker_nightly_research_penalty" in agentic:
                flat["portfolio_allocator_critic_weaker_nightly_research_penalty"] = float(
                    agentic["portfolio_allocator_critic_weaker_nightly_research_penalty"]
                )
            if "portfolio_allocator_critic_block_threshold" in agentic:
                flat["portfolio_allocator_critic_block_threshold"] = float(
                    agentic["portfolio_allocator_critic_block_threshold"]
                )
            if "portfolio_allocator_nightly_feedback_enabled" in agentic:
                flat["portfolio_allocator_nightly_feedback_enabled"] = bool(
                    agentic["portfolio_allocator_nightly_feedback_enabled"]
                )
            if "portfolio_allocator_nightly_feedback_lookback_reports" in agentic:
                flat["portfolio_allocator_nightly_feedback_lookback_reports"] = int(
                    agentic["portfolio_allocator_nightly_feedback_lookback_reports"]
                )
            if "portfolio_allocator_nightly_refresh_bonus_weight" in agentic:
                flat["portfolio_allocator_nightly_refresh_bonus_weight"] = float(
                    agentic["portfolio_allocator_nightly_refresh_bonus_weight"]
                )
            if "portfolio_allocator_nightly_promote_bonus_weight" in agentic:
                flat["portfolio_allocator_nightly_promote_bonus_weight"] = float(
                    agentic["portfolio_allocator_nightly_promote_bonus_weight"]
                )
            if "portfolio_allocator_nightly_report_dir" in agentic:
                flat["portfolio_allocator_nightly_report_dir"] = Path(
                    agentic["portfolio_allocator_nightly_report_dir"]
                )
            if "portfolio_allocator_freshness_preference_enabled" in agentic:
                flat["portfolio_allocator_freshness_preference_enabled"] = bool(
                    agentic["portfolio_allocator_freshness_preference_enabled"]
                )
            if "portfolio_allocator_freshness_preference_weight" in agentic:
                flat["portfolio_allocator_freshness_preference_weight"] = float(
                    agentic["portfolio_allocator_freshness_preference_weight"]
                )
            if "portfolio_allocator_same_target_balance_enabled" in agentic:
                flat["portfolio_allocator_same_target_balance_enabled"] = bool(
                    agentic["portfolio_allocator_same_target_balance_enabled"]
                )
            if "portfolio_allocator_same_target_balance_penalty" in agentic:
                flat["portfolio_allocator_same_target_balance_penalty"] = float(
                    agentic["portfolio_allocator_same_target_balance_penalty"]
                )
            if "model_status_nightly_alignment_enabled" in agentic:
                flat["model_status_nightly_alignment_enabled"] = bool(
                    agentic["model_status_nightly_alignment_enabled"]
                )
            if "model_status_nightly_alignment_lookback_reports" in agentic:
                flat["model_status_nightly_alignment_lookback_reports"] = int(
                    agentic["model_status_nightly_alignment_lookback_reports"]
                )
            if "model_status_nightly_report_dir" in agentic:
                flat["model_status_nightly_report_dir"] = Path(
                    agentic["model_status_nightly_report_dir"]
                )
            if "model_status_nightly_alignment_recommendation_ratio" in agentic:
                flat["model_status_nightly_alignment_recommendation_ratio"] = float(
                    agentic["model_status_nightly_alignment_recommendation_ratio"]
                )
            if "acceptance_bundle_nightly_alignment_min_enabled_reports" in agentic:
                flat["acceptance_bundle_nightly_alignment_min_enabled_reports"] = int(
                    agentic["acceptance_bundle_nightly_alignment_min_enabled_reports"]
                )
            if "acceptance_bundle_nightly_alignment_warn_ratio" in agentic:
                flat["acceptance_bundle_nightly_alignment_warn_ratio"] = float(
                    agentic["acceptance_bundle_nightly_alignment_warn_ratio"]
                )
            if "acceptance_bundle_nightly_refreshed_alignment_warn_ratio" in agentic:
                flat["acceptance_bundle_nightly_refreshed_alignment_warn_ratio"] = float(
                    agentic["acceptance_bundle_nightly_refreshed_alignment_warn_ratio"]
                )
            if "acceptance_bundle_team_research_alignment_warn_ratio" in agentic:
                flat["acceptance_bundle_team_research_alignment_warn_ratio"] = float(
                    agentic["acceptance_bundle_team_research_alignment_warn_ratio"]
                )
            if "acceptance_bundle_team_discovery_alignment_warn_ratio" in agentic:
                flat["acceptance_bundle_team_discovery_alignment_warn_ratio"] = float(
                    agentic["acceptance_bundle_team_discovery_alignment_warn_ratio"]
                )
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
                if "allowed_chat_ids" in telegram:
                    flat["telegram_allowed_chat_ids"] = str(telegram["allowed_chat_ids"])
                if "admin_chat_ids" in telegram:
                    flat["telegram_admin_chat_ids"] = str(telegram["admin_chat_ids"])
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
                if "signal_max_concurrent_requests" in telegram:
                    flat["signal_max_concurrent_requests"] = int(
                        telegram["signal_max_concurrent_requests"]
                    )
                if "signal_max_concurrent_per_user" in telegram:
                    flat["signal_max_concurrent_per_user"] = int(
                        telegram["signal_max_concurrent_per_user"]
                    )
                if "signal_max_pending_per_user" in telegram:
                    flat["signal_max_pending_per_user"] = int(
                        telegram["signal_max_pending_per_user"]
                    )
                if "signal_timeout_seconds" in telegram:
                    flat["signal_timeout_seconds"] = int(telegram["signal_timeout_seconds"])
                if "signal_slow_command_timeout_seconds" in telegram:
                    flat["signal_slow_command_timeout_seconds"] = int(
                        telegram["signal_slow_command_timeout_seconds"]
                    )
                if "signal_result_ttl_seconds" in telegram:
                    flat["signal_result_ttl_seconds"] = int(telegram["signal_result_ttl_seconds"])
                if "signal_cache_enabled" in telegram:
                    flat["signal_cache_enabled"] = bool(telegram["signal_cache_enabled"])
                if "signal_intent_cache_ttl_seconds" in telegram:
                    flat["signal_intent_cache_ttl_seconds"] = int(
                        telegram["signal_intent_cache_ttl_seconds"]
                    )
                if "signal_image_cache_ttl_seconds" in telegram:
                    flat["signal_image_cache_ttl_seconds"] = int(
                        telegram["signal_image_cache_ttl_seconds"]
                    )
                if "signal_rl_mode" in telegram:
                    flat["signal_rl_mode"] = str(telegram["signal_rl_mode"])
                if "signal_ml_enabled_in_request_path" in telegram:
                    flat["signal_ml_enabled_in_request_path"] = bool(
                        telegram["signal_ml_enabled_in_request_path"]
                    )
                if "signal_expert_commands_visible" in telegram:
                    flat["signal_expert_commands_visible"] = bool(
                        telegram["signal_expert_commands_visible"]
                    )
                if "signal_reply_style" in telegram:
                    flat["signal_reply_style"] = str(telegram["signal_reply_style"])
                if "signal_image_input_enabled" in telegram:
                    flat["signal_image_input_enabled"] = bool(telegram["signal_image_input_enabled"])
                if "signal_image_max_bytes" in telegram:
                    flat["signal_image_max_bytes"] = int(telegram["signal_image_max_bytes"])
                if "signal_image_temp_dir" in telegram:
                    flat["signal_image_temp_dir"] = Path(telegram["signal_image_temp_dir"])
                if "signal_image_temp_retention_minutes" in telegram:
                    flat["signal_image_temp_retention_minutes"] = int(
                        telegram["signal_image_temp_retention_minutes"]
                    )
                if "signal_agent_timeout_seconds" in telegram:
                    flat["signal_agent_timeout_seconds"] = int(
                        telegram["signal_agent_timeout_seconds"]
                    )
                if "signal_request_audit_retention_days" in telegram:
                    flat["signal_request_audit_retention_days"] = int(
                        telegram["signal_request_audit_retention_days"]
                    )
                if "signal_request_audit_retention_count" in telegram:
                    flat["signal_request_audit_retention_count"] = int(
                        telegram["signal_request_audit_retention_count"]
                    )
                if "signal_session_retention_days" in telegram:
                    flat["signal_session_retention_days"] = int(
                        telegram["signal_session_retention_days"]
                    )
                if "signal_secret_redaction_enabled" in telegram:
                    flat["signal_secret_redaction_enabled"] = bool(
                        telegram["signal_secret_redaction_enabled"]
                    )
                if "signal_openai_required_for_images" in telegram:
                    flat["signal_openai_required_for_images"] = bool(
                        telegram["signal_openai_required_for_images"]
                    )
                if "signal_forecast_enabled" in telegram:
                    flat["signal_forecast_enabled"] = bool(telegram["signal_forecast_enabled"])
                if "signal_forecast_trigger_confidence_threshold" in telegram:
                    flat["signal_forecast_trigger_confidence_threshold"] = float(
                        telegram["signal_forecast_trigger_confidence_threshold"]
                    )
                if "signal_forecast_cache_ttl_seconds" in telegram:
                    flat["signal_forecast_cache_ttl_seconds"] = int(
                        telegram["signal_forecast_cache_ttl_seconds"]
                    )
                if "signal_forecast_max_bars" in telegram:
                    flat["signal_forecast_max_bars"] = int(
                        telegram["signal_forecast_max_bars"]
                    )
                if "signal_forecast_max_tasks_per_request" in telegram:
                    flat["signal_forecast_max_tasks_per_request"] = int(
                        telegram["signal_forecast_max_tasks_per_request"]
                    )
                if "signal_replay_enabled" in telegram:
                    flat["signal_replay_enabled"] = bool(telegram["signal_replay_enabled"])
                if "signal_replay_cache_ttl_seconds" in telegram:
                    flat["signal_replay_cache_ttl_seconds"] = int(
                        telegram["signal_replay_cache_ttl_seconds"]
                    )
                if "signal_replay_max_bars" in telegram:
                    flat["signal_replay_max_bars"] = int(telegram["signal_replay_max_bars"])
                if "signal_replay_hold_bars" in telegram:
                    flat["signal_replay_hold_bars"] = int(telegram["signal_replay_hold_bars"])
                if "signal_futures_oi_enabled" in telegram:
                    flat["signal_futures_oi_enabled"] = bool(
                        telegram["signal_futures_oi_enabled"]
                    )
                if "signal_code_execution_enabled" in telegram:
                    flat["signal_code_execution_enabled"] = bool(
                        telegram["signal_code_execution_enabled"]
                    )
                if "signal_code_execution_timeout_seconds" in telegram:
                    flat["signal_code_execution_timeout_seconds"] = int(
                        telegram["signal_code_execution_timeout_seconds"]
                    )
                if "signal_code_execution_max_output_chars" in telegram:
                    flat["signal_code_execution_max_output_chars"] = int(
                        telegram["signal_code_execution_max_output_chars"]
                    )
                if "signal_code_execution_temp_dir" in telegram:
                    flat["signal_code_execution_temp_dir"] = Path(
                        telegram["signal_code_execution_temp_dir"]
                    )
                if "signal_code_execution_retention_minutes" in telegram:
                    flat["signal_code_execution_retention_minutes"] = int(
                        telegram["signal_code_execution_retention_minutes"]
                    )
                if "signal_code_execution_allow_live_generation" in telegram:
                    flat["signal_code_execution_allow_live_generation"] = bool(
                        telegram["signal_code_execution_allow_live_generation"]
                    )
                if "signal_code_execution_fallback_only" in telegram:
                    flat["signal_code_execution_fallback_only"] = bool(
                        telegram["signal_code_execution_fallback_only"]
                    )
                if "signal_forecast_audit_enabled" in telegram:
                    flat["signal_forecast_audit_enabled"] = bool(
                        telegram["signal_forecast_audit_enabled"]
                    )
                if "openai_model" in telegram:
                    flat["openai_model"] = str(telegram["openai_model"])
                if "openai_base_url" in telegram:
                    flat["openai_base_url"] = str(telegram["openai_base_url"])
                if "openai_timeout_seconds" in telegram:
                    flat["openai_timeout_seconds"] = int(telegram["openai_timeout_seconds"])
                if "openai_max_output_tokens" in telegram:
                    flat["openai_max_output_tokens"] = int(
                        telegram["openai_max_output_tokens"]
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

    def telegram_allowed_chat_id_set(self) -> set[str]:
        values = set(_split_csv_tokens(self.telegram_allowed_chat_ids))
        if not values and str(self.telegram_chat_id).strip():
            values.add(str(self.telegram_chat_id).strip())
        return values

    def telegram_admin_chat_id_set(self) -> set[str]:
        return set(_split_csv_tokens(self.telegram_admin_chat_ids))


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
