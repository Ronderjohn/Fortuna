from __future__ import annotations

from pathlib import Path

from fortuna.app.operator_preflight import render_preflight, run_operator_preflight
from fortuna.config.settings import Settings


def test_operator_preflight_passes_deterministic_mode(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        default_timeframe="5m",
        default_days=10,
        agentic_enabled=False,
        telegram_enabled=False,
    )
    result = run_operator_preflight(settings)
    assert result.ok is True
    assert "deterministic-only" in render_preflight(result)


def test_operator_preflight_warns_for_missing_registry_pointers(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        default_timeframe="5m",
        default_days=10,
        agentic_enabled=True,
        agentic_ml_scorer_enabled=True,
        model_registry_enabled=True,
        model_promotion_required=True,
        ml_scorer_artifact_dir=Path("models/ml_signal_scorer/validated"),
    )
    result = run_operator_preflight(settings)
    names = {check.name: check.status for check in result.checks}
    assert names["ml_scorer"] == "warn"
    assert names["rl_policy"] == "warn"


def test_operator_preflight_strict_fails_enabled_missing_subsystems(tmp_path):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        default_symbol="RELIANCE.NS",
        default_timeframe="5m",
        default_days=10,
        agentic_enabled=True,
        agentic_ml_scorer_enabled=True,
        telegram_enabled=True,
        model_registry_enabled=True,
        model_promotion_required=True,
        ml_scorer_artifact_dir=Path("models/ml_signal_scorer/validated"),
    )
    result = run_operator_preflight(settings, strict=True)
    assert result.ok is False
    failed_names = {check.name for check in result.failures}
    assert "ml_scorer" in failed_names
    assert "telegram" in failed_names
