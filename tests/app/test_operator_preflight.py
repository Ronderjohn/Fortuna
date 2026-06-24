from __future__ import annotations

from pathlib import Path

from fortuna.agentic.contracts import (
    AgenticStatusSummary,
    MlModelStatus,
    ModelHealthResponse,
    RegimeModelStatus,
    RlModelStatus,
)
from fortuna.app.operator_preflight import (
    OperatorPreflightResult,
    PreflightCheck,
    render_preflight,
    run_operator_preflight,
)
from fortuna.app.runtime_readiness import build_runtime_readiness_summary
from fortuna.config.settings import Settings
from fortuna.observability.sinks import read_workflow_events


def _base_settings(tmp_path: Path, **overrides) -> Settings:
    payload = {
        "env": "test",
        "log_level": "WARNING",
        "data_cache_dir": tmp_path / "cache",
        "duckdb_path": tmp_path / "cache" / "fortuna.duckdb",
        "project_root": tmp_path,
        "default_symbol": "RELIANCE.NS",
        "default_timeframe": "5m",
        "default_days": 10,
    }
    payload.update(overrides)
    return Settings(**payload)


def test_operator_preflight_passes_deterministic_mode(tmp_path):
    settings = _base_settings(
        tmp_path,
        agentic_enabled=False,
        telegram_enabled=False,
    )
    result = run_operator_preflight(settings)
    assert result.ok is True
    assert result.runtime_readiness is not None
    assert result.runtime_readiness.overall_posture == "deterministic_only"
    assert result.runtime_readiness.deterministic_fallback_ok is True
    assert result.runtime_readiness.nightly_posture == "manual_only"
    rendered = render_preflight(result)
    assert "Runtime posture: deterministic_only" in rendered
    assert "Nightly posture: manual_only" in rendered


def test_operator_preflight_emits_observability_event(tmp_path):
    settings = _base_settings(
        tmp_path,
        agentic_enabled=False,
        observability_enabled=True,
        observability_log_dir=tmp_path / "logs" / "observability",
    )
    run_operator_preflight(settings)
    log_path = tmp_path / "logs" / "observability" / "workflow_events.jsonl"
    rows = read_workflow_events(log_path)
    assert len(rows) == 2
    assert rows[0]["workflow_id"] == "operator_preflight"
    assert rows[1]["event_name"] == "operator_preflight.completed"
    assert rows[1]["context"]["overall_posture"] == "deterministic_only"


def test_operator_preflight_warns_for_missing_registry_pointers(tmp_path):
    settings = _base_settings(
        tmp_path,
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
    readiness = result.runtime_readiness
    assert readiness is not None
    assert readiness.overall_posture == "advisory_partial"
    warning_names = {row.name for row in readiness.warnings}
    assert "ml_scorer" in warning_names
    assert "rl_policy" in warning_names


def test_operator_preflight_strict_fails_enabled_missing_subsystems(tmp_path):
    settings = _base_settings(
        tmp_path,
        agentic_enabled=True,
        agentic_ml_scorer_enabled=True,
        telegram_enabled=True,
        telegram_bot_token="",
        telegram_chat_id="",
        model_registry_enabled=True,
        model_promotion_required=True,
        ml_scorer_artifact_dir=Path("models/ml_signal_scorer/validated"),
    )
    result = run_operator_preflight(settings, strict=True)
    assert result.ok is False
    failed_names = {check.name for check in result.failures}
    assert "ml_scorer" in failed_names
    assert "telegram" in failed_names
    readiness = result.runtime_readiness
    assert readiness is not None
    assert readiness.overall_posture == "blocked"
    blocker_names = {row.name for row in readiness.blockers}
    assert "ml_scorer" in blocker_names


def test_runtime_readiness_advisory_ready_with_model_health(tmp_path):
    settings = _base_settings(
        tmp_path,
        agentic_enabled=True,
        agentic_ml_scorer_enabled=True,
    )
    preflight = OperatorPreflightResult(
        checks=[
            PreflightCheck(name="agentic", status="pass", detail="enabled"),
            PreflightCheck(name="ml_scorer", status="pass", detail="live pointer ok"),
            PreflightCheck(name="rl_policy", status="pass", detail="live pointer ok"),
        ]
    )
    model_health = ModelHealthResponse(
        registry_enabled=True,
        promotion_required=True,
        rl=RlModelStatus(available=True, advisory_ready=True, live_pointer="rl.json"),
        ml=MlModelStatus(
            available=True,
            enabled=True,
            advisory_ready=True,
            live_pointer="ml.json",
        ),
        regime=RegimeModelStatus(available=False),
        agentic=AgenticStatusSummary(),
    )
    readiness = build_runtime_readiness_summary(
        settings,
        preflight=preflight,
        model_health=model_health,
    )
    assert readiness.overall_posture == "advisory_ready"
    assert readiness.ml_advisory_ready is True
    assert readiness.rl_advisory_ready is True


def test_runtime_readiness_nightly_manual_only_by_default(tmp_path):
    settings = _base_settings(tmp_path, agentic_enabled=False)
    result = run_operator_preflight(settings)
    assert result.runtime_readiness is not None
    assert result.runtime_readiness.nightly_posture == "manual_only"


def test_runtime_readiness_nightly_blocked_when_enabled_with_blockers(tmp_path):
    settings = _base_settings(
        tmp_path,
        agentic_enabled=True,
        agentic_ml_scorer_enabled=True,
        nightly_automation_enabled=True,
        model_registry_enabled=True,
        model_promotion_required=True,
        ml_scorer_artifact_dir=Path("models/ml_signal_scorer/validated"),
    )
    result = run_operator_preflight(settings, strict=True)
    readiness = result.runtime_readiness
    assert readiness is not None
    assert readiness.nightly_posture == "blocked"
    nightly_blockers = [row for row in readiness.blockers if row.name == "nightly_automation"]
    assert nightly_blockers


def test_operator_preflight_warns_when_image_intake_requires_openai(tmp_path):
    settings = _base_settings(
        tmp_path,
        telegram_enabled=True,
        telegram_bot_token="token",
        telegram_chat_id="12345",
        signal_image_input_enabled=True,
        signal_openai_required_for_images=True,
        openai_api_key="",
    )
    result = run_operator_preflight(settings)
    telegram = next(check for check in result.checks if check.name == "telegram")
    assert telegram.status == "warn"
    assert "OpenAI API key missing" in telegram.detail
