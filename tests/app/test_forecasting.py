from __future__ import annotations

from pathlib import Path

from support.conversation_eval_fakes import FakeEvalEngine, build_eval_settings

from fortuna.agentic.contracts import ForecastTaskRequest, SignalResponse
from fortuna.app.forecast_sandbox import (
    build_generated_code,
    cleanup_code_temp_dirs,
    execute_generated_code,
    validate_generated_code,
)
from fortuna.app.forecasting import merge_forecasts_into_signal, run_forecast_lane


def _signal_response() -> SignalResponse:
    return SignalResponse(
        resolved_symbol="RELIANCE.NS",
        segment="EQUITY",
        timeframe="5m",
        lookback_days=20,
        signal_verdict="BUY",
        confidence=0.55,
        summary="Borderline setup.",
        reasons=("Momentum is mixed.",),
        risk_notes=("Wait for confirmation.",),
        action_plan="Be selective.",
        data_freshness="Last bar: 2026-01-06 09:35",
    )


def test_validate_generated_code_blocks_forbidden_import():
    assert validate_generated_code("import os\nresult = {}") == "import_blocked:os"


def test_execute_generated_code_returns_sandbox_result(tmp_path: Path):
    settings = build_eval_settings(tmp_path).model_copy(
        update={
            "signal_code_execution_temp_dir": tmp_path / "tmp" / "forecast_code",
            "signal_code_execution_timeout_seconds": 3,
        }
    )
    request = ForecastTaskRequest(
        task_name="support_resistance",
        symbol="RELIANCE.NS",
        timeframe="5m",
        lookback_days=20,
        latest_bar_iso="2026-01-06T09:35:00",
        closes=(100.0, 101.0, 102.5, 101.5, 103.0),
        highs=(101.0, 102.0, 103.0, 102.0, 104.0),
        lows=(99.0, 100.0, 101.0, 100.5, 102.0),
    )
    sandbox, result = execute_generated_code(
        request,
        code=build_generated_code("support_resistance"),
        settings=settings,
    )
    assert sandbox.ok is True
    assert result is not None
    assert result.mode == "sandbox"
    assert "support" in result.metrics

    temp_dir = settings.resolve_path(settings.signal_code_execution_temp_dir)
    assert list(temp_dir.glob("forecast_*"))
    removed = cleanup_code_temp_dirs(temp_dir, retention_minutes=0)
    assert removed == 0


def test_run_forecast_lane_uses_builtin_fallback_and_merges(tmp_path: Path):
    settings = build_eval_settings(tmp_path).model_copy(
        update={
            "signal_code_execution_enabled": False,
            "signal_forecast_enabled": True,
        }
    )
    engine = FakeEvalEngine()
    response = type(
        "Response",
        (),
        {
            "ok": True,
            "decision": type("Decision", (), {"confidence": 0.4})(),
        },
    )()
    signal = _signal_response()
    forecasts = run_forecast_lane(
        raw_text="Give me a more precise forecast for Reliance",
        response=response,
        signal_response=signal,
        settings=settings,
        engine_factory=lambda: engine,
    )
    assert forecasts
    assert all(item.mode == "builtin" for item in forecasts)

    merged = merge_forecasts_into_signal(signal, tuple(forecasts))
    assert merged.forecast_used is True
    assert merged.forecast_summaries
    assert any(status.startswith("forecast:") for status in merged.agent_statuses)
