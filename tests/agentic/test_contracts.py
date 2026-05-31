from __future__ import annotations

import json
from datetime import datetime

from fortuna.agentic.contracts import (
    AdvisoryError,
    AdvisoryErrorCode,
    AgenticStatusSummary,
    DecisionSummary,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSnapshot,
    MlModelStatus,
    ModelHealthResponse,
    RegimeModelStatus,
    RlModelStatus,
    action_plan_for,
)
from fortuna.agentic.models import (
    ActionRecommendation,
    AgentDecision,
    DecisionRationale,
)


def test_decision_summary_from_agent_decision():
    decision = AgentDecision(
        symbol="RELIANCE.NS",
        action=ActionRecommendation.BUY,
        confidence=0.78,
        bar_time=datetime(2026, 1, 6, 9, 30),
        bar_close=104.0,
        rationale=DecisionRationale(
            summary="BUY won",
            reasons=["breakout", "flat book"],
            risk_notes=["wide spread"],
        ),
    )
    summary = DecisionSummary.from_agent_decision(decision)
    assert summary.action == "BUY"
    assert summary.confidence == 0.78
    assert summary.reasons == ("breakout", "flat book")
    assert summary.risk_notes == ("wide spread",)
    assert "long side" in summary.action_plan


def test_action_plan_for_known_actions():
    assert "long side" in action_plan_for("BUY")
    assert "short entry" in action_plan_for("SELL")
    assert "exiting" in action_plan_for("EXIT_LONG")
    assert "Stay flat" in action_plan_for("DO_NOT_ENTER")
    assert "clearer setup" in action_plan_for("HOLD")


def test_instrument_analysis_response_to_dict_is_json_safe():
    response = InstrumentAnalysisResponse(
        ok=True,
        request=InstrumentAnalysisRequest(symbol="RELIANCE.NS"),
        instrument=InstrumentSnapshot(
            requested_symbol="RELIANCE",
            resolved_symbol="RELIANCE.NS",
            segment_label="EQUITY",
            timeframe="5m",
            lookback_days=30,
            last_bar_time=datetime(2026, 1, 6, 9, 30),
            last_close=104.0,
        ),
        decision=DecisionSummary(
            action="BUY",
            confidence=0.5,
            summary="test",
            action_plan=action_plan_for("BUY"),
        ),
    )
    payload = response.to_dict()
    json.dumps(payload)
    assert payload["instrument"]["last_close"] == 104.0
    assert "ohlcv" not in payload


def test_advisory_error_to_dict():
    err = AdvisoryError(
        code=AdvisoryErrorCode.RESOLVE_FAILED,
        message="missing symbol",
        details={"symbol": "FOO"},
    )
    payload = err.to_dict()
    assert payload["code"] == "resolve_failed"
    assert payload["details"]["symbol"] == "FOO"


def test_model_health_response_to_dict_matches_nested_shape():
    status = ModelHealthResponse(
        registry_enabled=True,
        promotion_required=True,
        rl=RlModelStatus(symbol="RELIANCE.NS", load_error="no_generator"),
        ml=MlModelStatus(enabled=False, load_error="not_loaded"),
        regime=RegimeModelStatus(),
        agentic=AgenticStatusSummary(),
    )
    payload = status.to_dict()
    assert payload["rl"]["load_error"] == "no_generator"
    assert payload["ml"]["load_error"] == "not_loaded"
    assert payload["agentic"]["learning_summary"]["total_rows"] == 0
