"""Tests for typed workflow event schema."""

from __future__ import annotations

from fortuna.observability.events import WorkflowEvent


def test_workflow_event_to_dict_round_trip() -> None:
    event = WorkflowEvent(
        event_name="operator_preflight.completed",
        module="fortuna.app.operator_preflight",
        status="ok",
        ts="2026-06-13T10:00:00+00:00",
        run_id="run-1",
        workflow_id="operator_preflight",
        duration_ms=12,
        context={"ok": True},
    )
    payload = event.to_dict()
    assert payload["event_name"] == "operator_preflight.completed"
    assert payload["run_id"] == "run-1"
    assert payload["duration_ms"] == 12
    assert payload["context"]["ok"] is True


def test_workflow_event_to_otel_attributes_stable_keys() -> None:
    event = WorkflowEvent(
        event_name="market_universe.completed",
        module="fortuna.app.market_universe",
        status="ok",
        ts="2026-06-13T10:00:00+00:00",
        run_id="run-2",
        workflow_id="market_universe",
        symbol="RELIANCE.NS",
        context={"candidate_count": 5},
    )
    attrs = event.to_otel_attributes()
    assert attrs["fortuna.event_name"] == "market_universe.completed"
    assert attrs["fortuna.symbol"] == "RELIANCE.NS"
    assert attrs["fortuna.context.candidate_count"] == 5
