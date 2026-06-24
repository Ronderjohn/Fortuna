"""Tests for workflow event recorder and local JSONL sink."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fortuna.config.settings import Settings
from fortuna.observability.events import WorkflowEvent
from fortuna.observability.recorder import (
    get_workflow_recorder,
    workflow_boundary,
    workflow_run_context,
)
from fortuna.observability.sinks import JsonlLocalSink, read_workflow_events


def _settings(tmp_path: Path, **overrides) -> Settings:
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    payload = {
        "env": "test",
        "log_level": "WARNING",
        "data_cache_dir": tmp_path / "cache",
        "duckdb_path": tmp_path / "cache" / "fortuna.duckdb",
        "project_root": tmp_path,
        "observability_enabled": True,
        "observability_log_dir": tmp_path / "logs" / "observability",
    }
    payload.update(overrides)
    return Settings(**payload)


def test_recorder_disabled_writes_nothing(tmp_path: Path) -> None:
    settings = _settings(tmp_path, observability_enabled=False)
    log_path = settings.resolve_path(settings.observability_log_dir) / "workflow_events.jsonl"
    recorder = get_workflow_recorder(settings)
    recorder.emit(
        WorkflowEvent(
            event_name="test.event",
            module="tests.observability",
            status="ok",
            ts="2026-06-13T10:00:00+00:00",
            run_id="run-disabled",
            workflow_id="test",
        )
    )
    assert not log_path.exists()


def test_recorder_appends_jsonl_events(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    log_path = settings.resolve_path(settings.observability_log_dir) / "workflow_events.jsonl"
    with workflow_boundary(
        settings,
        event_name="test_boundary",
        module="tests.observability",
        workflow_id="test",
    ) as span:
        span.set_context(sample=True)
    rows = read_workflow_events(log_path)
    assert len(rows) == 2
    assert rows[0]["status"] == "started"
    assert rows[1]["status"] == "ok"
    assert rows[1]["context"]["sample"] is True


def test_recorder_fail_soft_on_sink_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    recorder = get_workflow_recorder(settings)
    recorder.sink = MagicMock()
    recorder.sink.emit.side_effect = OSError("disk full")
    recorder.emit(
        WorkflowEvent(
            event_name="test.event",
            module="tests.observability",
            status="ok",
            ts="2026-06-13T10:00:00+00:00",
            run_id="run-soft",
            workflow_id="test",
        )
    )


def test_workflow_run_context_propagates_run_id(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    log_path = settings.resolve_path(settings.observability_log_dir) / "workflow_events.jsonl"
    with workflow_run_context("parent-run") as run_id:
        assert run_id == "parent-run"
        get_workflow_recorder(settings).emit_step(
            event_name="child.step",
            module="tests.observability",
            workflow_id="nightly_train",
            status="ok",
        )
    rows = read_workflow_events(log_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "parent-run"


def test_jsonl_local_sink_append(tmp_path: Path) -> None:
    path = tmp_path / "workflow_events.jsonl"
    sink = JsonlLocalSink(path)
    sink.emit(
        WorkflowEvent(
            event_name="one",
            module="m",
            status="ok",
            ts="t",
            run_id="r",
            workflow_id="w",
        )
    )
    sink.emit(
        WorkflowEvent(
            event_name="two",
            module="m",
            status="ok",
            ts="t",
            run_id="r",
            workflow_id="w",
        )
    )
    assert len(read_workflow_events(path)) == 2
