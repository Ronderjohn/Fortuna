"""Fail-soft workflow event recorder and boundary helpers."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional
from uuid import uuid4

from fortuna.config.settings import Settings
from fortuna.observability.events import WorkflowEvent, new_run_id
from fortuna.observability.sinks import (
    CompositeSink,
    EventSink,
    JsonlLocalSink,
    NoopSink,
    OtlpEventExporter,
)

logger = logging.getLogger(__name__)

_current_run_id: ContextVar[Optional[str]] = ContextVar(
    "fortuna_observability_run_id",
    default=None,
)


def get_current_run_id() -> Optional[str]:
    return _current_run_id.get()


def set_current_run_id(run_id: Optional[str]) -> None:
    _current_run_id.set(run_id)


@dataclass
class WorkflowBoundary:
    recorder: "WorkflowEventRecorder"
    event_name: str
    module: str
    workflow_id: str
    run_id: str
    symbol: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    _started_at: float = field(default=0.0, init=False, repr=False)

    def set_context(self, **values: Any) -> None:
        self.context.update({key: value for key, value in values.items() if value is not None})

    def set_status(self, status: str) -> None:
        self.status = status

    def __enter__(self) -> WorkflowBoundary:
        self._started_at = time.perf_counter()
        self.recorder.emit(
            WorkflowEvent.started(
                event_name=self.event_name,
                module=self.module,
                workflow_id=self.workflow_id,
                run_id=self.run_id,
                symbol=self.symbol,
                context=dict(self.context),
            )
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        duration_ms = int((time.perf_counter() - self._started_at) * 1000)
        if exc_type is not None:
            self.status = "error"
            self.recorder.emit(
                WorkflowEvent.completed(
                    event_name=self.event_name,
                    module=self.module,
                    workflow_id=self.workflow_id,
                    run_id=self.run_id,
                    status="error",
                    symbol=self.symbol,
                    duration_ms=duration_ms,
                    error_code=getattr(exc_type, "__name__", "error"),
                    error_message=str(exc) if exc is not None else None,
                    context=dict(self.context),
                )
            )
            return None
        self.recorder.emit(
            WorkflowEvent.completed(
                event_name=self.event_name,
                module=self.module,
                workflow_id=self.workflow_id,
                run_id=self.run_id,
                status=self.status,
                symbol=self.symbol,
                duration_ms=duration_ms,
                context=dict(self.context),
            )
        )
        return None


class WorkflowEventRecorder:
    def __init__(self, sink: EventSink) -> None:
        self.sink = sink

    def emit(self, event: WorkflowEvent) -> None:
        try:
            self.sink.emit(event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("workflow event emit failed: %s", exc)

    def emit_step(
        self,
        *,
        event_name: str,
        module: str,
        workflow_id: str,
        run_id: Optional[str] = None,
        status: str = "ok",
        symbol: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        resolved_run_id = run_id or get_current_run_id() or new_run_id()
        self.emit(
            WorkflowEvent.step(
                event_name=event_name,
                module=module,
                workflow_id=workflow_id,
                run_id=resolved_run_id,
                status=status,
                symbol=symbol,
                context=context,
            )
        )

    def boundary(
        self,
        *,
        event_name: str,
        module: str,
        workflow_id: str,
        run_id: Optional[str] = None,
        symbol: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> WorkflowBoundary:
        resolved_run_id = run_id or get_current_run_id() or new_run_id()
        return WorkflowBoundary(
            recorder=self,
            event_name=event_name,
            module=module,
            workflow_id=workflow_id,
            run_id=resolved_run_id,
            symbol=symbol,
            context=dict(context or {}),
        )


def build_event_sink(settings: Settings) -> EventSink:
    if not bool(getattr(settings, "observability_enabled", False)):
        return NoopSink()

    exporter_mode = str(getattr(settings, "observability_exporter", "local") or "local").lower()
    sinks: list[EventSink] = []
    if exporter_mode in {"local", "both"}:
        log_dir = settings.resolve_path(
            Path(getattr(settings, "observability_log_dir", "logs/observability"))
        )
        sinks.append(JsonlLocalSink(log_dir / "workflow_events.jsonl"))
    if exporter_mode in {"otlp", "both"}:
        endpoint = str(getattr(settings, "observability_otlp_endpoint", "") or "").strip()
        if endpoint:
            sinks.append(OtlpEventExporter(endpoint))
    if exporter_mode == "none" or not sinks:
        return NoopSink()
    if len(sinks) == 1:
        return sinks[0]
    return CompositeSink(sinks)


def get_workflow_recorder(settings: Settings) -> WorkflowEventRecorder:
    return WorkflowEventRecorder(build_event_sink(settings))


def emit_workflow_step(
    settings: Settings,
    *,
    event_name: str,
    module: str,
    workflow_id: str,
    run_id: Optional[str] = None,
    status: str = "ok",
    symbol: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
) -> None:
    get_workflow_recorder(settings).emit_step(
        event_name=event_name,
        module=module,
        workflow_id=workflow_id,
        run_id=run_id,
        status=status,
        symbol=symbol,
        context=context,
    )


@contextmanager
def workflow_boundary(
    settings: Settings,
    *,
    event_name: str,
    module: str,
    workflow_id: str,
    run_id: Optional[str] = None,
    symbol: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
) -> Iterator[WorkflowBoundary]:
    recorder = get_workflow_recorder(settings)
    with recorder.boundary(
        event_name=event_name,
        module=module,
        workflow_id=workflow_id,
        run_id=run_id,
        symbol=symbol,
        context=context,
    ) as span:
        yield span


@contextmanager
def workflow_run_context(run_id: Optional[str] = None) -> Iterator[str]:
    token = _current_run_id.set(run_id or str(uuid4()))
    try:
        yield _current_run_id.get() or new_run_id()
    finally:
        _current_run_id.reset(token)

