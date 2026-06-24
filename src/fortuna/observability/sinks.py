"""Local and optional OTLP sinks for workflow events."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from fortuna.observability.events import WorkflowEvent

logger = logging.getLogger(__name__)


class EventSink(ABC):
    @abstractmethod
    def emit(self, event: WorkflowEvent) -> None:
        raise NotImplementedError


class NoopSink(EventSink):
    def emit(self, event: WorkflowEvent) -> None:
        return None


class JsonlLocalSink(EventSink):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def emit(self, event: WorkflowEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True, default=str) + "\n")


class CompositeSink(EventSink):
    def __init__(self, sinks: list[EventSink]) -> None:
        self.sinks = list(sinks)

    def emit(self, event: WorkflowEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)


class OtlpEventExporter(EventSink):
    """Optional OTLP exporter; no-ops when SDK or endpoint is unavailable."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = str(endpoint or "").strip()
        self._exporter: Any = None
        self._provider: Any = None

    def emit(self, event: WorkflowEvent) -> None:
        if not self.endpoint:
            return
        try:
            self._ensure_exporter()
            if self._exporter is None:
                return
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind

            tracer = trace.get_tracer("fortuna.observability")
            with tracer.start_as_current_span(
                event.event_name,
                kind=SpanKind.INTERNAL,
                attributes=event.to_otel_attributes(),
            ):
                pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("OTLP export skipped: %s", exc)

    def _ensure_exporter(self) -> None:
        if self._provider is not None:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            logger.debug("OpenTelemetry SDK not installed; OTLP export disabled")
            return

        resource = Resource.create({"service.name": "fortuna"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=self.endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self._provider = provider
        self._exporter = exporter


def read_workflow_events(path: Path | str, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows
