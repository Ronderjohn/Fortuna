"""Typed workflow event schema for cross-module observability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkflowEvent:
    event_name: str
    module: str
    status: str
    ts: str
    run_id: str
    workflow_id: str
    symbol: Optional[str] = None
    duration_ms: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def started(
        cls,
        *,
        event_name: str,
        module: str,
        workflow_id: str,
        run_id: str,
        symbol: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> WorkflowEvent:
        return cls(
            event_name=f"{event_name}.started",
            module=module,
            status="started",
            ts=utc_now_iso(),
            run_id=run_id,
            workflow_id=workflow_id,
            symbol=symbol,
            context=dict(context or {}),
        )

    @classmethod
    def completed(
        cls,
        *,
        event_name: str,
        module: str,
        workflow_id: str,
        run_id: str,
        status: str = "ok",
        symbol: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> WorkflowEvent:
        return cls(
            event_name=f"{event_name}.completed",
            module=module,
            status=status,
            ts=utc_now_iso(),
            run_id=run_id,
            workflow_id=workflow_id,
            symbol=symbol,
            duration_ms=duration_ms,
            error_code=error_code,
            error_message=error_message,
            context=dict(context or {}),
        )

    @classmethod
    def step(
        cls,
        *,
        event_name: str,
        module: str,
        workflow_id: str,
        run_id: str,
        status: str = "ok",
        symbol: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> WorkflowEvent:
        return cls(
            event_name=event_name,
            module=module,
            status=status,
            ts=utc_now_iso(),
            run_id=run_id,
            workflow_id=workflow_id,
            symbol=symbol,
            context=dict(context or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_name": self.event_name,
            "module": self.module,
            "status": self.status,
            "ts": self.ts,
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
        }
        if self.symbol is not None:
            payload["symbol"] = self.symbol
        if self.duration_ms is not None:
            payload["duration_ms"] = self.duration_ms
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        if self.context:
            payload["context"] = self.context
        return payload

    def to_otel_attributes(self) -> dict[str, str | int | float | bool]:
        attrs: dict[str, str | int | float | bool] = {
            "fortuna.event_name": self.event_name,
            "fortuna.module": self.module,
            "fortuna.status": self.status,
            "fortuna.run_id": self.run_id,
            "fortuna.workflow_id": self.workflow_id,
        }
        if self.symbol is not None:
            attrs["fortuna.symbol"] = self.symbol
        if self.duration_ms is not None:
            attrs["fortuna.duration_ms"] = int(self.duration_ms)
        if self.error_code is not None:
            attrs["fortuna.error_code"] = self.error_code
        for key, value in self.context.items():
            if isinstance(value, (str, int, float, bool)):
                attrs[f"fortuna.context.{key}"] = value
            elif value is not None:
                attrs[f"fortuna.context.{key}"] = json.dumps(value, default=str)
        return attrs


def new_run_id() -> str:
    return str(uuid4())
