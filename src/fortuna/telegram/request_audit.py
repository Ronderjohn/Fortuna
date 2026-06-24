"""Append-only JSONL audit for interactive Telegram request handling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fortuna.agentic.redaction import redact_secrets
from fortuna.config.settings import Settings
from fortuna.telegram.assistant import InteractionResult

_RAW_TEXT_MAX = 500


@dataclass(frozen=True)
class TelegramRequestAuditEntry:
    request_id: str
    ts: str
    channel: str = "telegram"
    chat_id: str = ""
    update_id: int | None = None
    raw_text: str = ""
    parsed_kind: str = ""
    query: str = ""
    symbol: str = ""
    timeframe: str = ""
    days: int = 0
    route_action: str = ""
    clarification: str | None = None
    source: str = ""
    source_confidence: float | None = None
    source_rationale: str | None = None
    tool: str | None = None
    tool_ok: bool | None = None
    error_code: str | None = None
    resolved_symbol: str | None = None
    decision_action: str | None = None
    hit_count: int | None = None
    modality: str = "text"
    attachment_kind: str | None = None
    forecast_used: bool | None = None
    response_chars: int = 0
    delivery_ok: bool | None = None
    delivery_error: str | None = None


class TelegramRequestAuditStore:
    """Append-only audit log for inbound Telegram assistant interactions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def from_settings(cls, settings: Settings) -> TelegramRequestAuditStore:
        log_path = settings.resolve_path(Path("logs/telegram/requests.jsonl"))
        return cls(log_path)

    def append(self, entry: TelegramRequestAuditEntry | dict[str, Any]) -> None:
        payload = entry if isinstance(entry, dict) else _compact_dict(asdict(entry))
        self._append(payload)

    def read_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return _read_jsonl(self.path, limit=limit)

    def apply_retention(self, *, keep_count: int, keep_days: int) -> int:
        rows = _read_jsonl(self.path)
        if not rows:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(keep_days)))
        kept: list[dict[str, Any]] = []
        pruned = 0
        for row in rows:
            ts = str(row.get("ts", "") or "")
            keep = True
            if ts:
                try:
                    keep = datetime.fromisoformat(ts) >= cutoff
                except ValueError:
                    keep = True
            if keep:
                kept.append(row)
            else:
                pruned += 1
        if len(kept) > keep_count:
            pruned += len(kept) - keep_count
            kept = kept[-keep_count:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return pruned

    def _append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def build_audit_entry(
    *,
    request_id: str,
    chat_id: str,
    update_id: int | None,
    interaction: InteractionResult,
    delivery_ok: bool | None = None,
    delivery_error: str | None = None,
) -> TelegramRequestAuditEntry:
    req = interaction.request
    route = interaction.route
    raw_text = redact_secrets(
        req.raw_text,
        enabled=True,
    )
    if len(raw_text) > _RAW_TEXT_MAX:
        raw_text = raw_text[:_RAW_TEXT_MAX]

    clarification: str | None = None
    if route.clarification is not None:
        clarification = route.clarification.value

    return TelegramRequestAuditEntry(
        request_id=request_id,
        ts=datetime.now(timezone.utc).isoformat(),
        chat_id=str(chat_id),
        update_id=update_id,
        raw_text=raw_text,
        parsed_kind=req.kind.value,
        query=req.query,
        symbol=req.symbol,
        timeframe=req.timeframe,
        days=req.days,
        route_action=route.action.value,
        clarification=clarification,
        source=interaction.source,
        source_confidence=interaction.source_confidence,
        source_rationale=interaction.source_rationale,
        tool=interaction.tool,
        tool_ok=interaction.tool_ok,
        error_code=interaction.error_code,
        resolved_symbol=interaction.resolved_symbol,
        decision_action=interaction.decision_action,
        hit_count=interaction.hit_count,
        modality=str(getattr(interaction, "modality", "text") or "text"),
        attachment_kind=getattr(interaction, "attachment_kind", None),
        forecast_used=(
            bool(interaction.signal_response.forecast_used)
            if interaction.signal_response is not None
            else None
        ),
        response_chars=len(interaction.reply),
        delivery_ok=delivery_ok,
        delivery_error=delivery_error,
    )


def _compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _read_jsonl(path: Path, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
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
