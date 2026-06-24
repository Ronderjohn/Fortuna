from __future__ import annotations

from pathlib import Path

from fortuna.telegram.assistant import InteractionResult
from fortuna.telegram.parser import parse_telegram_request
from fortuna.telegram.request_audit import (
    TelegramRequestAuditEntry,
    TelegramRequestAuditStore,
    build_audit_entry,
)
from fortuna.telegram.router import (
    ClarificationCode,
    ConversationRoute,
    RouteAction,
    route_telegram_request,
)


def _interaction(
    text: str,
    *,
    reply: str = "reply",
    tool: str | None = None,
    tool_ok: bool | None = None,
    error_code: str | None = None,
    resolved_symbol: str | None = None,
    decision_action: str | None = None,
    hit_count: int | None = None,
) -> InteractionResult:
    req = parse_telegram_request(text)
    route = route_telegram_request(req)
    return InteractionResult(
        reply=reply,
        request=req,
        route=route,
        tool=tool,
        tool_ok=tool_ok,
        error_code=error_code,
        resolved_symbol=resolved_symbol,
        decision_action=decision_action,
        hit_count=hit_count,
    )


def test_audit_store_append_and_read_recent(tmp_path: Path):
    store = TelegramRequestAuditStore(tmp_path / "requests.jsonl")
    store.append(
        TelegramRequestAuditEntry(
            request_id="abc123",
            ts="2026-05-30T10:00:00+00:00",
            chat_id="1",
            raw_text="/help",
            parsed_kind="help",
            route_action="show_help",
            response_chars=42,
            delivery_ok=True,
        )
    )
    store.append(
        {
            "request_id": "def456",
            "ts": "2026-05-30T10:01:00+00:00",
            "route_action": "search",
        }
    )

    rows = store.read_recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["request_id"] == "abc123"
    assert rows[1]["request_id"] == "def456"


def test_build_audit_entry_help():
    interaction = _interaction("/help", reply="help text")
    entry = build_audit_entry(
        request_id="req1",
        chat_id="99",
        update_id=7,
        interaction=interaction,
        delivery_ok=True,
    )
    assert entry.request_id == "req1"
    assert entry.chat_id == "99"
    assert entry.update_id == 7
    assert entry.parsed_kind == "help"
    assert entry.route_action == "show_help"
    assert entry.source == "command"
    assert entry.tool is None
    assert entry.response_chars == len("help text")
    assert entry.delivery_ok is True


def test_build_audit_entry_search():
    interaction = _interaction(
        "/search RELIANCE",
        tool="search_instruments",
        tool_ok=True,
        hit_count=2,
    )
    entry = build_audit_entry(
        request_id="req2",
        chat_id="99",
        update_id=None,
        interaction=interaction,
    )
    assert entry.parsed_kind == "search"
    assert entry.route_action == "search"
    assert entry.query == "RELIANCE"
    assert entry.tool == "search_instruments"
    assert entry.tool_ok is True
    assert entry.hit_count == 2


def test_build_audit_entry_analyze():
    interaction = _interaction(
        "/analyze RELIANCE",
        tool="analyze_instrument",
        tool_ok=True,
        resolved_symbol="RELIANCE.NS",
        decision_action="BUY",
    )
    entry = build_audit_entry(
        request_id="req3",
        chat_id="99",
        update_id=3,
        interaction=interaction,
        delivery_ok=False,
        delivery_error="send failed",
    )
    assert entry.route_action == "analyze"
    assert entry.symbol == "RELIANCE"
    assert entry.resolved_symbol == "RELIANCE.NS"
    assert entry.decision_action == "BUY"
    assert entry.delivery_ok is False
    assert entry.delivery_error == "send failed"


def test_build_audit_entry_clarify():
    req = parse_telegram_request("/analyze")
    route = ConversationRoute(
        RouteAction.CLARIFY,
        req,
        clarification=ClarificationCode.EMPTY_ANALYZE_SYMBOL,
    )
    interaction = InteractionResult(reply="clarify", request=req, route=route)
    entry = build_audit_entry(
        request_id="req4",
        chat_id="99",
        update_id=4,
        interaction=interaction,
    )
    assert entry.route_action == "clarify"
    assert entry.clarification == "empty_analyze_symbol"
    assert entry.tool is None


def test_build_audit_entry_unsupported():
    interaction = _interaction("hello there", reply="unsupported")
    entry = build_audit_entry(
        request_id="req5",
        chat_id="99",
        update_id=5,
        interaction=interaction,
    )
    assert entry.parsed_kind == "unknown"
    assert entry.route_action == "unsupported"


def test_build_audit_entry_caps_raw_text():
    long_text = "/search " + ("x" * 600)
    interaction = _interaction(long_text)
    entry = build_audit_entry(
        request_id="req6",
        chat_id="99",
        update_id=6,
        interaction=interaction,
    )
    assert len(entry.raw_text) == 500


def test_build_audit_entry_omits_none_fields_in_store(tmp_path: Path):
    interaction = _interaction("/help")
    entry = build_audit_entry(
        request_id="req7",
        chat_id="99",
        update_id=None,
        interaction=interaction,
    )
    store = TelegramRequestAuditStore(tmp_path / "requests.jsonl")
    store.append(entry)
    row = store.read_recent(limit=1)[0]
    assert "tool" not in row
    assert "delivery_ok" not in row


def test_build_audit_entry_captures_conversational_source():
    req = parse_telegram_request("How is RELIANCE looking?")
    route = ConversationRoute(RouteAction.ANALYZE, req)
    interaction = InteractionResult(
        reply="reply",
        request=req,
        route=route,
        source="conversational",
        source_confidence=0.82,
        source_rationale="freeform request was normalized into typed analysis",
        tool="analyze_instrument",
        tool_ok=True,
    )
    entry = build_audit_entry(
        request_id="req8",
        chat_id="99",
        update_id=8,
        interaction=interaction,
    )
    assert entry.source == "conversational"
    assert entry.source_confidence == 0.82
    assert "normalized" in str(entry.source_rationale)


def test_build_audit_entry_redacts_secret_like_text():
    interaction = _interaction("/search token=sk-secret-1234567890abcdefghijklmnop")
    entry = build_audit_entry(
        request_id="req9",
        chat_id="99",
        update_id=9,
        interaction=interaction,
    )
    assert "sk-secret" not in entry.raw_text
    assert "[REDACTED]" in entry.raw_text


def test_audit_store_apply_retention_keeps_recent_rows(tmp_path: Path):
    store = TelegramRequestAuditStore(tmp_path / "requests.jsonl")
    store.append(
        TelegramRequestAuditEntry(
            request_id="old",
            ts="2026-01-01T00:00:00+00:00",
            chat_id="1",
            raw_text="/help",
            parsed_kind="help",
            route_action="show_help",
            response_chars=4,
        )
    )
    store.append(
        TelegramRequestAuditEntry(
            request_id="new",
            ts="2026-06-21T00:00:00+00:00",
            chat_id="1",
            raw_text="/help",
            parsed_kind="help",
            route_action="show_help",
            response_chars=4,
        )
    )
    pruned = store.apply_retention(keep_count=10, keep_days=30)
    rows = store.read_recent(limit=10)
    assert pruned == 1
    assert [row["request_id"] for row in rows] == ["new"]
