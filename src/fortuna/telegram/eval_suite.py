"""Deterministic evaluation harness for conversational advisory requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

from fortuna.agentic.tools import FortunaAdvisoryTools
from fortuna.telegram.assistant import TelegramAnalysisAssistant
from fortuna.telegram.parser import TelegramRequestKind, parse_telegram_request
from fortuna.telegram.router import route_telegram_request


class ConversationEvalFailure(AssertionError):
    def __init__(self, case_id: str, message: str) -> None:
        super().__init__(f"[{case_id}] {message}")
        self.case_id = case_id


@dataclass(frozen=True)
class ConversationEvalExpectation:
    kind: Optional[str] = None
    query: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    days: Optional[int] = None
    tool: Optional[str] = None
    ok: Optional[bool] = None
    error_code: Optional[str] = None
    resolved_symbol: Optional[str] = None
    segment_contains: Optional[str] = None
    decision_action: Optional[str] = None
    hit_symbols: tuple[str, ...] = ()
    text_contains: tuple[str, ...] = ()
    route_action: Optional[str] = None
    clarification: Optional[str] = None


@dataclass(frozen=True)
class ConversationEvalCase:
    id: str
    layer: Literal["parse", "route", "tool", "assistant"]
    input: str
    expect: ConversationEvalExpectation


def default_eval_fixture_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tests/fixtures/conversation_eval/v1.yaml"


def load_eval_cases(path: Path | str) -> list[ConversationEvalCase]:
    fixture_path = Path(path)
    raw = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid eval fixture: {fixture_path}")
    version = raw.get("version")
    if version != 1:
        raise ValueError(f"Unsupported eval fixture version: {version!r}")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise ValueError(f"Invalid eval fixture cases: {fixture_path}")

    cases: list[ConversationEvalCase] = []
    for row in cases_raw:
        if not isinstance(row, dict):
            raise ValueError("Each eval case must be a mapping")
        case_id = str(row.get("id", "")).strip()
        layer = str(row.get("layer", "")).strip()
        if layer not in {"parse", "route", "tool", "assistant"}:
            raise ValueError(f"Unknown eval layer {layer!r} in case {case_id!r}")
        expect_raw = row.get("expect") or {}
        if not isinstance(expect_raw, dict):
            raise ValueError(f"Invalid expect block in case {case_id!r}")
        hit_symbols = expect_raw.get("hit_symbols") or []
        text_contains = expect_raw.get("text_contains") or []
        days = expect_raw.get("days")
        cases.append(
            ConversationEvalCase(
                id=case_id,
                layer=layer,  # type: ignore[arg-type]
                input=str(row.get("input", "")),
                expect=ConversationEvalExpectation(
                    kind=expect_raw.get("kind"),
                    query=expect_raw.get("query"),
                    symbol=expect_raw.get("symbol"),
                    timeframe=expect_raw.get("timeframe"),
                    days=int(days) if days is not None else None,
                    tool=expect_raw.get("tool"),
                    ok=expect_raw.get("ok"),
                    error_code=expect_raw.get("error_code"),
                    resolved_symbol=expect_raw.get("resolved_symbol"),
                    segment_contains=expect_raw.get("segment_contains"),
                    decision_action=expect_raw.get("decision_action"),
                    hit_symbols=tuple(str(s) for s in hit_symbols),
                    text_contains=tuple(str(s) for s in text_contains),
                    route_action=expect_raw.get("route_action"),
                    clarification=expect_raw.get("clarification"),
                ),
            )
        )
    return cases


def evaluate_parse(case: ConversationEvalCase) -> None:
    req = parse_telegram_request(case.input)
    expect = case.expect
    if expect.kind is not None:
        _assert_equal(case.id, "kind", req.kind.value, expect.kind)
    if expect.query is not None:
        _assert_equal(case.id, "query", req.query, expect.query)
    if expect.symbol is not None:
        _assert_equal(case.id, "symbol", req.symbol, expect.symbol)
    if expect.timeframe is not None:
        _assert_equal(case.id, "timeframe", req.timeframe, expect.timeframe)
    if expect.days is not None:
        _assert_equal(case.id, "days", req.days, expect.days)


def evaluate_route(case: ConversationEvalCase) -> None:
    req = parse_telegram_request(case.input)
    route = route_telegram_request(req)
    expect = case.expect
    if expect.route_action is not None:
        _assert_equal(case.id, "route_action", route.action.value, expect.route_action)
    if expect.clarification is not None:
        if route.clarification is None:
            raise ConversationEvalFailure(case.id, "expected clarification code")
        _assert_equal(
            case.id,
            "clarification",
            route.clarification.value,
            expect.clarification,
        )


def evaluate_tool(case: ConversationEvalCase, tools: FortunaAdvisoryTools) -> None:
    expect = case.expect
    tool_name = expect.tool
    if tool_name is None:
        raise ConversationEvalFailure(case.id, "tool expectation is required for tool layer")

    req = parse_telegram_request(case.input)
    if tool_name == "search_instruments":
        query = req.query if req.kind == TelegramRequestKind.SEARCH else case.input
        response = tools.search_instruments(query)
    elif tool_name == "analyze_instrument":
        response = tools.analyze_instrument(
            req.symbol,
            timeframe=req.timeframe,
            days=req.days,
        )
    else:
        raise ConversationEvalFailure(case.id, f"unsupported tool {tool_name!r}")

    if expect.ok is not None:
        _assert_equal(case.id, "ok", response.ok, expect.ok)
    if expect.error_code is not None:
        if response.error is None:
            raise ConversationEvalFailure(case.id, "expected error but response.error is None")
        _assert_equal(case.id, "error_code", response.error.code.value, expect.error_code)
    if expect.resolved_symbol is not None:
        if response.instrument is None:
            raise ConversationEvalFailure(case.id, "expected instrument snapshot")
        _assert_equal(
            case.id,
            "resolved_symbol",
            response.instrument.resolved_symbol,
            expect.resolved_symbol,
        )
    if expect.segment_contains is not None:
        if response.instrument is None:
            raise ConversationEvalFailure(case.id, "expected instrument snapshot")
        if expect.segment_contains not in response.instrument.segment_label:
            raise ConversationEvalFailure(
                case.id,
                f"segment_label {response.instrument.segment_label!r} "
                f"does not contain {expect.segment_contains!r}",
            )
    if expect.decision_action is not None:
        if response.decision is None:
            raise ConversationEvalFailure(case.id, "expected decision summary")
        _assert_equal(case.id, "decision_action", response.decision.action, expect.decision_action)
    if expect.hit_symbols:
        symbols = {hit.symbol for hit in response.hits}
        for symbol in expect.hit_symbols:
            if symbol not in symbols:
                raise ConversationEvalFailure(
                    case.id,
                    f"expected hit symbol {symbol!r} in {sorted(symbols)!r}",
                )


def evaluate_assistant(
    case: ConversationEvalCase,
    assistant: TelegramAnalysisAssistant,
) -> None:
    text = assistant.handle_text(case.input)
    for fragment in case.expect.text_contains:
        if fragment not in text:
            raise ConversationEvalFailure(
                case.id,
                f"expected text fragment {fragment!r} in response",
            )


def _assert_equal(case_id: str, field_name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ConversationEvalFailure(
            case_id,
            f"{field_name}: expected {expected!r}, got {actual!r}",
        )
