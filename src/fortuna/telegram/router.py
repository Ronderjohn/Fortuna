"""Route parsed Telegram requests to tool execution or static responses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fortuna.config.settings import get_settings
from fortuna.observability.recorder import get_workflow_recorder
from fortuna.telegram.parser import TelegramRequest, TelegramRequestKind

_OPTION_TYPE_TOKENS = frozenset({"CE", "PE", "C", "P"})


class RouteAction(str, Enum):
    SHOW_HELP = "show_help"
    SEARCH = "search"
    ANALYZE = "analyze"
    RISK = "risk"
    UNIVERSE = "universe"
    RESEARCH = "research"
    BRIEF = "brief"
    ALLOCATE = "allocate"
    CANDIDATES = "candidates"
    WORKFLOW = "workflow"
    HEALTH = "health"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"


class ClarificationCode(str, Enum):
    EMPTY_SEARCH_QUERY = "empty_search_query"
    EMPTY_ANALYZE_SYMBOL = "empty_analyze_symbol"
    INCOMPLETE_OPTION_SYNTAX = "incomplete_option_syntax"
    FREEFORM_INSTRUMENT_NEEDED = "freeform_instrument_needed"


@dataclass(frozen=True)
class ConversationRoute:
    action: RouteAction
    request: TelegramRequest
    clarification: ClarificationCode | None = None


def route_telegram_request(req: TelegramRequest) -> ConversationRoute:
    route = _route_telegram_request_impl(req)
    try:
        settings = get_settings()
        recorder = get_workflow_recorder(settings)
        clarification = route.clarification.value if route.clarification is not None else None
        recorder.emit_step(
            event_name="telegram_route",
            module="fortuna.telegram.router",
            workflow_id="telegram_route",
            status="ok" if route.action != RouteAction.UNSUPPORTED else "warn",
            symbol=req.symbol or None,
            context={
                "route_action": route.action.value,
                "parsed_kind": req.kind.value,
                "clarification": clarification,
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return route


def _route_telegram_request_impl(req: TelegramRequest) -> ConversationRoute:
    if req.kind == TelegramRequestKind.HELP:
        return ConversationRoute(RouteAction.SHOW_HELP, req)

    if req.kind == TelegramRequestKind.SEARCH:
        if not req.query.strip():
            return ConversationRoute(
                RouteAction.CLARIFY,
                req,
                clarification=ClarificationCode.EMPTY_SEARCH_QUERY,
            )
        return ConversationRoute(RouteAction.SEARCH, req)

    if req.kind == TelegramRequestKind.ANALYZE:
        if not req.symbol.strip():
            return ConversationRoute(
                RouteAction.CLARIFY,
                req,
                clarification=ClarificationCode.EMPTY_ANALYZE_SYMBOL,
            )
        if _looks_like_incomplete_option(req):
            return ConversationRoute(
                RouteAction.CLARIFY,
                req,
                clarification=ClarificationCode.INCOMPLETE_OPTION_SYNTAX,
            )
        return ConversationRoute(RouteAction.ANALYZE, req)

    if req.kind == TelegramRequestKind.RISK:
        if not req.symbol.strip():
            return ConversationRoute(
                RouteAction.CLARIFY,
                req,
                clarification=ClarificationCode.EMPTY_ANALYZE_SYMBOL,
            )
        if _looks_like_incomplete_option(req):
            return ConversationRoute(
                RouteAction.CLARIFY,
                req,
                clarification=ClarificationCode.INCOMPLETE_OPTION_SYNTAX,
            )
        return ConversationRoute(RouteAction.RISK, req)

    if req.kind == TelegramRequestKind.UNIVERSE:
        return ConversationRoute(RouteAction.UNIVERSE, req)

    if req.kind == TelegramRequestKind.RESEARCH:
        return ConversationRoute(RouteAction.RESEARCH, req)

    if req.kind == TelegramRequestKind.BRIEF:
        return ConversationRoute(RouteAction.BRIEF, req)

    if req.kind == TelegramRequestKind.ALLOCATE:
        return ConversationRoute(RouteAction.ALLOCATE, req)

    if req.kind == TelegramRequestKind.CANDIDATES:
        return ConversationRoute(RouteAction.CANDIDATES, req)

    if req.kind == TelegramRequestKind.WORKFLOW:
        return ConversationRoute(RouteAction.WORKFLOW, req)

    if req.kind == TelegramRequestKind.HEALTH:
        return ConversationRoute(RouteAction.HEALTH, req)

    return ConversationRoute(RouteAction.UNSUPPORTED, req)


def _looks_like_incomplete_option(req: TelegramRequest) -> bool:
    if ".OPT." in req.symbol.upper():
        return False
    tokens = [tok for tok in req.query.replace(",", " ").split() if tok]
    if len(tokens) < 2:
        return False
    return tokens[1].upper() in _OPTION_TYPE_TOKENS
