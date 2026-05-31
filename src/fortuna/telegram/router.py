"""Route parsed Telegram requests to tool execution or static responses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fortuna.telegram.parser import TelegramRequest, TelegramRequestKind

_OPTION_TYPE_TOKENS = frozenset({"CE", "PE", "C", "P"})


class RouteAction(str, Enum):
    SHOW_HELP = "show_help"
    SEARCH = "search"
    ANALYZE = "analyze"
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

    return ConversationRoute(RouteAction.UNSUPPORTED, req)


def _looks_like_incomplete_option(req: TelegramRequest) -> bool:
    if ".OPT." in req.symbol.upper():
        return False
    tokens = [tok for tok in req.query.replace(",", " ").split() if tok]
    if len(tokens) < 2:
        return False
    return tokens[1].upper() in _OPTION_TYPE_TOKENS
