from __future__ import annotations

from fortuna.telegram.parser import parse_telegram_request
from fortuna.telegram.router import (
    ClarificationCode,
    RouteAction,
    route_telegram_request,
)


def test_route_search_equity():
    route = route_telegram_request(parse_telegram_request("/search RELIANCE"))
    assert route.action == RouteAction.SEARCH


def test_route_search_empty():
    route = route_telegram_request(parse_telegram_request("/search"))
    assert route.action == RouteAction.CLARIFY
    assert route.clarification == ClarificationCode.EMPTY_SEARCH_QUERY


def test_route_analyze_empty():
    route = route_telegram_request(parse_telegram_request("/analyze"))
    assert route.action == RouteAction.CLARIFY
    assert route.clarification == ClarificationCode.EMPTY_ANALYZE_SYMBOL


def test_route_analyze_timeframe_only():
    route = route_telegram_request(parse_telegram_request("/analyze 15m"))
    assert route.action == RouteAction.CLARIFY
    assert route.clarification == ClarificationCode.EMPTY_ANALYZE_SYMBOL


def test_route_analyze_equity():
    route = route_telegram_request(parse_telegram_request("/analyze RELIANCE"))
    assert route.action == RouteAction.ANALYZE


def test_route_help():
    route = route_telegram_request(parse_telegram_request("/help"))
    assert route.action == RouteAction.SHOW_HELP


def test_route_unknown():
    route = route_telegram_request(parse_telegram_request("what is the weather"))
    assert route.action == RouteAction.UNSUPPORTED


def test_route_analyze_incomplete_option():
    route = route_telegram_request(parse_telegram_request("/analyze NIFTY CE 25000"))
    assert route.action == RouteAction.CLARIFY
    assert route.clarification == ClarificationCode.INCOMPLETE_OPTION_SYNTAX
