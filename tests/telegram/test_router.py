from __future__ import annotations

from fortuna.config.settings import Settings
from fortuna.observability.sinks import read_workflow_events
from fortuna.telegram.parser import parse_telegram_request
from fortuna.telegram.router import (
    ClarificationCode,
    RouteAction,
    route_telegram_request,
)


def test_route_search_equity():
    route = route_telegram_request(parse_telegram_request("/search RELIANCE"))
    assert route.action == RouteAction.SEARCH


def test_route_emits_observability_event(tmp_path, monkeypatch):
    settings = Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        observability_enabled=True,
        observability_log_dir=tmp_path / "logs" / "observability",
    )
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("fortuna.telegram.router.get_settings", lambda: settings)
    route_telegram_request(parse_telegram_request("/universe 1d 30d limit=12"))
    rows = read_workflow_events(tmp_path / "logs" / "observability" / "workflow_events.jsonl")
    assert len(rows) == 1
    assert rows[0]["event_name"] == "telegram_route"
    assert rows[0]["context"]["route_action"] == "universe"


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


def test_route_universe():
    route = route_telegram_request(parse_telegram_request("/universe 1d 30d limit=12"))
    assert route.action == RouteAction.UNIVERSE
    assert route.request.limit == 12


def test_route_brief():
    route = route_telegram_request(parse_telegram_request("/brief 5m 20d limit=6"))
    assert route.action == RouteAction.BRIEF
    assert route.request.limit == 6
    assert route.request.timeframe == "5m"


def test_route_candidates():
    route = route_telegram_request(parse_telegram_request("/candidates rl 15m 30d limit=7"))
    assert route.action == RouteAction.CANDIDATES
    assert route.request.limit == 7
    assert route.request.timeframe == "15m"
    assert route.request.target == "rl"


def test_route_workflow():
    route = route_telegram_request(parse_telegram_request("/workflow 5m 20d limit=6"))
    assert route.action == RouteAction.WORKFLOW
    assert route.request.limit == 6
    assert route.request.timeframe == "5m"


def test_route_health():
    route = route_telegram_request(parse_telegram_request("/health"))
    assert route.action == RouteAction.HEALTH


def test_route_unknown():
    route = route_telegram_request(parse_telegram_request("what is the weather"))
    assert route.action == RouteAction.UNSUPPORTED


def test_route_analyze_incomplete_option():
    route = route_telegram_request(parse_telegram_request("/analyze NIFTY CE 25000"))
    assert route.action == RouteAction.CLARIFY
    assert route.clarification == ClarificationCode.INCOMPLETE_OPTION_SYNTAX
