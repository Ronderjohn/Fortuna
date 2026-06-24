from __future__ import annotations

from pathlib import Path

from fortuna.agentic.openai_router import OpenAIConversationPlanner
from fortuna.config.settings import Settings
from fortuna.telegram.router import RouteAction


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        openai_api_key="test-key",
        openai_model="gpt-5.4-mini",
    )


def test_openai_planner_maps_analyze_tool(tmp_path: Path):
    def _transport(url, payload, headers, timeout):
        return {
            "id": "resp_123",
            "output": [
                {
                    "type": "function_call",
                    "name": "analyze_instrument",
                    "arguments": '{"symbol":"RELIANCE.NS","timeframe":"15m","days":20}',
                }
            ],
        }

    planner = OpenAIConversationPlanner(_settings(tmp_path), transport=_transport)
    plan = planner.plan("How is Reliance looking on 15m for 20d?")
    assert plan.tool_name == "analyze_instrument"
    assert plan.route.action == RouteAction.ANALYZE
    assert plan.request.symbol == "RELIANCE.NS"
    assert plan.request.timeframe == "15m"
    assert plan.request.days == 20


def test_openai_planner_maps_universe_tool(tmp_path: Path):
    def _transport(url, payload, headers, timeout):
        return {
            "id": "resp_456",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_market_universe",
                    "arguments": '{"timeframe":"1d","days":30,"limit":12,"source":"screener"}',
                }
            ],
        }

    planner = OpenAIConversationPlanner(_settings(tmp_path), transport=_transport)
    plan = planner.plan("Find liquid high-volume stocks for training")
    assert plan.tool_name == "get_market_universe"
    assert plan.route.action == RouteAction.UNIVERSE
    assert plan.request.limit == 12
    assert plan.request.source == "screener"


def test_openai_planner_maps_health_tool(tmp_path: Path):
    def _transport(url, payload, headers, timeout):
        return {
            "id": "resp_789",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_model_health",
                    "arguments": "{}",
                }
            ],
        }

    planner = OpenAIConversationPlanner(_settings(tmp_path), transport=_transport)
    plan = planner.plan("Show me model health")
    assert plan.route.action == RouteAction.HEALTH


def test_openai_planner_maps_shortlist_briefing_tool(tmp_path: Path):
    def _transport(url, payload, headers, timeout):
        return {
            "id": "resp_990",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_shortlist_briefing",
                    "arguments": '{"timeframe":"5m","days":20,"limit":5,"source":"auto"}',
                }
            ],
        }

    planner = OpenAIConversationPlanner(_settings(tmp_path), transport=_transport)
    plan = planner.plan("Summarize the top setups from active stocks")
    assert plan.tool_name == "get_shortlist_briefing"
    assert plan.route.action == RouteAction.BRIEF
    assert plan.request.limit == 5


def test_openai_planner_maps_training_candidates_tool(tmp_path: Path):
    def _transport(url, payload, headers, timeout):
        return {
            "id": "resp_991",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_training_candidates",
                    "arguments": (
                        '{"timeframe":"15m","days":30,"limit":8,'
                        '"source":"auto","target":"rl"}'
                    ),
                }
            ],
        }

    planner = OpenAIConversationPlanner(_settings(tmp_path), transport=_transport)
    plan = planner.plan("Find RL training candidates from active liquid stocks")
    assert plan.tool_name == "get_training_candidates"
    assert plan.route.action == RouteAction.CANDIDATES
    assert plan.request.target == "rl"


def test_openai_planner_maps_portfolio_allocation_tool(tmp_path: Path):
    def _transport(url, payload, headers, timeout):
        return {
            "id": "resp_991a",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_portfolio_allocation",
                    "arguments": (
                        '{"timeframe":"5m","days":20,"limit":5,"source":"auto",'
                        '"max_positions":3,"max_per_exposure":1,"max_same_side":2}'
                    ),
                }
            ],
        }

    planner = OpenAIConversationPlanner(_settings(tmp_path), transport=_transport)
    plan = planner.plan("Allocate the top setups into a small portfolio")
    assert plan.tool_name == "get_portfolio_allocation"
    assert plan.route.action == RouteAction.ALLOCATE
    assert plan.request.max_positions == 3


def test_openai_planner_maps_workflow_summary_tool(tmp_path: Path):
    def _transport(url, payload, headers, timeout):
        return {
            "id": "resp_992",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_workflow_summary",
                    "arguments": '{"timeframe":"5m","days":20,"limit":8,"source":"auto"}',
                }
            ],
        }

    planner = OpenAIConversationPlanner(_settings(tmp_path), transport=_transport)
    plan = planner.plan("Show me the full workflow for active stocks")
    assert plan.tool_name == "get_workflow_summary"
    assert plan.route.action == RouteAction.WORKFLOW
