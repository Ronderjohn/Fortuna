from __future__ import annotations

from support.conversation_eval_fakes import (
    FakeEvalCatalog,
    FakeEvalRegistry,
    build_eval_settings,
)

from fortuna.agentic.conversational_adapter import AdapterSource, FortunaConversationalAdapter
from fortuna.agentic.openai_router import OpenAIPlan
from fortuna.telegram.parser import TelegramRequest, TelegramRequestKind
from fortuna.telegram.router import ClarificationCode, RouteAction


def test_adapter_keeps_explicit_command_path(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("/analyze RELIANCE")
    assert plan.source == AdapterSource.COMMAND
    assert plan.route.action == RouteAction.ANALYZE
    assert plan.request.symbol == "RELIANCE"


def test_adapter_maps_freeform_stock_to_analyze(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("How is Reliance looking on 15m for 20d?")
    assert plan.source == AdapterSource.CONVERSATIONAL
    assert plan.route.action == RouteAction.ANALYZE
    assert plan.request.symbol == "RELIANCE"
    assert plan.request.timeframe == "15m"
    assert plan.request.days == 20


def test_adapter_maps_freeform_search_intent(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("find reliance")
    assert plan.route.action == RouteAction.SEARCH
    assert plan.request.query == "RELIANCE"


def test_adapter_maps_freeform_future_intent(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("How is Reliance future looking?")
    assert plan.route.action == RouteAction.ANALYZE
    assert plan.request.symbol == "RELIANCE.FUT"


def test_adapter_maps_freeform_futures_plural_intent(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("Reliance futures")
    assert plan.route.action == RouteAction.ANALYZE
    assert plan.request.symbol == "RELIANCE.FUT"


def test_adapter_maps_freeform_futures_risk_intent(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("Can I take one lot of Reliance futures right now?")
    assert plan.route.action == RouteAction.RISK
    assert plan.request.symbol == "RELIANCE.FUT"


def test_adapter_maps_freeform_option_intent(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("Should I enter NIFTY CE 25000 28MAY2026?")
    assert plan.route.action == RouteAction.ANALYZE
    assert plan.request.symbol == "NIFTY.OPT.CE.25000.28MAY2026"


def test_adapter_clarifies_when_no_instrument_can_be_inferred(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("what should I do now?")
    assert plan.route.action == RouteAction.CLARIFY
    assert plan.route.clarification == ClarificationCode.FREEFORM_INSTRUMENT_NEEDED


def test_adapter_maps_universe_scan_request(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("Show liquid volume dense stocks")
    assert plan.route.action == RouteAction.UNIVERSE
    assert plan.request.limit == 10


def test_adapter_maps_freeform_shortlist_briefing(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("Summarize the top liquid stock setups for me")
    assert plan.source == AdapterSource.CONVERSATIONAL
    assert plan.route.action == RouteAction.BRIEF
    assert plan.request.limit == 5


def test_adapter_maps_training_candidate_request(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("Find RL training candidates from liquid stocks")
    assert plan.source == AdapterSource.CONVERSATIONAL
    assert plan.route.action == RouteAction.CANDIDATES
    assert plan.request.target == "rl"


def test_adapter_maps_portfolio_allocation_request(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan(
        "How would you allocate the top liquid stock setups into a small portfolio?"
    )
    assert plan.source == AdapterSource.CONVERSATIONAL
    assert plan.route.action == RouteAction.ALLOCATE
    assert plan.request.limit == 5


def test_adapter_maps_workflow_request(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("Show the full workflow for liquid stocks")
    assert plan.source == AdapterSource.CONVERSATIONAL
    assert plan.route.action == RouteAction.WORKFLOW


def test_adapter_maps_health_request(tmp_path):
    adapter = FortunaConversationalAdapter(
        build_eval_settings(tmp_path),
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("show model health")
    assert plan.route.action == RouteAction.HEALTH


def test_adapter_uses_openai_mode_when_enabled(tmp_path, monkeypatch):
    settings = build_eval_settings(tmp_path).model_copy(
        update={
            "conversational_adapter_mode": "openai",
            "openai_api_key": "test-key",
        }
    )

    def _fake_plan(self, text):
        req = TelegramRequest(
            TelegramRequestKind.UNIVERSE,
            raw_text=text,
            timeframe="1d",
            days=30,
            limit=8,
            source="auto",
        )
        from fortuna.telegram.router import route_telegram_request

        return OpenAIPlan(
            request=req,
            route=route_telegram_request(req),
            tool_name="get_market_universe",
            confidence=0.9,
            rationale="fake planner",
        )

    monkeypatch.setattr("fortuna.agentic.openai_router.OpenAIConversationPlanner.plan", _fake_plan)
    adapter = FortunaConversationalAdapter(
        settings,
        registry=FakeEvalRegistry(),
        catalog=FakeEvalCatalog(),
    )
    plan = adapter.plan("Find active stocks")
    assert plan.source == AdapterSource.OPENAI
    assert plan.route.action == RouteAction.UNIVERSE
