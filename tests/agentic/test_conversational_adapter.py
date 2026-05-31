from __future__ import annotations

from support.conversation_eval_fakes import (
    FakeEvalCatalog,
    FakeEvalRegistry,
    build_eval_settings,
)

from fortuna.agentic.conversational_adapter import AdapterSource, FortunaConversationalAdapter
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
