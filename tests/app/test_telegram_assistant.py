from __future__ import annotations

from support.conversation_eval_fakes import FakeEvalEngine, build_eval_assistant

from fortuna.telegram.router import ClarificationCode, RouteAction


def test_assistant_search_formats_hits(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    text = assistant.handle_text("/search RELIANCE")
    assert "RELIANCE.NS" in text
    assert "[EQUITY]" in text


def test_assistant_analyze_formats_decision(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    text = assistant.handle_text("/analyze RELIANCE")
    assert "Fortuna analysis: RELIANCE.NS" in text
    assert "Decision: BUY" in text
    assert "Winning strategy: orb" in text


def test_assistant_analyze_future_formats_sell_advisory(tmp_path):
    engine = FakeEvalEngine()
    assistant = build_eval_assistant(tmp_path, engine=engine)
    text = assistant.handle_text("/analyze RELIANCE FUT")
    assert engine.loaded_symbol == "RELIANCE.FUT"
    assert "Segment: FUTURE" in text
    assert "Decision: SELL" in text
    assert "Action plan:" in text


def test_assistant_analyze_option_formats_do_not_enter(tmp_path):
    engine = FakeEvalEngine()
    assistant = build_eval_assistant(tmp_path, engine=engine)
    text = assistant.handle_text("/analyze NIFTY CE 25000 28MAY2026")
    assert engine.loaded_symbol == "NIFTY.OPT.CE.25000.28MAY2026"
    assert "Segment: OPTION" in text
    assert "Decision: DO_NOT_ENTER" in text
    assert "Advisory only" in text


def test_handle_interaction_analyze_without_symbol_clarifies(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    result = assistant.handle_interaction("/analyze")
    assert result.route.action == RouteAction.CLARIFY
    assert result.route.clarification == ClarificationCode.EMPTY_ANALYZE_SYMBOL
    assert result.tool is None


def test_handle_interaction_analyze_unknown_sets_resolve_failed(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    result = assistant.handle_interaction("/analyze UNKNOWN")
    assert result.route.action == RouteAction.ANALYZE
    assert result.tool == "analyze_instrument"
    assert result.tool_ok is False
    assert result.error_code == "resolve_failed"


def test_handle_interaction_freeform_stock_query_uses_conversational_source(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    result = assistant.handle_interaction("How is Reliance looking on 15m for 20d?")
    assert result.source == "conversational"
    assert result.route.action == RouteAction.ANALYZE
    assert result.request.timeframe == "15m"
    assert result.request.days == 20
    assert "Fortuna analysis: RELIANCE.NS" in result.reply


def test_handle_interaction_freeform_option_query_uses_conversational_source(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    result = assistant.handle_interaction("Should I enter NIFTY CE 25000 28MAY2026?")
    assert result.source == "conversational"
    assert result.route.action == RouteAction.ANALYZE
    assert result.request.symbol == "NIFTY.OPT.CE.25000.28MAY2026"
    assert "Decision: DO_NOT_ENTER" in result.reply


def test_handle_interaction_freeform_unknown_clarifies(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    result = assistant.handle_interaction("what should I do now?")
    assert result.source == "conversational"
    assert result.route.action == RouteAction.CLARIFY
    assert "clearer instrument request" in result.reply


def test_assistant_incomplete_option_returns_syntax_guidance(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    text = assistant.handle_text("/analyze NIFTY CE 25000")
    assert "28MAY2026" in text
    assert "CE" in text


def test_assistant_unresolved_symbol_suggests_search(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    text = assistant.handle_text("/analyze UNKNOWNXYZ")
    assert "/search" in text


def test_assistant_empty_search_suggests_next_steps(tmp_path):
    from fortuna.agentic.tools import FortunaAdvisoryTools

    class EmptyCatalog:
        def ensure_loaded(self, force_refresh=False):
            return None

        def search(self, query, limit=8):
            return []

    class EmptyRegistry:
        def ensure_loaded(self, force_refresh=False):
            return None

        def search_options(self, query, limit=4):
            return []

    settings = build_eval_assistant(tmp_path).settings
    tools = FortunaAdvisoryTools(
        settings=settings,
        registry=EmptyRegistry(),
        catalog=EmptyCatalog(),
    )
    assistant = build_eval_assistant(tmp_path)
    assistant.tools = tools
    text = assistant.handle_text("/search NOTAVALIDSYMBOL")
    assert "No instruments found" in text
    assert "/analyze NIFTY CE 25000 28MAY2026" in text
