from __future__ import annotations

from support.conversation_eval_fakes import FakeEvalEngine, build_eval_assistant

from fortuna.agentic.contracts import (
    AgenticStatusSummary,
    AgentRoleStatus,
    AllocationDecision,
    BriefingItem,
    ChartImageAnalysisResult,
    ConfidenceComponent,
    DecisionSummary,
    FuturesLotRiskAssessment,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSnapshot,
    LearningSummaryStatus,
    MarketUniverseCandidate,
    MarketUniverseResponse,
    MediaAttachment,
    MlModelStatus,
    ModelHealthResponse,
    MultiAgentWorkflowResponse,
    NightlyAlignmentRow,
    NightlyAlignmentStatus,
    PortfolioAllocationResponse,
    PortfolioCritique,
    PortfolioExposureNote,
    RegimeModelStatus,
    ReplayRiskSummary,
    RiskCritique,
    RlModelStatus,
    ShortlistAnalysisItem,
    ShortlistAnalysisResponse,
    ShortlistBriefingResponse,
    StructuredSignalDetails,
    TrainingCandidate,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
    TrainingResearchRow,
)
from fortuna.telegram.router import ClarificationCode, RouteAction
from fortuna.telegram.session_store import TelegramConversationState


def test_assistant_search_formats_hits(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    text = assistant.handle_text("/search RELIANCE")
    assert "RELIANCE.NS" in text
    assert "[EQUITY]" in text


def test_assistant_analyze_formats_decision(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    text = assistant.handle_text("/analyze RELIANCE")
    assert "RELIANCE.NS" in text
    assert "Verdict: BUY" in text
    assert "Confidence:" in text


def test_assistant_analyze_future_formats_sell_advisory(tmp_path):
    engine = FakeEvalEngine()
    assistant = build_eval_assistant(tmp_path, engine=engine)
    text = assistant.handle_text("/analyze RELIANCE FUT")
    assert engine.loaded_symbol == "RELIANCE.FUT"
    assert "RELIANCE.FUT" in text
    assert "Verdict: SELL" in text
    assert "Next:" in text


def test_assistant_analyze_option_formats_do_not_enter(tmp_path):
    engine = FakeEvalEngine()
    assistant = build_eval_assistant(tmp_path, engine=engine)
    text = assistant.handle_text("/analyze NIFTY CE 25000 28MAY2026")
    assert engine.loaded_symbol == "NIFTY.OPT.CE.25000.28MAY2026"
    assert "NIFTY.OPT.CE.25000.28MAY2026" in text
    assert "Verdict: DO_NOT_ENTER" in text
    assert "Last bar:" in text


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
    assert "RELIANCE.NS" in result.reply


def test_handle_interaction_freeform_option_query_uses_conversational_source(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    result = assistant.handle_interaction("Should I enter NIFTY CE 25000 28MAY2026?")
    assert result.source == "conversational"
    assert result.route.action == RouteAction.ANALYZE
    assert result.request.symbol == "NIFTY.OPT.CE.25000.28MAY2026"
    assert "Verdict: DO_NOT_ENTER" in result.reply


def test_handle_interaction_follow_up_uses_per_user_context(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    state = TelegramConversationState(
        chat_id="12345",
        last_symbol="RELIANCE.NS",
        last_timeframe="5m",
        last_days=20,
    )
    result = assistant.handle_interaction("What about 15m?", conversation_state=state)
    assert result.route.action == RouteAction.ANALYZE
    assert result.request.symbol == "RELIANCE.NS"
    assert result.request.timeframe == "15m"
    assert result.request.days == 20


def test_handle_interaction_follow_up_risk_uses_per_user_context(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    state = TelegramConversationState(
        chat_id="12345",
        last_symbol="RELIANCE.FUT",
        last_resolved_symbol="RELIANCE.FUT",
        last_timeframe="5m",
        last_days=20,
    )
    result = assistant.handle_interaction("What stop should I use?", conversation_state=state)
    assert result.route.action == RouteAction.RISK
    assert result.request.symbol == "RELIANCE.FUT"
    assert result.request.timeframe == "5m"
    assert result.request.days == 20


def test_help_advanced_hides_and_shows_expert_commands(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    default_text = assistant.handle_text("/help")
    advanced_text = assistant.handle_text("/help advanced")
    assert "/workflow" not in default_text
    assert "/workflow" in advanced_text


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


def test_assistant_universe_formats_ranked_candidates(tmp_path, monkeypatch):
    expected = MarketUniverseResponse(
        ok=True,
        source="registry",
        timeframe="1d",
        lookback_days=30,
        nightly_alignment_summary="recent=1/2 over 2 run(s); target=all force_refresh=1",
        nightly_alignment_target="all",
        nightly_alignment_force_refresh=True,
        candidates=(
            MarketUniverseCandidate(
                symbol="RELIANCE.NS",
                display_name="RELIANCE",
                source="registry",
                liquidity_score=12.5,
                trend_pct=4.2,
                notes=("adaptive_boost=+0.10 avg_pnl=+1.50% rows=2",),
            ),
        ),
    )
    monkeypatch.setattr(
        "fortuna.agentic.tools.build_market_universe",
        lambda **kwargs: expected,
    )
    assistant = build_eval_assistant(tmp_path)
    text = assistant.handle_text("/universe 1d 30d limit=5")
    assert "Fortuna market universe" in text
    assert "Nightly posture: recent=1/2 over 2 run(s); target=all force_refresh=1" in text
    assert "Nightly target: all force_refresh=1" in text
    assert "RELIANCE.NS" in text
    assert "adaptive_boost=+0.10" in text


def test_assistant_research_formats_training_plan(tmp_path):
    assistant = build_eval_assistant(tmp_path)

    class ResearchTools:
        def search_instruments(self, query):
            raise AssertionError("not used")

        def analyze_instrument(self, symbol, timeframe="5m", days=30):
            raise AssertionError("not used")

        def get_market_universe(self, **kwargs):
            raise AssertionError("not used")

        def get_training_research_plan(self, **kwargs):
            return TrainingResearchPlanResponse(
                ok=True,
                source="screener",
                timeframe="5m",
                lookback_days=20,
                selection_policy="diversified",
                refresh_target="rl",
                nightly_alignment_summary="recent=1/2 over 2 run(s); target=rl force_refresh=1",
                nightly_alignment_target="rl",
                nightly_alignment_force_refresh=True,
                rows=(
                    TrainingResearchRow(
                        symbol="RELIANCE.NS",
                        target="both",
                        selection_rank=1,
                        decision_action="BUY",
                        critique_verdict="candidate",
                        market_regime="TRENDING",
                        refreshed=True,
                    ),
                    TrainingResearchRow(
                        symbol="SBIN.NS",
                        target="rl",
                        selection_rank=2,
                        decision_action="SELL",
                        critique_verdict="candidate",
                        market_regime="TRENDING",
                    ),
                ),
            )

    assistant.tools = ResearchTools()
    text = assistant.handle_text("/research rl 5m 20d limit=8")
    assert "Fortuna training research plan" in text
    assert "Target: rl" in text
    assert "Nightly posture: recent=1/2 over 2 run(s); target=rl force_refresh=1" in text
    assert "Nightly target: rl force_refresh=1" in text
    assert "RELIANCE.NS" in text
    assert "SBIN.NS" in text


def test_assistant_health_formats_model_summary(tmp_path):
    assistant = build_eval_assistant(tmp_path)

    class HealthTools:
        def search_instruments(self, query):
            raise AssertionError("not used")

        def analyze_instrument(self, symbol, timeframe="5m", days=30):
            raise AssertionError("not used")

        def get_market_universe(self, **kwargs):
            raise AssertionError("not used")

        def get_model_health(self):
            return ModelHealthResponse(
                registry_enabled=True,
                promotion_required=True,
                rl=RlModelStatus(available=True, advisory_ready=True, run_id="rl_1"),
                ml=MlModelStatus(available=True, advisory_ready=True, run_id="ml_1"),
                regime=RegimeModelStatus(),
                agentic=AgenticStatusSummary(
                    learning_summary=LearningSummaryStatus(total_rows=12),
                    nightly_alignment=NightlyAlignmentStatus(
                        report_count=2,
                        enabled_reports=2,
                        aligned_reports=1,
                        latest_target_mix="both=1, rl=1",
                        latest_refreshed_target_mix="both=1",
                        recommended_action=(
                            "Refresh the full ML/RL training research plan and review "
                            "candidate-selection policy before the next promotion "
                            "or nightly cycle."
                        ),
                        recommended_refresh_target="all",
                        effective_refresh_target="all",
                        latest_workflow_research_alignment_summary="basket/research drifting 1/3",
                        latest_workflow_research_alignment_target_mix="ml=1",
                        latest_workflow_research_recommended_target="rl",
                        latest_workflow_discovery_alignment_summary="discovery drifting 1/3",
                        latest_workflow_discovery_overlap_symbols="RELIANCE.NS",
                        latest_workflow_discovery_warning=True,
                        recommended_force_refresh=True,
                        workflow_target_mismatch=True,
                        recommended_cli_command=(
                            "uv run python scripts/build_training_research_plan.py "
                            "--source auto --selection-policy diversified "
                            "--timeframe 5m --days 30 --refresh-data --refresh-target all "
                            "--force-refresh"
                        ),
                        recommended_discovery_action=(
                            "Rebuild the market universe from Screener liquidity and recent "
                            "activity inputs before the next promotion or nightly cycle. "
                            "Latest overlap: RELIANCE.NS."
                        ),
                        recommended_discovery_cli_command=(
                            "uv run python scripts/build_market_universe.py "
                            "--source auto --timeframe 5m --days 30 --limit 15"
                        ),
                        recent_rows=(
                            NightlyAlignmentRow(
                                path="reports/nightly/20260603_0100.json",
                                overall_status="ok",
                                basket_size=2,
                                alignment_enabled=True,
                                target_mix="both=1, rl=1",
                            ),
                        ),
                    ),
                ),
            )

    assistant.tools = HealthTools()
    text = assistant.handle_text("/health")
    assert "Fortuna model health" in text
    assert "rl_1" in text
    assert "12" in text
    assert "refreshed=both=1" in text
    assert "Recent nightly alignment:" in text
    assert "ratio=0.50" in text
    assert "Suggested next step:" in text
    assert "Suggested refresh target: all" in text
    assert "Nightly target hint: all" in text
    assert "Latest workflow basket/research: basket/research drifting 1/3 | mix=ml=1" in text
    assert "Latest workflow target hint: rl" in text
    assert "Latest workflow discovery: discovery drifting 1/3 | overlap=RELIANCE.NS" in text
    assert "Latest workflow discovery needs follow-up." in text
    assert "Target mismatch: nightly=all current_workflow=rl" in text
    assert "Suggested mode: force refresh" in text
    assert "Suggested command:" in text
    assert "Suggested discovery follow-up:" in text
    assert "Suggested discovery command:" in text


def test_assistant_brief_formats_shortlist_summary(tmp_path):
    assistant = build_eval_assistant(tmp_path)

    class BriefTools:
        def search_instruments(self, query):
            raise AssertionError("not used")

        def analyze_instrument(self, symbol, timeframe="5m", days=30):
            raise AssertionError("not used")

        def get_market_universe(self, **kwargs):
            raise AssertionError("not used")

        def get_model_health(self):
            raise AssertionError("not used")

        def get_shortlist_briefing(self, **kwargs):
            return ShortlistBriefingResponse(
                ok=True,
                source="auto",
                timeframe="5m",
                lookback_days=20,
                headline="2 candidate setups, 3 reviewed",
                items=(
                    BriefingItem(
                        symbol="RELIANCE.NS",
                        action="BUY",
                        confidence=0.82,
                        verdict="candidate",
                        summary="Trend and liquidity aligned",
                        rationale=("Momentum intact", "Liquidity strong"),
                    ),
                ),
                portfolio=PortfolioCritique(
                    summary="Portfolio looks balanced enough for shortlist review",
                    notes=(
                        PortfolioExposureNote(
                            level="info",
                            message="Existing open positions: 2",
                        ),
                    ),
                ),
            )

    assistant.tools = BriefTools()
    text = assistant.handle_text("/brief 5m 20d limit=5")
    assert "Fortuna shortlist briefing" in text
    assert "RELIANCE.NS" in text
    assert "Portfolio:" in text


def test_assistant_candidates_formats_training_selection(tmp_path):
    assistant = build_eval_assistant(tmp_path)

    class CandidateTools:
        def search_instruments(self, query):
            raise AssertionError("not used")

        def analyze_instrument(self, symbol, timeframe="5m", days=30):
            raise AssertionError("not used")

        def get_market_universe(self, **kwargs):
            raise AssertionError("not used")

        def get_shortlist_briefing(self, **kwargs):
            raise AssertionError("not used")

        def get_model_health(self):
            raise AssertionError("not used")

        def get_training_candidates(self, **kwargs):
            return TrainingCandidateResponse(
                ok=True,
                source="auto",
                timeframe="15m",
                lookback_days=30,
                candidates=(
                    TrainingCandidate(
                        symbol="SBIN.NS",
                        shortlist_rank=1,
                        decision_action="BUY",
                        decision_confidence=0.78,
                        critique_verdict="candidate",
                        ml_candidate=True,
                        rl_candidate=True,
                        rationale=("Liquidity strong", "Reward/risk acceptable"),
                    ),
                    TrainingCandidate(
                        symbol="ITC.NS",
                        shortlist_rank=2,
                        decision_action="DO_NOT_ENTER",
                        decision_confidence=0.61,
                        critique_verdict="watch",
                        ml_candidate=True,
                        rl_candidate=False,
                    ),
                ),
            )

    assistant.tools = CandidateTools()
    text = assistant.handle_text("/candidates rl 15m 30d limit=8")
    assert "Fortuna training candidates" in text
    assert "Target: rl" in text
    assert "SBIN.NS" in text
    assert "ITC.NS" not in text


def test_assistant_allocate_formats_portfolio_selection(tmp_path):
    assistant = build_eval_assistant(tmp_path)

    class AllocationTools:
        def search_instruments(self, query):
            raise AssertionError("not used")

        def analyze_instrument(self, symbol, timeframe="5m", days=30):
            raise AssertionError("not used")

        def get_market_universe(self, **kwargs):
            raise AssertionError("not used")

        def get_shortlist_briefing(self, **kwargs):
            raise AssertionError("not used")

        def get_model_health(self):
            raise AssertionError("not used")

        def get_training_candidates(self, **kwargs):
            raise AssertionError("not used")

        def get_portfolio_allocation(self, **kwargs):
            return PortfolioAllocationResponse(
                ok=True,
                source="auto",
                timeframe="5m",
                lookback_days=20,
                headline="1 selected, 1 skipped under portfolio limits",
                max_positions=1,
                items=(
                    AllocationDecision(
                        symbol="RELIANCE.NS",
                        action="BUY",
                        verdict="candidate",
                        selected=True,
                        reason="selected under current portfolio constraints",
                        selection_rank=1,
                        allocation_rank=1,
                        allocation_weight=1.0,
                        exposure_key="RELIANCE",
                    ),
                    AllocationDecision(
                        symbol="RELIANCE.FUT",
                        action="BUY",
                        verdict="candidate",
                        selected=False,
                        reason="exposure limit reached for RELIANCE",
                        selection_rank=2,
                        exposure_key="RELIANCE",
                    ),
                ),
                notes=("max_positions=1",),
            )

    assistant.tools = AllocationTools()
    text = assistant.handle_text("/allocate 5m 20d limit=5 maxpos=1 perexp=1 sameside=1")
    assert "Fortuna portfolio allocation" in text
    assert "RELIANCE.NS" in text
    assert "Skipped:" in text


def test_assistant_workflow_formats_stage_summary(tmp_path):
    assistant = build_eval_assistant(tmp_path)

    class WorkflowTools:
        def search_instruments(self, query):
            raise AssertionError("not used")

        def analyze_instrument(self, symbol, timeframe="5m", days=30):
            raise AssertionError("not used")

        def get_model_health(self):
            raise AssertionError("not used")

        def get_multi_agent_workflow(self, **kwargs):
            universe = MarketUniverseResponse(
                ok=True,
                source="auto",
                timeframe="5m",
                lookback_days=20,
                candidates=(
                    MarketUniverseCandidate(
                        symbol="RELIANCE.NS",
                        display_name="RELIANCE",
                        source="auto",
                        liquidity_score=12.5,
                        trend_pct=2.2,
                    ),
                ),
            )
            shortlist = ShortlistAnalysisResponse(
                ok=True,
                source="auto",
                timeframe="5m",
                lookback_days=20,
                items=(
                    ShortlistAnalysisItem(
                        symbol="RELIANCE.NS",
                        selection_rank=1,
                        priority_score=1.2,
                        exposure_penalty=0.0,
                        decision=DecisionSummary(action="BUY", confidence=0.82, summary="BUY won"),
                        critique=RiskCritique(severity="low", summary="clean", verdict="candidate"),
                    ),
                ),
            )
            briefing = ShortlistBriefingResponse(
                ok=True,
                source="auto",
                timeframe="5m",
                lookback_days=20,
                headline="1 candidate setups, 1 reviewed",
                items=(
                    BriefingItem(
                        symbol="RELIANCE.NS",
                        action="BUY",
                        confidence=0.82,
                        verdict="candidate",
                        summary="Trend aligned",
                        selection_rank=1,
                    ),
                ),
                portfolio=PortfolioCritique(summary="Balanced", notes=()),
            )
            allocation = PortfolioAllocationResponse(
                ok=True,
                source="auto",
                timeframe="5m",
                lookback_days=20,
                headline="1 selected, 0 skipped under portfolio limits",
                max_positions=1,
                selected_symbols=("RELIANCE.NS",),
                items=(
                    AllocationDecision(
                        symbol="RELIANCE.NS",
                        action="BUY",
                        verdict="candidate",
                        selected=True,
                        reason="selected under current portfolio constraints",
                        selection_rank=1,
                        allocation_rank=1,
                        allocation_weight=1.0,
                        exposure_key="RELIANCE",
                    ),
                ),
                notes=("max_positions=1",),
            )
            return MultiAgentWorkflowResponse(
                ok=True,
                source="auto",
                timeframe="5m",
                lookback_days=20,
                headline=(
                    "Multi-agent flow prepared 1 universe names, 1 analyzed setups, "
                    "1 selected basket names, and 0 ML/RL research rows"
                ),
                roles=(
                    AgentRoleStatus(
                        name="universe_scout",
                        ok=True,
                        headline="1 liquid/trend candidates from auto",
                        focus_symbols=("RELIANCE.NS",),
                        notes=(
                            "discovery aligned 1/1",
                            "discovery_overlap=RELIANCE.NS",
                        ),
                    ),
                    AgentRoleStatus(
                        name="briefing_agent",
                        ok=True,
                        headline="1 candidate setups ready for operator briefing",
                        focus_symbols=("RELIANCE.NS",),
                    ),
                    AgentRoleStatus(
                        name="portfolio_allocator",
                        ok=True,
                        headline="1 selected, 0 skipped under portfolio limits",
                        focus_symbols=("RELIANCE.NS",),
                    ),
                    AgentRoleStatus(
                        name="research_planner",
                        ok=True,
                        headline="1 ML/RL research rows prepared",
                        focus_symbols=("RELIANCE.NS",),
                        notes=(
                            "basket/research aligned 1/1",
                            "basket_research_mix=both=1",
                        ),
                    ),
                ),
                universe=universe,
                shortlist=shortlist,
                briefing=briefing,
                allocation=allocation,
                research=None,
                nightly_alignment=NightlyAlignmentStatus(
                    report_count=2,
                    enabled_reports=2,
                    aligned_reports=1,
                    latest_target_mix="both=1, rl=1",
                    recommended_action="Refresh the full ML/RL training research plan.",
                    recommended_refresh_target="all",
                    recommended_force_refresh=True,
                    recommended_discovery_action=(
                        "Rebuild the market universe from Screener liquidity inputs."
                    ),
                    recommended_discovery_cli_command=(
                        "uv run python scripts/build_market_universe.py --source screener"
                    ),
                    recent_rows=(
                        NightlyAlignmentRow(
                            path="reports/nightly/nightly_20260603_220000.json",
                            overall_status="ok",
                            basket_size=1,
                            alignment_enabled=True,
                            target_mix="both=1, rl=1",
                            refreshed_target_mix="both=1",
                        ),
                    ),
                ),
            )

    assistant.tools = WorkflowTools()
    text = assistant.handle_text("/workflow 5m 20d limit=8")
    assert "Fortuna multi-agent workflow" in text
    assert "universe_scout" in text
    assert "briefing_agent" in text
    assert "portfolio_allocator" in text
    assert "Coverage: universe=1 shortlist=1 selected=1 research=0" in text
    assert "Discovery posture: discovery aligned 1/1 | overlap=RELIANCE.NS" in text
    assert "Basket/research posture: basket/research aligned 1/1 | mix=both=1" in text
    assert "Basket/research suggested target: all" in text
    assert "Nightly alignment: reports=2 enabled=2 aligned=1 ratio=0.50" in text
    assert "Suggested next step: Refresh the full ML/RL training research plan." in text
    assert "Suggested refresh target: all" in text
    assert (
        "Suggested discovery follow-up: Rebuild the market universe from Screener liquidity inputs."
    ) in text
    assert (
        "Suggested discovery command: uv run python scripts/build_market_universe.py "
        "--source screener"
    ) in text
    assert "Suggested mode: force refresh" in text
    assert "Recent trend: 1/1 aligned over last 1 run(s); latest=ok basket=1" in text


def test_assistant_workflow_surfaces_target_mismatch(tmp_path):
    assistant = build_eval_assistant(tmp_path)

    class WorkflowTools:
        def search_instruments(self, query):
            raise AssertionError("not used")

        def analyze_instrument(self, symbol, timeframe="5m", days=30):
            raise AssertionError("not used")

        def get_model_health(self):
            raise AssertionError("not used")

        def get_multi_agent_workflow(self, **kwargs):
            return MultiAgentWorkflowResponse(
                ok=True,
                source="auto",
                timeframe="5m",
                lookback_days=20,
                headline="Multi-agent flow prepared 1 universe names and 1 ML/RL research rows",
                roles=(
                    AgentRoleStatus(
                        name="research_planner",
                        ok=True,
                        headline="1 ML/RL research rows prepared",
                        focus_symbols=("RELIANCE.NS",),
                        notes=(
                            "basket/research drifting 0/1",
                            "basket_research_mix=ml=1",
                        ),
                    ),
                ),
                nightly_alignment=NightlyAlignmentStatus(
                    report_count=2,
                    enabled_reports=2,
                    aligned_reports=1,
                    recommended_action="Refresh the full ML/RL training research plan.",
                    recommended_refresh_target="all",
                    recommended_discovery_action=(
                        "Rebuild the market universe from Screener liquidity inputs."
                    ),
                ),
            )

    assistant.tools = WorkflowTools()
    text = assistant.handle_text("/workflow 5m 20d limit=8")
    assert "Basket/research suggested target: rl" in text
    assert "Target mismatch: nightly=all current_basket=rl" in text
    assert (
        "Suggested discovery follow-up: Rebuild the market universe from Screener liquidity inputs."
    ) in text


def test_assistant_image_interaction_disabled_returns_clear_reply(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    attachment_path = tmp_path / "chart.png"
    attachment_path.write_bytes(b"fake-image")
    result = assistant.handle_interaction(
        "",
        attachment=MediaAttachment(
            kind="image",
            file_name="chart.png",
            mime_type="image/png",
            local_path=str(attachment_path),
            content_hash="abc123",
            size_bytes=10,
        ),
    )
    assert result.error_code == "image_disabled"
    assert "Image analysis is disabled" in result.reply


def test_assistant_image_interaction_uses_chart_analyzer(tmp_path):
    class FakeChartAnalyzer:
        def available(self):
            return True

        def analyze(self, *, image_bytes: bytes, caption: str = ""):
            assert image_bytes == b"chart-bytes"
            assert "Crompton Futures" in caption
            return ChartImageAnalysisResult(
                ok=True,
                intent="analyze",
                symbol="RELIANCE.FUT",
                timeframe="15m",
                confidence=0.91,
                summary="Front-month futures chart matched.",
            )

    settings = build_eval_assistant(tmp_path).settings.model_copy(
        update={"signal_image_input_enabled": True}
    )
    engine = FakeEvalEngine()
    assistant = build_eval_assistant(tmp_path, engine=engine)
    assistant.settings = settings
    assistant.chart_analyzer = FakeChartAnalyzer()
    attachment_path = tmp_path / "chart.png"
    attachment_path.write_bytes(b"chart-bytes")

    result = assistant.handle_interaction(
        "Crompton Futures",
        request_id="req-image",
        attachment=MediaAttachment(
            kind="image",
            file_name="chart.png",
            mime_type="image/png",
            local_path=str(attachment_path),
            content_hash="img123",
            size_bytes=11,
        ),
    )

    assert result.route.action == RouteAction.ANALYZE
    assert result.tool == "analyze_instrument"
    assert result.modality == "image_with_caption"
    assert result.attachment_kind == "image"
    assert result.signal_response is not None
    assert result.signal_response.request_modality == "image_with_caption"
    assert "RELIANCE.FUT" in result.reply


def test_assistant_cache_does_not_mix_admin_and_default_replies(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    default_result = assistant.handle_interaction("/analyze RELIANCE", is_admin=False)
    admin_result = assistant.handle_interaction("/analyze RELIANCE", is_admin=True)

    assert "Layers:" not in default_result.reply
    assert "Layers:" in admin_result.reply


def test_assistant_risk_command_surfaces_lot_risk(tmp_path):
    assistant = build_eval_assistant(tmp_path)

    class RiskTools:
        def analyze_instrument(self, symbol, timeframe="5m", days=30):
            return InstrumentAnalysisResponse(
                ok=True,
                request=InstrumentAnalysisRequest(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                ),
                instrument=InstrumentSnapshot(
                    requested_symbol="RELIANCE FUT",
                    resolved_symbol="RELIANCE.FUT",
                    segment_label="FUTURE front",
                    timeframe=timeframe,
                    lookback_days=days,
                ),
                decision=DecisionSummary(
                    action="BUY",
                    confidence=0.73,
                    summary="BUY bias from futures breakout.",
                    reasons=("Trend alignment holds.",),
                    risk_notes=("risk_reward=1.80",),
                    action_plan="Buy only while the setup holds.",
                ),
                structured_signal=StructuredSignalDetails(
                    verdict="BUY",
                    setup_type="breakout_continuation",
                    trend_context="bullish_trend",
                    invalidation="Invalidate if price loses 100.00.",
                    entry_price=104.0,
                    stop_loss=100.0,
                    target_price=111.2,
                    risk_reward=1.8,
                    confidence_components=(
                        ConfidenceComponent("trend_alignment", 0.2, "aligned"),
                    ),
                ),
                replay_risk=ReplayRiskSummary(
                    ok=True,
                    matched_setups=6,
                    stop_hit_rate=0.33,
                    target_hit_rate=0.5,
                    summary="Replay matched 6 similar setups; stops hit 33%, targets hit 50%.",
                ),
                futures_lot_risk=FuturesLotRiskAssessment(
                    available=True,
                    lot_size=250,
                    contract_price=104.0,
                    contract_notional=26000.0,
                    stop_distance_per_unit=4.0,
                    stop_loss_amount_per_lot=1000.0,
                    cost_adjusted_loss_per_lot=1045.0,
                    volatility_bucket="normal",
                    entry_risk_bucket="acceptable",
                    summary=(
                        "One lot carries roughly 1,045 INR risk to the stop; "
                        "current volatility is normal."
                    ),
                ),
            )

    assistant.tools = RiskTools()
    result = assistant.handle_interaction("/risk RELIANCE FUT 5m 20d")
    assert result.route.action == RouteAction.RISK
    assert "Lot risk:" in result.reply
    assert "Stop: 100.00 | Target: 111.20" in result.reply
    assert result.signal_response is not None
    assert "One lot carries roughly" in result.signal_response.lot_risk_summary


def test_assistant_freeform_risk_query_routes_to_risk(tmp_path):
    assistant = build_eval_assistant(tmp_path)
    result = assistant.handle_interaction("Can I take one lot of Reliance futures?")
    assert result.source == "conversational"
    assert result.route.action == RouteAction.RISK
    assert result.request.symbol == "RELIANCE.FUT"
