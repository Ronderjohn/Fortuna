"""Format typed advisory contracts for Telegram replies."""

from __future__ import annotations

import pandas as pd

from fortuna.agentic.contracts import (
    AdvisoryErrorCode,
    AgentRoleStatus,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSearchResponse,
    MarketUniverseResponse,
    ModelHealthResponse,
    MultiAgentWorkflowResponse,
    PortfolioAllocationResponse,
    ShortlistAnalysisResponse,
    ShortlistBriefingResponse,
    SignalResponse,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
)
from fortuna.app.operator_workflow import (
    summarize_market_universe,
    summarize_portfolio_allocation,
    summarize_shortlist_analysis,
    summarize_shortlist_briefing,
    summarize_training_candidates,
)
from fortuna.telegram.parser import TelegramRequest
from fortuna.telegram.router import ClarificationCode

_OPTION_SYNTAX_LINES = [
    "Option syntax requires all four parts:",
    "  underlying  CE|PE  strike  expiry",
    "",
    "Examples:",
    "/analyze NIFTY CE 25000 28MAY2026",
    "/analyze BANKNIFTY PE 52000 28MAY2026 5m 10d",
    "",
    "Canonical symbol form:",
    "  BASE.OPT.CE.<STRIKE>.<DDMMMYYYY>",
]


def format_help(*, advanced: bool = False, expert_commands_visible: bool = False) -> str:
    lines = [
        "Fortuna signal assistant",
        "",
        "Start here:",
        "/search RELIANCE",
        "/analyze RELIANCE",
        "/analyze RELIANCE 15m 20d",
        "/risk RELIANCE FUT",
        "/analyze RELIANCE FUT",
        "/analyze NIFTY CE 25000 28MAY2026",
        "",
        "You can also ask naturally:",
        "How is Reliance looking on 15m for 20d?",
        "How risky is Reliance Futures right now?",
        "Can I take one lot of Crompton Futures?",
        "Should I enter NIFTY CE 25000 28MAY2026?",
        "",
        "Timeframe and lookback:",
        "1m 5m 15m 30m 1h 1d",
        "20d or days=20",
        "",
        "Use /help advanced for expert workflow commands.",
    ]
    if advanced or expert_commands_visible:
        lines.extend(
            [
                "",
                "Expert commands:",
                "/universe 1d 30d limit=10",
                "/brief 5m 20d limit=5",
                "/allocate 5m 20d limit=5 maxpos=3 perexp=1 sameside=2",
                "/candidates rl 15m 30d limit=8",
                "/research all 5m 20d limit=8",
                "/workflow 5m 20d limit=8",
                "/health",
            ]
        )
    return "\n".join(lines)


def format_clarification(code: ClarificationCode | None) -> str:
    if code == ClarificationCode.EMPTY_SEARCH_QUERY:
        return "\n".join(
            [
                "Search needs a query after /search.",
                "",
                "Examples:",
                "/search RELIANCE",
                "/search NIFTY",
            ]
        )
    if code == ClarificationCode.EMPTY_ANALYZE_SYMBOL:
        return "\n".join(
            [
                "Analyze needs a symbol after /analyze.",
                "",
                "Examples:",
                "/analyze RELIANCE",
                "/risk RELIANCE FUT",
                "/analyze RELIANCE FUT",
                "/analyze NIFTY CE 25000 28MAY2026",
                "",
                "Timeframe-only input such as '/analyze 15m' is not enough.",
                "Add a symbol first, then optional timeframe/lookback.",
            ]
        )
    if code == ClarificationCode.INCOMPLETE_OPTION_SYNTAX:
        return "\n".join(
            [
                "That option request is incomplete.",
                "",
                *_OPTION_SYNTAX_LINES,
                "",
                "Tip: /search NIFTY can help you confirm contract formatting.",
            ]
        )
    if code == ClarificationCode.FREEFORM_INSTRUMENT_NEEDED:
        return "\n".join(
            [
                "I need a clearer instrument request before I can analyze it.",
                "",
                "Try one of these:",
                "/search RELIANCE",
                "/analyze RELIANCE",
                "/analyze RELIANCE FUT",
                "/analyze NIFTY CE 25000 28MAY2026",
            ]
        )
    return format_unsupported()


def format_unsupported(*, expert_commands_visible: bool = False) -> str:
    lines = [
        "I couldn't understand that.",
        "",
        "Try one of these:",
        "/search RELIANCE",
        "/analyze RELIANCE",
        "/risk RELIANCE FUT",
        "/analyze RELIANCE FUT",
        "/analyze NIFTY CE 25000 28MAY2026",
    ]
    if expert_commands_visible:
        lines.extend(["", "Advanced help: /help advanced"])
    else:
        lines.extend(["", "Send /help for quick usage."])
    return "\n".join(lines)


def format_instrument_search(response: InstrumentSearchResponse) -> str:
    if not response.ok:
        if response.error is not None:
            return response.error.message
        return "Search failed."

    if not response.hits:
        return "\n".join(
            [
                f"No instruments found for '{response.query}'.",
                "",
                "Try:",
                "- a shorter or alternate query (/search RELIANCE)",
                "- checking spelling",
                "- for options, use explicit analyze syntax:",
                "  /analyze NIFTY CE 25000 28MAY2026",
            ]
        )
    lines = [f"Search results for '{response.query}':"]
    for hit in response.hits:
        lines.append(f"- [{hit.segment}] {hit.symbol} :: {hit.display}")
    return "\n".join(lines)


def instrument_analysis_request_from_telegram(req: TelegramRequest) -> InstrumentAnalysisRequest:
    return InstrumentAnalysisRequest(
        symbol=req.symbol,
        timeframe=req.timeframe,
        days=req.days,
        force_refresh=False,
    )


def _looks_like_option_request(symbol: str) -> bool:
    upper_symbol = symbol.upper()
    if ".OPT." in upper_symbol:
        return True
    tokens = [tok for tok in symbol.replace(",", " ").split() if tok]
    if len(tokens) >= 2 and tokens[1].upper() in {"CE", "PE", "C", "P"}:
        return True
    return False


def format_analysis_error(response: InstrumentAnalysisResponse) -> str:
    error = response.error
    if error is None:
        return "Analysis failed."

    request = response.request
    if error.code == AdvisoryErrorCode.RESOLVE_FAILED:
        if _looks_like_option_request(request.symbol):
            return "\n".join(
                [
                    f"Could not resolve option contract: {request.symbol}",
                    "",
                    *_OPTION_SYNTAX_LINES,
                    "",
                    "Try /search NIFTY or verify strike and expiry.",
                ]
            )
        return "\n".join(
            [
                f"Could not resolve instrument: {request.symbol}",
                "",
                "Try searching first:",
                f"/search {request.symbol.split('.')[0]}",
                "",
                "Examples:",
                "/analyze RELIANCE",
                "/analyze RELIANCE FUT",
                "/analyze NIFTY CE 25000 28MAY2026",
            ]
        )

    if error.code == AdvisoryErrorCode.INVALID_REQUEST:
        return error.message

    if error.code == AdvisoryErrorCode.LOAD_FAILED:
        return "\n".join(
            [
                error.message,
                "",
                "Check that the instrument master is loaded and run:",
                "  uv run python scripts/run_operator_preflight.py",
            ]
        )

    return error.message


def format_instrument_analysis(
    response: InstrumentAnalysisResponse,
    *,
    signal_response: SignalResponse | None = None,
    reply_style: str = "compact",
    expert: bool = False,
) -> str:
    if not response.ok:
        return format_analysis_error(response)

    instrument = response.instrument
    if instrument is None:
        return "Analysis failed."

    if signal_response is not None and reply_style == "compact":
        lines = [
            f"{signal_response.resolved_symbol}",
            f"Verdict: {signal_response.signal_verdict}",
            f"Confidence: {signal_response.confidence * 100:.0f}%",
            f"Why: {signal_response.summary}",
        ]
        if signal_response.reasons:
            lines.append(f"Reasons: {'; '.join(signal_response.reasons[:3])}")
        if signal_response.setup_type and expert:
            lines.append(f"Setup: {signal_response.setup_type}")
        if signal_response.stop_loss is not None:
            stop_text = f"Stop: {signal_response.stop_loss:.2f}"
            if signal_response.target_price is not None:
                stop_text += f" | Target: {signal_response.target_price:.2f}"
            lines.append(stop_text)
        if signal_response.risk_notes:
            lines.append(f"Risk: {'; '.join(signal_response.risk_notes[:2])}")
        if signal_response.replay_summary and expert:
            lines.append(f"Replay: {signal_response.replay_summary}")
        if signal_response.lot_risk_summary:
            lines.append(f"Lot risk: {signal_response.lot_risk_summary}")
        if signal_response.action_plan:
            lines.append(f"Next: {signal_response.action_plan}")
        if signal_response.data_freshness:
            lines.append(signal_response.data_freshness)
        if signal_response.forecast_used and signal_response.forecast_summaries:
            lines.append(f"Forecast: {'; '.join(signal_response.forecast_summaries[:2])}")
        if signal_response.oi_summary and expert:
            lines.append(f"OI: {signal_response.oi_summary}")
        if signal_response.market_context and expert:
            lines.append(signal_response.market_context)
        if expert:
            if signal_response.agent_statuses:
                lines.append(f"Agents: {' | '.join(signal_response.agent_statuses[:4])}")
            if signal_response.expert_notes:
                lines.append(f"Notes: {' | '.join(signal_response.expert_notes[:3])}")
            lines.append(
                "Layers: "
                f"deterministic={signal_response.used_layers.deterministic}, "
                f"ml={signal_response.used_layers.ml}, "
                f"rl={signal_response.used_layers.rl}"
            )
        return "\n".join(lines)

    lines = [
        f"Fortuna analysis: {instrument.resolved_symbol}",
        f"Segment: {instrument.segment_label}",
        f"Timeframe: {instrument.timeframe} | Lookback: {instrument.lookback_days}d",
    ]
    if instrument.last_bar_time is not None and instrument.last_close is not None:
        lines.append(
            f"Last bar: {pd.Timestamp(instrument.last_bar_time)} "
            f"close={instrument.last_close:.2f}"
        )

    if response.decision is not None:
        decision = response.decision
        lines.extend(
            [
                f"Decision: {decision.action}",
                f"Confidence: {decision.confidence * 100:.0f}%",
                f"Summary: {decision.summary}",
            ]
        )
        if decision.reasons:
            lines.append(f"Why: {'; '.join(decision.reasons[:3])}")
        if decision.risk_notes:
            lines.append(f"Risk: {'; '.join(decision.risk_notes[:2])}")
        lines.append(f"Action plan: {decision.action_plan}")
    else:
        lines.append("Decision: unavailable; no agentic advisory result was produced.")

    if response.signals:
        signal_text = ", ".join(f"{sig.name}={sig.action}" for sig in response.signals)
        lines.append(f"Signals: {signal_text}")

    if response.winning_strategy:
        lines.append(f"Winning strategy: {response.winning_strategy}")

    lines.append(response.disclaimer)
    return "\n".join(lines)


def format_market_universe(response: MarketUniverseResponse) -> str:
    if not response.ok:
        if response.error is not None:
            return response.error.message
        return "Universe ranking failed."
    lines = [
        "Fortuna market universe",
        f"Source: {response.source}",
        f"Timeframe: {response.timeframe} | Lookback: {response.lookback_days}d",
    ]
    nightly_summary = str(getattr(response, "nightly_alignment_summary", "") or "").strip()
    if nightly_summary:
        lines.append(f"Nightly posture: {nightly_summary}")
    nightly_target = str(getattr(response, "nightly_alignment_target", "") or "").strip()
    if nightly_target:
        nightly_force_refresh = 1 if bool(
            getattr(response, "nightly_alignment_force_refresh", False)
        ) else 0
        lines.append(
            "Nightly target: "
            f"{nightly_target} "
            f"force_refresh={nightly_force_refresh}"
        )
    if not response.candidates:
        lines.append("No ranked candidates available.")
        return "\n".join(lines)
    for idx, row in enumerate(response.candidates, start=1):
        score = f"{row.liquidity_score:.2f}" if row.liquidity_score is not None else "NA"
        trend = f"{row.trend_pct:+.2f}%" if row.trend_pct is not None else "NA"
        regime = str(getattr(row, "regime", "") or "").upper()
        adaptive_note = next(
            (note for note in row.notes if str(note).startswith("adaptive_")),
            "",
        )
        lines.append(
            f"{idx}. {row.symbol} | score={score} | trend={trend}"
            + (f" | regime={regime}" if regime else "")
            + (f" | {adaptive_note}" if adaptive_note else "")
        )
    lines.append(response.disclaimer)
    return "\n".join(lines)


def format_training_research_plan(
    response: TrainingResearchPlanResponse,
    *,
    target: str = "all",
) -> str:
    if not response.ok:
        if response.error is not None:
            return response.error.message
        return "Training research plan failed."
    lines = [
        "Fortuna training research plan",
        f"Target: {target}",
        f"Source: {response.source}",
        f"Timeframe: {response.timeframe} | Lookback: {response.lookback_days}d",
        f"Selection policy: {response.selection_policy}",
    ]
    nightly_summary = str(getattr(response, "nightly_alignment_summary", "") or "").strip()
    if nightly_summary:
        lines.append(f"Nightly posture: {nightly_summary}")
    nightly_target = str(getattr(response, "nightly_alignment_target", "") or "").strip()
    if nightly_target:
        lines.append(
            "Nightly target: "
            f"{nightly_target} "
            "force_refresh="
            f"{1 if bool(getattr(response, 'nightly_alignment_force_refresh', False)) else 0}"
        )
    if not response.rows:
        lines.append("No research rows available.")
        lines.append(response.disclaimer)
        return "\n".join(lines)
    for idx, row in enumerate(response.rows, start=1):
        action = str(getattr(row, "decision_action", "") or "NA")
        verdict = str(getattr(row, "critique_verdict", "") or "NA")
        regime = str(getattr(row, "market_regime", "") or "").upper()
        nightly_note = " refreshed" if bool(getattr(row, "refreshed", False)) else ""
        lines.append(
            f"{idx}. {row.symbol} | target={row.target} | action={action} | verdict={verdict}"
            + (f" | regime={regime}" if regime else "")
            + nightly_note
        )
    lines.append(response.disclaimer)
    return "\n".join(lines)


def format_model_health(response: ModelHealthResponse) -> str:
    lines = [
        "Fortuna model health",
        f"Registry enabled: {response.registry_enabled}",
        f"Promotion required: {response.promotion_required}",
        (
            f"RL: available={response.rl.available} advisory_ready={response.rl.advisory_ready} "
            f"run_id={response.rl.run_id or 'NA'}"
        ),
        (
            f"ML: available={response.ml.available} advisory_ready="
            f"{response.ml.advisory_ready if response.ml.advisory_ready is not None else 'NA'} "
            f"run_id={response.ml.run_id or 'NA'}"
        ),
        f"Recent learning rows: {response.agentic.learning_summary.total_rows}",
    ]
    nightly = response.agentic.nightly_alignment
    if nightly.report_count:
        ratio = (
            float(nightly.aligned_reports) / float(nightly.enabled_reports)
            if nightly.enabled_reports > 0
            else 0.0
        )
        lines.append(
            "Recent nightly alignment: "
            f"reports={nightly.report_count} enabled={nightly.enabled_reports} "
            f"aligned={nightly.aligned_reports} ratio={ratio:.2f}"
            + (
                f" latest={nightly.latest_target_mix}"
                if nightly.latest_target_mix
                else ""
            )
            + (
                f" refreshed={nightly.latest_refreshed_target_mix}"
                if nightly.latest_refreshed_target_mix
                else ""
            )
        )
        if nightly.recommended_action:
            lines.append(f"Suggested next step: {nightly.recommended_action}")
        if nightly.effective_refresh_target:
            lines.append(f"Suggested refresh target: {nightly.effective_refresh_target}")
        if nightly.recommended_refresh_target:
            lines.append(f"Nightly target hint: {nightly.recommended_refresh_target}")
        if nightly.latest_workflow_research_alignment_summary:
            line = (
                "Latest workflow basket/research: "
                f"{nightly.latest_workflow_research_alignment_summary}"
            )
            if nightly.latest_workflow_research_alignment_target_mix:
                line += f" | mix={nightly.latest_workflow_research_alignment_target_mix}"
            lines.append(line)
        if nightly.latest_workflow_research_recommended_target:
            lines.append(
                "Latest workflow target hint: "
                f"{nightly.latest_workflow_research_recommended_target}"
            )
        if nightly.latest_workflow_discovery_alignment_summary:
            line = (
                "Latest workflow discovery: "
                f"{nightly.latest_workflow_discovery_alignment_summary}"
            )
            if nightly.latest_workflow_discovery_overlap_symbols:
                line += f" | overlap={nightly.latest_workflow_discovery_overlap_symbols}"
            lines.append(line)
        if nightly.latest_workflow_discovery_warning:
            lines.append("Latest workflow discovery needs follow-up.")
        if nightly.workflow_target_mismatch:
            lines.append(
                "Target mismatch: "
                f"nightly={nightly.recommended_refresh_target} "
                f"current_workflow={nightly.latest_workflow_research_recommended_target}"
            )
        if nightly.recommended_force_refresh:
            lines.append("Suggested mode: force refresh")
        if nightly.recommended_cli_command:
            lines.append(f"Suggested command: {nightly.recommended_cli_command}")
        if nightly.recommended_discovery_action:
            lines.append(f"Suggested discovery follow-up: {nightly.recommended_discovery_action}")
        if nightly.recommended_discovery_cli_command:
            lines.append(
                f"Suggested discovery command: {nightly.recommended_discovery_cli_command}"
            )
    return "\n".join(lines)


def format_shortlist_briefing(response: ShortlistBriefingResponse) -> str:
    if not response.ok:
        if response.error is not None:
            return response.error.message
        return "Shortlist briefing failed."

    lines = [
        "Fortuna shortlist briefing",
        response.headline,
        f"Source: {response.source}",
        f"Timeframe: {response.timeframe} | Lookback: {response.lookback_days}d",
    ]
    if not response.items:
        lines.append("No shortlisted setups available.")
    else:
        for idx, item in enumerate(response.items, start=1):
            confidence = f"{item.confidence * 100:.0f}%"
            rank_text = (
                f"sel={int(getattr(item, 'selection_rank', 0) or idx)}"
                if getattr(item, "selection_rank", 0) or idx
                else ""
            )
            penalty = (
                f"penalty={item.exposure_penalty:.2f}"
                if getattr(item, "exposure_penalty", 0.0)
                else ""
            )
            lines.append(
                f"{idx}. {item.symbol} | {item.action} | {item.verdict} | conf={confidence}"
                + (f" | {rank_text}" if rank_text else "")
                + (f" | {penalty}" if penalty else "")
            )
            lines.append(f"   {item.summary}")
            if item.rationale:
                lines.append(f"   Why: {'; '.join(item.rationale[:2])}")
    if response.portfolio is not None:
        lines.append(f"Portfolio: {response.portfolio.summary}")
        for note in response.portfolio.notes[:3]:
            suffix = f" [{', '.join(note.symbols)}]" if note.symbols else ""
            lines.append(f"- {note.level.upper()}: {note.message}{suffix}")
    lines.append(response.disclaimer)
    return "\n".join(lines)


def format_portfolio_allocation(response: PortfolioAllocationResponse) -> str:
    if not response.ok:
        if response.error is not None:
            return response.error.message
        return "Portfolio allocation failed."

    lines = [
        "Fortuna portfolio allocation",
        response.headline,
        f"Source: {response.source}",
        f"Timeframe: {response.timeframe} | Lookback: {response.lookback_days}d",
        f"Max positions: {response.max_positions}",
    ]
    selected = [row for row in response.items if row.selected]
    skipped = [row for row in response.items if not row.selected]
    if not selected:
        lines.append("No setups were selected under the current portfolio limits.")
    else:
        lines.append("Selected:")
        for row in selected:
            weight = (
                f"{float(row.allocation_weight or 0.0) * 100:.1f}%"
                if row.allocation_weight is not None
                else "NA"
            )
            penalty = f" | penalty={row.exposure_penalty:.2f}" if row.exposure_penalty else ""
            critic = (
                f" | critic={row.critic_penalty:.2f}"
                if getattr(row, "critic_penalty", 0.0)
                else ""
            )
            research = (
                f" | research={row.research_target}"
                if getattr(row, "research_target", "")
                else ""
            )
            regime = f" | regime={row.regime}" if getattr(row, "regime", "") else ""
            risk = f" | risk={row.risk_bucket}" if getattr(row, "risk_bucket", "") else ""
            sizing = f" | size={row.sizing_hint}" if getattr(row, "sizing_hint", "") else ""
            lines.append(
                f"{row.allocation_rank}. {row.symbol} | {row.action} | wt={weight}"
                f" | exp={row.exposure_key}{research}{regime}{risk}{sizing}{penalty}{critic}"
            )
            lines.append(f"   {row.reason}")
    if skipped:
        lines.append("Skipped:")
        for row in skipped[:4]:
            lines.append(f"- {row.symbol} | {row.reason}")
    for note in response.notes[:3]:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def format_training_candidates(response: TrainingCandidateResponse, *, target: str = "all") -> str:
    if not response.ok:
        if response.error is not None:
            return response.error.message
        return "Training candidate scan failed."

    target_key = (target or "all").strip().lower()
    lines = [
        "Fortuna training candidates",
        f"Target: {target_key}",
        f"Source: {response.source}",
        f"Timeframe: {response.timeframe} | Lookback: {response.lookback_days}d",
    ]
    candidates = list(response.candidates)
    if target_key == "ml":
        candidates = [row for row in candidates if row.ml_candidate]
    elif target_key == "rl":
        candidates = [row for row in candidates if row.rl_candidate]
    if not candidates:
        lines.append("No matching training candidates available.")
        lines.append(response.disclaimer)
        return "\n".join(lines)
    for idx, row in enumerate(candidates, start=1):
        conf = (
            f"{row.decision_confidence * 100:.0f}%"
            if row.decision_confidence is not None
            else "NA"
        )
        flags: list[str] = []
        if row.ml_candidate:
            flags.append("ml")
        if row.rl_candidate:
            flags.append("rl")
        flag_text = "/".join(flags) if flags else "watch"
        regime = str(getattr(row, "market_regime", "") or "").upper()
        strategy = str(getattr(row, "winning_strategy", "") or "")
        penalty = (
            f"penalty={row.exposure_penalty:.2f}"
            if getattr(row, "exposure_penalty", 0.0)
            else ""
        )
        adaptive = (
            f"bias={float(row.adaptive_score_adjustment or 0.0):+.2f}"
            if getattr(row, "adaptive_score_adjustment", None) is not None
            else ""
        )
        rank_text = (
            f"sel={int(getattr(row, 'selection_rank', 0) or idx)}"
            if getattr(row, "selection_rank", 0) or idx
            else ""
        )
        lines.append(
            f"{idx}. {row.symbol} | {row.decision_action or 'NA'} | {row.critique_verdict} | "
            f"conf={conf} | {flag_text}"
            + (f" | strat={strategy}" if strategy else "")
            + (f" | regime={regime}" if regime else "")
            + (f" | {adaptive}" if adaptive else "")
            + (f" | {rank_text}" if rank_text else "")
            + (f" | {penalty}" if penalty else "")
        )
        if row.rationale:
            lines.append(f"   Why: {'; '.join(row.rationale[:2])}")
    lines.append(response.disclaimer)
    return "\n".join(lines)


def format_workflow_summary(
    *,
    universe: MarketUniverseResponse,
    shortlist: ShortlistAnalysisResponse,
    briefing: ShortlistBriefingResponse,
    candidates: TrainingCandidateResponse,
    allocation: PortfolioAllocationResponse,
) -> str:
    universe_summary = summarize_market_universe(universe)
    shortlist_summary = summarize_shortlist_analysis(shortlist)
    briefing_summary = summarize_shortlist_briefing(briefing)
    candidate_summary = summarize_training_candidates(candidates)
    allocation_summary = summarize_portfolio_allocation(allocation)

    for summary in (
        universe_summary,
        shortlist_summary,
        briefing_summary,
        candidate_summary,
        allocation_summary,
    ):
        if summary.error:
            return summary.error

    lines = [
        "Fortuna workflow summary",
        (
            f"Universe: {universe_summary.metrics.get('count', 0)} names "
            f"(source={universe_summary.metrics.get('source', 'auto')})"
        ),
        (
            f"Shortlist: {shortlist_summary.metrics.get('count', 0)} ranked "
            f"setups"
        ),
        (
            f"Briefing: {briefing_summary.metrics.get('candidates', 0)} candidate "
            f"setups"
        ),
        (
            f"Training: ML={candidate_summary.metrics.get('ml_count', 0)} "
            f"RL={candidate_summary.metrics.get('rl_count', 0)}"
        ),
        (
            f"Allocation: selected={allocation_summary.metrics.get('selected_count', 0)} "
            f"skipped={allocation_summary.metrics.get('skipped_count', 0)}"
        ),
    ]
    if universe_summary.rows:
        top = universe_summary.rows[0]
        lines.append(
            f"Top liquid: {top.get('symbol')} score={top.get('liquidity_score')} "
            f"trend={top.get('trend_pct')}"
        )
        if top.get("adaptive_note"):
            lines.append(f"Adaptive: {top.get('adaptive_note')}")
    if shortlist_summary.rows:
        top = shortlist_summary.rows[0]
        lines.append(
            f"Top shortlist: {top.get('symbol')} {top.get('action')} "
            f"rank={top.get('selection_rank')} "
            f"penalty={top.get('exposure_penalty') or 0.0}"
        )
    if briefing_summary.rows:
        top = briefing_summary.rows[0]
        lines.append(
            f"Top setup: {top.get('symbol')} {top.get('action')} "
            f"{top.get('verdict')} conf={top.get('confidence_pct')}%"
        )
    if candidate_summary.rows:
        top = candidate_summary.rows[0]
        lines.append(
            f"Top candidate: {top.get('symbol')} rank={top.get('selection_rank')} "
            f"ml={top.get('ml_candidate')} rl={top.get('rl_candidate')}"
        )
    selected_rows = [row for row in allocation_summary.rows if row.get("selected")]
    if selected_rows:
        top = selected_rows[0]
        lines.append(
            f"Top allocation: {top.get('symbol')} wt={top.get('allocation_weight')} "
            f"rank={top.get('allocation_rank')}"
        )
    for note in briefing_summary.notes[:2]:
        lines.append(f"Risk: {note}")
    for note in universe_summary.notes[:1]:
        lines.append(f"Universe note: {note}")
    lines.append("Use /brief, /allocate, or /candidates for deeper detail.")
    return "\n".join(lines)


def format_multi_agent_workflow(response: MultiAgentWorkflowResponse) -> str:
    if not response.ok:
        if response.error is not None:
            return response.error.message
        return "Multi-agent workflow unavailable."

    lines = [
        "Fortuna multi-agent workflow",
        response.headline,
    ]
    for role in response.roles:
        lines.append(_format_agent_role(role))
        if role.notes:
            lines.append(f"   Notes: {'; '.join(role.notes[:2])}")

    universe_count = len(getattr(getattr(response, "universe", None), "candidates", ()) or ())
    shortlist_count = len(getattr(getattr(response, "shortlist", None), "items", ()) or ())
    selected_count = len(
        getattr(getattr(response, "allocation", None), "selected_symbols", ()) or ()
    )
    research_count = len(getattr(getattr(response, "research", None), "rows", ()) or ())
    lines.append(
        f"Coverage: universe={universe_count} shortlist={shortlist_count} "
        f"selected={selected_count} research={research_count}"
    )
    nightly = getattr(response, "nightly_alignment", None)
    research_role = next(
        (role for role in response.roles if str(getattr(role, "name", "")) == "research_planner"),
        None,
    )
    if research_role is not None:
        alignment_summary = next(
            (
                str(note)
                for note in getattr(research_role, "notes", ())
                if str(note).startswith("basket/research ")
            ),
            "",
        )
        alignment_mix = next(
            (
                str(note).split("=", 1)[1].strip()
                for note in getattr(research_role, "notes", ())
                if str(note).startswith("basket_research_mix=")
            ),
            "",
        )
        if alignment_summary:
            line = f"Basket/research posture: {alignment_summary}"
            if alignment_mix:
                line += f" | mix={alignment_mix}"
            lines.append(line)
            inferred_target = _target_from_alignment_mix(alignment_mix)
            if inferred_target:
                lines.append(f"Basket/research suggested target: {inferred_target}")
                nightly_target = str(
                    getattr(nightly, "recommended_refresh_target", "") or ""
                ).strip()
                if nightly_target and nightly_target != inferred_target:
                    lines.append(
                        "Target mismatch: "
                        f"nightly={nightly_target} current_basket={inferred_target}"
                    )
    universe_role = next(
        (role for role in response.roles if str(getattr(role, "name", "")) == "universe_scout"),
        None,
    )
    if universe_role is not None:
        discovery_summary = next(
            (
                str(note)
                for note in getattr(universe_role, "notes", ())
                if str(note).startswith("discovery ")
            ),
            "",
        )
        discovery_overlap = next(
            (
                str(note).split("=", 1)[1].strip()
                for note in getattr(universe_role, "notes", ())
                if str(note).startswith("discovery_overlap=")
            ),
            "",
        )
        if discovery_summary:
            line = f"Discovery posture: {discovery_summary}"
            if discovery_overlap:
                line += f" | overlap={discovery_overlap}"
            lines.append(line)
    if nightly is not None and int(getattr(nightly, "report_count", 0) or 0) > 0:
        enabled_reports = int(getattr(nightly, "enabled_reports", 0) or 0)
        aligned_reports = int(getattr(nightly, "aligned_reports", 0) or 0)
        ratio = (
            float(aligned_reports) / float(enabled_reports)
            if enabled_reports > 0
            else 0.0
        )
        lines.append(
            "Nightly alignment: "
            f"reports={int(getattr(nightly, 'report_count', 0) or 0)} "
            f"enabled={enabled_reports} aligned={aligned_reports} ratio={ratio:.2f}"
        )
        recommended_action = str(getattr(nightly, "recommended_action", "") or "").strip()
        if recommended_action:
            lines.append(f"Suggested next step: {recommended_action}")
        recommended_target = str(
            getattr(nightly, "recommended_refresh_target", "") or ""
        ).strip()
        if recommended_target:
            lines.append(f"Suggested refresh target: {recommended_target}")
        recommended_discovery_action = str(
            getattr(nightly, "recommended_discovery_action", "") or ""
        ).strip()
        if recommended_discovery_action:
            lines.append(f"Suggested discovery follow-up: {recommended_discovery_action}")
        if bool(getattr(nightly, "recommended_force_refresh", False)):
            lines.append("Suggested mode: force refresh")
        recommended_discovery_command = str(
            getattr(nightly, "recommended_discovery_cli_command", "") or ""
        ).strip()
        if recommended_discovery_command:
            lines.append(f"Suggested discovery command: {recommended_discovery_command}")
        recent_window = int(getattr(nightly, "recent_trend_window", 0) or 0)
        recent_enabled = int(getattr(nightly, "recent_trend_enabled", 0) or 0)
        recent_aligned = int(getattr(nightly, "recent_trend_aligned", 0) or 0)
        latest_status = str(getattr(nightly, "recent_trend_latest_status", "") or "").strip()
        latest_basket = int(getattr(nightly, "recent_trend_latest_basket_size", 0) or 0)
        if recent_window <= 0:
            recent_rows = tuple(getattr(nightly, "recent_rows", ()) or ())
            recent_window_rows = recent_rows[:3]
            if recent_window_rows:
                recent_window = len(recent_window_rows)
                recent_enabled = sum(
                    1 for row in recent_window_rows if getattr(row, "alignment_enabled", False)
                )
                recent_aligned = sum(
                    1
                    for row in recent_window_rows
                    if getattr(row, "alignment_enabled", False)
                    and str(getattr(row, "target_mix", "") or "").strip()
                )
                latest_row = recent_window_rows[0]
                latest_status = str(
                    getattr(latest_row, "overall_status", "") or ""
                ).strip()
                latest_basket = int(getattr(latest_row, "basket_size", 0) or 0)
        if recent_window > 0:
            lines.append(
                "Recent trend: "
                f"{recent_aligned}/{recent_enabled} aligned over last {recent_window} "
                f"run(s); latest={latest_status or '—'} basket={latest_basket}"
            )
    lines.append("Use /brief, /allocate, or /candidates for deeper detail.")
    lines.append(response.disclaimer)
    return "\n".join(lines)


def _format_agent_role(role: AgentRoleStatus) -> str:
    status = "ok" if role.ok else "unavailable"
    focus = ", ".join(role.focus_symbols[:3]) if role.focus_symbols else "—"
    return f"- {role.name}: {status} | {role.headline} | focus={focus}"


def _target_from_alignment_mix(mix: str) -> str:
    text = str(mix or "").strip().lower()
    if not text:
        return ""
    keys = set()
    for part in text.split(","):
        chunk = part.strip()
        if not chunk or "=" not in chunk:
            continue
        key = chunk.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    if not keys:
        return ""
    if "none" in keys or "both" in keys:
        return "all"
    if "ml" in keys and "rl" not in keys:
        return "rl"
    if "rl" in keys and "ml" not in keys:
        return "ml"
    if "ml" in keys and "rl" in keys:
        return "all"
    return ""
