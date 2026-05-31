"""Format typed advisory contracts for Telegram replies."""

from __future__ import annotations

import pandas as pd

from fortuna.agentic.contracts import (
    AdvisoryErrorCode,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSearchResponse,
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


def format_help() -> str:
    return "\n".join(
        [
            "Fortuna Telegram assistant",
            "",
            "Commands:",
            "/help",
            "/search RELIANCE",
            "/analyze RELIANCE",
            "/analyze RELIANCE FUT",
            "/analyze NIFTY CE 25000 28MAY2026",
            "",
            "Tag aliases: #search, #analyze, #help",
            "",
            "Timeframe and lookback (optional on /analyze):",
            "  1m 5m 15m 30m 1h 1d",
            "  20d or days=20 for lookback days",
            "  Example: /analyze ICICIBANK 15m 20d",
            "",
            "Futures:",
            "  /analyze RELIANCE FUT",
            "  /analyze NIFTY FUT 15m",
            "  /analyze RELIANCE.FUT.27NOV2025",
            "",
            "Options (explicit contract required):",
            "  /analyze NIFTY CE 25000 28MAY2026",
            "  /analyze NIFTY.OPT.CE.25000.28MAY2026",
            "",
            "Suggested loop:",
            "  1) /search RELIANCE",
            "  2) /analyze RELIANCE",
            "  3) refine with /analyze RELIANCE 15m 20d if needed",
        ]
    )


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


def format_unsupported() -> str:
    return "\n".join(
        [
            "I couldn't understand that.",
            "",
            "Supported commands:",
            "/help",
            "/search RELIANCE",
            "/analyze RELIANCE",
            "/analyze RELIANCE FUT",
            "/analyze NIFTY CE 25000 28MAY2026",
            "",
            "Send /help for full usage.",
        ]
    )


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


def format_instrument_analysis(response: InstrumentAnalysisResponse) -> str:
    if not response.ok:
        return format_analysis_error(response)

    instrument = response.instrument
    if instrument is None:
        return "Analysis failed."

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
