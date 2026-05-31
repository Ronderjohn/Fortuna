from __future__ import annotations

from fortuna.telegram.formatters import (
    format_analysis_error,
    format_clarification,
    format_help,
    format_unsupported,
)
from fortuna.telegram.router import ClarificationCode


def test_format_help_lists_core_commands():
    text = format_help()
    assert "/search RELIANCE" in text
    assert "/analyze NIFTY CE 25000 28MAY2026" in text
    assert "Timeframe and lookback" in text


def test_format_clarification_incomplete_option():
    text = format_clarification(ClarificationCode.INCOMPLETE_OPTION_SYNTAX)
    assert "28MAY2026" in text
    assert "CE" in text


def test_format_unsupported_lists_commands():
    text = format_unsupported()
    assert "couldn't understand" in text
    assert "/help" in text


def test_format_analysis_error_resolve_failed_suggests_search():
    from fortuna.agentic.contracts import (
        AdvisoryError,
        AdvisoryErrorCode,
        InstrumentAnalysisRequest,
        InstrumentAnalysisResponse,
    )

    response = InstrumentAnalysisResponse(
        ok=False,
        request=InstrumentAnalysisRequest(symbol="UNKNOWNXYZ"),
        error=AdvisoryError(
            code=AdvisoryErrorCode.RESOLVE_FAILED,
            message="Could not resolve instrument: 'UNKNOWNXYZ'",
        ),
    )
    text = format_analysis_error(response)
    assert "/search" in text
    assert "UNKNOWNXYZ" in text
