"""Build typed advisory contracts from Fortuna runtime state."""

from __future__ import annotations

from typing import Any, Callable, Optional

from fortuna.agentic.contracts import (
    AdvisoryError,
    AdvisoryErrorCode,
    DecisionSummary,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSnapshot,
    SignalSummary,
)
from fortuna.agentic.models import AgentDecision
from fortuna.app.session_engine import FortunaSessionEngine
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRef, InstrumentRegistry


def segment_label(ref: InstrumentRef) -> str:
    if ref.is_option:
        expiry = ref.expiry.strftime("%d-%b-%Y") if ref.expiry else "NA"
        strike = f"{ref.strike:g}" if ref.strike is not None else "NA"
        return f"OPTION {ref.option_type or ''} {strike} {expiry}".strip()
    if ref.is_future:
        expiry = ref.expiry.strftime("%d-%b-%Y") if ref.expiry else "front"
        return f"FUTURE {expiry}"
    return "EQUITY"


def analysis_symbol(
    resolved: InstrumentRef,
    requested: str,
    registry: InstrumentRegistry,
) -> str:
    if resolved.is_future:
        if resolved.expiry is None:
            return resolved.symbol
        base = resolved.name or resolved.symbol.replace(".FUT", "")
        front = registry.resolve(f"{base}.FUT")
        if front.tradingsymbol == resolved.tradingsymbol:
            return f"{base}.FUT"
        return requested.upper().strip()
    if resolved.is_option:
        return requested.upper().strip()
    return f"{resolved.symbol}.NS"


def analyze_instrument(
    *,
    request: InstrumentAnalysisRequest,
    settings: Settings,
    engine_factory: Optional[Callable[[], Any]] = None,
    registry: Optional[InstrumentRegistry] = None,
) -> InstrumentAnalysisResponse:
    if not request.symbol.strip():
        return InstrumentAnalysisResponse(
            ok=False,
            request=request,
            error=AdvisoryError(
                code=AdvisoryErrorCode.INVALID_REQUEST,
                message="Usage: /analyze RELIANCE or /analyze NIFTY CE 25000 28MAY2026",
            ),
        )

    reg = registry or InstrumentRegistry()
    reg.ensure_loaded()

    try:
        resolved = reg.resolve(request.symbol)
    except KeyError as exc:
        return InstrumentAnalysisResponse(
            ok=False,
            request=request,
            error=AdvisoryError(
                code=AdvisoryErrorCode.RESOLVE_FAILED,
                message=f"Could not resolve instrument: {exc}",
            ),
        )

    symbol = analysis_symbol(resolved, request.symbol, reg)
    engine = engine_factory() if engine_factory is not None else FortunaSessionEngine(settings)
    try:
        state = engine.load_symbol(
            symbol,
            timeframe=request.timeframe,
            days=request.days,
            force_refresh=request.force_refresh,
        )
    except Exception as exc:  # noqa: BLE001
        return InstrumentAnalysisResponse(
            ok=False,
            request=request,
            error=AdvisoryError(
                code=AdvisoryErrorCode.INTERNAL,
                message=f"Analysis failed for {symbol}: {exc}",
            ),
        )

    if getattr(state, "load_error", None):
        return InstrumentAnalysisResponse(
            ok=False,
            request=request,
            error=AdvisoryError(
                code=AdvisoryErrorCode.LOAD_FAILED,
                message=f"Analysis failed for {symbol}: {state.load_error}",
            ),
        )

    last_bar_time = getattr(state, "last_bar_time", None)
    last_close = None
    if state.ohlcv is not None and not state.ohlcv.empty:
        last_close = float(state.ohlcv.iloc[-1]["close"])
    if last_bar_time is not None and hasattr(last_bar_time, "to_pydatetime"):
        last_bar_time = last_bar_time.to_pydatetime()

    instrument = InstrumentSnapshot(
        requested_symbol=request.symbol,
        resolved_symbol=symbol,
        segment_label=segment_label(resolved),
        timeframe=request.timeframe,
        lookback_days=request.days,
        last_bar_time=last_bar_time,
        last_close=last_close,
    )

    decisions = engine.agent_decisions()
    live_signals = engine.live_signals()
    decision_obj = decisions.get(symbol) or decisions.get(request.symbol)
    decision = None
    if isinstance(decision_obj, AgentDecision):
        decision = DecisionSummary.from_agent_decision(decision_obj)

    signals = _actionable_signals(live_signals)
    winning_strategy = None
    batch = getattr(state, "batch", None)
    if batch is not None and getattr(batch, "winner", None) is not None:
        winning_strategy = batch.winner.strategy_name

    return InstrumentAnalysisResponse(
        ok=True,
        request=request,
        instrument=instrument,
        decision=decision,
        signals=signals,
        winning_strategy=winning_strategy,
    )


def _actionable_signals(live_signals: dict[str, Any]) -> tuple[SignalSummary, ...]:
    out: list[SignalSummary] = []
    for name, sig in live_signals.items():
        action = getattr(sig, "action", "HOLD")
        if action not in {"HOLD", "IN_LONG", "IN_SHORT"}:
            out.append(SignalSummary(name=name, action=str(action)))
    return tuple(out[:5])
