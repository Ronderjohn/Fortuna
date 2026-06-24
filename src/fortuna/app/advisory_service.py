"""Build typed advisory contracts from Fortuna runtime state."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional

from fortuna.agentic.contracts import (
    AdvisoryError,
    AdvisoryErrorCode,
    DecisionSummary,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSnapshot,
    OpenInterestEnrichment,
    SignalSummary,
)
from fortuna.agentic.models import AgentDecision
from fortuna.app.open_interest import fetch_open_interest_enrichment
from fortuna.app.session_engine import FortunaSessionEngine
from fortuna.app.signal_engine import analyze_signal_bundle
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


def _winning_strategy_name(batch: Any) -> Optional[str]:
    winner = getattr(batch, "winner", None)
    if winner is None:
        return None
    if isinstance(winner, str):
        text = winner.strip()
        return text or None
    strategy_name = getattr(winner, "strategy_name", None)
    if isinstance(strategy_name, str):
        text = strategy_name.strip()
        return text or None
    if isinstance(winner, dict):
        for key in ("strategy_name", "strategy", "winner"):
            value = winner.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _expired_contract_error(
    *,
    request: InstrumentAnalysisRequest,
    resolved: InstrumentRef,
) -> Optional[AdvisoryError]:
    expiry = resolved.expiry
    if expiry is None or expiry >= date.today():
        return None
    expiry_text = expiry.strftime("%d-%b-%Y")
    if resolved.is_option:
        base = resolved.name or resolved.symbol.replace(".OPT", "")
        return AdvisoryError(
            code=AdvisoryErrorCode.INVALID_REQUEST,
            message=(
                f"Option contract {request.symbol} expired on {expiry_text}. "
                f"Use /search {base} to find an active contract, then analyze that one."
            ),
        )
    if resolved.is_future:
        base = resolved.name or resolved.symbol.replace(".FUT", "")
        return AdvisoryError(
            code=AdvisoryErrorCode.INVALID_REQUEST,
            message=(
                f"Futures contract {request.symbol} expired on {expiry_text}. "
                f"Use /analyze {base} FUT for the current front-month contract "
                f"or /search {base} for the full active chain."
            ),
        )
    return None


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
    expired_error = _expired_contract_error(request=request, resolved=resolved)
    if expired_error is not None:
        return InstrumentAnalysisResponse(
            ok=False,
            request=request,
            error=expired_error,
        )
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
    if batch is not None:
        winning_strategy = _winning_strategy_name(batch)

    structured_signal = None
    replay_risk = None
    futures_lot_risk = None
    oi_enrichment: OpenInterestEnrichment | None = None
    try:
        if state.ohlcv is not None and not state.ohlcv.empty:
            if resolved.is_future and bool(getattr(settings, "signal_futures_oi_enabled", True)):
                try:
                    oi_enrichment = fetch_open_interest_enrichment(
                        instrument=resolved,
                        timeframe=request.timeframe,
                        days=request.days,
                        settings=settings,
                        registry=reg,
                    )
                except Exception:  # noqa: BLE001
                    oi_enrichment = OpenInterestEnrichment(
                        available=False,
                        summary="OI enrichment unavailable on this request.",
                        warnings=("oi_internal_error",),
                    )
            bundle = analyze_signal_bundle(
                ohlcv=state.ohlcv,
                instrument=resolved,
                timeframe=request.timeframe,
                settings=settings,
                oi_enrichment=oi_enrichment,
            )
            structured_signal = bundle.signal
            replay_risk = bundle.replay
            futures_lot_risk = bundle.lot_risk
            avoid_tags = set(getattr(structured_signal, "reason_to_avoid", ()) or ())
            if structured_signal.verdict and "insufficient_history" not in avoid_tags:
                decision = DecisionSummary(
                    action=structured_signal.verdict,
                    confidence=_signal_confidence(structured_signal),
                    summary=_signal_summary(structured_signal, replay_risk=replay_risk),
                    reasons=tuple(structured_signal.reasons or ()),
                    risk_notes=_signal_risk_notes(
                        structured_signal,
                        replay_risk=replay_risk,
                        lot_risk=futures_lot_risk,
                    ),
                    action_plan=_signal_action_plan(structured_signal),
                )
    except Exception:  # noqa: BLE001
        structured_signal = None
        replay_risk = None
        futures_lot_risk = None

    return InstrumentAnalysisResponse(
        ok=True,
        request=request,
        instrument=instrument,
        decision=decision,
        signals=signals,
        winning_strategy=winning_strategy,
        structured_signal=structured_signal,
        replay_risk=replay_risk,
        futures_lot_risk=futures_lot_risk,
        oi_enrichment=oi_enrichment,
    )


def _actionable_signals(live_signals: dict[str, Any]) -> tuple[SignalSummary, ...]:
    out: list[SignalSummary] = []
    for name, sig in live_signals.items():
        action = getattr(sig, "action", "HOLD")
        if action not in {"HOLD", "IN_LONG", "IN_SHORT"}:
            out.append(SignalSummary(name=name, action=str(action)))
    return tuple(out[:5])


def _signal_confidence(signal: Any) -> float:
    components = getattr(signal, "confidence_components", ()) or ()
    score = sum(float(getattr(item, "score", 0.0) or 0.0) for item in components)
    if getattr(signal, "verdict", "") == "NO_TRADE":
        return max(0.38, min(0.7, score))
    return max(0.45, min(0.92, score))


def _signal_summary(signal: Any, *, replay_risk: Any | None) -> str:
    verdict = str(getattr(signal, "verdict", "") or "NO_TRADE")
    setup = str(getattr(signal, "setup_type", "") or "setup")
    trend = str(getattr(signal, "trend_context", "") or "mixed")
    if verdict == "NO_TRADE":
        return (
            f"No trade: {setup.replace('_', ' ')} lacks enough edge in "
            f"{trend.replace('_', ' ')} conditions."
        )
    replay_text = ""
    if replay_risk is not None and getattr(replay_risk, "ok", False):
        target_hit_rate = getattr(replay_risk, "target_hit_rate", None)
        if target_hit_rate is not None:
            replay_text = f" Replay target follow-through is {float(target_hit_rate):.0%}."
    return (
        f"{verdict} bias from {setup.replace('_', ' ')} within "
        f"{trend.replace('_', ' ')} conditions.{replay_text}"
    )


def _signal_risk_notes(
    signal: Any,
    *,
    replay_risk: Any | None,
    lot_risk: Any | None,
) -> tuple[str, ...]:
    notes = list(getattr(signal, "reason_to_avoid", ()) or ())
    risk_reward = getattr(signal, "risk_reward", None)
    if risk_reward is not None:
        notes.append(f"risk_reward={float(risk_reward):.2f}")
    if replay_risk is not None and getattr(replay_risk, "ok", False):
        stop_hit = getattr(replay_risk, "stop_hit_rate", None)
        if stop_hit is not None:
            notes.append(f"replay_stop_rate={float(stop_hit):.0%}")
    if lot_risk is not None and getattr(lot_risk, "available", False):
        notes.append(
            "one_lot_risk="
            f"{float(getattr(lot_risk, 'cost_adjusted_loss_per_lot', 0.0) or 0.0):,.0f} INR"
        )
    out: list[str] = []
    seen: set[str] = set()
    for note in notes:
        text = str(note or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out[:5])


def _signal_action_plan(signal: Any) -> str:
    verdict = str(getattr(signal, "verdict", "") or "NO_TRADE")
    stop = getattr(signal, "stop_loss", None)
    target = getattr(signal, "target_price", None)
    if verdict == "BUY" and stop is not None and target is not None:
        return (
            f"Buy only while the setup holds; stop {float(stop):.2f}, "
            f"initial target {float(target):.2f}."
        )
    if verdict == "SELL" and stop is not None and target is not None:
        return (
            f"Sell only while the setup holds; stop {float(stop):.2f}, "
            f"initial target {float(target):.2f}."
        )
    return "Wait for a cleaner setup with stronger reward-to-risk."
