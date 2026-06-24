"""Typed short-horizon forecast lane for conversational signal refinement."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

import pandas as pd

from fortuna.agentic.contracts import ForecastTaskRequest, ForecastTaskResult, SignalResponse
from fortuna.app.forecast_sandbox import (
    build_generated_code,
    cleanup_code_temp_dirs,
    execute_generated_code,
    supported_forecast_tasks,
)
from fortuna.app.session_engine import FortunaSessionEngine
from fortuna.config.settings import Settings

_FORECAST_HINTS = ("forecast", "precise", "precision", "probability", "projection", "project")


def should_run_forecast(*, raw_text: str, response: Any, settings: Settings) -> bool:
    if not bool(getattr(settings, "signal_forecast_enabled", True)):
        return False
    text = str(raw_text or "").lower()
    if any(token in text for token in _FORECAST_HINTS):
        return True
    decision = getattr(response, "decision", None)
    threshold = float(
        getattr(settings, "signal_forecast_trigger_confidence_threshold", 0.62) or 0.62
    )
    if decision is None:
        return True
    try:
        confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence < threshold


def forecast_tasks_for_request(*, raw_text: str, max_tasks: int) -> tuple[str, ...]:
    tasks = ["trend_continuation", "support_resistance"]
    lowered = str(raw_text or "").lower()
    if "vol" in lowered or "forecast" in lowered or "probability" in lowered:
        tasks.append("realized_volatility")
    if "atr" in lowered or "range" in lowered:
        tasks.append("atr_posture")
    allowed = [task for task in tasks if task in supported_forecast_tasks()]
    return tuple(allowed[: max(1, int(max_tasks))])


def build_forecast_requests(
    *,
    ohlcv: pd.DataFrame,
    signal_response: SignalResponse,
    tasks: tuple[str, ...],
    settings: Settings,
) -> tuple[ForecastTaskRequest, ...]:
    if ohlcv is None or ohlcv.empty:
        return ()
    max_bars = max(20, int(getattr(settings, "signal_forecast_max_bars", 240) or 240))
    trimmed = ohlcv.tail(max_bars)
    latest_bar_iso = ""
    if len(trimmed.index) > 0:
        latest = trimmed.index[-1]
        latest_bar_iso = latest.isoformat() if hasattr(latest, "isoformat") else str(latest)
    closes = tuple(float(value) for value in trimmed["close"].tolist())
    highs = tuple(float(value) for value in trimmed["high"].tolist())
    lows = tuple(float(value) for value in trimmed["low"].tolist())
    return tuple(
        ForecastTaskRequest(
            task_name=task,
            symbol=signal_response.resolved_symbol,
            timeframe=signal_response.timeframe,
            lookback_days=signal_response.lookback_days,
            latest_bar_iso=latest_bar_iso,
            closes=closes,
            highs=highs,
            lows=lows,
        )
        for task in tasks
    )


def run_forecast_lane(
    *,
    raw_text: str,
    response: Any,
    signal_response: SignalResponse,
    settings: Settings,
    engine_factory: Optional[Callable[[], Any]] = None,
) -> tuple[ForecastTaskResult, ...]:
    if not should_run_forecast(raw_text=raw_text, response=response, settings=settings):
        return ()
    engine = engine_factory() if engine_factory is not None else FortunaSessionEngine(settings)
    state = engine.load_symbol(
        signal_response.resolved_symbol,
        timeframe=signal_response.timeframe,
        days=signal_response.lookback_days,
        force_refresh=False,
    )
    ohlcv = getattr(state, "ohlcv", None)
    tasks = forecast_tasks_for_request(
        raw_text=raw_text,
        max_tasks=int(getattr(settings, "signal_forecast_max_tasks_per_request", 2) or 2),
    )
    requests = build_forecast_requests(
        ohlcv=ohlcv,
        signal_response=signal_response,
        tasks=tasks,
        settings=settings,
    )
    if not requests:
        return ()
    results: list[ForecastTaskResult] = []
    allow_live = bool(getattr(settings, "signal_code_execution_allow_live_generation", True))
    fallback_only = bool(getattr(settings, "signal_code_execution_fallback_only", False))
    for request in requests:
        if (
            allow_live
            and not fallback_only
            and bool(getattr(settings, "signal_code_execution_enabled", True))
        ):
            code = build_generated_code(request.task_name)
            sandbox_result, task_result = execute_generated_code(
                request,
                code=code,
                settings=settings,
            )
            if sandbox_result.ok and task_result is not None:
                results.append(task_result)
                continue
        results.append(_builtin_forecast(request))
    return tuple(result for result in results if _forecast_is_useful(result))


def merge_forecasts_into_signal(
    signal_response: SignalResponse,
    forecasts: tuple[ForecastTaskResult, ...],
) -> SignalResponse:
    if not forecasts:
        return signal_response
    forecast_lines = tuple(result.summary for result in forecasts if result.summary)
    reasons = tuple(signal_response.reasons or ())
    enhanced_reasons = reasons + forecast_lines[:2]
    agent_statuses = tuple(signal_response.agent_statuses or ()) + tuple(
        f"forecast:{result.task_name}:{result.mode}:{result.reliability:.2f}"
        for result in forecasts
    )
    return replace(
        signal_response,
        forecast_used=True,
        forecast_summaries=forecast_lines,
        reasons=enhanced_reasons[:5],
        agent_statuses=agent_statuses[:8],
    )


def cleanup_forecast_artifacts(settings: Settings) -> int:
    temp_dir = settings.resolve_path(settings.signal_code_execution_temp_dir)
    retention = int(getattr(settings, "signal_code_execution_retention_minutes", 10) or 10)
    return cleanup_code_temp_dirs(temp_dir, retention_minutes=retention)


def _builtin_forecast(request: ForecastTaskRequest) -> ForecastTaskResult:
    closes = list(request.closes)
    if request.task_name == "trend_continuation":
        if len(closes) < 3:
            return ForecastTaskResult(
                task_name=request.task_name,
                ok=True,
                summary="Trend continuation unavailable on short history.",
                warnings=("insufficient_history",),
                reliability=0.0,
                mode="builtin",
            )
        deltas = [b - a for a, b in zip(closes[:-1], closes[1:])]
        positive = sum(1 for delta in deltas if delta > 0)
        negative = sum(1 for delta in deltas if delta < 0)
        direction = "upside" if deltas[-1] >= 0 else "downside"
        bias = positive / len(deltas) if direction == "upside" else negative / len(deltas)
        return ForecastTaskResult(
            task_name=request.task_name,
            ok=True,
            summary=f"Short-horizon {direction} follow-through bias is {bias:.0%}.",
            metrics={"continuation_score": round(bias, 6)},
            reliability=min(0.85, 0.35 + len(deltas) / 40.0),
            mode="builtin",
        )
    if request.task_name == "support_resistance":
        window = closes[-20:] if len(closes) >= 20 else closes
        if not window:
            return ForecastTaskResult(
                task_name=request.task_name,
                ok=True,
                summary="Support/resistance unavailable on short history.",
                warnings=("insufficient_history",),
                reliability=0.0,
                mode="builtin",
            )
        support = min(window)
        resistance = max(window)
        last_close = window[-1]
        nearer = (
            "support" if abs(last_close - support) <= abs(resistance - last_close) else "resistance"
        )
        return ForecastTaskResult(
            task_name=request.task_name,
            ok=True,
            summary=f"Price is trading closer to {nearer}.",
            metrics={"support": round(support, 4), "resistance": round(resistance, 4)},
            reliability=min(0.8, 0.4 + len(window) / 50.0),
            mode="builtin",
        )
    if request.task_name == "realized_volatility":
        if len(closes) < 5:
            return ForecastTaskResult(
                task_name=request.task_name,
                ok=True,
                summary="Volatility estimate unavailable on short history.",
                warnings=("insufficient_history",),
                reliability=0.0,
                mode="builtin",
            )
        returns = [
            (closes[idx] / closes[idx - 1]) - 1.0
            for idx in range(1, len(closes))
            if closes[idx - 1]
        ]
        mean = sum(returns) / len(returns)
        variance = sum((item - mean) ** 2 for item in returns) / max(1, len(returns) - 1)
        volatility = variance**0.5
        posture = "elevated" if volatility >= 0.018 else "contained"
        return ForecastTaskResult(
            task_name=request.task_name,
            ok=True,
            summary=f"Realized volatility is {posture}.",
            metrics={"volatility": round(volatility, 6)},
            reliability=min(0.82, 0.3 + len(returns) / 50.0),
            mode="builtin",
        )
    highs = list(request.highs)
    lows = list(request.lows)
    if request.task_name == "atr_posture" and len(closes) >= 3 and highs and lows:
        true_ranges = []
        for idx in range(1, min(len(closes), len(highs), len(lows))):
            true_ranges.append(
                max(
                    highs[idx] - lows[idx],
                    abs(highs[idx] - closes[idx - 1]),
                    abs(lows[idx] - closes[idx - 1]),
                )
            )
        atr = sum(true_ranges[-14:]) / max(1, min(14, len(true_ranges)))
        ratio = atr / closes[-1] if closes[-1] else 0.0
        posture = "expanding" if ratio >= 0.018 else "compressed"
        return ForecastTaskResult(
            task_name=request.task_name,
            ok=True,
            summary=f"Range posture is {posture}.",
            metrics={"atr_ratio": round(ratio, 6)},
            reliability=min(0.8, 0.3 + len(true_ranges) / 50.0),
            mode="builtin",
        )
    return ForecastTaskResult(
        task_name=request.task_name,
        ok=False,
        summary=f"{request.task_name} unavailable.",
        warnings=("unsupported_task",),
        reliability=0.0,
        mode="builtin",
    )


def _forecast_is_useful(result: ForecastTaskResult) -> bool:
    if not result.ok:
        return False
    if float(result.reliability or 0.0) < 0.35:
        return False
    return True
