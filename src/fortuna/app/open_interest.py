"""Request-scoped futures open-interest enrichment."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from fortuna.agentic.contracts import OpenInterestEnrichment
from fortuna.config.settings import Settings
from fortuna.config.smartapi_settings import get_smartapi_settings
from fortuna.data.instruments import InstrumentRef, InstrumentRegistry
from fortuna.data.sources.smartapi_historical import SmartAPIHistoricalSource


def fetch_open_interest_enrichment(
    *,
    instrument: InstrumentRef,
    timeframe: str,
    days: int,
    settings: Settings,
    registry: Optional[InstrumentRegistry] = None,
) -> OpenInterestEnrichment:
    del settings
    if not instrument.is_future:
        return OpenInterestEnrichment(
            available=False,
            summary="OI enrichment is only supported for futures in this advisory path.",
            warnings=("not_a_future",),
        )
    if not get_smartapi_settings().configured:
        return OpenInterestEnrichment(
            available=False,
            summary="SmartAPI credentials are not configured for OI enrichment.",
            warnings=("oi_not_configured",),
        )
    try:
        source = SmartAPIHistoricalSource(registry=registry)
        symbol = _instrument_symbol(instrument)
        df = source.fetch(symbol, timeframe, days=days)
    except Exception:
        return OpenInterestEnrichment(
            available=False,
            summary="OI enrichment unavailable from SmartAPI on this request.",
            warnings=("oi_fetch_failed",),
        )
    if "open_interest" not in df.columns or df["open_interest"].dropna().empty:
        return OpenInterestEnrichment(
            available=False,
            summary="SmartAPI did not return usable open-interest bars for this contract.",
            warnings=("oi_missing",),
        )
    series = pd.to_numeric(df["open_interest"], errors="coerce").dropna()
    if len(series) < 3:
        return OpenInterestEnrichment(
            available=False,
            summary="Open-interest history is too short for a stable read.",
            warnings=("oi_insufficient_history",),
        )
    latest = float(series.iloc[-1])
    prev = float(series.iloc[-2])
    base = float(series.iloc[max(0, len(series) - min(10, len(series)))])
    change = latest - prev
    change_pct = (change / prev) if prev else None
    trend_change = latest - base
    price_change = float(df["close"].iloc[-1] - df["close"].iloc[-min(5, len(df))])
    posture = _classify_posture(price_change=price_change, oi_change=trend_change)
    confirms_direction: Optional[bool] = None
    if posture in {"long_build_up", "short_covering"}:
        confirms_direction = True
    elif posture in {"short_build_up", "long_unwinding"}:
        confirms_direction = False
    return OpenInterestEnrichment(
        available=True,
        latest_open_interest=round(latest, 2),
        change=round(change, 2),
        change_pct=round(change_pct, 6) if change_pct is not None else None,
        posture=posture,
        confirms_direction=confirms_direction,
        summary=_summary(posture, change_pct),
    )


def _instrument_symbol(instrument: InstrumentRef) -> str:
    if instrument.expiry is None:
        return instrument.symbol
    base = instrument.name or instrument.symbol.replace(".FUT", "")
    return f"{base}.FUT.{instrument.expiry.strftime('%d%b%Y').upper()}"


def _classify_posture(*, price_change: float, oi_change: float) -> str:
    if price_change >= 0 and oi_change >= 0:
        return "long_build_up"
    if price_change < 0 and oi_change >= 0:
        return "short_build_up"
    if price_change >= 0 and oi_change < 0:
        return "short_covering"
    return "long_unwinding"


def _summary(posture: str, change_pct: Optional[float]) -> str:
    pct_text = f"{change_pct * 100:.1f}%" if change_pct is not None else "n/a"
    label = posture.replace("_", " ")
    return f"OI posture looks like {label} ({pct_text} latest change)."
