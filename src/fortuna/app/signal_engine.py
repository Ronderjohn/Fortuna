"""Structured signal engine, bounded replay risk, and futures lot-risk helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from fortuna.agentic.contracts import (
    ConfidenceComponent,
    FuturesLotRiskAssessment,
    OpenInterestEnrichment,
    ReplayRiskSummary,
    StructuredSignalDetails,
)
from fortuna.backtesting.standard.config import MarketCostModel
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRef
from fortuna.indicators.registry import (
    compute_atr,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_volume_sma,
    compute_vwap,
)


@dataclass(frozen=True)
class SignalEngineBundle:
    signal: StructuredSignalDetails
    replay: ReplayRiskSummary
    lot_risk: Optional[FuturesLotRiskAssessment]


_BUNDLE_CACHE: dict[tuple[str, ...], tuple[float, SignalEngineBundle]] = {}


def analyze_signal_bundle(
    *,
    ohlcv: pd.DataFrame,
    instrument: InstrumentRef,
    timeframe: str,
    settings: Settings,
    oi_enrichment: Optional[OpenInterestEnrichment] = None,
) -> SignalEngineBundle:
    cache_key = _bundle_cache_key(
        ohlcv=ohlcv,
        instrument=instrument,
        timeframe=timeframe,
    )
    cached = _bundle_cache_get(
        cache_key,
        ttl_seconds=max(1, int(getattr(settings, "signal_replay_cache_ttl_seconds", 60) or 60)),
    )
    if cached is not None:
        return cached
    signal = build_structured_signal(
        ohlcv=ohlcv,
        instrument=instrument,
        timeframe=timeframe,
        oi_enrichment=oi_enrichment,
    )
    replay = (
        evaluate_replay_risk(
            ohlcv=ohlcv,
            instrument=instrument,
            timeframe=timeframe,
            settings=settings,
            current_signal=signal,
            oi_enrichment=oi_enrichment,
        )
        if bool(getattr(settings, "signal_replay_enabled", True))
        else ReplayRiskSummary(
            ok=False,
            summary="Replay disabled by settings.",
            warnings=("disabled",),
        )
    )
    lot_risk = None
    if instrument.is_future:
        lot_risk = assess_futures_lot_risk(
            instrument=instrument,
            signal=signal,
            replay=replay,
            settings=settings,
        )
    bundle = SignalEngineBundle(signal=signal, replay=replay, lot_risk=lot_risk)
    _BUNDLE_CACHE[cache_key] = (time.time(), bundle)
    return bundle


def build_structured_signal(
    *,
    ohlcv: pd.DataFrame,
    instrument: InstrumentRef,
    timeframe: str,
    oi_enrichment: Optional[OpenInterestEnrichment] = None,
) -> StructuredSignalDetails:
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 30:
        return StructuredSignalDetails(
            verdict="NO_TRADE",
            trend_context="insufficient_history",
            confirmation="not enough bars to form a reliable setup",
            invalidation="wait for more data",
            reason_to_avoid=("insufficient_history",),
        )

    df = _prepare_frame(ohlcv)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["close"])
    atr = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    atr_ratio = atr / close if close else 0.0
    prior_high = float(last["prior_high"]) if pd.notna(last["prior_high"]) else close
    prior_low = float(last["prior_low"]) if pd.notna(last["prior_low"]) else close
    recent_support = float(last["recent_support"]) if pd.notna(last["recent_support"]) else close
    recent_resistance = (
        float(last["recent_resistance"]) if pd.notna(last["recent_resistance"]) else close
    )
    volume_ratio = float(last["volume_ratio"]) if pd.notna(last["volume_ratio"]) else 0.0
    ema9 = float(last["ema_9"]) if pd.notna(last["ema_9"]) else close
    ema21 = float(last["ema_21"]) if pd.notna(last["ema_21"]) else close
    ema50 = float(last["ema_50"]) if pd.notna(last["ema_50"]) else close
    ema9_slope = float(last["ema_9_slope"]) if pd.notna(last["ema_9_slope"]) else 0.0
    ema21_slope = float(last["ema_21_slope"]) if pd.notna(last["ema_21_slope"]) else 0.0
    vwap = float(last["vwap"]) if pd.notna(last["vwap"]) else close
    price_vs_vwap = (close / vwap - 1.0) if vwap else 0.0
    rsi = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
    macd_hist = float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else 0.0
    body_strength = float(last["body_strength"]) if pd.notna(last["body_strength"]) else 0.0
    lower_wick_ratio = (
        float(last["lower_wick_ratio"]) if pd.notna(last["lower_wick_ratio"]) else 0.0
    )
    upper_wick_ratio = (
        float(last["upper_wick_ratio"]) if pd.notna(last["upper_wick_ratio"]) else 0.0
    )

    bullish_trend = close > ema9 > ema21 > ema50 and ema9_slope > 0 and ema21_slope >= 0
    bearish_trend = close < ema9 < ema21 < ema50 and ema9_slope < 0 and ema21_slope <= 0
    trend_context = (
        "bullish_trend"
        if bullish_trend
        else "bearish_trend"
        if bearish_trend
        else "mixed"
    )

    breakout = close > prior_high and prev["close"] <= prev["prior_high"]
    breakdown = close < prior_low and prev["close"] >= prev["prior_low"]
    vwap_reclaim = close > vwap and prev["close"] <= prev["vwap"]
    vwap_reject = close < vwap and prev["close"] >= prev["vwap"]

    bull_score = 0.0
    bear_score = 0.0
    components: list[ConfidenceComponent] = []
    if bullish_trend:
        bull_score += 0.24
        components.append(ConfidenceComponent("trend_alignment", 0.24, "ema 9/21/50 aligned up"))
    elif bearish_trend:
        bear_score += 0.24
        components.append(ConfidenceComponent("trend_alignment", 0.24, "ema 9/21/50 aligned down"))
    if breakout:
        bull_score += 0.19
        components.append(
            ConfidenceComponent("structure_break", 0.19, "price broke prior range high")
        )
    if breakdown:
        bear_score += 0.19
        components.append(
            ConfidenceComponent("structure_break", 0.19, "price broke prior range low")
        )
    if price_vs_vwap > 0:
        bull_score += 0.12
        components.append(ConfidenceComponent("vwap_posture", 0.12, "price above VWAP"))
    elif price_vs_vwap < 0:
        bear_score += 0.12
        components.append(ConfidenceComponent("vwap_posture", 0.12, "price below VWAP"))
    if volume_ratio >= 1.1:
        if bullish_trend or breakout or vwap_reclaim:
            bull_score += 0.09
        if bearish_trend or breakdown or vwap_reject:
            bear_score += 0.09
        components.append(
            ConfidenceComponent("volume_confirmation", 0.09, "volume expansion present")
        )
    if rsi >= 57:
        bull_score += 0.08
        components.append(ConfidenceComponent("rsi", 0.08, "RSI supports upside"))
    elif rsi <= 43:
        bear_score += 0.08
        components.append(ConfidenceComponent("rsi", 0.08, "RSI supports downside"))
    if macd_hist > 0:
        bull_score += 0.06
    elif macd_hist < 0:
        bear_score += 0.06
    if body_strength >= 0.55:
        if close >= prev["close"]:
            bull_score += 0.05
        else:
            bear_score += 0.05
    if oi_enrichment is not None and oi_enrichment.available:
        if oi_enrichment.posture in {"long_build_up", "short_covering"}:
            bull_score += 0.08
        elif oi_enrichment.posture in {"short_build_up", "long_unwinding"}:
            bear_score += 0.08

    setup_type = "watch"
    direction = "none"
    reasons: list[str] = []
    avoid: list[str] = []
    confirmation = "Mixed confirmation."
    invalidation = "Wait for a clearer setup."
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward: Optional[float] = None
    market_regime = _volatility_bucket(atr_ratio)

    if bull_score >= bear_score and bull_score >= 0.43:
        direction = "BUY"
        setup_type = (
            "breakout_continuation"
            if breakout
            else "vwap_reclaim"
            if vwap_reclaim
            else "trend_pullback"
        )
        stop_anchor = max(recent_support, close - max(atr * 1.25, close * 0.0045))
        stop_loss = min(close - 0.01, stop_anchor)
        risk = max(close - stop_loss, max(atr * 0.35, close * 0.002))
        target_anchor = max(recent_resistance, close + risk * 1.7)
        target_price = target_anchor
        risk_reward = (target_price - close) / risk if risk > 0 else None
        reasons.extend(_unique_nonempty([
            "Bullish trend alignment holds across EMA 9/21/50.",
            "Price is trading above session VWAP." if price_vs_vwap > 0 else "",
            "Breakout above recent structure confirmed." if breakout else "",
            "Volume is expanding into the move." if volume_ratio >= 1.1 else "",
            oi_enrichment.summary if oi_enrichment and oi_enrichment.summary else "",
        ]))
        confirmation = "Trend, VWAP, and structure favor continuation higher."
        invalidation = f"Invalidate if price loses {stop_loss:.2f}."
        if upper_wick_ratio > 0.45:
            avoid.append("upper_wick_rejection")
    elif bear_score > bull_score and bear_score >= 0.43:
        direction = "SELL"
        setup_type = (
            "breakdown_continuation"
            if breakdown
            else "vwap_rejection"
            if vwap_reject
            else "trend_pullback"
        )
        stop_anchor = min(recent_resistance, close + max(atr * 1.25, close * 0.0045))
        stop_loss = max(close + 0.01, stop_anchor)
        risk = max(stop_loss - close, max(atr * 0.35, close * 0.002))
        target_anchor = min(recent_support, close - risk * 1.7)
        target_price = target_anchor
        risk_reward = (close - target_price) / risk if risk > 0 else None
        reasons.extend(_unique_nonempty([
            "Bearish trend alignment holds across EMA 9/21/50.",
            "Price is trading below session VWAP." if price_vs_vwap < 0 else "",
            "Breakdown below recent structure confirmed." if breakdown else "",
            "Volume is expanding into the move." if volume_ratio >= 1.1 else "",
            oi_enrichment.summary if oi_enrichment and oi_enrichment.summary else "",
        ]))
        confirmation = "Trend, VWAP, and structure favor continuation lower."
        invalidation = f"Invalidate if price reclaims {stop_loss:.2f}."
        if lower_wick_ratio > 0.45:
            avoid.append("lower_wick_absorption")
    else:
        direction = "NO_TRADE"
        reasons.extend(_unique_nonempty([
            "Trend alignment is mixed.",
            "No decisive breakout or breakdown is active.",
        ]))
        if volume_ratio < 0.9:
            avoid.append("thin_confirmation")

    if atr_ratio >= 0.028:
        avoid.append("volatility_extreme")
    elif atr_ratio >= 0.018:
        avoid.append("volatility_elevated")
    if volume_ratio < 0.8:
        avoid.append("volume_below_average")
    if risk_reward is not None and risk_reward < 1.35:
        avoid.append("risk_reward_too_thin")
    if abs(price_vs_vwap) > 0.028:
        avoid.append("stretched_from_vwap")

    if direction != "NO_TRADE" and avoid:
        if "risk_reward_too_thin" in avoid or "volatility_extreme" in avoid:
            direction = "NO_TRADE"
            reasons.append("Reward-to-risk is not strong enough for entry right now.")

    confidence = max(bull_score, bear_score)
    if direction == "NO_TRADE":
        confidence = max(0.35, min(0.72, confidence))
    else:
        confidence = max(0.45, min(0.92, confidence))

    feature_summary = {
        "close": round(close, 4),
        "atr": round(atr, 4),
        "atr_ratio": round(atr_ratio, 6),
        "volume_ratio": round(volume_ratio, 4),
        "price_vs_vwap": round(price_vs_vwap, 6),
        "rsi": round(rsi, 4),
        "macd_hist": round(macd_hist, 6),
        "ema_9_slope": round(ema9_slope, 6),
        "ema_21_slope": round(ema21_slope, 6),
    }
    return StructuredSignalDetails(
        verdict=direction,
        setup_type=setup_type,
        trend_context=trend_context,
        confirmation=confirmation,
        invalidation=invalidation,
        entry_price=round(close, 4),
        stop_loss=round(stop_loss, 4) if stop_loss is not None else None,
        target_price=round(target_price, 4) if target_price is not None else None,
        risk_reward=round(risk_reward, 4) if risk_reward is not None else None,
        market_regime=market_regime,
        volatility_bucket=market_regime,
        reason_to_avoid=tuple(avoid),
        reasons=tuple(reasons[:5]),
        confidence_components=tuple(components[:8]),
        feature_summary=feature_summary,
        oi_summary=oi_enrichment.summary if oi_enrichment is not None else "",
    )


def evaluate_replay_risk(
    *,
    ohlcv: pd.DataFrame,
    instrument: InstrumentRef,
    timeframe: str,
    settings: Settings,
    current_signal: StructuredSignalDetails,
    oi_enrichment: Optional[OpenInterestEnrichment] = None,
) -> ReplayRiskSummary:
    del instrument, timeframe, oi_enrichment
    if (
        ohlcv is None
        or ohlcv.empty
        or current_signal.verdict not in {"BUY", "SELL"}
        or len(ohlcv) < 70
    ):
        return ReplayRiskSummary(
            ok=False,
            summary="Replay risk unavailable on the current setup.",
            warnings=("insufficient_setup_context",),
        )

    max_bars = max(80, int(getattr(settings, "signal_replay_max_bars", 180) or 180))
    lookahead_bars = max(4, int(getattr(settings, "signal_replay_hold_bars", 12) or 12))
    trimmed = ohlcv.tail(max_bars)
    candidate_metrics: list[dict[str, float]] = []
    start_index = max(35, len(trimmed) - max_bars + 35)
    for end_idx in range(start_index, len(trimmed) - lookahead_bars):
        hist = trimmed.iloc[: end_idx + 1]
        signal = build_structured_signal(
            ohlcv=hist,
            instrument=InstrumentRef(
                symbol="X",
                tradingsymbol="X",
                symboltoken="0",
                exchange="NSE",
            ),
            timeframe="5m",
            oi_enrichment=None,
        )
        if (
            signal.verdict != current_signal.verdict
            or signal.setup_type != current_signal.setup_type
        ):
            continue
        if signal.entry_price is None or signal.stop_loss is None or signal.target_price is None:
            continue
        metric = _simulate_trade_outcome(
            trimmed.iloc[end_idx + 1 : end_idx + 1 + lookahead_bars],
            signal=signal,
        )
        if metric is not None:
            candidate_metrics.append(metric)
        if len(candidate_metrics) >= 24:
            break

    if not candidate_metrics:
        return ReplayRiskSummary(
            ok=False,
            sample_size=max(0, len(trimmed) - start_index),
            matched_setups=0,
            summary="Replay found too few similar setups for a stable read.",
            warnings=("no_matching_setups",),
        )

    stop_hits = sum(1 for row in candidate_metrics if row["outcome"] == -1)
    target_hits = sum(1 for row in candidate_metrics if row["outcome"] == 1)
    matched = len(candidate_metrics)
    stop_hit_rate = stop_hits / matched
    target_hit_rate = target_hits / matched
    avg_hold = sum(row["hold_bars"] for row in candidate_metrics) / matched
    avg_mae = sum(row["mae_pct"] for row in candidate_metrics) / matched
    avg_mfe = sum(row["mfe_pct"] for row in candidate_metrics) / matched
    worst_mae = max(row["mae_pct"] for row in candidate_metrics)
    avg_rr = sum(row["rr"] for row in candidate_metrics) / matched
    volatility_posture = (
        "elevated" if avg_mae >= 0.018 else "normal" if avg_mae >= 0.009 else "contained"
    )
    summary = (
        f"Replay matched {matched} similar setups; stops hit {stop_hit_rate:.0%}, "
        f"targets hit {target_hit_rate:.0%}, average hold {avg_hold:.1f} bars."
    )
    warnings: list[str] = []
    if stop_hit_rate > target_hit_rate:
        warnings.append("stops_outnumber_targets")
    if worst_mae >= 0.03:
        warnings.append("adverse_excursion_high")
    return ReplayRiskSummary(
        ok=True,
        sample_size=len(trimmed),
        matched_setups=matched,
        stop_hit_rate=round(stop_hit_rate, 6),
        target_hit_rate=round(target_hit_rate, 6),
        avg_hold_bars=round(avg_hold, 4),
        avg_mae_pct=round(avg_mae, 6),
        avg_mfe_pct=round(avg_mfe, 6),
        worst_adverse_excursion_pct=round(worst_mae, 6),
        avg_realized_risk_reward=round(avg_rr, 6),
        volatility_posture=volatility_posture,
        summary=summary,
        warnings=tuple(warnings),
    )


def assess_futures_lot_risk(
    *,
    instrument: InstrumentRef,
    signal: StructuredSignalDetails,
    replay: ReplayRiskSummary,
    settings: Settings,
) -> FuturesLotRiskAssessment:
    lot_size = int(instrument.lot_size or 0)
    if lot_size <= 0 or signal.entry_price is None or signal.stop_loss is None:
        return FuturesLotRiskAssessment(
            available=False,
            summary="Lot-risk assessment unavailable for this contract.",
            warnings=("missing_lot_or_stop",),
        )
    price = float(signal.entry_price)
    stop_distance = abs(price - float(signal.stop_loss))
    notional = price * lot_size
    stop_loss_amount = stop_distance * lot_size
    costs = MarketCostModel(slippage_rate=float(getattr(settings, "slippage", 0.0005) or 0.0005))
    cost_adjusted_loss = stop_loss_amount + (notional * costs.round_trip_rate)
    atr = float(signal.feature_summary.get("atr", 0.0) or 0.0)
    atr_noise = atr * lot_size
    atr_ratio = float(signal.feature_summary.get("atr_ratio", 0.0) or 0.0)
    volatility_bucket = _volatility_bucket(atr_ratio)
    stop_pct = stop_distance / price if price else 0.0
    replay_stop_rate = float(replay.stop_hit_rate or 0.0)
    if stop_pct <= 0.009 and replay_stop_rate <= 0.42:
        risk_bucket = "acceptable"
    elif stop_pct <= 0.016 and replay_stop_rate <= 0.52:
        risk_bucket = "stretched"
    elif stop_pct <= 0.024:
        risk_bucket = "high_risk"
    else:
        risk_bucket = "avoid"
    summary = (
        f"One lot carries roughly {cost_adjusted_loss:,.0f} INR risk to the stop; "
        f"current volatility is {volatility_bucket}."
    )
    warnings: list[str] = []
    if risk_bucket in {"high_risk", "avoid"}:
        warnings.append("lot_risk_elevated")
    if replay_stop_rate > 0.55:
        warnings.append("historical_stop_pressure")
    return FuturesLotRiskAssessment(
        available=True,
        lot_size=lot_size,
        contract_price=round(price, 4),
        contract_notional=round(notional, 2),
        stop_distance_per_unit=round(stop_distance, 4),
        stop_loss_amount_per_lot=round(stop_loss_amount, 2),
        cost_adjusted_loss_per_lot=round(cost_adjusted_loss, 2),
        atr_noise_per_lot=round(atr_noise, 2),
        volatility_bucket=volatility_bucket,
        entry_risk_bucket=risk_bucket,
        summary=summary,
        warnings=tuple(warnings),
    )


def _prepare_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    out = ohlcv.copy()
    out["ema_9"] = compute_ema(out, "close", 9)
    out["ema_21"] = compute_ema(out, "close", 21)
    out["ema_50"] = compute_ema(out, "close", 50)
    out["atr"] = compute_atr(out, 14)
    out["vwap"] = compute_vwap(out)
    out["volume_sma"] = compute_volume_sma(out, 20)
    out["volume_ratio"] = out["volume"] / out["volume_sma"].replace(0, pd.NA)
    out["rsi"] = compute_rsi(out, "close", 14)
    out["macd_hist"] = compute_macd(out, "close")["macd_hist"]
    out["prior_high"] = out["high"].rolling(20).max().shift(1)
    out["prior_low"] = out["low"].rolling(20).min().shift(1)
    out["recent_support"] = out["low"].rolling(8).min().shift(1)
    out["recent_resistance"] = out["high"].rolling(8).max().shift(1)
    out["ema_9_slope"] = out["ema_9"] - out["ema_9"].shift(3)
    out["ema_21_slope"] = out["ema_21"] - out["ema_21"].shift(3)
    spread = (out["high"] - out["low"]).replace(0, pd.NA)
    out["body_strength"] = (out["close"] - out["open"]).abs() / spread
    out["upper_wick_ratio"] = (out["high"] - out[["close", "open"]].max(axis=1)) / spread
    out["lower_wick_ratio"] = (out[["close", "open"]].min(axis=1) - out["low"]) / spread
    return out


def _simulate_trade_outcome(
    future_bars: pd.DataFrame,
    *,
    signal: StructuredSignalDetails,
) -> Optional[dict[str, float]]:
    if future_bars is None or future_bars.empty:
        return None
    entry = float(signal.entry_price or 0.0)
    stop = float(signal.stop_loss or 0.0)
    target = float(signal.target_price or 0.0)
    if not entry or not stop or not target:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    outcome = 0
    hold_bars = 0
    mae_pct = 0.0
    mfe_pct = 0.0
    for idx, row in enumerate(future_bars.itertuples(index=False), start=1):
        high = float(getattr(row, "high"))
        low = float(getattr(row, "low"))
        if signal.verdict == "BUY":
            adverse = max(0.0, (entry - low) / entry)
            favorable = max(0.0, (high - entry) / entry)
            if low <= stop:
                outcome = -1
                hold_bars = idx
                mae_pct = max(mae_pct, adverse)
                mfe_pct = max(mfe_pct, favorable)
                break
            if high >= target:
                outcome = 1
                hold_bars = idx
                mae_pct = max(mae_pct, adverse)
                mfe_pct = max(mfe_pct, favorable)
                break
        else:
            adverse = max(0.0, (high - entry) / entry)
            favorable = max(0.0, (entry - low) / entry)
            if high >= stop:
                outcome = -1
                hold_bars = idx
                mae_pct = max(mae_pct, adverse)
                mfe_pct = max(mfe_pct, favorable)
                break
            if low <= target:
                outcome = 1
                hold_bars = idx
                mae_pct = max(mae_pct, adverse)
                mfe_pct = max(mfe_pct, favorable)
                break
        mae_pct = max(mae_pct, adverse)
        mfe_pct = max(mfe_pct, favorable)
        hold_bars = idx
    realized = 0.0
    if outcome == 1:
        realized = abs(target - entry) / risk
    elif outcome == -1:
        realized = -1.0
    else:
        last_close = float(
            future_bars.iloc[min(len(future_bars) - 1, hold_bars - 1)]["close"]
        )
        realized = (
            (last_close - entry) / risk
            if signal.verdict == "BUY"
            else (entry - last_close) / risk
        )
    return {
        "outcome": float(outcome),
        "hold_bars": float(hold_bars),
        "mae_pct": float(mae_pct),
        "mfe_pct": float(mfe_pct),
        "rr": float(realized),
    }


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _volatility_bucket(atr_ratio: float) -> str:
    if atr_ratio >= 0.028:
        return "high"
    if atr_ratio >= 0.018:
        return "elevated"
    if atr_ratio >= 0.009:
        return "normal"
    return "calm"


def _bundle_cache_key(
    *,
    ohlcv: pd.DataFrame,
    instrument: InstrumentRef,
    timeframe: str,
) -> tuple[str, ...]:
    last_ts = ""
    last_close = ""
    if ohlcv is not None and not ohlcv.empty:
        ts = ohlcv.index[-1]
        last_ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        last_close = f"{float(ohlcv.iloc[-1]['close']):.4f}"
    return (
        instrument.symbol,
        instrument.tradingsymbol,
        timeframe,
        last_ts,
        last_close,
        str(len(ohlcv) if ohlcv is not None else 0),
    )


def _bundle_cache_get(key: tuple[str, ...], *, ttl_seconds: int) -> Optional[SignalEngineBundle]:
    entry = _BUNDLE_CACHE.get(key)
    if entry is None:
        return None
    created_at, bundle = entry
    if time.time() - created_at > max(1, ttl_seconds):
        _BUNDLE_CACHE.pop(key, None)
        return None
    return bundle
