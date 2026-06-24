from __future__ import annotations

from datetime import date

import pandas as pd

from fortuna.agentic.contracts import OpenInterestEnrichment
from fortuna.app.signal_engine import (
    assess_futures_lot_risk,
    build_structured_signal,
    evaluate_replay_risk,
)
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRef


def _trend_frame(periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 09:15", periods=periods, freq="5min")
    closes = [100.0 + i * 0.35 for i in range(periods)]
    opens = [value - 0.12 for value in closes]
    highs = [value + 0.35 for value in closes]
    lows = [value - 0.28 for value in closes]
    volume = [1000 + (i % 7) * 120 for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        },
        index=idx,
    )


def test_build_structured_signal_returns_buy_for_bullish_trend():
    df = _trend_frame()
    instrument = InstrumentRef("RELIANCE", "RELIANCE-EQ", "2885", "NSE")
    signal = build_structured_signal(
        ohlcv=df,
        instrument=instrument,
        timeframe="5m",
        oi_enrichment=OpenInterestEnrichment(
            available=True,
            posture="long_build_up",
            summary="OI posture looks like long build up.",
        ),
    )
    assert signal.verdict == "BUY"
    assert signal.entry_price is not None
    assert signal.stop_loss is not None
    assert signal.target_price is not None
    assert signal.risk_reward is not None
    assert signal.risk_reward > 1.3
    assert signal.setup_type


def test_evaluate_replay_risk_returns_metrics_for_actionable_setup(tmp_path):
    df = _trend_frame()
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )
    instrument = InstrumentRef("RELIANCE", "RELIANCE-EQ", "2885", "NSE")
    signal = build_structured_signal(ohlcv=df, instrument=instrument, timeframe="5m")
    replay = evaluate_replay_risk(
        ohlcv=df,
        instrument=instrument,
        timeframe="5m",
        settings=settings,
        current_signal=signal,
    )
    assert replay.ok is True
    assert replay.matched_setups >= 1
    assert replay.stop_hit_rate is not None
    assert replay.target_hit_rate is not None


def test_assess_futures_lot_risk_uses_lot_size_and_stop(tmp_path):
    df = _trend_frame()
    signal = build_structured_signal(
        ohlcv=df,
        instrument=InstrumentRef(
            "RELIANCE.FUT",
            "RELIANCE30JUN26FUT",
            "5001",
            "NFO",
            instrumenttype="FUTSTK",
            expiry=date(2026, 6, 30),
            lot_size=250,
            name="RELIANCE",
        ),
        timeframe="5m",
    )
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
    )
    replay = evaluate_replay_risk(
        ohlcv=df,
        instrument=InstrumentRef("RELIANCE", "RELIANCE-EQ", "2885", "NSE"),
        timeframe="5m",
        settings=settings,
        current_signal=signal,
    )
    lot_risk = assess_futures_lot_risk(
        instrument=InstrumentRef(
            "RELIANCE.FUT",
            "RELIANCE30JUN26FUT",
            "5001",
            "NFO",
            instrumenttype="FUTSTK",
            expiry=date(2026, 6, 30),
            lot_size=250,
            name="RELIANCE",
        ),
        signal=signal,
        replay=replay,
        settings=settings,
    )
    assert lot_risk.available is True
    assert lot_risk.lot_size == 250
    assert lot_risk.contract_notional is not None
    assert lot_risk.stop_loss_amount_per_lot is not None
    assert lot_risk.cost_adjusted_loss_per_lot >= lot_risk.stop_loss_amount_per_lot
