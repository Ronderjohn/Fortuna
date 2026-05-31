"""Aggregate SmartAPI snap quotes into OHLCV bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

_IST = ZoneInfo("Asia/Kolkata")

_TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "60m": 60,
}


def bucket_start_for(timeframe: str, ts: pd.Timestamp) -> pd.Timestamp:
    """IST wall-clock start of the OHLCV bucket that contains ``ts`` (tz-naive)."""
    minutes = _TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"Unsupported bar timeframe: {timeframe}")
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_IST).tz_localize(None)
    minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=minute, second=0, microsecond=0, nanosecond=0)


def current_bucket_start(timeframe: str) -> pd.Timestamp:
    """IST bucket start for the current wall-clock minute (tz-naive)."""
    return bucket_start_for(timeframe, pd.Timestamp.now(tz=_IST))


@dataclass
class OhlcvBar:
    """Single completed OHLCV bar."""

    datetime: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarAggregator:
    """
    Bucket snap-quote ticks into OHLCV bars.

    Uses exchange_timestamp when present (IST); otherwise wall-clock IST.
    NSE regular session is not enforced here — bars align to clock buckets.
    """

    def __init__(
        self,
        timeframe: str,
        *,
        on_bar: Optional[Callable[[OhlcvBar], None]] = None,
    ) -> None:
        minutes = _TIMEFRAME_MINUTES.get(timeframe)
        if minutes is None:
            raise ValueError(f"Unsupported bar timeframe: {timeframe}")
        self.timeframe = timeframe
        self._minutes = minutes
        self._on_bar = on_bar
        self._on_tick: Optional[Callable[[OhlcvBar], None]] = None
        self._current_bucket: Optional[pd.Timestamp] = None
        self._bar: Optional[OhlcvBar] = None

    @property
    def current_bar(self) -> Optional[OhlcvBar]:
        """In-progress (forming) bar, if any."""
        return self._bar

    def set_on_tick(self, callback: Optional[Callable[[OhlcvBar], None]]) -> None:
        self._on_tick = callback

    def _bucket_start(self, ts: pd.Timestamp) -> pd.Timestamp:
        return bucket_start_for(self.timeframe, ts)

    def _extract_ts(self, tick: dict) -> pd.Timestamp:
        """Tick time for bucketing.

        Uses SmartAPI ``exchange_timestamp`` when present; otherwise uses
        wall-clock IST.
        """
        if "exchange_timestamp" in tick and tick["exchange_timestamp"]:
            raw = tick["exchange_timestamp"]
            if raw > 10_000_000_000:
                raw = raw / 1000.0
            return pd.Timestamp(datetime.fromtimestamp(raw, tz=_IST))
        return pd.Timestamp.now(tz=_IST)

    def _price(self, tick: dict) -> float:
        for key in ("last_traded_price", "ltp", "closed_price"):
            if key in tick and tick[key] is not None:
                return float(tick[key]) / 100.0
        raise ValueError(f"No price in tick: {tick!r}")

    def _volume(self, tick: dict) -> float:
        for key in ("volume_trade_for_the_day", "total_traded_volume", "volume"):
            if key in tick and tick[key] is not None:
                return float(tick[key])
        return 0.0

    def update(self, tick: dict) -> Optional[OhlcvBar]:
        """Ingest one parsed snap-quote dict; returns bar if previous bucket closed."""
        ts = self._extract_ts(tick)
        bucket = self._bucket_start(ts)
        price = self._price(tick)
        vol = self._volume(tick)
        closed: Optional[OhlcvBar] = None

        if self._current_bucket is None:
            self._current_bucket = bucket
            self._bar = OhlcvBar(bucket, price, price, price, price, vol)
            return None

        if bucket > self._current_bucket:
            closed = self._bar
            if closed and self._on_bar:
                self._on_bar(closed)
            self._current_bucket = bucket
            self._bar = OhlcvBar(bucket, price, price, price, price, vol)
            return closed

        if self._bar is None:
            return None
        b = self._bar
        b.high = max(b.high, price)
        b.low = min(b.low, price)
        b.close = price
        b.volume = max(b.volume, vol)
        if self._on_tick and self._bar is not None:
            self._on_tick(self._bar)
        return None

    def flush(self) -> Optional[OhlcvBar]:
        """Emit the in-progress bar (e.g. on shutdown)."""
        bar = self._bar
        if bar and self._on_bar:
            self._on_bar(bar)
        self._bar = None
        self._current_bucket = None
        return bar
