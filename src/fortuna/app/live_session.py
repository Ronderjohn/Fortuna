"""Live SmartAPI feed bridge: in-memory OHLCV + event queue for dashboard."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import pandas as pd

from fortuna.data.manager import MarketDataManager
from fortuna.data.sources.smartapi_bar_aggregator import (
    OhlcvBar,
)
from fortuna.data.sources.smartapi_live import SmartAPILiveBarFeed
from fortuna.utils.logging import get_logger
from fortuna.utils.nse_session import in_nse_session

logger = get_logger(__name__)


def _to_naive_ist(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert any tz-aware timestamp to tz-naive IST so the live and historical
    OHLCV indices can be safely concatenated/sorted (the cache is tz-naive IST)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
    return ts


class LiveEventType(str, Enum):
    TICK = "tick"
    BAR_CLOSED = "bar_closed"


@dataclass
class LiveEvent:
    type: LiveEventType
    bar: Optional[OhlcvBar] = None


class LiveSessionBridge:
    """
    Maintains in-memory OHLCV updated from SmartAPI WebSocket.

    Closed bars append to memory and Parquet; forming bar exposed for chart tail.
    """

    def __init__(
        self,
        manager: MarketDataManager,
        symbol: str,
        timeframe: str,
        initial_ohlcv: pd.DataFrame,
        *,
        on_bar_closed: Optional[Callable[[], None]] = None,
    ) -> None:
        self._manager = manager
        self._symbol = symbol
        self._timeframe = timeframe
        self._lock = threading.RLock()
        self._ohlcv = initial_ohlcv.copy()
        self._events: queue.Queue[LiveEvent] = queue.Queue()
        self._on_bar_closed = on_bar_closed
        self._feed: Optional[SmartAPILiveBarFeed] = None
        self._running = False

    @property
    def ohlcv(self) -> pd.DataFrame:
        with self._lock:
            return self._ohlcv.copy()

    @property
    def is_running(self) -> bool:
        return self._running

    def forming_bar_df(self) -> Optional[pd.DataFrame]:
        if self._feed is None:
            return None
        bar = self._feed.aggregator.current_bar
        if bar is None:
            return None
        if not in_nse_session(bar.datetime):
            return None
        ts = _to_naive_ist(bar.datetime)
        with self._lock:
            if not self._ohlcv.empty:
                last_closed = pd.Timestamp(self._ohlcv.index.max())
                if ts < last_closed:
                    return None
        return pd.DataFrame(
            {
                "open": [bar.open],
                "high": [bar.high],
                "low": [bar.low],
                "close": [bar.close],
                "volume": [bar.volume],
            },
            index=pd.DatetimeIndex([ts], name="datetime"),
        )

    def _append_bar(self, bar: OhlcvBar) -> None:
        if not in_nse_session(bar.datetime):
            logger.debug("dropping off-session bar %s %s", self._symbol, bar.datetime)
            return
        ts = _to_naive_ist(bar.datetime)
        row = pd.DataFrame(
            {
                "open": [bar.open],
                "high": [bar.high],
                "low": [bar.low],
                "close": [bar.close],
                "volume": [bar.volume],
            },
            index=pd.DatetimeIndex([ts], name="datetime"),
        )
        with self._lock:
            self._ohlcv = pd.concat([self._ohlcv, row]).sort_index()
            self._ohlcv = self._ohlcv[~self._ohlcv.index.duplicated(keep="last")]
        row = row.copy()
        row["symbol"] = self._symbol
        self._manager.append_bar(self._symbol, self._timeframe, row)

    def _handle_bar(self, bar: OhlcvBar) -> None:
        if not in_nse_session(bar.datetime):
            return
        self._append_bar(bar)
        self._events.put(LiveEvent(LiveEventType.BAR_CLOSED, bar))
        if self._on_bar_closed:
            try:
                self._on_bar_closed()
            except Exception as e:
                logger.warning("on_bar_closed error: %s", e)

    def _handle_tick(self, bar: OhlcvBar) -> None:
        if not in_nse_session(bar.datetime):
            return
        self._events.put(LiveEvent(LiveEventType.TICK, bar))

    def start(self) -> None:
        if self._running:
            return
        self._feed = SmartAPILiveBarFeed(
            self._manager,
            self._symbol,
            self._timeframe,
            on_bar=self._handle_bar,
            on_tick=self._handle_tick,
        )
        self._feed.start(background=True)
        self._running = True
        logger.info("Live feed started for %s %s", self._symbol, self._timeframe)

    def stop(self) -> None:
        if self._feed is not None:
            self._feed.stop()
            self._feed = None
        self._running = False

    def drain_events(self) -> int:
        n = 0
        while True:
            try:
                self._events.get_nowait()
                n += 1
            except queue.Empty:
                break
        return n
