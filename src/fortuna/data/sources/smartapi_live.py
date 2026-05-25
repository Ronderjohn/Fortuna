"""SmartAPI WebSocket 2.0 live feed → OHLCV bars → cache."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

import pandas as pd

from fortuna.config.smartapi_settings import SmartAPISettings, get_smartapi_settings
from fortuna.data.instruments import InstrumentRegistry, InstrumentRef
from fortuna.data.sources.smartapi_bar_aggregator import BarAggregator, OhlcvBar
from fortuna.data.sources.smartapi_session import SmartAPISession
from fortuna.data.sources.smartapi_ws_decode import decode_ws_message
from fortuna.data.sources.symbols import normalize_for_smartapi
from fortuna.utils.logging import get_logger

if TYPE_CHECKING:
    from fortuna.data.manager import MarketDataManager

logger = get_logger(__name__)


class SmartAPILiveBarFeed:
    """
    Subscribe to SmartAPI WebSocket 2.0 snap quotes and write closed bars to cache.

    Requires SMARTAPI_USE_LIVE_FEED=true and valid credentials.
    """

    def __init__(
        self,
        manager: "MarketDataManager",
        symbol: str,
        timeframe: str = "5m",
        *,
        session: Optional[SmartAPISession] = None,
        registry: Optional[InstrumentRegistry] = None,
        settings: Optional[SmartAPISettings] = None,
        on_bar: Optional[Callable[[OhlcvBar], None]] = None,
        on_tick: Optional[Callable[[OhlcvBar], None]] = None,
    ) -> None:
        self._manager = manager
        self._symbol = normalize_for_smartapi(symbol)
        self._timeframe = timeframe
        self._settings = settings or get_smartapi_settings()
        self._session = session or SmartAPISession.get(self._settings)
        self._registry = registry or InstrumentRegistry(self._settings)
        self._inst: Optional[InstrumentRef] = None
        self._ws: Any = None
        self._thread: Optional[threading.Thread] = None
        self._aggregator = BarAggregator(
            timeframe,
            on_bar=on_bar or self._on_bar_closed,
        )
        if on_tick is not None:
            self._aggregator.set_on_tick(on_tick)

    @property
    def aggregator(self) -> BarAggregator:
        return self._aggregator

    def _on_bar_closed(self, bar: OhlcvBar) -> None:
        row = pd.DataFrame(
            {
                "open": [bar.open],
                "high": [bar.high],
                "low": [bar.low],
                "close": [bar.close],
                "volume": [bar.volume],
            },
            index=pd.DatetimeIndex([bar.datetime], name="datetime"),
        )
        row["symbol"] = self._symbol
        self._manager.append_bar(self._symbol, self._timeframe, row)
        logger.info("Live bar %s %s close=%.2f", self._symbol, self._timeframe, bar.close)

    def _build_ws(self) -> Any:
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        except ImportError as e:
            raise ImportError(
                "smartapi-python is required. Install with: uv sync --group smartapi"
            ) from e

        tokens = self._session.get_tokens()
        return SmartWebSocketV2(
            tokens.jwt_token,
            self._settings.api_key,
            tokens.client_code,
            tokens.feed_token,
            max_retry_attempt=3,
        )

    def _on_data(self, _wsapp: object, message: object) -> None:
        try:
            tick = decode_ws_message(message)
            if tick is None:
                return
            self._aggregator.update(tick)
        except Exception as e:
            logger.warning("Tick parse error: %s", e)

    def _on_error(self, _wsapp: object, error: object) -> None:
        logger.error("SmartAPI WebSocket error: %s", error)

    def _on_close(self, _wsapp: object, close_status_code: object, close_msg: object) -> None:
        logger.info("SmartAPI WebSocket closed: %s %s", close_status_code, close_msg)

    def _on_open(self, wsapp: object) -> None:
        assert self._inst is not None
        sws = self._ws
        assert sws is not None
        token_list = [
            {
                "exchangeType": self._inst.exchange_type,
                "tokens": [self._inst.symboltoken],
            }
        ]
        correlation_id = f"fortuna_{self._symbol}"
        sws.subscribe(correlation_id, sws.SNAP_QUOTE, token_list)
        logger.info("Subscribed snap quote %s token=%s", self._symbol, self._inst.symboltoken)

    def start(self, *, background: bool = True) -> None:
        if not self._settings.use_live_feed:
            raise RuntimeError("Live feed disabled. Set SMARTAPI_USE_LIVE_FEED=true")
        self._inst = self._registry.resolve(self._symbol)
        self._ws = self._build_ws()
        self._ws.on_data = self._on_data
        self._ws.on_open = self._on_open
        self._ws.on_error = self._on_error
        self._ws.on_close = self._on_close

        def run() -> None:
            assert self._ws is not None
            self._ws.connect()

        if background:
            self._thread = threading.Thread(
                target=run,
                name=f"smartapi-live-{self._symbol}",
                daemon=True,
            )
            self._thread.start()
        else:
            run()

    def stop(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close_connection()
            except Exception as e:
                logger.debug("WebSocket close: %s", e)
        self._aggregator.flush()
