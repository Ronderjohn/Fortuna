"""Angel One SmartAPI historical OHLCV (getCandleData)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from fortuna.config.smartapi_settings import SmartAPISettings, get_smartapi_settings
from fortuna.data.instruments import InstrumentRegistry
from fortuna.data.sources.smartapi_normalize import candles_to_dataframe
from fortuna.data.sources.smartapi_session import SmartAPISession
from fortuna.data.sources.symbols import normalize_for_smartapi
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

# SmartAPI signals a rate-limit hit as HTTP 403 in the REST path, but the
# SmartConnect SDK wraps that as a plain "Invalid Token" string — completely
# misleading. We treat any of these as "back off and retry".
_RATE_LIMIT_HINTS = ("rate", "exceeding access", "ag8001", "invalid token", "403")

_INTERVAL_MAP = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "60m": "ONE_HOUR",
    "1d": "ONE_DAY",
    "1D": "ONE_DAY",
}


class SmartAPIHistoricalSource:
    """Fetch OHLCV via SmartAPI getCandleData."""

    def __init__(
        self,
        session: Optional[SmartAPISession] = None,
        registry: Optional[InstrumentRegistry] = None,
        settings: Optional[SmartAPISettings] = None,
    ) -> None:
        self._settings = settings or get_smartapi_settings()
        self._session = session or SmartAPISession.get(self._settings)
        self._registry = registry or InstrumentRegistry(self._settings)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        sym = normalize_for_smartapi(symbol)
        interval = _INTERVAL_MAP.get(timeframe, timeframe)
        if interval not in _INTERVAL_MAP.values():
            raise ValueError(f"Unsupported SmartAPI timeframe: {timeframe}")

        inst = self._registry.resolve(sym)
        end_dt = datetime.now() if end is None else pd.Timestamp(end).to_pydatetime()
        if start:
            start_dt = pd.Timestamp(start).to_pydatetime()
        elif days:
            start_dt = end_dt - timedelta(days=days)
        else:
            start_dt = end_dt - timedelta(days=30)

        span_days = (end_dt - start_dt).days + 1
        chunk_days = 20 if interval in ("FIVE_MINUTE", "ONE_MINUTE", "FIFTEEN_MINUTE") else 90
        if span_days <= chunk_days:
            return self._fetch_range(
                sym, inst.exchange, inst.symboltoken, interval, start_dt, end_dt
            )

        frames: list[pd.DataFrame] = []
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)
            logger.info(
                "SmartAPI chunk %s %s -> %s",
                sym,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            part = self._fetch_range(
                sym, inst.exchange, inst.symboltoken, interval, chunk_start, chunk_end
            )
            if not part.empty:
                frames.append(part)
            chunk_start = chunk_end + timedelta(minutes=1)
            self._session.rate_limit()

        if not frames:
            return candles_to_dataframe([], sym)
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    def _fetch_range(
        self,
        sym: str,
        exchange: str,
        symboltoken: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        fromdate = start_dt.strftime("%Y-%m-%d %H:%M")
        todate = end_dt.strftime("%Y-%m-%d %H:%M")
        params = {
            "exchange": exchange,
            "symboltoken": symboltoken,
            "interval": interval,
            "fromdate": fromdate,
            "todate": todate,
        }
        self._session.rate_limit()
        logger.info(
            "SmartAPI getCandleData %s token=%s %s %s -> %s",
            sym,
            symboltoken,
            interval,
            fromdate,
            todate,
        )
        response = self._get_candles_with_retries(params, sym)
        if not response or not response.get("status"):
            msg = response.get("message", "unknown error") if response else "empty response"
            raise RuntimeError(f"SmartAPI getCandleData failed for {sym}: {msg}")
        candles = response.get("data") or []
        return candles_to_dataframe(candles, sym)

    @staticmethod
    def _is_rate_limited(response: Optional[dict], err: Optional[Exception]) -> bool:
        haystacks: list[str] = []
        if response:
            haystacks.append(str(response.get("message") or "").lower())
            haystacks.append(str(response.get("errorCode") or "").lower())
        if err is not None:
            haystacks.append(str(err).lower())
        return any(any(h in s for h in _RATE_LIMIT_HINTS) for s in haystacks if s)

    def _get_candles_with_retries(
        self, params: dict, sym: str, *, max_attempts: int = 4
    ) -> dict:
        """Call getCandleData with exponential backoff on rate-limit errors."""
        delay = 4.0
        last_resp: Optional[dict] = None
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._get_candles(params)
            except Exception as e:  # noqa: BLE001
                resp, last_err = None, e
            else:
                last_resp = resp
                if resp and resp.get("status"):
                    return resp
            if not self._is_rate_limited(resp, last_err) or attempt == max_attempts:
                if last_err is not None:
                    raise last_err
                return resp or {}
            logger.warning(
                "SmartAPI rate-limit hit for %s (attempt %d/%d) — backing off %.1fs",
                sym,
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            delay *= 2.0
        return last_resp or {}

    def _get_candles(self, params: dict) -> dict:
        # Prefer the REST path: it surfaces the real HTTP status (e.g. 403 for
        # rate-limit) instead of the SDK's misleading "Invalid Token" message.
        from fortuna.data.sources.smartapi_rest import get_candle_data

        try:
            tokens = self._session.get_tokens()
            return get_candle_data(
                api_key=self._settings.api_key or "",
                jwt_token=tokens.jwt_token,
                params=params,
            )
        except Exception as e:  # noqa: BLE001
            # If REST itself failed (network / TLS), fall back to the SDK path.
            try:
                api = self._session.get_client()
                return api.getCandleData(params)
            except ImportError:
                raise e
