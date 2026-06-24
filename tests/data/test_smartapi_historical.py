"""SmartAPI historical source tests (mocked REST)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fortuna.config.smartapi_settings import SmartAPISettings
from fortuna.data.instruments import InstrumentRef, InstrumentRegistry
from fortuna.data.sources.smartapi_historical import SmartAPIHistoricalSource
from fortuna.data.sources.smartapi_session import SmartAPISession


@pytest.fixture(autouse=True)
def _reset_session() -> None:
    SmartAPISession.reset()
    yield
    SmartAPISession.reset()


def test_fetch_normalizes_candles() -> None:
    settings = SmartAPISettings(
        api_key="k",
        client_code="C",
        password="p",
        totp_secret="JBSWY3DPEHPK3PXP",
    )
    inst = InstrumentRef(
        symbol="RELIANCE",
        tradingsymbol="RELIANCE-EQ",
        symboltoken="2885",
        exchange="NSE",
    )
    registry = MagicMock(spec=InstrumentRegistry)
    registry.resolve.return_value = inst

    # The fetcher now prefers the REST helper (so it can detect HTTP 403 rate
    # limits cleanly). Mock that path directly.
    fake_response = {
        "status": True,
        "data": [
            ["2024-01-02 09:15:00", 100.0, 101.0, 99.0, 100.5, 1000],
            ["2024-01-02 09:20:00", 100.5, 102.0, 100.0, 101.0, 1200],
        ],
    }
    session = MagicMock(spec=SmartAPISession)
    session.get_tokens.return_value = MagicMock(jwt_token="jwt")

    src = SmartAPIHistoricalSource(session=session, registry=registry, settings=settings)
    with patch(
        "fortuna.data.sources.smartapi_rest.get_candle_data",
        return_value=fake_response,
    ) as rest:
        df = src.fetch("RELIANCE.NS", "5m", days=1)

    assert rest.called
    assert len(df) == 2
    assert list(df.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "datetime"


def test_fetch_retries_on_rate_limit() -> None:
    settings = SmartAPISettings(
        api_key="k",
        client_code="C",
        password="p",
        totp_secret="JBSWY3DPEHPK3PXP",
    )
    inst = InstrumentRef(
        symbol="RELIANCE",
        tradingsymbol="RELIANCE-EQ",
        symboltoken="2885",
        exchange="NSE",
    )
    registry = MagicMock(spec=InstrumentRegistry)
    registry.resolve.return_value = inst

    rate_limited = {
        "status": False,
        "errorCode": "RATE_LIMIT",
        "message": "HTTP 403 rate limit: Access denied because of exceeding access rate",
    }
    success = {
        "status": True,
        "data": [["2024-01-02 09:15:00", 100.0, 101.0, 99.0, 100.5, 1000]],
    }

    session = MagicMock(spec=SmartAPISession)
    session.get_tokens.return_value = MagicMock(jwt_token="jwt")
    src = SmartAPIHistoricalSource(session=session, registry=registry, settings=settings)

    with patch(
        "fortuna.data.sources.smartapi_rest.get_candle_data",
        side_effect=[rate_limited, success],
    ) as rest, patch("fortuna.data.sources.smartapi_historical.time.sleep") as sleep:
        df = src.fetch("RELIANCE.NS", "5m", days=1)

    assert rest.call_count == 2
    assert sleep.called
    assert len(df) == 1


def test_fetch_preserves_open_interest_when_candles_include_it() -> None:
    settings = SmartAPISettings(
        api_key="k",
        client_code="C",
        password="p",
        totp_secret="JBSWY3DPEHPK3PXP",
    )
    inst = InstrumentRef(
        symbol="RELIANCE.FUT",
        tradingsymbol="RELIANCE30JUN26FUT",
        symboltoken="5001",
        exchange="NFO",
        instrumenttype="FUTSTK",
    )
    registry = MagicMock(spec=InstrumentRegistry)
    registry.resolve.return_value = inst
    fake_response = {
        "status": True,
        "data": [
            ["2024-01-02 09:15:00", 100.0, 101.0, 99.0, 100.5, 1000, 25000],
            ["2024-01-02 09:20:00", 100.5, 102.0, 100.0, 101.0, 1200, 25200],
        ],
    }
    session = MagicMock(spec=SmartAPISession)
    session.get_tokens.return_value = MagicMock(jwt_token="jwt")

    src = SmartAPIHistoricalSource(session=session, registry=registry, settings=settings)
    with patch(
        "fortuna.data.sources.smartapi_rest.get_candle_data",
        return_value=fake_response,
    ):
        df = src.fetch("RELIANCE.FUT", "5m", days=1)

    assert "open_interest" in df.columns
    assert float(df["open_interest"].iloc[-1]) == 25200.0


def test_fetch_strips_bearer_prefix_from_sdk_jwt() -> None:
    """Regression test: SDK login returns 'Bearer <jwt>'. The REST helper adds
    its own 'Bearer ' prefix, so without stripping the Authorization header
    becomes 'Bearer Bearer <jwt>' and SmartAPI rejects every call with
    'Invalid Token'."""
    from fortuna.data.sources.smartapi_session import SmartAPISession

    settings = SmartAPISettings(
        api_key="k",
        client_code="C",
        password="p",
        totp_secret="JBSWY3DPEHPK3PXP",
    )
    SmartAPISession.reset()
    sess = SmartAPISession(settings)

    fake_session_response = {
        "status": True,
        "data": {
            "jwtToken": "Bearer eyJhbGciOi.example.token",
            "feedToken": "feed",
            "refreshToken": "refresh",
        },
    }

    with patch.object(sess, "_login_session", return_value=fake_session_response), patch.object(
        sess, "_connect", return_value=MagicMock()
    ):
        tokens = sess.login(force=True)

    assert not tokens.jwt_token.lower().startswith("bearer "), (
        "JWT must be stored without 'Bearer ' prefix to avoid double-prefixing"
    )
    assert tokens.jwt_token == "eyJhbGciOi.example.token"
