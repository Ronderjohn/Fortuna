"""SmartAPI session tests (mocked login)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fortuna.config.smartapi_settings import SmartAPISettings
from fortuna.data.sources.smartapi_session import SmartAPISession


@pytest.fixture(autouse=True)
def _reset_session() -> None:
    SmartAPISession.reset()
    yield
    SmartAPISession.reset()


def test_login_returns_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SmartAPISettings(
        api_key="k",
        client_code="C123",
        password="pin",
        totp_secret="JBSWY3DPEHPK3PXP",
    )
    mock_api = MagicMock()
    mock_api.generateSession.return_value = {
        "status": True,
        "data": {
            "jwtToken": "jwt",
            "feedToken": "feed",
            "refreshToken": "refresh",
        },
    }

    with patch(
        "fortuna.data.sources.smartapi_session.SmartAPISession._import_smart_connect",
        return_value=lambda api_key=None: mock_api,
    ):
        with patch("pyotp.TOTP") as totp_cls:
            totp_cls.return_value.now.return_value = "123456"
            session = SmartAPISession(settings)
            tokens = session.login()

    assert tokens.jwt_token == "jwt"
    assert tokens.feed_token == "feed"
    mock_api.generateSession.assert_called_once()
