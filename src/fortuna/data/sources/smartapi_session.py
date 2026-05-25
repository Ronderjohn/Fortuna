"""Thread-safe SmartAPI login session (jwt + feed tokens)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fortuna.config.smartapi_settings import SmartAPISettings, get_smartapi_settings
from fortuna.data.sources.smartapi_throttle import RateLimiter
from fortuna.utils.insecure_ssl import maybe_disable_ssl_verification
from fortuna.utils.logging import get_logger

# Idempotent — applies the bypass only when FORTUNA_INSECURE_SSL is set.
maybe_disable_ssl_verification()

logger = get_logger(__name__)

_IST = ZoneInfo("Asia/Kolkata")


@dataclass
class SmartAPISessionTokens:
    jwt_token: str
    feed_token: str
    refresh_token: str
    client_code: str


class SmartAPISession:
    """
    Singleton-style session for SmartConnect login + token refresh.

    Session expires around midnight IST; refresh when stale.
    """

    _instance: Optional["SmartAPISession"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        settings: Optional[SmartAPISettings] = None,
        *,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._settings = settings or get_smartapi_settings()
        self._limiter = rate_limiter or RateLimiter(self._settings.rate_limit_per_sec)
        self._lock = threading.Lock()
        self._tokens: Optional[SmartAPISessionTokens] = None
        self._smart_connect: Any = None
        self._login_at: Optional[datetime] = None

    @classmethod
    def get(cls, settings: Optional[SmartAPISettings] = None) -> "SmartAPISession":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(settings)
            elif settings is not None:
                cls._instance._settings = settings
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear singleton (tests)."""
        with cls._instance_lock:
            cls._instance = None

    def _require_config(self) -> None:
        if not self._settings.configured:
            raise ValueError(
                "SmartAPI credentials missing. Set SMARTAPI_API_KEY, SMARTAPI_CLIENT_CODE, "
                "SMARTAPI_PASSWORD, SMARTAPI_TOTP_SECRET in .env"
            )

    def _import_smart_connect(self) -> Any:
        try:
            from SmartApi.smartConnect import SmartConnect
        except ImportError as e:
            raise ImportError(
                "smartapi-python is required. Install with: uv sync --group smartapi"
            ) from e
        return SmartConnect

    def _connect(self) -> Any:
        if self._smart_connect is None:
            SmartConnect = self._import_smart_connect()
            self._smart_connect = SmartConnect(api_key=self._settings.api_key)
        return self._smart_connect

    def _totp(self) -> str:
        try:
            import pyotp
        except ImportError:
            from fortuna.data.sources.smartapi_rest import totp_now

            return totp_now(self._settings.totp_secret or "")
        secret = self._settings.totp_secret or ""
        try:
            return pyotp.TOTP(secret).now()
        except Exception:
            from fortuna.data.sources.smartapi_rest import totp_now

            return totp_now(secret)

    def _midnight_ist(self, when: Optional[datetime] = None) -> datetime:
        when = when or datetime.now(tz=_IST)
        return when.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    def is_stale(self) -> bool:
        if self._tokens is None or self._login_at is None:
            return True
        now = datetime.now(tz=_IST)
        if now >= self._midnight_ist(self._login_at):
            return True
        return (now - self._login_at) > timedelta(hours=23)

    def login(self, *, force: bool = False) -> SmartAPISessionTokens:
        with self._lock:
            if not force and self._tokens is not None and not self.is_stale():
                return self._tokens

            self._require_config()
            self._limiter.wait()
            logger.info("SmartAPI login for client %s", self._settings.client_code)
            session = self._login_session()
            if not session or not session.get("status"):
                msg = session.get("message", "login failed") if session else "empty response"
                raise RuntimeError(f"SmartAPI login failed: {msg}")

            data = session.get("data") or {}
            jwt = data.get("jwtToken") or data.get("jwt_token")
            feed = data.get("feedToken") or data.get("feed_token")
            refresh = data.get("refreshToken") or data.get("refresh_token")
            if not all([jwt, feed, refresh]):
                raise RuntimeError(f"SmartAPI login missing tokens: {data!r}")

            # The SmartConnect SDK sometimes returns the JWT already prefixed
            # with "Bearer "; downstream REST helpers add their own prefix, so
            # we'd otherwise produce a malformed Authorization header and the
            # server would reject every historical call with "Invalid Token".
            jwt = jwt.strip()
            if jwt.lower().startswith("bearer "):
                jwt = jwt[len("bearer ") :].strip()

            self._tokens = SmartAPISessionTokens(
                jwt_token=jwt,
                feed_token=feed,
                refresh_token=refresh,
                client_code=self._settings.client_code or "",
            )
            self._login_at = datetime.now(tz=_IST)
            if self._smart_connect is not None:
                api = self._connect()
                api.setAccessToken(jwt)
                api.setRefreshToken(refresh)
                api.setFeedToken(feed)
            return self._tokens

    def _login_session(self) -> dict:
        """Prefer SmartConnect SDK; fall back to stdlib REST."""
        try:
            api = self._connect()
            totp = self._totp()
            return api.generateSession(
                self._settings.client_code,
                self._settings.password,
                totp,
            )
        except ImportError:
            from fortuna.data.sources.smartapi_rest import login_by_password

            return login_by_password(
                api_key=self._settings.api_key or "",
                client_code=self._settings.client_code or "",
                password=self._settings.password or "",
                totp_secret=self._settings.totp_secret or "",
            )

    def get_tokens(self, *, force_refresh: bool = False) -> SmartAPISessionTokens:
        if force_refresh or self.is_stale():
            return self.login(force=True)
        with self._lock:
            if self._tokens is None:
                return self.login(force=True)
            return self._tokens

    def get_client(self) -> Any:
        """Authenticated SmartConnect instance."""
        self.get_tokens()
        return self._connect()

    def rate_limit(self) -> None:
        self._limiter.wait()
