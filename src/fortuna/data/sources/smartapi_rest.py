"""Minimal SmartAPI REST login (stdlib) when smartapi-python is not installed."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import struct
import time
import urllib.error
import urllib.request
from typing import Any

_LOGIN_URL = (
    "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/loginByPassword"
)
_CANDLE_URL = "https://apiconnect.angelone.in/rest/secure/angelbroking/historical/v1/getCandleData"


def _b32_secret(secret: str) -> bytes:
    s = secret.replace(" ", "").strip().upper()
    pad = (-len(s)) % 8
    if pad:
        s += "=" * pad
    return base64.b32decode(s, casefold=True)


def totp_now(secret: str, *, period: int = 30, digits: int = 6) -> str:
    """RFC 6238 TOTP from base32 secret (no pyotp dependency)."""
    key = _b32_secret(secret)
    counter = int(time.time() // period)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(code % (10**digits)).zfill(digits)


def _default_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-UserType": "USER",
        "X-SourceID": "WEB",
        "X-ClientLocalIP": "127.0.0.1",
        "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress": "00:00:00:00:00:00",
        "X-PrivateKey": api_key,
    }


def login_by_password(
    *,
    api_key: str,
    client_code: str,
    password: str,
    totp_secret: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Return parsed JSON from loginByPassword."""
    body = json.dumps(
        {
            "clientcode": client_code,
            "password": password,
            "totp": totp_now(totp_secret),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _LOGIN_URL,
        data=body,
        headers=_default_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError(f"HTTP {e.code}: {raw}") from e


def get_candle_data(
    *,
    api_key: str,
    jwt_token: str,
    params: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Historical candles (getCandleData).

    Returns the parsed JSON. On HTTP errors a synthetic response with
    ``status=False`` is returned so the caller can detect rate-limit hits
    (HTTP 403 → ``errorCode='RATE_LIMIT'``) instead of swallowing the cause.
    """
    headers = _default_headers(api_key)
    headers["Authorization"] = f"Bearer {jwt_token}"
    body = json.dumps(params).encode("utf-8")
    req = urllib.request.Request(
        _CANDLE_URL,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"status": False, "message": raw}
        payload.setdefault("status", False)
        if e.code == 403:
            payload.setdefault("errorCode", "RATE_LIMIT")
            payload["message"] = (
                f"HTTP 403 rate limit: {payload.get('message') or raw or 'no body'}"
            )
        else:
            payload["message"] = f"HTTP {e.code}: {payload.get('message') or raw}"
        return payload
