#!/usr/bin/env python
"""Verify SmartAPI credentials and download instrument master (no orders)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.config.smartapi_settings import get_smartapi_settings
from fortuna.data.instruments import InstrumentRegistry
from fortuna.data.sources.smartapi_session import SmartAPISession


def _diagnose_login_error(msg: str) -> str:
    m = msg.lower()
    hints: list[str] = []
    if "invalid" in m and ("otp" in m or "totp" in m or "2fa" in m):
        hints.append("TOTP wrong or clock skew — check SMARTAPI_TOTP_SECRET (base32, no spaces).")
    if "password" in m or "pin" in m:
        hints.append("Check SMARTAPI_PASSWORD (trading PIN for API login, not app login only).")
    if "client" in m or "user" in m:
        hints.append("Check SMARTAPI_CLIENT_CODE matches Angel client ID exactly.")
    if "api" in m and "key" in m:
        hints.append("Check SMARTAPI_API_KEY matches SmartAPI app private key.")
    if not hints:
        hints.append("See Angel SmartAPI portal: app active, TOTP enabled, IP whitelist if configured.")
    return " ".join(hints)


def main() -> int:
    settings = get_smartapi_settings()
    if not settings.configured:
        print(
            "ERROR: Set in .env:\n"
            "  SMARTAPI_API_KEY\n"
            "  SMARTAPI_CLIENT_CODE\n"
            "  SMARTAPI_PASSWORD\n"
            "  SMARTAPI_TOTP_SECRET",
            file=sys.stderr,
        )
        return 1

    try:
        import SmartApi  # noqa: F401
    except ImportError:
        print("Note: smartapi-python not installed — using stdlib REST login.", flush=True)

    print("Logging in to SmartAPI...", flush=True)
    try:
        tokens = SmartAPISession.get(settings).login()
    except Exception as e:
        print(f"LOGIN FAILED: {e}", file=sys.stderr)
        print(_diagnose_login_error(str(e)), file=sys.stderr)
        return 1

    print(f"OK — client={tokens.client_code}, jwt=...{tokens.jwt_token[-8:]}, feed=...{tokens.feed_token[-8:]}")

    print("Loading instrument master...", flush=True)
    try:
        reg = InstrumentRegistry(settings)
        reg.ensure_loaded()
        ref = reg.resolve("RELIANCE.NS")
    except Exception as e:
        print(f"INSTRUMENT MASTER FAILED: {e}", file=sys.stderr)
        print("Check network/SSL or refresh SMARTAPI_SCRIP_MASTER_URL.", file=sys.stderr)
        return 1

    print(f"RELIANCE.NS -> token {ref.symboltoken} ({ref.tradingsymbol})")
    print(f"Cache: {reg.cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
