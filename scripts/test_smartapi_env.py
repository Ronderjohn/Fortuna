#!/usr/bin/env python
"""
Standalone SmartAPI connectivity test (verbose).

Checks .env variables, logs in, resolves RELIANCE.NS, fetches historical
5m candles via REST. Does not place orders.

Usage (PowerShell, from repo root):

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts\\test_smartapi_env.py

Optional:

    .\\.venv\\Scripts\\python.exe scripts\\test_smartapi_env.py --symbol RELIANCE.NS --days 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_VARS = (
    "SMARTAPI_API_KEY",
    "SMARTAPI_CLIENT_CODE",
    "SMARTAPI_PASSWORD",
    "SMARTAPI_TOTP_SECRET",
)

INTERVAL_MAP = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "1h": "ONE_HOUR",
    "60m": "ONE_HOUR",
    "1d": "ONE_DAY",
}


def _step(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _mask(value: str | None) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 4:
        return "****"
    return f"{value[:2]}...{value[-2:]}"


def _load_dotenv() -> Path:
    env_path = ROOT / ".env"
    _step(f"Loading environment from {env_path}")
    if not env_path.is_file():
        print(f"ERROR: Missing file {env_path}", file=sys.stderr)
        print("Copy .env.example to .env and fill SmartAPI credentials.", file=sys.stderr)
        sys.exit(1)
    try:
        from dotenv import load_dotenv

        loaded = load_dotenv(env_path, override=True)
        print(f"  dotenv loaded: {loaded}", flush=True)
    except ImportError:
        print("ERROR: python-dotenv not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)
    return env_path


def _check_env() -> dict[str, str]:
    _step("Checking required environment variables")
    out: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_VARS:
        val = os.environ.get(name, "").strip()
        if not val and name == "SMARTAPI_API_KEY":
            val = os.environ.get("ANGEL_API_KEY", "").strip()
        out[name] = val
        status = "OK" if val else "MISSING"
        print(f"  {name}: {status} ({_mask(val)})", flush=True)
        if not val:
            missing.append(name)
    optional = os.environ.get("SMARTAPI_USE_LIVE_FEED", "")
    print(f"  SMARTAPI_USE_LIVE_FEED: {optional or '(not set)'}", flush=True)
    if missing:
        print("\nERROR: Set missing variables in .env", file=sys.stderr)
        sys.exit(1)
    return out


def _diagnose(message: str) -> None:
    m = message.lower()
    print("\n--- Troubleshooting ---", flush=True)
    if "otp" in m or "totp" in m or "2fa" in m:
        print("  • TOTP: use base32 secret from Angel app (not the 6-digit code).", flush=True)
        print("  • Sync Windows clock (Settings → Time).", flush=True)
    if "password" in m or "pin" in m:
        print("  • PASSWORD: API trading PIN from SmartAPI login, not only mobile password.", flush=True)
    if "client" in m:
        print("  • CLIENT_CODE: must match Angel client ID (e.g. S1234567).", flush=True)
    if "api" in m and "key" in m:
        print("  • API_KEY: Private Key from smartapi.angelbroking.com app.", flush=True)
    if "ssl" in m or "certificate" in m:
        print("  • SSL: install certifi or fix corporate proxy / antivirus.", flush=True)
    print("  • Portal: app enabled, TOTP on, IP whitelist if you configured it.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test SmartAPI .env and historical fetch")
    parser.add_argument("--symbol", default="RELIANCE.NS", help="NSE symbol (default RELIANCE.NS)")
    parser.add_argument("--timeframe", default="5m", choices=sorted(INTERVAL_MAP.keys()))
    parser.add_argument("--days", type=int, default=7, help="Calendar days of history")
    parser.add_argument("--token", default=None, help="Override symboltoken (skip scrip master)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("Fortuna SmartAPI environment test", flush=True)
    print("=" * 60, flush=True)

    _load_dotenv()
    env = _check_env()

    _step("Importing Fortuna SmartAPI modules")
    try:
        from fortuna.data.instruments import InstrumentRegistry
        from fortuna.data.sources.smartapi_normalize import candles_to_dataframe
        from fortuna.data.sources.smartapi_rest import get_candle_data, login_by_password, totp_now
        from fortuna.config.smartapi_settings import SmartAPISettings
    except Exception:
        print("ERROR: Failed to import Fortuna modules:", file=sys.stderr)
        traceback.print_exc()
        return 1
    print("  imports OK", flush=True)

    _step("Generating TOTP (clock check)")
    try:
        code = totp_now(env["SMARTAPI_TOTP_SECRET"])
        print(f"  TOTP code (changes every 30s): {code}", flush=True)
    except Exception as e:
        print(f"ERROR: Invalid SMARTAPI_TOTP_SECRET: {e}", file=sys.stderr)
        _diagnose(str(e))
        return 1

    _step("Login (loginByPassword)")
    t0 = time.perf_counter()
    try:
        login_resp = login_by_password(
            api_key=env["SMARTAPI_API_KEY"],
            client_code=env["SMARTAPI_CLIENT_CODE"],
            password=env["SMARTAPI_PASSWORD"],
            totp_secret=env["SMARTAPI_TOTP_SECRET"],
            timeout=60.0,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"ERROR: Login request failed after {elapsed:.1f}s: {e}", file=sys.stderr)
        traceback.print_exc()
        _diagnose(str(e))
        return 1

    elapsed = time.perf_counter() - t0
    status = login_resp.get("status")
    message = login_resp.get("message", "")
    print(f"  Response in {elapsed:.1f}s — status={status!r} message={message!r}", flush=True)
    if not status:
        print(f"ERROR: Full response: {login_resp}", file=sys.stderr)
        _diagnose(message)
        return 1

    data = login_resp.get("data") or {}
    jwt = data.get("jwtToken") or data.get("jwt_token")
    feed = data.get("feedToken") or data.get("feed_token")
    if not jwt:
        print("ERROR: Login succeeded but no jwtToken in response", file=sys.stderr)
        return 1
    print(f"  jwtToken: ...{_mask(jwt)}", flush=True)
    print(f"  feedToken: ...{_mask(feed)}", flush=True)

    if args.token:
        symboltoken = args.token
        exchange = "NSE"
        _step(f"Using manual symboltoken={symboltoken}")
    else:
        _step("Loading instrument master (OpenAPIScripMaster.json)")
        try:
            settings = SmartAPISettings()
            registry = InstrumentRegistry(settings)
            registry.ensure_loaded()
            ref = registry.resolve(args.symbol)
            symboltoken = ref.symboltoken
            exchange = ref.exchange
            print(
                f"  {args.symbol} -> {ref.tradingsymbol} token={symboltoken} exchange={exchange}",
                flush=True,
            )
        except Exception as e:
            print(f"ERROR: Instrument lookup failed: {e}", file=sys.stderr)
            traceback.print_exc()
            print("  Try: --token 2885 for RELIANCE", flush=True)
            return 1

    interval = INTERVAL_MAP[args.timeframe]
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=args.days)
    params = {
        "exchange": exchange,
        "symboltoken": symboltoken,
        "interval": interval,
        "fromdate": start_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": end_dt.strftime("%Y-%m-%d %H:%M"),
    }

    _step(f"Historical candles getCandleData ({args.timeframe}, {args.days} days)")
    print(f"  params: exchange={exchange} token={symboltoken} interval={interval}", flush=True)
    print(f"  from={params['fromdate']} to={params['todate']}", flush=True)

    t0 = time.perf_counter()
    try:
        candle_resp = get_candle_data(
            api_key=env["SMARTAPI_API_KEY"],
            jwt_token=jwt,
            params=params,
            timeout=60.0,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"ERROR: getCandleData failed after {elapsed:.1f}s: {e}", file=sys.stderr)
        traceback.print_exc()
        _diagnose(str(e))
        return 1

    elapsed = time.perf_counter() - t0
    c_status = candle_resp.get("status")
    c_msg = candle_resp.get("message", "")
    print(f"  Response in {elapsed:.1f}s — status={c_status!r} message={c_msg!r}", flush=True)
    if not c_status:
        print(f"ERROR: Full response: {candle_resp}", file=sys.stderr)
        _diagnose(c_msg)
        return 1

    raw = candle_resp.get("data") or []
    print(f"  Raw candle rows: {len(raw)}", flush=True)
    if not raw:
        print("ERROR: No candle rows returned (market holiday or bad token/range?)", file=sys.stderr)
        return 1

    _step("Normalizing to DataFrame")
    try:
        df = candles_to_dataframe(raw, args.symbol)
    except Exception as e:
        print(f"ERROR: Normalize failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"  Bars: {len(df)}", flush=True)
    print(f"  Range: {df.index.min()} -> {df.index.max()}", flush=True)
    print("  Last 3 bars:", flush=True)
    print(df.tail(3).to_string(), flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SUCCESS: Environment and historical API are working.", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        raise SystemExit(130)
