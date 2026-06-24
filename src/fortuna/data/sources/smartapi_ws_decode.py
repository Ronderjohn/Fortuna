"""Decode SmartAPI WebSocket 2.0 messages to tick dicts for BarAggregator."""

from __future__ import annotations

import json
import struct
from typing import Any


def _parse_token(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore").strip("\x00").strip()


def parse_snap_quote_binary(data: bytes | bytearray) -> dict[str, Any] | None:
    """Parse SmartWebSocketV2 binary frames (LTP / QUOTE / SNAP_QUOTE).

    Mirrors ``SmartWebSocketV2._parse_binary_data`` field layout so ticks decode
    even when the SDK passes raw bytes to ``on_data``.
    """
    if len(data) < 51:
        return None
    try:
        subscription_mode = struct.unpack_from("<B", data, 0)[0]
        exchange_type = struct.unpack_from("<B", data, 1)[0]
        token = _parse_token(bytes(data[2:27]))
        sequence_number = struct.unpack_from("<q", data, 27)[0]
        exchange_timestamp = struct.unpack_from("<q", data, 35)[0]
        last_traded_price = struct.unpack_from("<q", data, 43)[0]
        parsed: dict[str, Any] = {
            "subscription_mode": subscription_mode,
            "exchange_type": exchange_type,
            "token": token,
            "sequence_number": sequence_number,
            "exchange_timestamp": exchange_timestamp,
            "last_traded_price": last_traded_price,
        }
        # QUOTE / SNAP_QUOTE (modes 2 and 3)
        if subscription_mode in (2, 3) and len(data) >= 123:
            parsed["last_traded_quantity"] = struct.unpack_from("<q", data, 51)[0]
            parsed["average_traded_price"] = struct.unpack_from("<q", data, 59)[0]
            parsed["volume_trade_for_the_day"] = struct.unpack_from("<q", data, 67)[0]
            parsed["open_price_of_the_day"] = struct.unpack_from("<q", data, 91)[0]
            parsed["high_price_of_the_day"] = struct.unpack_from("<q", data, 99)[0]
            parsed["low_price_of_the_day"] = struct.unpack_from("<q", data, 107)[0]
            parsed["closed_price"] = struct.unpack_from("<q", data, 115)[0]
        return parsed
    except struct.error:
        return None


def decode_ws_message(message: Any) -> dict[str, Any] | None:
    """
    Normalize SmartWebSocketV2 ``on_data`` payloads.

    The official client may pass a parsed dict or raw bytes depending on version.
    Returns None for heartbeats / unparseable frames.
    """
    if message is None:
        return None
    if isinstance(message, dict):
        return message
    if isinstance(message, (bytes, bytearray)):
        try:
            text = message.decode("utf-8")
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        parsed = parse_snap_quote_binary(message)
        if parsed is not None:
            return parsed
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2

            ws = SmartWebSocketV2("", "", "", "")
            if hasattr(ws, "_parse_binary_data"):
                return ws._parse_binary_data(message)  # type: ignore[attr-defined]
        except Exception:
            pass
    return None
