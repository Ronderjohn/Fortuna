"""Decode SmartAPI WebSocket 2.0 messages to tick dicts for BarAggregator."""

from __future__ import annotations

import json
from typing import Any


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
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2

            if hasattr(SmartWebSocketV2, "parse_binary_data"):
                return SmartWebSocketV2.parse_binary_data(message)  # type: ignore[attr-defined]
        except Exception:
            pass
    return None
