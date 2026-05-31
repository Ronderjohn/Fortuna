"""Small Telegram Bot API client for polling and sending messages."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TelegramBotClient:
    bot_token: str
    opener: Optional[callable] = None
    timeout: float = 30.0

    def get_updates(
        self,
        *,
        offset: Optional[int] = None,
        timeout: int = 25,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        payload = urllib.parse.urlencode(params).encode("utf-8")
        doc = self._call("getUpdates", payload)
        return list(doc.get("result") or [])

    def send_message(self, chat_id: str, text: str) -> dict[str, Any]:
        payload = urllib.parse.urlencode(
            {
                "chat_id": str(chat_id),
                "text": text,
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")
        return self._call("sendMessage", payload)

    def _call(self, method: str, payload: bytes) -> dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        opener = self.opener or urllib.request.urlopen
        with opener(req, self.timeout) as resp:  # type: ignore[misc]
            body = resp.read().decode("utf-8")
        doc = json.loads(body) if body else {}
        if not bool(doc.get("ok")):
            raise RuntimeError(str(doc))
        return doc
