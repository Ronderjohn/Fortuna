"""Notification adapters for agentic decisions."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from fortuna.agentic.models import NotificationIntent, NotificationResult


class Notifier(ABC):
    @abstractmethod
    def send(self, intent: NotificationIntent) -> NotificationResult:
        ...


@dataclass
class TelegramNotifier(Notifier):
    """Minimal Telegram bot sender using stdlib HTTP."""

    bot_token: str
    chat_id: str
    opener: Optional[Callable[[urllib.request.Request, float], object]] = None
    timeout: float = 15.0

    def send(self, intent: NotificationIntent) -> NotificationResult:
        if not all([self.bot_token, self.chat_id]):
            return NotificationResult(
                sent=False,
                provider="telegram",
                error="Telegram settings are incomplete",
            )
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = urllib.parse.urlencode(
            {
                "chat_id": intent.destination or self.chat_id,
                "text": intent.message,
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            opener = self.opener or urllib.request.urlopen
            with opener(req, self.timeout) as resp:  # type: ignore[misc]
                body = resp.read().decode("utf-8")
            doc = json.loads(body) if body else {}
            result = doc.get("result") or {}
            return NotificationResult(
                sent=True,
                provider="telegram",
                message_id=str(result.get("message_id", "")) or None,
            )
        except Exception as exc:  # noqa: BLE001
            return NotificationResult(
                sent=False,
                provider="telegram",
                error=str(exc),
            )


class NullNotifier(Notifier):
    def send(self, intent: NotificationIntent) -> NotificationResult:
        return NotificationResult(sent=False, provider="none", error="disabled")
