from __future__ import annotations

import json

from fortuna.agentic import ActionRecommendation, NotificationIntent
from fortuna.agentic.notifiers import TelegramNotifier


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({"ok": True, "result": {"message_id": 123}}).encode("utf-8")


def test_telegram_notifier_posts_form_payload():
    seen = {}

    def opener(req, timeout):
        seen["url"] = req.full_url
        seen["data"] = req.data.decode("utf-8")
        return FakeResponse()

    notifier = TelegramNotifier(
        bot_token="bot-secret",
        chat_id="123456",
        opener=opener,
    )
    result = notifier.send(
        NotificationIntent(
            symbol="RELIANCE.NS",
            action=ActionRecommendation.BUY,
            message="Fortuna: BUY RELIANCE.NS",
            dedupe_key="k",
        )
    )

    assert result.sent is True
    assert result.message_id == "123"
    assert "/botbot-secret/sendMessage" in seen["url"]
    assert "chat_id=123456" in seen["data"]
    assert "Fortuna%3A+BUY+RELIANCE.NS" in seen["data"]


def test_telegram_notifier_reports_incomplete_settings():
    notifier = TelegramNotifier(
        bot_token="",
        chat_id="",
    )

    result = notifier.send(
        NotificationIntent(
            symbol="RELIANCE.NS",
            action=ActionRecommendation.BUY,
            message="x",
            dedupe_key="k",
        )
    )

    assert result.sent is False
    assert "incomplete" in result.error.lower()
