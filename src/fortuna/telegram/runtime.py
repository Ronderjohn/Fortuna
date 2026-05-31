"""Polling runtime for the Fortuna Telegram assistant."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fortuna.config.settings import Settings
from fortuna.telegram.assistant import TelegramAnalysisAssistant
from fortuna.telegram.client import TelegramBotClient
from fortuna.telegram.request_audit import (
    TelegramRequestAuditStore,
    build_audit_entry,
)


class TelegramBotRuntime:
    def __init__(
        self,
        settings: Settings,
        *,
        client: Optional[TelegramBotClient] = None,
        assistant: Optional[TelegramAnalysisAssistant] = None,
        state_path: Optional[Path] = None,
        audit_store: Optional[TelegramRequestAuditStore] = None,
    ) -> None:
        self.settings = settings
        self.client = client or TelegramBotClient(settings.telegram_bot_token)
        self.assistant = assistant or TelegramAnalysisAssistant(settings)
        self.state_path = state_path or settings.resolve_path(Path("logs/telegram_bot/state.json"))
        self._audit_store = audit_store
        if audit_store is None and settings.telegram_request_audit_enabled:
            self._audit_store = TelegramRequestAuditStore.from_settings(settings)

    def run_forever(self, *, poll_timeout: int = 25, sleep_seconds: float = 1.0) -> None:
        offset = self._load_offset()
        while True:
            updates = self.client.get_updates(offset=offset, timeout=poll_timeout)
            for upd in updates:
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                self._handle_update(upd)
                self._save_offset(offset)
            time.sleep(max(0.0, float(sleep_seconds)))

    def _handle_update(self, update: dict) -> None:
        msg = update.get("message") or {}
        text = str(msg.get("text") or "").strip()
        if not text:
            return
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id") or "")
        if self.settings.telegram_chat_id and chat_id != str(self.settings.telegram_chat_id):
            return

        request_id = uuid.uuid4().hex[:16]
        update_id = update.get("update_id")
        interaction = self.assistant.handle_interaction(text)

        delivery_ok: bool | None = None
        delivery_error: str | None = None
        try:
            self.client.send_message(chat_id, interaction.reply)
            delivery_ok = True
        except Exception as exc:
            delivery_ok = False
            delivery_error = str(exc)

        if self._audit_store is not None:
            try:
                self._audit_store.append(
                    build_audit_entry(
                        request_id=request_id,
                        chat_id=chat_id,
                        update_id=int(update_id) if update_id is not None else None,
                        interaction=interaction,
                        delivery_ok=delivery_ok,
                        delivery_error=delivery_error,
                    )
                )
            except Exception:
                pass

    def _load_offset(self) -> int:
        path = self.state_path
        if not path.is_file():
            return 0
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            return int(doc.get("offset", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0

    def _save_offset(self, path_offset: int) -> None:
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"offset": int(path_offset)}), encoding="utf-8")
