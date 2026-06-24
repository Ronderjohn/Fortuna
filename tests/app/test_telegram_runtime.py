from __future__ import annotations

from pathlib import Path

from fortuna.config.settings import Settings
from fortuna.telegram.assistant import InteractionResult
from fortuna.telegram.parser import parse_telegram_request
from fortuna.telegram.request_audit import TelegramRequestAuditStore
from fortuna.telegram.router import route_telegram_request
from fortuna.telegram.runtime import TelegramBotRuntime


class _FakeClient:
    def __init__(self, *, fail_send: bool = False):
        self.sent = []
        self.fail_send = fail_send
        self.file_bytes = b"fake-image-bytes"
        self.updates = [
            {
                "update_id": 10,
                "message": {
                    "text": "/help",
                    "chat": {"id": "12345"},
                },
            }
        ]

    def get_updates(self, *, offset=None, timeout=25):
        out = list(self.updates)
        self.updates = []
        return out

    def send_message(self, chat_id, text):
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append((chat_id, text))
        return {"ok": True}

    def get_file(self, file_id):
        return {"file_path": f"photos/{file_id}.jpg"}

    def download_file(self, file_path):
        return self.file_bytes


class _FakeAssistant:
    def handle_interaction(self, text: str, **kwargs) -> InteractionResult:
        req = parse_telegram_request(text)
        route = route_telegram_request(req)
        attachment = kwargs.get("attachment")
        return InteractionResult(
            reply=f"handled::{text}",
            request=req,
            route=route,
            modality="image" if attachment is not None else "text",
            attachment_kind=(attachment.kind if attachment is not None else None),
        )


def _settings(tmp_path: Path, *, audit_enabled: bool = True) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        telegram_enabled=True,
        telegram_bot_token="x",
        telegram_chat_id="12345",
        telegram_allowed_chat_ids="",
        telegram_admin_chat_ids="",
        telegram_request_audit_enabled=audit_enabled,
    )


def test_runtime_handles_update_and_persists_offset(tmp_path: Path):
    settings = _settings(tmp_path, audit_enabled=False)
    client = _FakeClient()
    state_path = tmp_path / "telegram" / "state.json"
    runtime = TelegramBotRuntime(
        settings,
        client=client,
        assistant=_FakeAssistant(),
        state_path=state_path,
    )
    runtime._handle_update(client.updates[0])
    runtime._save_offset(11)

    assert client.sent == [("12345", "handled::/help")]
    assert state_path.exists()
    assert runtime._load_offset() == 11


def test_runtime_writes_audit_row_for_accepted_help(tmp_path: Path):
    settings = _settings(tmp_path)
    audit_path = tmp_path / "logs" / "telegram" / "requests.jsonl"
    audit_store = TelegramRequestAuditStore(audit_path)
    client = _FakeClient()
    runtime = TelegramBotRuntime(
        settings,
        client=client,
        assistant=_FakeAssistant(),
        audit_store=audit_store,
    )
    runtime._handle_update(
        {
            "update_id": 10,
            "message": {"text": "/help", "chat": {"id": "12345"}},
        }
    )

    rows = audit_store.read_recent(limit=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["chat_id"] == "12345"
    assert row["update_id"] == 10
    assert row["parsed_kind"] == "help"
    assert row["route_action"] == "show_help"
    assert row["delivery_ok"] is True
    assert "request_id" in row


def test_runtime_skips_audit_for_wrong_chat_id(tmp_path: Path):
    settings = _settings(tmp_path)
    audit_path = tmp_path / "logs" / "telegram" / "requests.jsonl"
    audit_store = TelegramRequestAuditStore(audit_path)
    client = _FakeClient()
    runtime = TelegramBotRuntime(
        settings,
        client=client,
        assistant=_FakeAssistant(),
        audit_store=audit_store,
    )
    runtime._handle_update(
        {
            "update_id": 11,
            "message": {"text": "/help", "chat": {"id": "99999"}},
        }
    )

    assert audit_store.read_recent(limit=10) == []
    assert client.sent == []


def test_runtime_audit_records_delivery_failure(tmp_path: Path):
    settings = _settings(tmp_path)
    audit_path = tmp_path / "logs" / "telegram" / "requests.jsonl"
    audit_store = TelegramRequestAuditStore(audit_path)
    client = _FakeClient(fail_send=True)
    runtime = TelegramBotRuntime(
        settings,
        client=client,
        assistant=_FakeAssistant(),
        audit_store=audit_store,
    )
    runtime._handle_update(
        {
            "update_id": 12,
            "message": {"text": "/help", "chat": {"id": "12345"}},
        }
    )

    row = audit_store.read_recent(limit=1)[0]
    assert row["delivery_ok"] is False
    assert "send failed" in row["delivery_error"]
    assert client.sent == []


def test_runtime_skips_audit_store_when_disabled(tmp_path: Path):
    settings = _settings(tmp_path, audit_enabled=False)
    runtime = TelegramBotRuntime(settings, client=_FakeClient(), assistant=_FakeAssistant())
    assert runtime._audit_store is None


def test_runtime_offset_restored_on_restart(tmp_path: Path):
    settings = _settings(tmp_path, audit_enabled=False)
    state_path = tmp_path / "telegram" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text('{"offset": 42}', encoding="utf-8")

    runtime = TelegramBotRuntime(
        settings,
        client=_FakeClient(),
        assistant=_FakeAssistant(),
        state_path=state_path,
    )
    assert runtime._load_offset() == 42


def test_runtime_polling_loop_persists_offset_per_update(tmp_path: Path):
    settings = _settings(tmp_path, audit_enabled=False)
    client = _FakeClient()
    state_path = tmp_path / "telegram" / "state.json"
    runtime = TelegramBotRuntime(
        settings,
        client=client,
        assistant=_FakeAssistant(),
        state_path=state_path,
    )

    offset = runtime._load_offset()
    updates = [
        {
            "update_id": 10,
            "message": {"text": "/help", "chat": {"id": "12345"}},
        },
        {
            "update_id": 11,
            "message": {"text": "/help", "chat": {"id": "12345"}},
        },
    ]
    for upd in updates:
        offset = max(offset, int(upd.get("update_id", 0)) + 1)
        runtime._handle_update(upd)
        runtime._save_offset(offset)

    assert offset == 12
    assert runtime._load_offset() == 12
    assert len(client.sent) == 2


def test_runtime_accepts_any_chat_when_chat_id_unconfigured(tmp_path: Path):
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        telegram_enabled=True,
        telegram_bot_token="x",
        telegram_chat_id="",
        telegram_allowed_chat_ids="",
        telegram_admin_chat_ids="",
        telegram_request_audit_enabled=False,
    )
    client = _FakeClient()
    runtime = TelegramBotRuntime(
        settings,
        client=client,
        assistant=_FakeAssistant(),
    )
    runtime._handle_update(
        {
            "update_id": 20,
            "message": {"text": "/help", "chat": {"id": "99999"}},
        }
    )

    assert client.sent == [("99999", "handled::/help")]


def test_runtime_accepts_multiple_allowed_chats(tmp_path: Path):
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        telegram_enabled=True,
        telegram_bot_token="x",
        telegram_chat_id="",
        telegram_allowed_chat_ids="12345,22222",
        telegram_admin_chat_ids="",
        telegram_request_audit_enabled=False,
    )
    client = _FakeClient()
    runtime = TelegramBotRuntime(settings, client=client, assistant=_FakeAssistant())

    runtime._handle_update(
        {
            "update_id": 30,
            "message": {"text": "/help", "chat": {"id": "22222"}},
        }
    )
    runtime._handle_update(
        {
            "update_id": 31,
            "message": {"text": "/help", "chat": {"id": "99999"}},
        }
    )

    assert client.sent == [("22222", "handled::/help")]


def test_runtime_persists_per_user_state_without_leakage(tmp_path: Path):
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        telegram_enabled=True,
        telegram_bot_token="x",
        telegram_chat_id="",
        telegram_allowed_chat_ids="",
        telegram_admin_chat_ids="",
        telegram_request_audit_enabled=False,
    )
    client = _FakeClient()
    runtime = TelegramBotRuntime(settings, client=client, assistant=_FakeAssistant())

    runtime._handle_update(
        {
            "update_id": 40,
            "message": {"text": "/analyze RELIANCE 15m 20d", "chat": {"id": "11111"}},
        }
    )
    runtime._handle_update(
        {
            "update_id": 41,
            "message": {"text": "/analyze SBIN 5m 10d", "chat": {"id": "22222"}},
        }
    )

    state_a = runtime.session_store.get_state("11111")
    state_b = runtime.session_store.get_state("22222")
    assert state_a.last_symbol == "RELIANCE"
    assert state_a.last_timeframe == "15m"
    assert state_a.last_days == 20
    assert state_b.last_symbol == "SBIN"
    assert state_b.last_timeframe == "5m"
    assert state_b.last_days == 10


def test_runtime_handles_photo_update_and_cleans_temp_file(tmp_path: Path):
    settings = Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        telegram_enabled=True,
        telegram_bot_token="x",
        telegram_chat_id="12345",
        telegram_allowed_chat_ids="",
        telegram_admin_chat_ids="",
        telegram_request_audit_enabled=True,
        signal_image_input_enabled=True,
        signal_image_temp_dir=tmp_path / "tmp" / "telegram_images",
    )
    audit_store = TelegramRequestAuditStore(tmp_path / "logs" / "telegram" / "requests.jsonl")
    client = _FakeClient()
    runtime = TelegramBotRuntime(
        settings,
        client=client,
        assistant=_FakeAssistant(),
        audit_store=audit_store,
    )

    runtime._handle_update(
        {
            "update_id": 50,
            "message": {
                "caption": "What do you think?",
                "photo": [{"file_id": "abc123", "file_size": 24}],
                "chat": {"id": "12345"},
            },
        }
    )

    assert client.sent == [("12345", "handled::What do you think?")]
    rows = audit_store.read_recent(limit=1)
    assert rows[0]["modality"] == "image"
    assert rows[0]["attachment_kind"] == "image"
    temp_dir = settings.resolve_path(settings.signal_image_temp_dir)
    assert not list(temp_dir.glob("*"))
