"""SQLite-backed per-chat session state for Telegram signal handling."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class TelegramConversationState:
    chat_id: str
    last_symbol: str = ""
    last_timeframe: str = "5m"
    last_days: int = 30
    last_request_text: str = ""
    last_resolved_symbol: str = ""
    recent_history: tuple[str, ...] = ()
    updated_at: str = ""


@dataclass(frozen=True)
class RequestAdmission:
    accepted: bool
    request_id: str
    chat_id: str
    reason: str = ""
    active_request_id: str | None = None


class TelegramSessionStore:
    """Persist small per-user conversational state and runtime coordination."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def get_state(self, chat_id: str) -> TelegramConversationState:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT last_symbol, last_timeframe, last_days, last_request_text,
                       last_resolved_symbol, recent_history_json, updated_at
                FROM telegram_sessions
                WHERE chat_id = ?
                """,
                (str(chat_id),),
            ).fetchone()
        if row is None:
            return TelegramConversationState(chat_id=str(chat_id))
        history = _decode_history(row[5])
        return TelegramConversationState(
            chat_id=str(chat_id),
            last_symbol=str(row[0] or ""),
            last_timeframe=str(row[1] or "5m"),
            last_days=max(1, int(row[2] or 30)),
            last_request_text=str(row[3] or ""),
            last_resolved_symbol=str(row[4] or ""),
            recent_history=history,
            updated_at=str(row[6] or ""),
        )

    def save_state(self, state: TelegramConversationState) -> None:
        payload = list(state.recent_history[-20:])
        updated_at = state.updated_at or _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO telegram_sessions (
                    chat_id,
                    last_symbol,
                    last_timeframe,
                    last_days,
                    last_request_text,
                    last_resolved_symbol,
                    recent_history_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    last_symbol = excluded.last_symbol,
                    last_timeframe = excluded.last_timeframe,
                    last_days = excluded.last_days,
                    last_request_text = excluded.last_request_text,
                    last_resolved_symbol = excluded.last_resolved_symbol,
                    recent_history_json = excluded.recent_history_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(state.chat_id),
                    str(state.last_symbol or ""),
                    str(state.last_timeframe or "5m"),
                    max(1, int(state.last_days or 30)),
                    str(state.last_request_text or ""),
                    str(state.last_resolved_symbol or ""),
                    json.dumps(payload),
                    updated_at,
                ),
            )
            conn.commit()

    def try_begin_request(
        self,
        *,
        chat_id: str,
        request_id: str,
        max_in_flight: int = 1,
        max_pending: int = 1,
    ) -> RequestAdmission:
        del max_pending  # current runtime rejects rather than queues a second pending request
        with self._lock, self._connect() as conn:
            active_rows = conn.execute(
                """
                SELECT request_id
                FROM telegram_inflight_requests
                WHERE chat_id = ? AND status = 'running'
                ORDER BY started_at ASC
                """,
                (str(chat_id),),
            ).fetchall()
            if len(active_rows) >= max(1, int(max_in_flight)):
                active_request_id = str(active_rows[0][0] or "") if active_rows else None
                return RequestAdmission(
                    accepted=False,
                    request_id=request_id,
                    chat_id=str(chat_id),
                    reason="processing_previous_request",
                    active_request_id=active_request_id or None,
                )
            conn.execute(
                """
                INSERT INTO telegram_inflight_requests (
                    request_id,
                    chat_id,
                    status,
                    started_at
                )
                VALUES (?, ?, 'running', ?)
                """,
                (request_id, str(chat_id), _utc_now()),
            )
            conn.commit()
        return RequestAdmission(accepted=True, request_id=request_id, chat_id=str(chat_id))

    def finish_request(self, *, chat_id: str, request_id: str, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE telegram_inflight_requests
                SET status = ?, finished_at = ?
                WHERE chat_id = ? AND request_id = ?
                """,
                (str(status), _utc_now(), str(chat_id), str(request_id)),
            )
            conn.commit()

    def prune_stale(self, *, session_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(session_days)))
        cutoff_iso = cutoff.isoformat()
        with self._lock, self._connect() as conn:
            before_sessions = int(
                conn.execute("SELECT COUNT(*) FROM telegram_sessions").fetchone()[0]
            )
            before_requests = int(
                conn.execute("SELECT COUNT(*) FROM telegram_inflight_requests").fetchone()[0]
            )
            conn.execute(
                "DELETE FROM telegram_sessions WHERE updated_at != '' AND updated_at < ?",
                (cutoff_iso,),
            )
            conn.execute(
                """
                DELETE FROM telegram_inflight_requests
                WHERE started_at != '' AND started_at < ?
                """,
                (cutoff_iso,),
            )
            conn.commit()
            after_sessions = int(
                conn.execute("SELECT COUNT(*) FROM telegram_sessions").fetchone()[0]
            )
            after_requests = int(
                conn.execute("SELECT COUNT(*) FROM telegram_inflight_requests").fetchone()[0]
            )
        return (before_sessions - after_sessions) + (before_requests - after_requests)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_sessions (
                    chat_id TEXT PRIMARY KEY,
                    last_symbol TEXT NOT NULL DEFAULT '',
                    last_timeframe TEXT NOT NULL DEFAULT '5m',
                    last_days INTEGER NOT NULL DEFAULT 30,
                    last_request_text TEXT NOT NULL DEFAULT '',
                    last_resolved_symbol TEXT NOT NULL DEFAULT '',
                    recent_history_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_inflight_requests (
                    request_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_telegram_inflight_chat_status
                ON telegram_inflight_requests (chat_id, status, started_at)
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def evolve_state(
    state: TelegramConversationState,
    *,
    request_text: str,
    symbol: str = "",
    timeframe: str | None = None,
    days: int | None = None,
    resolved_symbol: str = "",
) -> TelegramConversationState:
    history = [*state.recent_history[-19:], str(request_text or "").strip()]
    return TelegramConversationState(
        chat_id=state.chat_id,
        last_symbol=str(symbol or state.last_symbol or ""),
        last_timeframe=str(timeframe or state.last_timeframe or "5m"),
        last_days=max(1, int(days if days is not None else state.last_days or 30)),
        last_request_text=str(request_text or ""),
        last_resolved_symbol=str(resolved_symbol or state.last_resolved_symbol or ""),
        recent_history=tuple(item for item in history if item),
        updated_at=_utc_now(),
    )


def _decode_history(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
