"""Polling runtime for the Fortuna Telegram assistant."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fortuna.agentic.contracts import MediaAttachment
from fortuna.app.forecasting import cleanup_forecast_artifacts
from fortuna.config.settings import Settings
from fortuna.observability.recorder import get_workflow_recorder
from fortuna.telegram.assistant import InteractionResult, TelegramAnalysisAssistant
from fortuna.telegram.client import TelegramBotClient
from fortuna.telegram.parser import parse_telegram_request
from fortuna.telegram.request_audit import (
    TelegramRequestAuditStore,
    build_audit_entry,
)
from fortuna.telegram.router import route_telegram_request
from fortuna.telegram.session_store import (
    TelegramConversationState,
    TelegramSessionStore,
    evolve_state,
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
        session_store: Optional[TelegramSessionStore] = None,
    ) -> None:
        self.settings = settings
        self.client = client or TelegramBotClient(settings.telegram_bot_token)
        self.assistant = assistant or TelegramAnalysisAssistant(settings)
        self.state_path = state_path or settings.resolve_path(Path("logs/telegram_bot/state.json"))
        self.session_store = session_store or TelegramSessionStore(
            settings.resolve_path(Path("logs/telegram/runtime.sqlite3"))
        )
        self._audit_store = audit_store
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(getattr(settings, "signal_max_concurrent_requests", 4) or 4)),
            thread_name_prefix="fortuna-telegram",
        )
        if audit_store is None and settings.telegram_request_audit_enabled:
            self._audit_store = TelegramRequestAuditStore.from_settings(settings)
        self._cleanup_retained_artifacts()

    def run_forever(self, *, poll_timeout: int = 25, sleep_seconds: float = 1.0) -> None:
        offset = self._load_offset()
        futures: set[concurrent.futures.Future[None]] = set()
        while True:
            updates = self.client.get_updates(offset=offset, timeout=poll_timeout)
            for upd in updates:
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                futures.add(self._executor.submit(self._handle_update, upd))
                self._save_offset(offset)
            futures = {future for future in futures if not future.done()}
            time.sleep(max(0.0, float(sleep_seconds)))

    def _handle_update(self, update: dict) -> None:
        msg = update.get("message") or {}
        text = str(msg.get("text") or "").strip()
        caption = str(msg.get("caption") or "").strip()
        attachment = self._download_attachment(msg)
        effective_text = text or caption or ("analyze this chart" if attachment is not None else "")
        if not effective_text:
            return
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id") or "")
        if not self._is_allowed_chat(chat_id):
            return

        request_id = uuid.uuid4().hex[:16]
        update_id = update.get("update_id")
        admission = self.session_store.try_begin_request(
            chat_id=chat_id,
            request_id=request_id,
            max_in_flight=max(
                1, int(getattr(self.settings, "signal_max_concurrent_per_user", 1) or 1)
            ),
            max_pending=max(0, int(getattr(self.settings, "signal_max_pending_per_user", 1) or 1)),
        )
        if not admission.accepted:
            interaction = self._build_busy_interaction(text=text, request_id=request_id)
            self._deliver_interaction(
                chat_id=chat_id,
                update_id=update_id,
                request_id=request_id,
                interaction=interaction,
            )
            self._emit_runtime_event(
                request_id=request_id,
                chat_id=chat_id,
                status="busy",
                interaction=interaction,
            )
            self._cleanup_attachment(attachment)
            return

        state = self.session_store.get_state(chat_id)
        interaction, request_status = self._run_interaction(
            text=effective_text,
            chat_id=chat_id,
            request_id=request_id,
            state=state,
            attachment=attachment,
        )
        try:
            self._deliver_interaction(
                chat_id=chat_id,
                update_id=update_id,
                request_id=request_id,
                interaction=interaction,
            )
            self._emit_runtime_event(
                request_id=request_id,
                chat_id=chat_id,
                status=request_status,
                interaction=interaction,
            )
            self._persist_conversation_state(
                state,
                interaction=interaction,
                request_text=effective_text,
            )
        finally:
            self._cleanup_attachment(attachment)
            self.session_store.finish_request(
                chat_id=chat_id,
                request_id=request_id,
                status=request_status,
            )

    def _deliver_interaction(
        self,
        *,
        chat_id: str,
        update_id: int | None,
        request_id: str,
        interaction,
    ) -> None:
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

    def _is_allowed_chat(self, chat_id: str) -> bool:
        allowed = self.settings.telegram_allowed_chat_id_set()
        return not allowed or str(chat_id) in allowed

    def _is_admin_chat(self, chat_id: str) -> bool:
        return str(chat_id) in self.settings.telegram_admin_chat_id_set()

    def _timeout_seconds_for(self, text: str) -> int:
        req = parse_telegram_request(text)
        if req.kind.value in {"workflow", "research", "allocate", "candidates", "brief", "health"}:
            return max(
                1,
                int(getattr(self.settings, "signal_slow_command_timeout_seconds", 45) or 45),
            )
        return max(1, int(getattr(self.settings, "signal_timeout_seconds", 18) or 18))

    def _run_interaction(
        self,
        *,
        text: str,
        chat_id: str,
        request_id: str,
        state: TelegramConversationState,
        attachment: MediaAttachment | None = None,
    ):
        holder: dict[str, object] = {}
        failure: dict[str, BaseException] = {}

        def _target() -> None:
            try:
                try:
                    holder["result"] = self.assistant.handle_interaction(
                        text,
                        chat_id=chat_id,
                        request_id=request_id,
                        conversation_state=state,
                        is_admin=self._is_admin_chat(chat_id),
                        attachment=attachment,
                    )
                except TypeError:
                    holder["result"] = self.assistant.handle_interaction(text)
            except BaseException as exc:  # noqa: BLE001
                failure["exc"] = exc

        timeout_seconds = self._timeout_seconds_for(text)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_target)
        try:
            future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            return self._build_timeout_interaction(text=text, request_id=request_id), "timeout"
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if "exc" in failure:
            return self._build_failure_interaction(
                text=text,
                request_id=request_id,
                error=str(failure["exc"]),
            ), "error"
        return holder["result"], "completed"

    def _persist_conversation_state(
        self,
        state: TelegramConversationState,
        *,
        interaction,
        request_text: str,
    ) -> None:
        request = interaction.request
        resolved_symbol = str(interaction.resolved_symbol or "")
        symbol = resolved_symbol or str(request.symbol or "")
        next_state = evolve_state(
            state,
            request_text=request_text,
            symbol=symbol,
            timeframe=str(request.timeframe or state.last_timeframe or "5m"),
            days=max(1, int(request.days or state.last_days or 30)),
            resolved_symbol=resolved_symbol,
        )
        self.session_store.save_state(next_state)

    def _build_busy_interaction(self, *, text: str, request_id: str):
        req = parse_telegram_request(text)
        route = route_telegram_request(req)
        return InteractionResult(
            reply="Still processing your previous request. Please wait for that reply first.",
            request=req,
            route=route,
            source="command",
            source_confidence=1.0,
            source_rationale="per-user concurrency guard rejected overlapping request",
            error_code="busy",
            request_id=request_id,
        )

    def _build_timeout_interaction(self, *, text: str, request_id: str):
        req = parse_telegram_request(text)
        route = route_telegram_request(req)
        return InteractionResult(
            reply=(
                "This request is taking too long right now. Fortuna returned a degraded reply "
                "instead of hanging. Please retry in a few seconds."
            ),
            request=req,
            route=route,
            source="command",
            source_confidence=1.0,
            source_rationale="request timed out in Telegram runtime",
            error_code="timeout",
            request_id=request_id,
        )

    def _build_failure_interaction(self, *, text: str, request_id: str, error: str):
        req = parse_telegram_request(text)
        route = route_telegram_request(req)
        return InteractionResult(
            reply=(
                "Fortuna hit an internal error while processing that request. "
                "Deterministic fallback is still available; please retry once."
            ),
            request=req,
            route=route,
            source="command",
            source_confidence=1.0,
            source_rationale=f"request failed inside Telegram runtime: {error}",
            error_code="internal",
            request_id=request_id,
        )

    def _emit_runtime_event(
        self,
        *,
        request_id: str,
        chat_id: str,
        status: str,
        interaction,
    ) -> None:
        try:
            recorder = get_workflow_recorder(self.settings)
            recorder.emit_step(
                event_name="telegram_request",
                module="fortuna.telegram.runtime",
                workflow_id="telegram_signal_runtime",
                run_id=request_id,
                status="ok" if status == "completed" else "warn",
                symbol=interaction.resolved_symbol or interaction.request.symbol or None,
                context={
                    "chat_id": str(chat_id),
                    "route_action": interaction.route.action.value,
                    "request_status": status,
                    "tool": interaction.tool,
                    "tool_ok": interaction.tool_ok,
                    "error_code": interaction.error_code,
                    "source": interaction.source,
                    "decision_action": interaction.decision_action,
                    "modality": getattr(interaction, "modality", "text"),
                    "attachment_kind": getattr(interaction, "attachment_kind", None),
                    "forecast_used": (
                        bool(interaction.signal_response.forecast_used)
                        if interaction.signal_response is not None
                        else None
                    ),
                    "layer_deterministic": (
                        interaction.signal_response.used_layers.deterministic
                        if interaction.signal_response is not None
                        else None
                    ),
                    "layer_ml": (
                        interaction.signal_response.used_layers.ml
                        if interaction.signal_response is not None
                        else None
                    ),
                    "layer_rl": (
                        interaction.signal_response.used_layers.rl
                        if interaction.signal_response is not None
                        else None
                    ),
                },
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

    def _download_attachment(self, message: dict) -> MediaAttachment | None:
        photo = message.get("photo") or []
        if photo:
            chosen = max(photo, key=lambda item: int(item.get("file_size", 0) or 0))
            return self._materialize_attachment(
                file_id=str(chosen.get("file_id") or ""),
                kind="image",
                mime_type="image/jpeg",
                file_name="telegram_photo.jpg",
                size_bytes=int(chosen.get("file_size", 0) or 0),
            )
        document = message.get("document") or {}
        mime_type = str(document.get("mime_type") or "").lower()
        if document and mime_type.startswith("image/"):
            return self._materialize_attachment(
                file_id=str(document.get("file_id") or ""),
                kind="image",
                mime_type=mime_type,
                file_name=str(document.get("file_name") or "telegram_image"),
                size_bytes=int(document.get("file_size", 0) or 0),
            )
        return None

    def _materialize_attachment(
        self,
        *,
        file_id: str,
        kind: str,
        mime_type: str,
        file_name: str,
        size_bytes: int,
    ) -> MediaAttachment | None:
        if not file_id:
            return None
        max_bytes = max(
            1024,
            int(getattr(self.settings, "signal_image_max_bytes", 3_000_000) or 3_000_000),
        )
        if size_bytes and size_bytes > max_bytes:
            return None
        file_meta = self.client.get_file(file_id)
        remote_path = str(file_meta.get("file_path") or "").strip()
        if not remote_path:
            return None
        content = self.client.download_file(remote_path)
        if len(content) > max_bytes:
            return None
        temp_dir = self.settings.resolve_path(self.settings.signal_image_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        suffix = Path(file_name).suffix or ".bin"
        local_path = temp_dir / f"{digest[:16]}{suffix}"
        local_path.write_bytes(content)
        return MediaAttachment(
            kind=kind,
            file_name=file_name,
            mime_type=mime_type,
            local_path=str(local_path),
            content_hash=digest,
            size_bytes=len(content),
        )

    def _cleanup_attachment(self, attachment: MediaAttachment | None) -> None:
        if attachment is None or not attachment.local_path:
            return
        try:
            Path(attachment.local_path).unlink(missing_ok=True)
        except OSError:
            pass

    def _cleanup_retained_artifacts(self) -> None:
        if self._audit_store is not None:
            try:
                self._audit_store.apply_retention(
                    keep_count=max(
                        100,
                        int(
                            getattr(
                                self.settings,
                                "signal_request_audit_retention_count",
                                5_000,
                            )
                            or 5_000
                        ),
                    ),
                    keep_days=max(
                        1,
                        int(
                            getattr(
                                self.settings,
                                "signal_request_audit_retention_days",
                                7,
                            )
                            or 7
                        ),
                    ),
                )
            except Exception:
                pass
        try:
            self.session_store.prune_stale(
                session_days=max(
                    1,
                    int(getattr(self.settings, "signal_session_retention_days", 30) or 30),
                )
            )
        except Exception:
            pass
        try:
            cleanup_forecast_artifacts(self.settings)
        except Exception:
            pass
        try:
            self._cleanup_temp_dir(
                self.settings.resolve_path(self.settings.signal_image_temp_dir),
                retention_minutes=max(
                    1,
                    int(
                        getattr(
                            self.settings,
                            "signal_image_temp_retention_minutes",
                            10,
                        )
                        or 10
                    ),
                ),
            )
        except Exception:
            pass

    def _cleanup_temp_dir(self, path: Path, *, retention_minutes: int) -> int:
        if retention_minutes <= 0 or not path.exists():
            return 0
        removed = 0
        cutoff = time.time() - (retention_minutes * 60)
        for child in path.iterdir():
            try:
                if child.stat().st_mtime > cutoff:
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue
        return removed
