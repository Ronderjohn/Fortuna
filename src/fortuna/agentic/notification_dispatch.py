"""Dispatch advisory notifications with policy, dedupe, and audit."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Optional

from fortuna.agentic.models import AgentDecision, NotificationResult
from fortuna.agentic.notification_policy import NotificationPolicyEngine
from fortuna.agentic.notifiers import Notifier
from fortuna.agentic.store import NotificationAuditStore


class NotificationDispatcher:
    """Apply dedupe, throttle policy, send, and audit notification attempts."""

    def __init__(
        self,
        notifier: Optional[Notifier],
        *,
        policy_engine: NotificationPolicyEngine,
        audit_store: Optional[NotificationAuditStore] = None,
        dedupe_keys: Optional[set[str]] = None,
        dedupe_limit: int = 500,
    ) -> None:
        self._notifier = notifier
        self._policy = policy_engine
        self._audit = audit_store
        self._dedupe_keys = dedupe_keys if dedupe_keys is not None else set()
        self._dedupe_limit = max(1, int(dedupe_limit))
        self._dedupe_order: deque[str] = deque()

    def dispatch(self, decision: AgentDecision) -> AgentDecision:
        if self._notifier is None or decision.notification is None:
            return decision

        intent = decision.notification
        key = intent.dedupe_key
        provider = getattr(self._notifier, "__class__", type(self._notifier)).__name__
        if hasattr(self._notifier, "send"):
            if "Telegram" in provider:
                provider = "telegram"
            else:
                provider = provider.lower()

        if key in self._dedupe_keys:
            result = NotificationResult(
                sent=False,
                provider=provider,
                deduped=True,
                policy_reason="deduped",
            )
            self._audit_attempt(decision, result, message_preview=intent.message)
            return decision.with_notification_result(result)

        policy = self._policy.evaluate(
            symbol=intent.symbol,
            bar_time=decision.bar_time,
        )
        if not policy.allowed:
            result = NotificationResult(
                sent=False,
                provider=provider,
                throttled=True,
                policy_reason=policy.reason,
            )
            self._audit_attempt(decision, result, message_preview=intent.message)
            return decision.with_notification_result(result)

        result = self._notifier.send(intent)
        if not result.sent and result.error and not result.policy_reason:
            reason = (
                "incomplete_credentials"
                if "incomplete" in (result.error or "").lower()
                else "send_error"
            )
            result = NotificationResult(
                sent=result.sent,
                provider=result.provider or provider,
                message_id=result.message_id,
                error=result.error,
                policy_reason=reason,
            )
        else:
            self._remember_dedupe_key(key)
            if result.sent:
                self._policy.record_sent(symbol=intent.symbol, bar_time=decision.bar_time)

        self._audit_attempt(decision, result, message_preview=intent.message)
        return decision.with_notification_result(result)

    def load_dedupe_keys(self, keys: set[str]) -> None:
        for key in keys:
            self._remember_dedupe_key(key, skip_audit=True)

    def _remember_dedupe_key(self, key: str, *, skip_audit: bool = False) -> None:
        if key in self._dedupe_keys:
            return
        self._dedupe_keys.add(key)
        self._dedupe_order.append(key)
        while len(self._dedupe_keys) > self._dedupe_limit:
            old = self._dedupe_order.popleft()
            self._dedupe_keys.discard(old)
        if skip_audit:
            return

    def _audit_attempt(
        self,
        decision: AgentDecision,
        result: NotificationResult,
        *,
        message_preview: str,
    ) -> None:
        if self._audit is None:
            return
        intent = decision.notification
        if intent is None:
            return
        self._audit.append(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "decision_hash": decision.decision_hash,
                "dedupe_key": intent.dedupe_key,
                "symbol": intent.symbol,
                "action": intent.action.value,
                "bar_time": decision.bar_time.isoformat() if decision.bar_time else None,
                "sent": bool(result.sent),
                "deduped": bool(result.deduped),
                "throttled": bool(result.throttled),
                "provider": result.provider,
                "message_id": result.message_id,
                "error": result.error,
                "policy_reason": result.policy_reason,
                "message_preview": message_preview[:200],
            }
        )
