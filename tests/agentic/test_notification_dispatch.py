from __future__ import annotations

from datetime import datetime

from fortuna.agentic import (
    ActionRecommendation,
    AgentDecision,
    DecisionRationale,
    NotificationIntent,
    NotificationResult,
)
from fortuna.agentic.notification_dispatch import NotificationDispatcher
from fortuna.agentic.notification_policy import NotificationPolicy, NotificationPolicyEngine
from fortuna.agentic.store import NotificationAuditStore


def _decision(*, dedupe_key: str = "k1", symbol: str = "RELIANCE.NS") -> AgentDecision:
    return AgentDecision(
        symbol=symbol,
        action=ActionRecommendation.BUY,
        confidence=0.8,
        bar_time=datetime(2026, 1, 6, 10, 0),
        bar_close=2500.0,
        rationale=DecisionRationale(summary="buy"),
        notification=NotificationIntent(
            symbol=symbol,
            action=ActionRecommendation.BUY,
            message="Fortuna: BUY",
            dedupe_key=dedupe_key,
        ),
    )


class _RecordingNotifier:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, intent: NotificationIntent) -> NotificationResult:
        self.calls += 1
        return NotificationResult(sent=True, provider="fake", message_id="msg1")


def test_dispatch_dedupes_repeat_key(tmp_path):
    audit = NotificationAuditStore(tmp_path)
    dispatcher = NotificationDispatcher(
        _RecordingNotifier(),
        policy_engine=NotificationPolicyEngine(
            policy=NotificationPolicy(quiet_hours_enabled=False)
        ),
        audit_store=audit,
        dedupe_keys=set(),
    )
    first = dispatcher.dispatch(_decision(dedupe_key="same"))
    second = dispatcher.dispatch(_decision(dedupe_key="same"))

    assert first.notification_result is not None
    assert first.notification_result.sent is True
    assert second.notification_result is not None
    assert second.notification_result.deduped is True
    assert len(audit.read_recent(10)) == 2


def test_dispatch_throttles_by_interval(tmp_path):
    notifier = _RecordingNotifier()
    policy = NotificationPolicyEngine(
        policy=NotificationPolicy(
            max_per_symbol_per_session=10,
            min_interval_seconds=3600,
            quiet_hours_enabled=False,
        )
    )
    audit = NotificationAuditStore(tmp_path)
    dispatcher = NotificationDispatcher(
        notifier,
        policy_engine=policy,
        audit_store=audit,
    )
    first = dispatcher.dispatch(_decision(dedupe_key="k1"))
    second = dispatcher.dispatch(_decision(dedupe_key="k2"))

    assert first.notification_result.sent is True
    assert second.notification_result.throttled is True
    assert second.notification_result.policy_reason == "throttled_interval"
    assert notifier.calls == 1


def test_different_action_same_symbol_allowed(tmp_path):
    notifier = _RecordingNotifier()
    dispatcher = NotificationDispatcher(
        notifier,
        policy_engine=NotificationPolicyEngine(
            policy=NotificationPolicy(quiet_hours_enabled=False, min_interval_seconds=0)
        ),
        audit_store=NotificationAuditStore(tmp_path),
    )
    buy = dispatcher.dispatch(_decision(dedupe_key="buy-key"))
    sell = _decision(dedupe_key="sell-key")
    sell = AgentDecision(
        symbol=sell.symbol,
        action=ActionRecommendation.SELL,
        confidence=sell.confidence,
        bar_time=sell.bar_time,
        bar_close=sell.bar_close,
        rationale=sell.rationale,
        notification=NotificationIntent(
            symbol=sell.symbol,
            action=ActionRecommendation.SELL,
            message="Fortuna: SELL",
            dedupe_key="sell-key",
        ),
    )
    sell = dispatcher.dispatch(sell)

    assert buy.notification_result.sent is True
    assert sell.notification_result.sent is True
    assert notifier.calls == 2
