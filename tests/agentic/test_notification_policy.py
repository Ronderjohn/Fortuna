from __future__ import annotations

from datetime import datetime

import pytest

from fortuna.agentic.notification_policy import NotificationPolicy, NotificationPolicyEngine


@pytest.fixture
def engine() -> NotificationPolicyEngine:
    return NotificationPolicyEngine(
        policy=NotificationPolicy(
            max_per_symbol_per_session=2,
            min_interval_seconds=60,
            quiet_hours_enabled=True,
        )
    )


def test_quiet_hours_blocks_outside_nse_session(engine):
    bar_time = datetime(2026, 1, 6, 8, 0)  # before 09:15 IST
    decision = engine.evaluate(symbol="RELIANCE.NS", bar_time=bar_time)
    assert decision.allowed is False
    assert decision.reason == "quiet_hours"


def test_allows_during_nse_session(engine):
    bar_time = datetime(2026, 1, 6, 10, 0)
    decision = engine.evaluate(symbol="RELIANCE.NS", bar_time=bar_time)
    assert decision.allowed is True


def test_throttle_interval_blocks_second_send(engine):
    bar_time = datetime(2026, 1, 6, 10, 0)
    now = 1000.0
    assert engine.evaluate(symbol="RELIANCE.NS", bar_time=bar_time, now=now).allowed
    engine.record_sent("RELIANCE.NS", bar_time, now=now)
    blocked = engine.evaluate(symbol="RELIANCE.NS", bar_time=bar_time, now=now + 10)
    assert blocked.allowed is False
    assert blocked.reason == "throttled_interval"


def test_throttle_count_blocks_after_max(engine):
    bar_time = datetime(2026, 1, 6, 10, 0)
    base = 2000.0
    for i in range(2):
        assert engine.evaluate(symbol="RELIANCE.NS", bar_time=bar_time, now=base + i * 100).allowed
        engine.record_sent("RELIANCE.NS", bar_time, now=base + i * 100)
    blocked = engine.evaluate(symbol="RELIANCE.NS", bar_time=bar_time, now=base + 300)
    assert blocked.allowed is False
    assert blocked.reason == "throttled_count"


def test_different_symbols_independent(engine):
    bar_time = datetime(2026, 1, 6, 10, 0)
    engine.record_sent("RELIANCE.NS", bar_time, now=1000.0)
    other = engine.evaluate(symbol="TCS.NS", bar_time=bar_time, now=1001.0)
    assert other.allowed is True
