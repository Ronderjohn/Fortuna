from __future__ import annotations

from fortuna.agentic.store import NotificationAuditStore


def test_notification_audit_append_and_read(tmp_path):
    store = NotificationAuditStore(tmp_path)
    store.append({"dedupe_key": "a", "sent": True})
    store.append({"dedupe_key": "b", "sent": False, "deduped": True})

    rows = store.read_recent(10)
    assert len(rows) == 2
    assert rows[0]["dedupe_key"] == "a"
    assert rows[1]["dedupe_key"] == "b"


def test_recent_dedupe_keys_replay(tmp_path):
    store = NotificationAuditStore(tmp_path)
    for i in range(5):
        store.append({"dedupe_key": f"k{i}", "sent": True})

    keys = store.recent_dedupe_keys(limit=3)
    assert keys == {"k2", "k3", "k4"}


def test_recent_dedupe_keys_skip_failed_or_throttled_attempts(tmp_path):
    store = NotificationAuditStore(tmp_path)
    store.append({"dedupe_key": "sent-key", "sent": True})
    store.append({"dedupe_key": "deduped-key", "sent": False, "deduped": True})
    store.append({"dedupe_key": "throttled-key", "sent": False, "throttled": True})
    store.append({"dedupe_key": "error-key", "sent": False, "error": "boom"})

    keys = store.recent_dedupe_keys(limit=10)

    assert keys == {"sent-key", "deduped-key"}
