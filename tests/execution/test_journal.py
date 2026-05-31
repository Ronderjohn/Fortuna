"""OrderJournal: append, read, rotate, tail."""

from __future__ import annotations

import json
from datetime import date

from fortuna.execution.journal import OrderJournal, replay_events


def test_write_and_read_round_trip(tmp_path):
    j = OrderJournal(directory=tmp_path)
    j.write("order_placed", {"order_id": "X1", "symbol": "RELIANCE"})
    j.write("order_filled", {"order_id": "X1", "filled_price": 1.0})
    j.close()

    rows = j.read_today()
    assert len(rows) == 2
    assert rows[0]["type"] == "order_placed"
    assert rows[1]["type"] == "order_filled"
    assert rows[0]["order_id"] == "X1"


def test_tail_returns_last_n_only(tmp_path):
    j = OrderJournal(directory=tmp_path)
    for i in range(20):
        j.write("evt", {"i": i})
    j.close()
    last5 = j.tail(5)
    assert len(last5) == 5
    assert [r["i"] for r in last5] == [15, 16, 17, 18, 19]


def test_rotation_across_days(tmp_path):
    """When the calendar date changes, the journal opens a new file."""
    days = [date(2026, 1, 5), date(2026, 1, 6)]
    state = {"i": 0}

    def fake_today() -> date:
        return days[state["i"]]

    j = OrderJournal(directory=tmp_path, date_fn=fake_today)
    j.write("evt", {"day": 1})
    state["i"] = 1  # advance the calendar
    j.write("evt", {"day": 2})
    j.close()

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["2026-01-05.jsonl", "2026-01-06.jsonl"]


def test_replay_events_aggregates_multiple_files(tmp_path):
    """Useful for EOD reconciliation across symbols / sessions."""
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    f1.write_text(json.dumps({"type": "a"}) + "\n")
    f2.write_text(json.dumps({"type": "b"}) + "\n")
    rows = replay_events([f1, f2])
    assert [r["type"] for r in rows] == ["a", "b"]


def test_replay_skips_malformed_lines(tmp_path):
    f = tmp_path / "broken.jsonl"
    f.write_text("{not json\n" + json.dumps({"type": "ok"}) + "\n")
    rows = replay_events([f])
    assert rows == [{"type": "ok"}]


def test_journal_serializes_datetime_via_default_encoder(tmp_path):
    from datetime import datetime

    j = OrderJournal(directory=tmp_path)
    j.write("evt", {"ts": datetime(2026, 1, 6, 9, 30)})
    j.close()
    rows = j.read_today()
    assert rows[0]["ts"].startswith("2026-01-06T09:30")
