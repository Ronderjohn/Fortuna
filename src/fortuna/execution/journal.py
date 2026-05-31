"""Append-only JSONL event journal under ``logs/execution/<DATE>.jsonl``.

Every order placement / fill / rejection / cancellation / risk-breach is
written as one JSON object per line. The dashboard's Monitor tab tails
the file; the EOD reconciliation script replays it.

Crash-safe: every write is followed by ``flush()`` so a process kill
loses at most the in-flight event, never previously-written history.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional


class OrderJournal:
    """Thread-safe append-only JSONL logger for execution events.

    Constructor accepts ``directory`` (defaults to ``logs/execution``)
    and lazily opens one file per calendar date (rotates at midnight
    when the first post-midnight write arrives).
    """

    def __init__(
        self,
        directory: str | Path = "logs/execution",
        *,
        date_fn=date.today,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._date_fn = date_fn
        self._current_date: Optional[date] = None
        self._fh = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- write
    def write(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one event to the journal; return the full row written."""
        row = {
            "ts": datetime.now().isoformat(timespec="microseconds"),
            "type": event_type,
            **payload,
        }
        line = json.dumps(row, default=_json_default) + "\n"
        with self._lock:
            self._rotate_if_needed()
            assert self._fh is not None
            self._fh.write(line)
            self._fh.flush()
        return row

    # ----------------------------------------------------------------- read
    def read_today(self) -> list[dict[str, Any]]:
        """Return all events written to today's journal (in file order)."""
        path = self._path_for(self._date_fn())
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def tail(self, n: int = 100) -> list[dict[str, Any]]:
        """Return the most recent ``n`` events from today's journal."""
        all_rows = self.read_today()
        return all_rows[-n:] if len(all_rows) > n else all_rows

    # ----------------------------------------------------------------- lifecycle
    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None
                    self._current_date = None

    def __enter__(self) -> "OrderJournal":
        return self

    def __exit__(self, *args, **kwargs) -> None:  # noqa: D401
        self.close()

    # ----------------------------------------------------------------- private
    def _path_for(self, day: date) -> Path:
        return self.directory / f"{day.isoformat()}.jsonl"

    def _rotate_if_needed(self) -> None:
        """Open / re-open the file when the calendar date crosses over."""
        today = self._date_fn()
        if self._current_date == today and self._fh is not None:
            return
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
        path = self._path_for(today)
        self._fh = path.open("a", encoding="utf-8")
        self._current_date = today


def _json_default(obj: Any) -> Any:
    """Fallback encoder for datetime / Enum values that callers may pass."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def replay_events(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Replay events from one or more journal files in file order."""
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not Path(p).exists():
            continue
        with Path(p).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


__all__ = ["OrderJournal", "replay_events"]
