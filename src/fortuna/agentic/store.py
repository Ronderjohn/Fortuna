"""Persistence for agentic decisions, events, and learning rows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from fortuna.agentic.learning import (
    DEFAULT_HORIZON_BARS,
    LearningExample,
    apply_learning_event,
    bar_idx_for_decision,
    build_learning_rows,
    learning_row_from_decision,
    paper_outcome_for_tag,
    resolve_outcome,
    should_persist_decision,
)
from fortuna.agentic.models import AgentDecision, PaperLearningEvent


def _read_jsonl(path: Path, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


class AgenticDecisionStore:
    def __init__(self, directory: str | Path = "logs/agentic") -> None:
        self.directory = Path(directory)
        self.decisions_path = self.directory / "decisions.jsonl"
        self.learning_path = self.directory / "learning.jsonl"

    def append_decision(self, decision: AgentDecision) -> None:
        self._append(self.decisions_path, decision.to_dict())

    def append_learning_event(self, event: PaperLearningEvent) -> None:
        self._append(self.learning_path, event.to_dict())

    def read_decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        return _read_jsonl(self.decisions_path, limit=limit)

    def read_learning_events(self, limit: int = 100) -> list[dict[str, Any]]:
        return _read_jsonl(self.learning_path, limit=limit)

    def _append(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


class NotificationAuditStore:
    """Append-only audit log for outbound notification attempts."""

    def __init__(self, directory: str | Path = "logs/agentic") -> None:
        self.directory = Path(directory)
        self.notifications_path = self.directory / "notifications.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        self._append(self.notifications_path, record)

    def read_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return _read_jsonl(self.notifications_path, limit=limit)

    def recent_dedupe_keys(self, limit: int = 500) -> set[str]:
        keys: set[str] = set()
        for row in reversed(self.read_recent(limit)):
            if not bool(row.get("sent")) and not bool(row.get("deduped")):
                continue
            key = row.get("dedupe_key")
            if key:
                keys.add(str(key))
        return keys

    def _append(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


class AgenticLearningStore:
    """Pending/resolved learning rows for agentic advisory outcomes."""

    def __init__(self, directory: str | Path = "logs/agentic") -> None:
        self.directory = Path(directory)
        self.rows_path = self.directory / "learning_rows.jsonl"

    def append_pending(
        self,
        decision: AgentDecision,
        *,
        timeframe: str,
        bar_idx: Optional[int] = None,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> Optional[LearningExample]:
        if not should_persist_decision(decision.action):
            return None
        existing = self._rows_by_hash()
        if decision.decision_hash in existing:
            return existing[decision.decision_hash]

        idx = bar_idx
        if idx is None and ohlcv is not None and decision.bar_time is not None:
            idx = bar_idx_for_decision(ohlcv, decision.bar_time, timeframe=timeframe)

        row = learning_row_from_decision(decision, timeframe=timeframe, bar_idx=idx)
        self.upsert(row)
        return row

    def record_event(
        self,
        decision_hash: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> bool:
        """Idempotent event merge. Returns True if row was updated."""
        rows = self._rows_by_hash()
        row = rows.get(decision_hash)
        if row is None:
            return False
        updated = apply_learning_event(row, event_type, payload)
        if updated is row:
            return False
        self.upsert(updated)
        return True

    def resolve_with_ohlcv(
        self,
        decision_hash: str,
        ohlcv: pd.DataFrame,
        *,
        horizon_bars: int = DEFAULT_HORIZON_BARS,
        paper_outcome: Optional[dict[str, Any]] = None,
    ) -> Optional[LearningExample]:
        rows = self._rows_by_hash()
        row = rows.get(decision_hash)
        if row is None:
            return row
        if row.outcome.status == "resolved":
            if paper_outcome is not None:
                return self.enrich_paper_outcome(decision_hash, paper_outcome)
            return row

        resolved = resolve_outcome(
            row,
            ohlcv,
            horizon_bars=horizon_bars,
            paper_outcome=paper_outcome,
        )
        if resolved.outcome.status != "resolved":
            return row
        self.upsert(resolved)
        return resolved

    def enrich_paper_outcome(
        self,
        decision_hash: str,
        paper_outcome: dict[str, Any],
    ) -> Optional[LearningExample]:
        rows = self._rows_by_hash()
        row = rows.get(decision_hash)
        if row is None:
            return None

        outcome = row.outcome
        paper_filled = outcome.paper_filled
        paper_closed = outcome.paper_closed
        realized_pnl = outcome.realized_pnl_pct

        if paper_outcome.get("paper_filled") is not None:
            paper_filled = bool(paper_outcome["paper_filled"])
        if paper_outcome.get("paper_closed") is not None:
            paper_closed = bool(paper_outcome["paper_closed"])
        if paper_outcome.get("realized_pnl_pct") is not None:
            realized_pnl = float(paper_outcome["realized_pnl_pct"])

        if (
            paper_filled == outcome.paper_filled
            and paper_closed == outcome.paper_closed
            and realized_pnl == outcome.realized_pnl_pct
        ):
            return row

        updated = LearningExample(
            decision_hash=row.decision_hash,
            bar_time=row.bar_time,
            bar_idx=row.bar_idx,
            symbol=row.symbol,
            timeframe=row.timeframe,
            action=row.action,
            confidence=row.confidence,
            current_side=row.current_side,
            bar_close=row.bar_close,
            vote_summary=list(row.vote_summary),
            notification_sent=row.notification_sent,
            notification_deduped=row.notification_deduped,
            paper_submitted=row.paper_submitted,
            outcome=row.outcome.__class__(
                status=outcome.status,
                forward_return_pct=outcome.forward_return_pct,
                mfe_pct=outcome.mfe_pct,
                mae_pct=outcome.mae_pct,
                directionally_correct=outcome.directionally_correct,
                paper_filled=paper_filled,
                paper_closed=paper_closed,
                realized_pnl_pct=realized_pnl,
                risk_block_reason=outcome.risk_block_reason,
                resolved_at=outcome.resolved_at,
            ),
            metadata=dict(row.metadata),
            recorded_events=list(row.recorded_events),
        )
        self.upsert(updated)
        return updated

    def rows_for_symbol(self, symbol: str) -> list[LearningExample]:
        rows = [r for r in self._rows_by_hash().values() if r.symbol == str(symbol)]
        rows.sort(key=lambda r: r.bar_time or "")
        return rows

    def unresolved(
        self,
        limit: Optional[int] = 100,
        symbol: Optional[str] = None,
    ) -> list[LearningExample]:
        rows = [
            r for r in self._rows_by_hash().values() if r.outcome.status == "pending"
        ]
        if symbol is not None:
            sym = str(symbol)
            rows = [r for r in rows if r.symbol == sym]
        rows.sort(key=lambda r: r.bar_time or "")
        if limit is not None and limit > 0:
            return rows[:limit]
        return rows

    def read_all(self, limit: Optional[int] = None) -> list[LearningExample]:
        rows = list(self._rows_by_hash().values())
        rows.sort(key=lambda r: r.bar_time or "")
        if limit is not None and limit > 0:
            return rows[-limit:]
        return rows

    def upsert(self, row: LearningExample) -> None:
        rows = self._rows_by_hash()
        rows[row.decision_hash] = row
        self._write_rows(list(rows.values()))

    def _rows_by_hash(self) -> dict[str, LearningExample]:
        raw = _read_jsonl(self.rows_path)
        deduped = build_learning_rows([LearningExample.from_dict(r) for r in raw])
        return {r.decision_hash: r for r in deduped}

    def _write_rows(self, rows: list[LearningExample]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.rows_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), sort_keys=True, default=str) + "\n")
        os.replace(tmp, self.rows_path)


def resolve_pending_for_symbol(
    store: AgenticLearningStore,
    symbol: str,
    ohlcv: pd.DataFrame,
    trades: list[Any],
    *,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
) -> int:
    """Resolve pending rows and refresh paper-close data for a symbol."""
    resolved_count = 0
    for row in store.unresolved(limit=0, symbol=symbol):
        bar_idx = row.bar_idx
        if bar_idx is None and row.bar_time:
            bar_idx = bar_idx_for_decision(ohlcv, row.bar_time, timeframe=row.timeframe)
        if bar_idx is None or bar_idx + horizon_bars >= len(ohlcv):
            continue
        paper = paper_outcome_for_tag(trades, row.decision_hash)
        result = store.resolve_with_ohlcv(
            row.decision_hash,
            ohlcv,
            horizon_bars=horizon_bars,
            paper_outcome=paper,
        )
        if result is not None and result.outcome.status == "resolved":
            resolved_count += 1
    for row in store.rows_for_symbol(symbol):
        paper = paper_outcome_for_tag(trades, row.decision_hash)
        if paper is None:
            continue
        store.enrich_paper_outcome(row.decision_hash, paper)
    return resolved_count
