"""Agentic paper-learning dataset: pending rows, outcome resolution, ML export."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Iterable, Optional, Union

import pandas as pd

from fortuna.agentic.models import ActionRecommendation, AgentDecision
from fortuna.ml.labels import directional_multiplier, forward_return_at_bar
from fortuna.ml.types import SignalExample

DEFAULT_HORIZON_BARS = 3
AGENTIC_STRATEGY_NAME = "agentic"
AGENTIC_TAG_PREFIX = "agentic:"

_TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "1d": 86400,
}


@dataclass
class LearningOutcome:
    status: str = "pending"
    forward_return_pct: Optional[float] = None
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    directionally_correct: Optional[bool] = None
    paper_filled: Optional[bool] = None
    paper_closed: Optional[bool] = None
    realized_pnl_pct: Optional[float] = None
    risk_block_reason: Optional[str] = None
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "LearningOutcome":
        if not d:
            return cls()
        return cls(
            status=str(d.get("status", "pending")),
            forward_return_pct=_optional_float(d.get("forward_return_pct")),
            mfe_pct=_optional_float(d.get("mfe_pct")),
            mae_pct=_optional_float(d.get("mae_pct")),
            directionally_correct=_optional_bool(d.get("directionally_correct")),
            paper_filled=_optional_bool(d.get("paper_filled")),
            paper_closed=_optional_bool(d.get("paper_closed")),
            realized_pnl_pct=_optional_float(d.get("realized_pnl_pct")),
            risk_block_reason=d.get("risk_block_reason"),
            resolved_at=d.get("resolved_at"),
        )


@dataclass
class LearningExample:
    decision_hash: str
    bar_time: Optional[str]
    bar_idx: Optional[int]
    symbol: str
    timeframe: str
    action: str
    confidence: float
    current_side: Optional[str]
    bar_close: Optional[float]
    vote_summary: list[dict[str, Any]] = field(default_factory=list)
    notification_sent: Optional[bool] = None
    notification_deduped: Optional[bool] = None
    paper_submitted: bool = False
    outcome: LearningOutcome = field(default_factory=LearningOutcome)
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_hash": self.decision_hash,
            "bar_time": self.bar_time,
            "bar_idx": self.bar_idx,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "action": self.action,
            "confidence": self.confidence,
            "current_side": self.current_side,
            "bar_close": self.bar_close,
            "vote_summary": self.vote_summary,
            "notification_sent": self.notification_sent,
            "notification_deduped": self.notification_deduped,
            "paper_submitted": self.paper_submitted,
            "outcome": self.outcome.to_dict(),
            "metadata": self.metadata,
            "recorded_events": list(self.recorded_events),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LearningExample":
        return cls(
            decision_hash=str(d["decision_hash"]),
            bar_time=d.get("bar_time"),
            bar_idx=_optional_int(d.get("bar_idx")),
            symbol=str(d.get("symbol", "")),
            timeframe=str(d.get("timeframe", "")),
            action=str(d.get("action", "HOLD")),
            confidence=float(d.get("confidence", 0.0)),
            current_side=d.get("current_side"),
            bar_close=_optional_float(d.get("bar_close")),
            vote_summary=list(d.get("vote_summary") or []),
            notification_sent=_optional_bool(d.get("notification_sent")),
            notification_deduped=_optional_bool(d.get("notification_deduped")),
            paper_submitted=bool(d.get("paper_submitted", False)),
            outcome=LearningOutcome.from_dict(d.get("outcome")),
            metadata=dict(d.get("metadata") or {}),
            recorded_events=list(d.get("recorded_events") or []),
        )


def should_persist_decision(action: Union[str, ActionRecommendation]) -> bool:
    """Persist actionable decisions and DO_NOT_ENTER; skip ordinary HOLD."""
    val = getattr(action, "value", action)
    text = str(val or "").upper()
    if text == ActionRecommendation.DO_NOT_ENTER.value:
        return True
    if text in {
        ActionRecommendation.BUY.value,
        ActionRecommendation.SELL.value,
        ActionRecommendation.EXIT_LONG.value,
        ActionRecommendation.EXIT_SHORT.value,
    }:
        return True
    return False


def bar_period_seconds(timeframe: str) -> int:
    return _TIMEFRAME_SECONDS.get(str(timeframe).lower(), 300)


def _normalize_ts(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Kolkata").tz_localize(None)
    return t


def match_bar_time(
    ohlcv_index: pd.Index,
    bar_time: Any,
    *,
    timeframe: str = "5m",
) -> Optional[int]:
    """Exact or single-candidate toleranced match; never nearest-neighbor snap."""
    if bar_time is None or len(ohlcv_index) == 0:
        return None
    target = _normalize_ts(bar_time)
    norm_index = pd.Index([_normalize_ts(x) for x in ohlcv_index])

    for i, idx_ts in enumerate(norm_index):
        if idx_ts == target:
            return i

    tol = pd.Timedelta(seconds=bar_period_seconds(timeframe))
    candidates = [
        i for i, idx_ts in enumerate(norm_index) if abs(idx_ts - target) <= tol
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def bar_idx_for_decision(
    ohlcv: pd.DataFrame,
    bar_time: Any,
    *,
    timeframe: str = "5m",
) -> Optional[int]:
    if ohlcv is None or ohlcv.empty or bar_time is None:
        return None
    return match_bar_time(ohlcv.index, bar_time, timeframe=timeframe)


def _vote_summary(decision: AgentDecision) -> list[dict[str, Any]]:
    return [
        {
            "agent": v.agent,
            "action": v.action.value,
            "confidence": round(float(v.confidence), 4),
            "reason": v.reason,
            "source": v.source,
        }
        for v in decision.votes
    ]


def learning_row_from_decision(
    decision: AgentDecision,
    *,
    timeframe: str,
    bar_idx: Optional[int] = None,
) -> LearningExample:
    nr = decision.notification_result
    return LearningExample(
        decision_hash=decision.decision_hash,
        bar_time=decision.bar_time.isoformat() if decision.bar_time else None,
        bar_idx=bar_idx,
        symbol=decision.symbol,
        timeframe=timeframe,
        action=decision.action.value,
        confidence=float(decision.confidence),
        current_side=decision.current_side,
        bar_close=float(decision.bar_close) if decision.bar_close is not None else None,
        vote_summary=_vote_summary(decision),
        notification_sent=nr.sent if nr is not None else None,
        notification_deduped=nr.deduped if nr is not None else None,
        paper_submitted=False,
        outcome=LearningOutcome(status="pending"),
        metadata=dict(decision.metadata),
    )


def compute_mfe_mae(
    ohlcv: pd.DataFrame,
    bar_idx: int,
    action: str,
    horizon_bars: int,
) -> tuple[Optional[float], Optional[float]]:
    """Direction-adjusted MFE/MAE over bar_idx+1 .. bar_idx+horizon."""
    mult = directional_multiplier(action)
    if mult == 0.0 or bar_idx < 0 or bar_idx >= len(ohlcv):
        return None, None
    end = bar_idx + horizon_bars
    if end >= len(ohlcv) or "close" not in ohlcv.columns:
        return None, None

    ref = float(ohlcv.iloc[bar_idx]["close"])
    if ref == 0:
        return None, None

    directed: list[float] = []
    for i in range(bar_idx + 1, end + 1):
        close_i = float(ohlcv.iloc[i]["close"])
        raw_pct = (close_i - ref) / ref * 100.0
        directed.append(raw_pct * mult)

    if "high" in ohlcv.columns and "low" in ohlcv.columns:
        for i in range(bar_idx + 1, end + 1):
            hi = float(ohlcv.iloc[i]["high"])
            lo = float(ohlcv.iloc[i]["low"])
            directed.append((hi - ref) / ref * 100.0 * mult)
            directed.append((lo - ref) / ref * 100.0 * mult)

    return max(directed), min(directed)


def _directionally_correct(action: str, forward_return_pct: Optional[float]) -> Optional[bool]:
    if forward_return_pct is None:
        return None
    if directional_multiplier(action) == 0.0:
        return None
    return forward_return_pct > 0.0


def resolve_outcome(
    row: LearningExample,
    ohlcv: pd.DataFrame,
    *,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    paper_outcome: Optional[dict[str, Any]] = None,
) -> LearningExample:
    """Resolve pending row using stored bar_idx; backfill index once if missing."""
    bar_idx = row.bar_idx
    if bar_idx is None and row.bar_time:
        bar_idx = bar_idx_for_decision(ohlcv, row.bar_time, timeframe=row.timeframe)

    if bar_idx is None or bar_idx + horizon_bars >= len(ohlcv):
        return row

    fwd = forward_return_at_bar(ohlcv, bar_idx, row.action, horizon_bars)
    mfe, mae = compute_mfe_mae(ohlcv, bar_idx, row.action, horizon_bars)
    dir_ok = _directionally_correct(row.action, fwd)

    if row.action == ActionRecommendation.DO_NOT_ENTER.value:
        buy_fwd = forward_return_at_bar(ohlcv, bar_idx, "BUY", horizon_bars)
        sell_fwd = forward_return_at_bar(ohlcv, bar_idx, "SELL", horizon_bars)
        if buy_fwd is not None and sell_fwd is not None:
            dir_ok = buy_fwd <= 0.0 and sell_fwd <= 0.0

    paper_filled = row.outcome.paper_filled
    paper_closed = row.outcome.paper_closed
    realized_pnl = row.outcome.realized_pnl_pct
    if paper_outcome:
        if paper_outcome.get("paper_filled") is not None:
            paper_filled = bool(paper_outcome["paper_filled"])
        if paper_outcome.get("paper_closed") is not None:
            paper_closed = bool(paper_outcome["paper_closed"])
        if paper_outcome.get("realized_pnl_pct") is not None:
            realized_pnl = _optional_float(paper_outcome["realized_pnl_pct"])

    outcome = LearningOutcome(
        status="resolved",
        forward_return_pct=fwd,
        mfe_pct=mfe,
        mae_pct=mae,
        directionally_correct=dir_ok,
        paper_filled=paper_filled,
        paper_closed=paper_closed,
        realized_pnl_pct=realized_pnl,
        risk_block_reason=row.outcome.risk_block_reason,
        resolved_at=datetime.now().isoformat(timespec="seconds"),
    )

    return replace(row, bar_idx=bar_idx, outcome=outcome)


def agentic_trade_tag(decision_hash: str) -> str:
    return f"{AGENTIC_TAG_PREFIX}{decision_hash}"


def paper_outcome_for_tag(trades: Iterable[Any], decision_hash: str) -> Optional[dict[str, Any]]:
    """Match trade by exact ``agentic:{decision_hash}`` tag only."""
    tag = agentic_trade_tag(decision_hash)
    matched = [t for t in trades if getattr(t, "tag", None) == tag]
    if not matched:
        return None
    trade = matched[-1]
    exit_ts = getattr(trade, "exit_ts", None)
    closed = exit_ts is not None
    return {
        "paper_filled": True,
        "paper_closed": closed,
        "realized_pnl_pct": float(getattr(trade, "return_pct", 0.0)) if closed else None,
        "tag": tag,
    }


def event_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def event_record_key(event_type: str, payload: dict[str, Any]) -> str:
    return f"{event_type}:{event_fingerprint(payload)}"


def apply_learning_event(
    row: LearningExample,
    event_type: str,
    payload: dict[str, Any],
) -> LearningExample:
    """Idempotent merge of a learning event into a row."""
    key = event_record_key(event_type, payload)
    if key in row.recorded_events:
        return row

    recorded = [*row.recorded_events, key]
    paper_submitted = row.paper_submitted
    outcome = row.outcome

    if event_type == "paper_submitted":
        paper_submitted = True
    elif event_type == "risk_blocked":
        reason = str(payload.get("reason") or payload.get("rule") or "blocked")
        outcome = replace(outcome, risk_block_reason=reason)
    elif event_type == "paper_filled":
        outcome = replace(outcome, paper_filled=True)
    elif event_type == "paper_closed":
        outcome = replace(
            outcome,
            paper_closed=True,
            realized_pnl_pct=_optional_float(payload.get("realized_pnl_pct")),
        )

    return replace(
        row,
        paper_submitted=paper_submitted,
        outcome=outcome,
        recorded_events=recorded,
        metadata={**row.metadata, **payload},
    )


def build_learning_rows(
    examples: list[LearningExample],
    *,
    resolved_only: bool = False,
) -> list[LearningExample]:
    """Dedupe by decision_hash (latest wins)."""
    by_hash: dict[str, LearningExample] = {}
    for ex in examples:
        by_hash[ex.decision_hash] = ex
    rows = list(by_hash.values())
    if resolved_only:
        rows = [r for r in rows if r.outcome.status == "resolved"]
    return rows


def to_signal_examples(rows: list[LearningExample]) -> list[SignalExample]:
    """Export resolved rows to Phase 1 SignalExample (requires bar_idx)."""
    out: list[SignalExample] = []
    for row in rows:
        if row.outcome.status != "resolved" or row.bar_idx is None:
            continue
        if row.outcome.directionally_correct is None:
            continue
        label = 1 if row.outcome.directionally_correct else 0
        bar_time = pd.Timestamp(row.bar_time) if row.bar_time else pd.Timestamp("1970-01-01")
        out.append(
            SignalExample(
                bar_idx=int(row.bar_idx),
                bar_time=bar_time,
                symbol=row.symbol,
                strategy_name=AGENTIC_STRATEGY_NAME,
                action=row.action,
                label=label,
                forward_return_pct=float(row.outcome.forward_return_pct or 0.0),
                cost_adjusted_return_pct=float(row.outcome.forward_return_pct or 0.0),
            )
        )
    return out


def _optional_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _optional_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _optional_bool(val: Any) -> Optional[bool]:
    if val is None:
        return None
    return bool(val)
