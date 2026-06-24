"""Tests for agentic paper-learning dataset."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from fortuna.agentic import (
    ActionRecommendation,
    AgentDecision,
    AgenticDecisionStore,
    AgenticLearningStore,
    DecisionRationale,
    LearningExample,
    match_bar_time,
    paper_outcome_for_tag,
    resolve_outcome,
    should_persist_decision,
    to_signal_examples,
)
from fortuna.agentic.learning import apply_learning_event, event_record_key
from fortuna.execution.types import Side, Trade


def _ohlcv(closes: list[float], freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 09:15", periods=len(closes), freq=freq)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )


def _decision(
    action: ActionRecommendation,
    *,
    bar_time: datetime | None = None,
    regime: str | None = None,
) -> AgentDecision:
    ts = bar_time or datetime(2026, 1, 6, 9, 30)
    return AgentDecision(
        symbol="RELIANCE.NS",
        action=action,
        confidence=0.75,
        bar_time=ts,
        bar_close=100.0,
        rationale=DecisionRationale(summary="test"),
        regime=regime,
    )


def test_should_persist_decision():
    assert should_persist_decision(ActionRecommendation.BUY) is True
    assert should_persist_decision(ActionRecommendation.DO_NOT_ENTER) is True
    assert should_persist_decision(ActionRecommendation.HOLD) is False


def test_match_bar_time_exact_and_tolerance():
    ohlcv = _ohlcv([100.0] * 5)
    exact = match_bar_time(ohlcv.index, ohlcv.index[2], timeframe="5m")
    assert exact == 2

    sparse_idx = pd.DatetimeIndex(["2026-01-06 09:25:00", "2026-01-06 09:31:00"])
    near = match_bar_time(sparse_idx, pd.Timestamp("2026-01-06 09:25:30"), timeframe="5m")
    assert near == 0

    ambiguous = match_bar_time(
        ohlcv.index,
        ohlcv.index[2] + pd.Timedelta(seconds=150),
        timeframe="5m",
    )
    assert ambiguous is None


def test_append_pending_stores_bar_idx(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    decision = _decision(ActionRecommendation.BUY, bar_time=ohlcv.index[2].to_pydatetime())
    row = store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    assert row is not None
    assert row.bar_idx == 2
    pending = store.unresolved(symbol="RELIANCE.NS")
    assert len(pending) == 1
    assert pending[0].decision_hash == decision.decision_hash


def test_append_pending_persists_regime_metadata(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    decision = _decision(
        ActionRecommendation.BUY,
        bar_time=ohlcv.index[2].to_pydatetime(),
        regime="TRENDING",
    )
    row = store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    assert row is not None
    assert row.metadata["regime"] == "TRENDING"


def test_hold_not_appended(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0] * 8)
    row = store.append_pending(_decision(ActionRecommendation.HOLD), timeframe="5m", ohlcv=ohlcv)
    assert row is None
    assert store.unresolved() == []


def test_do_not_enter_appended(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0] * 8)
    row = store.append_pending(
        _decision(ActionRecommendation.DO_NOT_ENTER),
        timeframe="5m",
        ohlcv=ohlcv,
    )
    assert row is not None


def test_resolve_buy_rising(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0, 100.5, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    decision = _decision(ActionRecommendation.BUY, bar_time=ohlcv.index[2].to_pydatetime())
    store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    resolved = store.resolve_with_ohlcv(decision.decision_hash, ohlcv, horizon_bars=3)
    assert resolved is not None
    assert resolved.outcome.status == "resolved"
    assert resolved.bar_idx == 2
    assert resolved.outcome.forward_return_pct is not None
    assert resolved.outcome.forward_return_pct > 0
    assert resolved.outcome.directionally_correct is True


def test_resolve_sell_falling(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0, 99.5, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0])
    decision = _decision(ActionRecommendation.SELL, bar_time=ohlcv.index[2].to_pydatetime())
    store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    resolved = store.resolve_with_ohlcv(decision.decision_hash, ohlcv, horizon_bars=3)
    assert resolved is not None
    assert resolved.outcome.directionally_correct is True


def test_insufficient_horizon_stays_pending(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0, 101.0, 102.0])
    decision = _decision(ActionRecommendation.BUY, bar_time=ohlcv.index[1].to_pydatetime())
    store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    row = store.resolve_with_ohlcv(decision.decision_hash, ohlcv, horizon_bars=3)
    assert row is not None
    assert row.outcome.status == "pending"


def test_resolved_row_can_be_enriched_with_late_paper_close(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0, 100.5, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    decision = _decision(ActionRecommendation.BUY, bar_time=ohlcv.index[2].to_pydatetime())
    store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    resolved = store.resolve_with_ohlcv(decision.decision_hash, ohlcv, horizon_bars=3)
    assert resolved is not None
    assert resolved.outcome.paper_closed is None

    enriched = store.enrich_paper_outcome(
        decision.decision_hash,
        {
            "paper_filled": True,
            "paper_closed": True,
            "realized_pnl_pct": 1.25,
        },
    )
    assert enriched is not None
    assert enriched.outcome.status == "resolved"
    assert enriched.outcome.forward_return_pct == resolved.outcome.forward_return_pct
    assert enriched.outcome.paper_closed is True
    assert enriched.outcome.realized_pnl_pct == 1.25


def test_resolve_pending_for_symbol_refreshes_closed_paper_trade(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0, 100.5, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    decision = _decision(ActionRecommendation.BUY, bar_time=ohlcv.index[2].to_pydatetime())
    store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    store.resolve_with_ohlcv(decision.decision_hash, ohlcv, horizon_bars=3)

    trades = [
        Trade(
            symbol="RELIANCE.NS",
            side=Side.LONG,
            qty=1,
            entry_price=100.0,
            entry_ts=datetime(2026, 1, 6, 9, 30),
            exit_price=101.5,
            exit_ts=datetime(2026, 1, 6, 9, 45),
            gross_pnl=1.5,
            cost=0.1,
            tag=f"agentic:{decision.decision_hash}",
        )
    ]

    from fortuna.agentic.store import resolve_pending_for_symbol

    resolve_pending_for_symbol(store, "RELIANCE.NS", ohlcv, trades, horizon_bars=3)
    row = store.read_all()[0]
    assert row.outcome.status == "resolved"
    assert row.outcome.paper_closed is True
    assert row.outcome.realized_pnl_pct == trades[0].return_pct


def test_record_event_idempotent(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0] * 8)
    decision = _decision(ActionRecommendation.BUY, bar_time=ohlcv.index[2].to_pydatetime())
    store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    tag = f"agentic:{decision.decision_hash}"
    payload = {"rule": "cooldown", "reason": "blocked", "strategy": tag}
    assert store.record_event(decision.decision_hash, "risk_blocked", payload) is True
    assert store.record_event(decision.decision_hash, "risk_blocked", payload) is False
    row = store.read_all()[0]
    assert row.outcome.risk_block_reason == "blocked"
    assert len(row.recorded_events) == 1


def test_record_event_paper_submitted(tmp_path: Path):
    store = AgenticLearningStore(tmp_path)
    ohlcv = _ohlcv([100.0] * 8)
    decision = _decision(ActionRecommendation.BUY, bar_time=ohlcv.index[2].to_pydatetime())
    store.append_pending(decision, timeframe="5m", ohlcv=ohlcv)
    store.record_event(decision.decision_hash, "paper_submitted", {"bar_close": 100.0})
    row = store.read_all()[0]
    assert row.paper_submitted is True


def test_paper_outcome_for_tag_exact_match():
    trades = [
        Trade(
            symbol="RELIANCE.NS",
            side=Side.LONG,
            qty=1,
            entry_price=100.0,
            entry_ts=datetime(2026, 1, 6, 9, 30),
            exit_price=101.0,
            exit_ts=datetime(2026, 1, 6, 9, 45),
            gross_pnl=1.0,
            cost=0.1,
            tag="agentic:abc123",
        ),
        Trade(
            symbol="RELIANCE.NS",
            side=Side.LONG,
            qty=1,
            entry_price=100.0,
            entry_ts=datetime(2026, 1, 6, 10, 0),
            exit_price=102.0,
            exit_ts=datetime(2026, 1, 6, 10, 15),
            gross_pnl=2.0,
            cost=0.1,
            tag="agentic:other",
        ),
    ]
    out = paper_outcome_for_tag(trades, "abc123")
    assert out is not None
    assert out["paper_closed"] is True
    assert out["realized_pnl_pct"] == trades[0].return_pct
    assert paper_outcome_for_tag(trades, "missing") is None


def test_from_dict_missing_optional_fields():
    row = LearningExample.from_dict(
        {
            "decision_hash": "abc",
            "bar_time": "2026-01-06T09:30:00",
            "symbol": "X",
            "timeframe": "5m",
            "action": "BUY",
            "confidence": 0.5,
            "current_side": None,
            "bar_close": 100.0,
        }
    )
    assert row.bar_idx is None
    assert row.recorded_events == []


def test_read_decisions(tmp_path: Path):
    store = AgenticDecisionStore(tmp_path)
    store.append_decision(_decision(ActionRecommendation.BUY))
    rows = store.read_decisions(limit=10)
    assert len(rows) == 1
    assert rows[0]["action"] == "BUY"


def test_to_signal_examples_requires_bar_idx():
    ohlcv = _ohlcv([100.0, 100.5, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    base = LearningExample(
        decision_hash="h1",
        bar_time=str(ohlcv.index[2]),
        bar_idx=2,
        symbol="RELIANCE.NS",
        timeframe="5m",
        action="BUY",
        confidence=0.8,
        current_side=None,
        bar_close=101.0,
    )
    pending = resolve_outcome(base, ohlcv, horizon_bars=3)
    examples = to_signal_examples([pending])
    assert len(examples) == 1
    assert examples[0].bar_idx == 2
    assert examples[0].strategy_name == "agentic"


def test_apply_learning_event_key_stable():
    row = LearningExample(
        decision_hash="h",
        bar_time=None,
        bar_idx=0,
        symbol="X",
        timeframe="5m",
        action="BUY",
        confidence=0.5,
        current_side=None,
        bar_close=1.0,
    )
    payload = {"reason": "blocked", "rule": "halted"}
    key = event_record_key("risk_blocked", payload)
    updated = apply_learning_event(row, "risk_blocked", payload)
    assert key in updated.recorded_events
    again = apply_learning_event(updated, "risk_blocked", payload)
    assert again is updated
