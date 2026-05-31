"""Phase 1 completion 2.3: ``FoldRecord`` JSON round-trip helpers."""

from __future__ import annotations

import json

import numpy as np

from fortuna.paper.learner import FoldRecord, StrategyLearningState


def test_fold_record_to_dict_yields_plain_json_types():
    rec = FoldRecord(
        fold_id=np.int64(3),
        profit_pct=np.float64(1.25),
        win_ratio_pct=np.float64(55.0),
        total_trades=np.int64(42),
        candidate_id="abc",
        params_summary="ema=20",
    )
    d = rec.to_dict()
    serialized = json.dumps(d)  # must not raise
    loaded = json.loads(serialized)
    assert isinstance(loaded["fold_id"], int)
    assert isinstance(loaded["profit_pct"], float)
    assert isinstance(loaded["total_trades"], int)


def test_fold_record_round_trip():
    src = FoldRecord(
        fold_id=1,
        profit_pct=0.5,
        win_ratio_pct=60.0,
        total_trades=10,
        candidate_id="c1",
        params_summary="p=1",
    )
    rt = FoldRecord.from_dict(src.to_dict())
    assert rt == src


def test_strategy_learning_state_uses_fold_record_helpers():
    state = StrategyLearningState(
        strategy_stem="foo",
        fold_history=[
            FoldRecord(0, 1.0, 50.0, 12, "c0", "p0"),
            FoldRecord(1, 2.0, 60.0, 18, "c1", "p1"),
        ],
        cumulative_oos_profit_pct=3.0,
        generation=2,
    )
    payload = json.dumps(state.to_dict())  # must not raise
    rt = StrategyLearningState.from_dict(json.loads(payload))
    assert rt.strategy_stem == "foo"
    assert len(rt.fold_history) == 2
    assert rt.fold_history[1].profit_pct == 2.0
