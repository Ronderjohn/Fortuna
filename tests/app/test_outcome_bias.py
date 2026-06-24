from __future__ import annotations

from fortuna.agentic.learning import LearningExample, LearningOutcome
from fortuna.app.outcome_bias import (
    load_adaptive_outcome_policy,
    setup_family_key_from_row,
    setup_family_reinforcement_rationale,
)
from fortuna.config.settings import Settings


def _settings(tmp_path) -> Settings:
    return Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        project_root=tmp_path,
        agentic_log_dir=tmp_path / "logs" / "agentic",
    )


def test_setup_family_key_from_row_prefers_primary_signal():
    row = LearningExample(
        decision_hash="a1",
        bar_time="2026-01-10T10:00:00",
        bar_idx=1,
        symbol="RELIANCE.NS",
        timeframe="5m",
        action="BUY",
        confidence=0.7,
        current_side=None,
        bar_close=100.0,
        metadata={"primary_signal": "ORB", "winning_strategy": "MMTS"},
        outcome=LearningOutcome(status="resolved", paper_closed=True, realized_pnl_pct=1.0),
    )
    assert setup_family_key_from_row(row) == "ORB"


def test_setup_family_key_from_row_falls_back_to_winning_strategy():
    row = LearningExample(
        decision_hash="a2",
        bar_time="2026-01-10T10:05:00",
        bar_idx=2,
        symbol="TCS.NS",
        timeframe="5m",
        action="BUY",
        confidence=0.7,
        current_side=None,
        bar_close=200.0,
        metadata={"winning_strategy": "mmts"},
        outcome=LearningOutcome(status="resolved", paper_closed=True, realized_pnl_pct=-1.0),
    )
    assert setup_family_key_from_row(row) == "MMTS"


def test_load_adaptive_outcome_policy_builds_durable_setup_family_biases(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "training_candidate_setup_family_reinforcement_enabled": True,
            "training_candidate_setup_family_min_durable_rows": 2,
            "training_candidate_setup_family_max_boost": 0.05,
        }
    )

    class FakeStore:
        def read_all(self, limit=None):
            return [
                LearningExample(
                    decision_hash="w1",
                    bar_time="2026-01-10T10:00:00",
                    bar_idx=1,
                    symbol="TCS.NS",
                    timeframe="5m",
                    action="BUY",
                    confidence=0.7,
                    current_side=None,
                    bar_close=200.0,
                    metadata={"primary_signal": "ORB", "regime": "TRENDING"},
                    outcome=LearningOutcome(
                        status="resolved",
                        paper_closed=True,
                        realized_pnl_pct=1.9,
                    ),
                ),
                LearningExample(
                    decision_hash="w2",
                    bar_time="2026-01-10T10:05:00",
                    bar_idx=2,
                    symbol="SBIN.NS",
                    timeframe="5m",
                    action="BUY",
                    confidence=0.7,
                    current_side=None,
                    bar_close=100.0,
                    metadata={"primary_signal": "ORB", "regime": "TRENDING"},
                    outcome=LearningOutcome(
                        status="resolved",
                        paper_closed=True,
                        realized_pnl_pct=2.1,
                    ),
                ),
                LearningExample(
                    decision_hash="l1",
                    bar_time="2026-01-10T10:10:00",
                    bar_idx=3,
                    symbol="RELIANCE.NS",
                    timeframe="5m",
                    action="BUY",
                    confidence=0.7,
                    current_side=None,
                    bar_close=150.0,
                    metadata={"primary_signal": "MMTS", "regime": "RANGING"},
                    outcome=LearningOutcome(
                        status="resolved",
                        paper_closed=True,
                        realized_pnl_pct=-2.4,
                    ),
                ),
                LearningExample(
                    decision_hash="l2",
                    bar_time="2026-01-10T10:15:00",
                    bar_idx=4,
                    symbol="INFY.NS",
                    timeframe="5m",
                    action="BUY",
                    confidence=0.7,
                    current_side=None,
                    bar_close=160.0,
                    metadata={"primary_signal": "MMTS", "regime": "RANGING"},
                    outcome=LearningOutcome(
                        status="resolved",
                        paper_closed=True,
                        realized_pnl_pct=-2.0,
                    ),
                ),
            ]

    monkeypatch.setattr(
        "fortuna.app.outcome_bias.AgenticLearningStore",
        lambda *_args, **_kwargs: FakeStore(),
    )

    policy = load_adaptive_outcome_policy(
        settings,
        enabled=True,
        max_adjustment=0.18,
    )

    assert "ORB" in policy.durable_setup_family_biases
    assert policy.durable_setup_family_biases["ORB"].verdict == "winner"
    assert policy.durable_setup_family_biases["ORB"].paper_closed_rows == 2
    assert policy.durable_setup_family_biases["ORB"].score_adjustment <= 0.05
    assert "MMTS" in policy.durable_setup_family_biases
    assert policy.durable_setup_family_biases["MMTS"].verdict == "loser"
    assert "ORB" not in policy.signal_biases
    assert "Setup-family reinforcement ORB" in setup_family_reinforcement_rationale(
        policy.durable_setup_family_biases["ORB"]
    )


def test_load_adaptive_outcome_policy_skips_durable_families_when_disabled(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={"training_candidate_setup_family_reinforcement_enabled": False}
    )

    class FakeStore:
        def read_all(self, limit=None):
            return [
                LearningExample(
                    decision_hash="w1",
                    bar_time="2026-01-10T10:00:00",
                    bar_idx=1,
                    symbol="TCS.NS",
                    timeframe="5m",
                    action="BUY",
                    confidence=0.7,
                    current_side=None,
                    bar_close=200.0,
                    metadata={"primary_signal": "ORB"},
                    outcome=LearningOutcome(
                        status="resolved",
                        paper_closed=True,
                        realized_pnl_pct=1.9,
                    ),
                ),
                LearningExample(
                    decision_hash="w2",
                    bar_time="2026-01-10T10:05:00",
                    bar_idx=2,
                    symbol="SBIN.NS",
                    timeframe="5m",
                    action="BUY",
                    confidence=0.7,
                    current_side=None,
                    bar_close=100.0,
                    metadata={"primary_signal": "ORB"},
                    outcome=LearningOutcome(
                        status="resolved",
                        paper_closed=True,
                        realized_pnl_pct=2.1,
                    ),
                ),
            ]

    monkeypatch.setattr(
        "fortuna.app.outcome_bias.AgenticLearningStore",
        lambda *_args, **_kwargs: FakeStore(),
    )

    policy = load_adaptive_outcome_policy(
        settings,
        enabled=True,
        max_adjustment=0.18,
    )

    assert policy.durable_setup_family_biases == {}
    assert "ORB" in policy.signal_biases


def test_load_adaptive_outcome_policy_blends_long_horizon_symbol_biases(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "market_universe_adaptive_recency_halflife": 4.0,
            "market_universe_long_horizon_feedback_enabled": True,
            "market_universe_long_horizon_halflife": 40.0,
        }
    )

    class FakeStore:
        def read_all(self, limit=None):
            rows = []
            for idx in range(6):
                rows.append(
                    LearningExample(
                        decision_hash=f"tcs-{idx}",
                        bar_time=f"2026-01-0{1 + idx // 2}T10:0{idx % 2}:00",
                        bar_idx=idx,
                        symbol="TCS.NS",
                        timeframe="5m",
                        action="BUY",
                        confidence=0.7,
                        current_side=None,
                        bar_close=200.0,
                        outcome=LearningOutcome(
                            status="resolved",
                            paper_closed=True,
                            realized_pnl_pct=3.0 if idx < 4 else 0.6,
                        ),
                    )
                )
            return rows

    monkeypatch.setattr(
        "fortuna.app.outcome_bias.AgenticLearningStore",
        lambda *_args, **_kwargs: FakeStore(),
    )

    short_only = load_adaptive_outcome_policy(
        settings.model_copy(update={"market_universe_long_horizon_feedback_enabled": False}),
        enabled=True,
        max_adjustment=0.2,
    )
    blended = load_adaptive_outcome_policy(
        settings,
        enabled=True,
        max_adjustment=0.2,
    )

    assert "TCS" in short_only.symbol_biases
    assert "TCS" in blended.symbol_biases
    assert (
        blended.symbol_biases["TCS"].score_adjustment
        >= short_only.symbol_biases["TCS"].score_adjustment
    )
