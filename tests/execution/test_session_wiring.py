"""Smoke that the session engine builds the router when enabled.

We don't exercise the full data pipeline here (that needs SmartAPI auth);
we only assert the wiring: when ``settings.execution_enabled=True`` the
engine should expose a working ``execution_router`` + ``live_account``,
and when disabled it should expose ``None``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fortuna.app.session_engine import FortunaSessionEngine
from fortuna.config.settings import Settings
from fortuna.execution.types import Side, Trade


@pytest.fixture
def base_settings(tmp_path) -> Settings:
    cache = tmp_path / "cache"
    cache.mkdir()
    return Settings(
        env="test",
        log_level="WARNING",
        data_cache_dir=cache,
        duckdb_path=cache / "fortuna.duckdb",
        default_symbol="RELIANCE.NS",
        default_timeframe="5m",
        default_days=10,
        execution_enabled=False,
        execution_account_sync=False,
        agentic_enabled=False,
        agentic_paper_learning_enabled=False,
        telegram_enabled=False,
    )


def test_router_disabled_by_default(base_settings):
    eng = FortunaSessionEngine(base_settings)
    assert eng.execution_enabled is False
    assert eng.execution_router is None
    assert eng.live_account is None
    assert eng.execution_monitor is None
    assert eng.agent_decisions() == {}


def test_model_status_safe_when_subsystems_disabled(base_settings):
    eng = FortunaSessionEngine(base_settings)
    status = eng.model_status()
    assert status.ml.available is False
    assert status.rl.available is False
    assert status.ml.load_error == "not_loaded"
    assert status.rl.load_error == "no_generator"
    payload = status.to_dict()
    assert payload["registry_enabled"] is False


def test_session_engine_exposes_conversational_assistant(base_settings):
    eng = FortunaSessionEngine(base_settings)
    assistant = eng.conversational_assistant()
    assert assistant.settings is eng.settings
    assert assistant.tools is not None


def test_router_built_when_flag_set(base_settings):
    enabled = base_settings.model_copy(update={"execution_enabled": True})
    eng = FortunaSessionEngine(enabled)
    assert eng.execution_enabled is True
    assert eng.execution_router is not None
    assert eng.live_account is not None
    assert eng.execution_monitor is not None
    # Sane defaults bubble up from config/execution.yaml or RiskConfig().
    assert eng.live_account.init_cash > 0


def test_telegram_disabled_has_no_dispatcher(base_settings, tmp_path):
    enabled = base_settings.model_copy(update={
        "agentic_enabled": True,
        "agentic_log_dir": tmp_path / "agentic",
        "telegram_enabled": False,
    })
    eng = FortunaSessionEngine(enabled)
    assert eng._notifier is None
    assert eng._notification_dispatcher is None


def test_agentic_enabled_without_paper_keeps_router_disabled(base_settings, tmp_path):
    enabled = base_settings.model_copy(update={
        "agentic_enabled": True,
        "agentic_log_dir": tmp_path / "agentic",
    })
    eng = FortunaSessionEngine(enabled)
    assert eng.execution_enabled is False
    assert eng.execution_router is None


def test_agentic_paper_learning_builds_paper_router(base_settings, tmp_path):
    enabled = base_settings.model_copy(update={
        "agentic_enabled": True,
        "agentic_paper_learning_enabled": True,
        "agentic_log_dir": tmp_path / "agentic",
    })
    eng = FortunaSessionEngine(enabled)
    assert eng.execution_enabled is True
    assert eng.execution_router is not None


def test_agentic_decision_computed_from_signals(base_settings, tmp_path):
    from fortuna.app.live_signals import LiveSignal

    enabled = base_settings.model_copy(update={
        "agentic_enabled": True,
        "agentic_log_dir": tmp_path / "agentic",
    })
    eng = FortunaSessionEngine(enabled)
    ts = pd.Timestamp("2026-01-06 09:30")
    ohlcv = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10.0]},
        index=pd.DatetimeIndex([ts]),
    )
    sig = LiveSignal(
        strategy_name="orb",
        action="BUY",
        label="BUY",
        bar_time=ts,
        bar_close=100.5,
        enter=True,
        exit=False,
        in_position=False,
        rl_confidence="agree",
        rl_action="BUY",
    )
    rl_sig = LiveSignal(
        strategy_name="RL:test",
        action="BUY",
        label="BUY",
        bar_time=ts,
        bar_close=100.5,
        enter=True,
        exit=False,
        in_position=False,
    )

    decisions = eng._compute_agent_decisions(
        "RELIANCE.NS",
        {"orb": sig, "RL:test": rl_sig},
        ohlcv,
        timeframe="5m",
        notify=False,
    )

    assert decisions["RELIANCE.NS"].action.value == "BUY"
    assert (tmp_path / "agentic" / "decisions.jsonl").exists()
    assert (tmp_path / "agentic" / "learning_rows.jsonl").exists()


def test_agentic_paper_learning_writes_learning_rows(base_settings, tmp_path):
    from fortuna.app.live_signals import LiveSignal

    enabled = base_settings.model_copy(update={
        "agentic_enabled": True,
        "agentic_paper_learning_enabled": True,
        "agentic_log_dir": tmp_path / "agentic",
    })
    eng = FortunaSessionEngine(enabled)
    ts = pd.Timestamp("2026-01-06 09:30")
    ohlcv = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10.0]},
        index=pd.DatetimeIndex([ts]),
    )
    sig = LiveSignal(
        strategy_name="orb",
        action="BUY",
        label="BUY",
        bar_time=ts,
        bar_close=100.5,
        enter=True,
        exit=False,
        in_position=False,
    )
    eng._compute_agent_decisions(
        "RELIANCE.NS",
        {"orb": sig},
        ohlcv,
        timeframe="5m",
        notify=False,
    )
    assert (tmp_path / "agentic" / "learning_rows.jsonl").exists()


def test_agentic_learning_resolution_refreshes_late_paper_close(base_settings, tmp_path):
    from fortuna.app.live_signals import LiveSignal

    enabled = base_settings.model_copy(update={
        "agentic_enabled": True,
        "agentic_paper_learning_enabled": True,
        "agentic_log_dir": tmp_path / "agentic",
    })
    eng = FortunaSessionEngine(enabled)
    idx = pd.date_range("2026-01-06 09:15", periods=8, freq="5min")
    ohlcv = pd.DataFrame(
        {
            "open": [100.0, 100.2, 100.5, 101.0, 102.0, 103.0, 104.0, 105.0],
            "high": [100.5, 100.7, 101.0, 101.5, 102.5, 103.5, 104.5, 105.5],
            "low": [99.5, 99.7, 100.0, 100.5, 101.5, 102.5, 103.5, 104.5],
            "close": [100.0, 100.2, 100.5, 101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [10.0] * 8,
        },
        index=idx,
    )
    sig = LiveSignal(
        strategy_name="orb",
        action="BUY",
        label="BUY",
        bar_time=idx[2],
        bar_close=100.5,
        enter=True,
        exit=False,
        in_position=False,
    )
    decisions = eng._compute_agent_decisions(
        "RELIANCE.NS",
        {"orb": sig},
        ohlcv.iloc[:3],
        timeframe="5m",
        notify=False,
    )
    decision = decisions["RELIANCE.NS"]

    eng._resolve_agentic_learning_outcomes("RELIANCE.NS", ohlcv)
    rows = eng._agentic_learning_store.read_all()
    assert rows[0].outcome.status == "resolved"
    assert rows[0].outcome.paper_closed is None

    eng.live_account.trades.append(
        Trade(
            symbol="RELIANCE.NS",
            side=Side.LONG,
            qty=1,
            entry_price=100.5,
            entry_ts=idx[2].to_pydatetime(),
            exit_price=102.0,
            exit_ts=idx[4].to_pydatetime(),
            gross_pnl=1.5,
            cost=0.1,
            tag=f"agentic:{decision.decision_hash}",
        )
    )
    eng._resolve_agentic_learning_outcomes("RELIANCE.NS", ohlcv)
    updated = eng._agentic_learning_store.read_all()[0]
    assert updated.outcome.paper_closed is True
    assert updated.outcome.realized_pnl_pct is not None


def test_agentic_ml_enabled_without_artifact_fails_soft(base_settings, tmp_path):
    from fortuna.app.live_signals import LiveSignal

    enabled = base_settings.model_copy(update={
        "agentic_enabled": True,
        "agentic_ml_scorer_enabled": True,
        "ml_scorer_artifact_dir": tmp_path / "missing-ml-artifacts",
        "agentic_log_dir": tmp_path / "agentic",
    })
    eng = FortunaSessionEngine(enabled)
    ts = pd.Timestamp("2026-01-06 09:30")
    ohlcv = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10.0]},
        index=pd.DatetimeIndex([ts]),
    )
    sig = LiveSignal(
        strategy_name="orb",
        action="BUY",
        label="BUY",
        bar_time=ts,
        bar_close=100.5,
        enter=True,
        exit=False,
        in_position=False,
        rl_confidence="agree",
        rl_action="BUY",
    )
    rl_sig = LiveSignal(
        strategy_name="RL:test",
        action="BUY",
        label="BUY",
        bar_time=ts,
        bar_close=100.5,
        enter=True,
        exit=False,
        in_position=False,
    )

    decisions = eng._compute_agent_decisions(
        "RELIANCE.NS",
        {"orb": sig, "RL:test": rl_sig},
        ohlcv,
        timeframe="5m",
        notify=False,
    )

    decision = decisions["RELIANCE.NS"]
    assert decision.action.value == "BUY"
    assert decision.metadata["bar_idx"] == 0
    assert "enriched" not in decision.metadata


def test_drive_execution_router_no_ops_when_disabled(base_settings):
    """Even if you accidentally call the hook on a disabled engine, it must not raise."""
    eng = FortunaSessionEngine(base_settings)
    df = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10.0]},
        index=pd.DatetimeIndex(["2026-01-06 09:30"]),
    )
    eng._drive_execution_router("X", df, {})


def test_drive_execution_router_forwards_bar(base_settings):
    """When enabled, a bar + empty signals dict should produce an equity snapshot."""
    enabled = base_settings.model_copy(update={"execution_enabled": True})
    eng = FortunaSessionEngine(enabled)
    df = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10.0]},
        index=pd.DatetimeIndex(["2026-01-06 09:30"]),
    )
    eng._drive_execution_router("RELIANCE.NS", df, {})
    assert len(eng.live_account.equity_curve) == 1


def test_account_sync_disabled_when_flag_off(base_settings):
    """``execution_account_sync=False`` (default) → no account reader, no snapshot."""
    enabled = base_settings.model_copy(update={"execution_enabled": True})
    eng = FortunaSessionEngine(enabled)
    assert eng.account_sync_enabled is False
    assert eng.account_reader is None
    assert eng.broker_snapshot is None
    # Sync call should be a no-op (returns None, doesn't crash).
    assert eng.sync_account_from_broker() is None


def test_account_sync_disabled_when_creds_missing(base_settings, monkeypatch):
    """Even with the flag on, missing creds should degrade gracefully."""
    # Force the SmartAPI settings to report not-configured.
    from fortuna.config import smartapi_settings as sm
    sm.get_smartapi_settings.cache_clear()

    class _FakeSm:
        configured = False
    monkeypatch.setattr(sm, "get_smartapi_settings", lambda: _FakeSm())

    enabled = base_settings.model_copy(update={
        "execution_enabled": True,
        "execution_account_sync": True,
    })
    eng = FortunaSessionEngine(enabled)
    assert eng.account_sync_enabled is False
    assert eng.account_sync_error is not None
    assert "credentials" in eng.account_sync_error.lower()


def test_account_sync_invokes_seed_with_fake_reader(base_settings, monkeypatch):
    """When a reader is present, sync_account_from_broker should seed the account."""
    enabled = base_settings.model_copy(update={"execution_enabled": True})
    eng = FortunaSessionEngine(enabled)

    # Manually attach a fake reader bypassing the credentials check.
    from datetime import datetime

    from fortuna.execution.smartapi_account import (
        AccountSnapshot,
        BrokerHolding,
        BrokerPosition,
        FundsSnapshot,
    )
    from fortuna.execution.types import Side

    class FakeReader:
        def full_snapshot(self):
            return AccountSnapshot(
                funds=FundsSnapshot(
                    net=100_000.0, available_cash=100_000.0, available_margin=100_000.0,
                    used_margin=0.0,
                ),
                positions=[
                    BrokerPosition(
                        tradingsymbol="RELIANCE-EQ", symboltoken="2885", exchange="NSE",
                        producttype="INTRADAY", side=Side.LONG, quantity=10,
                        avg_price=2_000.0, fortuna_symbol="RELIANCE.NS",
                    )
                ],
                holdings=[
                    BrokerHolding(
                        tradingsymbol="ITC-EQ", symboltoken="1660", exchange="NSE",
                        quantity=100, avg_price=400.0, fortuna_symbol="ITC.NS",
                    )
                ],
                orders=[],
                trades=[],
                taken_at=datetime(2026, 5, 26, 9, 15),
            )

    eng._account_reader = FakeReader()
    applied = eng.sync_account_from_broker()
    assert applied is not None
    assert eng.broker_snapshot is not None
    # Position + holding should now be reflected in the account.
    assert "RELIANCE.NS" in eng.live_account.positions
    assert "ITC.NS" in eng.live_account.holdings
    assert eng.live_account.init_cash == 100_000.0


def test_run_reconciliation_returns_report_with_fake_reader(base_settings, tmp_path):
    from datetime import datetime

    from fortuna.execution.journal import OrderJournal
    from fortuna.execution.smartapi_account import (
        AccountSnapshot,
        BrokerTrade,
        FundsSnapshot,
    )

    enabled = base_settings.model_copy(update={"execution_enabled": True})
    eng = FortunaSessionEngine(enabled)

    # Swap the router's journal for a tmp one so the test doesn't pollute logs/.
    tmp_journal = OrderJournal(directory=tmp_path / "journal")
    eng.execution_router.journal = tmp_journal

    ts = datetime(2026, 5, 26, 9, 30)
    tmp_journal.write("order_filled", {
        "ack": {
            "order_id": "PAPER-1", "symbol": "RELIANCE.NS",
            "side": "LONG", "status": "FILLED",
            "qty": 10, "filled_qty": 10, "filled_price": 2_000.0,
            "ts": ts.isoformat(), "tag": "orb",
        },
    })

    class FakeReader:
        def full_snapshot(self):
            return AccountSnapshot(
                funds=FundsSnapshot(
                    net=0.0, available_cash=0.0, available_margin=0.0, used_margin=0.0,
                ),
                positions=[], holdings=[], orders=[], trades=[],
                taken_at=ts,
            )

        def snapshot_trades(self):
            return [BrokerTrade(
                order_id="REAL-1", tradingsymbol="RELIANCE-EQ", symboltoken="2885",
                exchange="NSE", transaction_type="BUY", quantity=10, fill_price=2_000.5,
                fill_time=ts, producttype="INTRADAY", fortuna_symbol="RELIANCE.NS",
            )]

    eng._account_reader = FakeReader()
    report = eng.run_reconciliation()
    assert report is not None
    assert report.paper_fills == 1
    assert report.real_fills == 1
    assert len(report.matched) == 1
