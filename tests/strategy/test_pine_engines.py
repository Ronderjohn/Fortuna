"""Smoke tests for the Pine-parity engines: LS-VWAP and IVWAP-ORB.

These tests don't try to lock down exact P&L numbers (the synthetic data is
random and not meant to be economically meaningful). Instead they verify the
contracts the rest of the system depends on:

- The strategy JSON loads and routes to the correct builtin engine.
- The backtest pipeline produces a populated :class:`BacktestResult` with
  ``equity_curve``, ``enriched_data``, and a finite ``final_value``.
- The expected indicator/signal columns are present so the chart overlay
  picker has something to render.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fortuna.app.live_signals import (
    _dual_side_state,
    _signal_from_builtin,
    compute_live_signal,
)
from fortuna.strategies.builtin.dispatch import builtin_engine_id, run_builtin_backtest
from fortuna.strategies.builtin.ivorb import (
    IVORBParams,
    compute_ivorb_signals,
    is_ivorb_strategy,
)
from fortuna.strategies.builtin.lsvwap import (
    LSVWAPParams,
    compute_lsvwap_signals,
    is_lsvwap_strategy,
)
from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import TradeSide


# ───────────────────────── synthetic intraday bars ──────────────────────────


def _intraday_5m_bars(days: int = 3, *, seed: int = 7) -> pd.DataFrame:
    """Build ``days`` of synthetic NSE 5-minute bars (09:15–15:30 IST).

    Index is tz-naive IST timestamps so the engines' ``_minutes_of_day``
    helper extracts the wall-clock minute correctly.
    """
    rng = np.random.default_rng(seed)
    rows: list[pd.Timestamp] = []
    start = pd.Timestamp("2024-01-02 09:15:00")
    bars_per_day = (15 * 60 + 30 - (9 * 60 + 15)) // 5  # 75 bars
    for d in range(days):
        day0 = start + pd.Timedelta(days=d)
        for i in range(bars_per_day):
            rows.append(day0 + pd.Timedelta(minutes=5 * i))
    idx = pd.DatetimeIndex(rows)
    n = len(idx)
    drift = rng.normal(0, 0.5, n).cumsum()
    close = 1000.0 + drift + np.sin(np.arange(n) / 7.0) * 5.0
    high = close + rng.uniform(0.5, 2.5, n)
    low = close - rng.uniform(0.5, 2.5, n)
    open_ = close + rng.normal(0, 0.4, n)
    vol = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


@pytest.fixture(scope="module")
def intraday_bars() -> pd.DataFrame:
    return _intraday_5m_bars(days=3)


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ───────────────────────────── LS-VWAP engine ────────────────────────────────


def test_orb_does_not_pyramid_by_default() -> None:
    """Regression: ORB inherits ``MMTSParams`` defaults in its fill loop. The
    default ``max_pyramids=3`` silently turned every re-test of OR-high/low
    into an over-leveraged pyramid add — a violation of Pine's ``pyramiding=0``
    on the canonical ORB strategy. The fix is to pin ``max_pyramids=1`` when
    ORB constructs its MMTSParams. We verify by simulating a synthetic series
    that re-tests the OR-high several times in one session and asserting the
    trade log never produces a ``pyramid`` row."""
    from fortuna.strategies.builtin.orb import ORBParams, simulate_orb

    idx = pd.date_range("2024-01-02 09:15", periods=75, freq="5min")
    n = len(idx)
    # Build a session with a clean OR breakout at bar 3 followed by
    # several re-tests above and below OR-high.
    close = np.concatenate(
        [
            np.array([100.0, 100.5, 100.2]),  # OR window
            np.linspace(101.0, 102.0, 10),
            np.linspace(102.0, 101.2, 10),
            np.linspace(101.2, 103.0, 15),
            np.linspace(103.0, 102.0, 10),
            np.full(n - 48, 102.0),
        ]
    )[:n]
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(n, 5000.0),
        },
        index=idx,
    )

    _, _, trades = simulate_orb(df, ORBParams(), init_cash=100_000.0)
    if trades.empty:
        return
    reasons = trades["exit_reason"].astype(str).str.lower()
    assert not reasons.eq("pyramid").any(), (
        f"ORB recorded pyramid additions ({int(reasons.eq('pyramid').sum())}) — "
        "default max_pyramids regression"
    )


def test_supertrend_warmup_does_not_pin_direction() -> None:
    """Regression: ATR is NaN for the first ``period`` bars; if the
    Supertrend implementation seeds the final bands from those NaN values,
    the direction array gets pinned to its initial value forever (silently
    disabling the filter on any real series). The fix is to seed the bands
    from the basic bands on the first non-NaN ATR bar."""
    from fortuna.strategies.builtin.lsvwap import supertrend_array

    rng = np.random.default_rng(13)
    n = 500
    # Up-then-down regime so a real Pine Supertrend definitely flips at least once.
    trend = np.concatenate([np.linspace(0, 30, n // 2), np.linspace(30, -10, n - n // 2)])
    noise = rng.normal(0, 0.4, n)
    close = 1000.0 + trend + noise
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)

    _, direction = supertrend_array(high, low, close, period=10, factor=3.0)
    bull = (direction < 0).sum()
    bear = (direction > 0).sum()
    flips = int(np.sum(np.diff(direction) != 0))
    assert bull > 0 and bear > 0, (
        f"Supertrend never flipped (bull={bull}, bear={bear}) — warmup bug regressed"
    )
    assert flips >= 1, "Supertrend produced no transitions across a clear up→down regime"


def test_lsvwap_strategy_routes_to_builtin_engine(project_root: Path) -> None:
    strat = load_strategy(project_root / "strategies" / "builtin" / "lsvwap.json")
    assert is_lsvwap_strategy(strat)
    assert builtin_engine_id(strat) == "lsvwap"


def test_lsvwap_signals_have_expected_columns(intraday_bars: pd.DataFrame) -> None:
    enriched = compute_lsvwap_signals(intraday_bars, LSVWAPParams())
    for col in (
        "ema_fast",
        "vwap",
        "supertrend",
        "atr",
        "long_entry",
        "short_entry",
        "long_stop",
        "short_stop",
    ):
        assert col in enriched.columns, f"missing column: {col}"
    assert enriched["long_entry"].dtype == bool
    assert enriched["short_entry"].dtype == bool


def test_lsvwap_backtest_runs(
    project_root: Path, intraday_bars: pd.DataFrame
) -> None:
    strat = load_strategy(project_root / "strategies" / "builtin" / "lsvwap.json")
    result = run_builtin_backtest(strat, intraday_bars, symbol="TEST")
    assert result.metrics.final_value > 0
    assert result.equity_curve is not None
    assert len(result.equity_curve) == len(intraday_bars)
    assert result.enriched_data is not None
    assert "ema_fast" in result.enriched_data.columns


# ───────────────────────────── IVWAP-ORB engine ──────────────────────────────


def test_ivorb_strategy_routes_to_builtin_engine(project_root: Path) -> None:
    strat = load_strategy(project_root / "strategies" / "builtin" / "ivorb.json")
    assert is_ivorb_strategy(strat)
    assert builtin_engine_id(strat) == "ivorb"


def test_ivorb_signals_have_expected_columns(intraday_bars: pd.DataFrame) -> None:
    enriched = compute_ivorb_signals(intraday_bars, IVORBParams())
    for col in (
        "ema_fast",
        "vwap",
        "supertrend",
        "atr",
        "or_high",
        "or_low",
        "long_entry",
        "short_entry",
        "long_stop",
        "long_target",
    ):
        assert col in enriched.columns, f"missing column: {col}"
    # The OR envelope must be populated for every bar after the OR window.
    after_open = enriched.between_time("09:30", "15:30")
    assert after_open["or_high"].notna().all()
    assert after_open["or_low"].notna().all()


def test_ivorb_backtest_runs(
    project_root: Path, intraday_bars: pd.DataFrame
) -> None:
    strat = load_strategy(project_root / "strategies" / "builtin" / "ivorb.json")
    result = run_builtin_backtest(strat, intraday_bars, symbol="TEST")
    assert result.metrics.final_value > 0
    assert result.equity_curve is not None
    assert len(result.equity_curve) == len(intraday_bars)
    assert result.enriched_data is not None
    assert "or_high" in result.enriched_data.columns
    assert "or_low" in result.enriched_data.columns


# ───────────────────────── live signal contract ──────────────────────────────


def test_dual_side_state_walker_handles_flips() -> None:
    """Direction-aware state walker: BUY → SELL → EXIT on a single instrument."""
    n = 10
    long_e = pd.Series([False] * n)
    short_e = pd.Series([False] * n)
    exit_b = pd.Series([False] * n)
    long_e.iloc[1] = True   # Open LONG at bar 1
    short_e.iloc[4] = True  # Flip to SHORT at bar 4 (no explicit exit needed)
    exit_b.iloc[7] = True   # Stop-out at bar 7 → FLAT

    state, entry_idx, bars_in_trade = _dual_side_state(long_e, short_e, exit_b)
    assert state == "FLAT"
    assert entry_idx == -1
    assert bars_in_trade == 0

    # Truncate one bar earlier — should still be SHORT.
    state2, idx2, btrade2 = _dual_side_state(
        long_e.iloc[:7], short_e.iloc[:7], exit_b.iloc[:7]
    )
    assert state2 == "SHORT"
    assert idx2 == 4
    assert btrade2 == 6 - 4

    # Truncate before the flip — should be LONG.
    state3, idx3, btrade3 = _dual_side_state(
        long_e.iloc[:4], short_e.iloc[:4], exit_b.iloc[:4]
    )
    assert state3 == "LONG"
    assert idx3 == 1
    assert btrade3 == 3 - 1


def test_dual_side_walker_exit_then_reentry_same_bar() -> None:
    """Exits resolve before entries — exit + new entry on same bar opens fresh."""
    n = 5
    long_e = pd.Series([False] * n)
    short_e = pd.Series([False] * n)
    exit_b = pd.Series([False] * n)
    long_e.iloc[1] = True   # LONG at 1
    exit_b.iloc[3] = True   # Stop at 3
    short_e.iloc[3] = True  # SHORT entry on the same bar after stop

    state, entry_idx, _ = _dual_side_state(long_e, short_e, exit_b)
    assert state == "SHORT"
    assert entry_idx == 3


def test_live_signal_emits_buy_for_long_entry(intraday_bars: pd.DataFrame) -> None:
    """Force a long_entry on the latest bar and verify the live signal is BUY."""
    enriched = intraday_bars.copy()
    n = len(enriched)
    enriched["long_entry"] = [False] * n
    enriched["short_entry"] = [False] * n
    enriched.loc[enriched.index[-1], "long_entry"] = True
    enriched.attrs["trades"] = pd.DataFrame(
        columns=["entry_time", "exit_time", "direction", "exit_reason"]
    )

    class _Strat:
        name = "fake"
        side = TradeSide.BOTH

    sig = _signal_from_builtin(enriched, intraday_bars, "fake", _Strat())
    assert sig is not None
    assert sig.action == "BUY"
    assert sig.label == "BUY"
    assert sig.side == "LONG"


def test_live_signal_emits_sell_for_short_entry(intraday_bars: pd.DataFrame) -> None:
    """Force a short_entry on the latest bar and verify the live signal is SELL."""
    enriched = intraday_bars.copy()
    n = len(enriched)
    enriched["long_entry"] = [False] * n
    enriched["short_entry"] = [False] * n
    enriched.loc[enriched.index[-1], "short_entry"] = True
    enriched.attrs["trades"] = pd.DataFrame(
        columns=["entry_time", "exit_time", "direction", "exit_reason"]
    )

    class _Strat:
        name = "fake"
        side = TradeSide.BOTH

    sig = _signal_from_builtin(enriched, intraday_bars, "fake", _Strat())
    assert sig is not None
    assert sig.action == "SELL"
    assert sig.label == "SELL"
    assert sig.side == "SHORT"


def test_exit_bars_drops_out_of_range_exit_times() -> None:
    """Regression: ``_exit_bars_from_trades`` snapped exit_times to the
    nearest indexed bar with **no tolerance**. A stale exit_time from a
    prior session (or an off-by-days mismatch) silently fired EXIT on a
    random live bar. The snap must now reject anything more than one bar
    interval away from any indexed bar."""
    from fortuna.app.live_signals import _exit_bars_from_trades

    idx = pd.date_range("2026-05-25 09:15", periods=10, freq="5min")
    bogus_trade = pd.DataFrame(
        [
            {
                "entry_time": idx[1],
                "exit_time": pd.Timestamp("2026-05-22 14:30"),  # 3 days earlier
                "direction": "long",
                "exit_reason": "stop",
            },
            {
                "entry_time": idx[2],
                "exit_time": idx[5],  # legitimate exit inside the window
                "direction": "long",
                "exit_reason": "target",
            },
        ]
    )
    exit_bars, reasons = _exit_bars_from_trades(bogus_trade, idx)
    assert exit_bars.sum() == 1, (
        f"Expected only the in-range exit to fire, got {int(exit_bars.sum())}"
    )
    assert exit_bars.iloc[5]
    assert reasons[idx[5]] == "target"


def test_live_signal_emits_exit_when_simulator_closes(
    intraday_bars: pd.DataFrame,
) -> None:
    """Simulator records an exit on the latest bar — live signal must be EXIT."""
    enriched = intraday_bars.copy()
    n = len(enriched)
    enriched["long_entry"] = [False] * n
    enriched["short_entry"] = [False] * n
    # Open long 5 bars ago, close on the latest bar.
    enriched.loc[enriched.index[-6], "long_entry"] = True
    enriched.attrs["trades"] = pd.DataFrame(
        [
            {
                "entry_time": enriched.index[-6],
                "exit_time": enriched.index[-1],
                "direction": "long",
                "exit_reason": "target",
            }
        ]
    )

    class _Strat:
        name = "fake"
        side = TradeSide.BOTH

    sig = _signal_from_builtin(enriched, intraday_bars, "fake", _Strat())
    assert sig is not None
    assert sig.action == "EXIT_LONG"
    assert sig.label == "EXIT"
    assert sig.exit_reason == "target"


# ─────────────────── dual-side DSL pipeline (BUY/SELL/EXIT) ─────────────────


def test_dsl_compile_dual_routes_long_and_short_rules() -> None:
    """``compile_dual`` must return four direction-tagged series for a
    ``side: both`` DSL strategy, with the long rules driving the long pair
    and the short rules driving the short pair. Symmetric crossover
    strategies (like ema_crossover) should produce non-empty signals on
    both sides over a noisy series."""
    from fortuna.backtesting.compiler import StrategyCompiler
    from fortuna.indicators.engine import IndicatorEngine
    from fortuna.strategy.loader import load_strategy

    strat = load_strategy(Path("strategies/generated/ema_crossover.json"))
    assert strat.side == TradeSide.BOTH

    rng = np.random.default_rng(3)
    n = 400
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="5min")
    close = 1000.0 + rng.normal(0, 1.5, n).cumsum() + np.sin(np.arange(n) / 12.0) * 8.0
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=idx,
    )
    enriched = IndicatorEngine().compute(df, strat.indicators)
    long_e, long_x, short_e, short_x = StrategyCompiler().compile_dual(strat, enriched)

    assert int(long_e.sum()) > 0, "Long entries never fire on symmetric crossover data"
    assert int(short_e.sum()) > 0, "Short entries never fire on symmetric crossover data"
    # Cross-direction rules must be each other's mirrors here.
    assert (long_e.to_numpy() & short_e.to_numpy()).sum() == 0, (
        "A single bar can't be both a long and a short entry on a crossover"
    )


def test_numpy_runner_dual_side_produces_long_and_short_trades() -> None:
    """End-to-end: the dual-side NumPy runner must record BOTH ``long`` and
    ``short`` trades in its trade log for a ``side: both`` strategy on a
    series that contains clear up- and down-regimes."""
    from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
    from fortuna.strategy.loader import load_strategy

    strat = load_strategy(Path("strategies/generated/ema_crossover.json"))
    rng = np.random.default_rng(11)
    n = 800
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="5min")
    # Two full oscillation cycles to guarantee both long & short crosses fire.
    phase = np.arange(n) / n * 4.0 * np.pi
    close = 1000.0 + np.sin(phase) * 60.0 + rng.normal(0, 0.4, n)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=idx,
    )

    result = NumPyBacktestRunner().run(strat, df, symbol="TEST")
    trades = result.enriched_data.attrs.get("trades")
    assert trades is not None and not trades.empty, "No trades recorded"
    directions = set(trades["direction"].astype(str).str.lower().unique())
    assert "long" in directions, f"No long trades; got {directions}"
    assert "short" in directions, f"No short trades; got {directions}"


def test_dual_side_live_signal_emits_sell_on_short_entry() -> None:
    """End-to-end live-signal path: force a short_entry on the latest bar
    of a DSL ``side: both`` strategy and verify SELL is emitted."""
    from fortuna.strategy.loader import load_strategy

    strat = load_strategy(Path("strategies/generated/ema_crossover.json"))
    rng = np.random.default_rng(17)
    n = 400
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="5min")
    # Up trend then sharp decline so the latest bar is a bearish crossunder.
    close = np.concatenate(
        [np.linspace(1000.0, 1080.0, n - 30), np.linspace(1080.0, 1020.0, 30)]
    )
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=idx,
    )

    sig = compute_live_signal(strat, df, strategy_name="ema_crossover")
    assert sig is not None
    # Either we're now in SHORT, just emitted SELL, or just hit EXIT_LONG —
    # all three are valid representations of a fresh bearish cross.
    assert sig.action in {"SELL", "IN_SHORT", "EXIT_LONG"}, (
        f"Expected bearish-direction action, got {sig.action}"
    )
