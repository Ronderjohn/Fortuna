"""Live BUY / SELL / EXIT / HOLD signals for the latest (forming) bar.

Convention (matches TradingView's ``strategy.entry`` / ``strategy.close``):

- ``BUY``  — a long entry fires on the latest bar (open a long, or flip from
  short to long).
- ``SELL`` — a short entry fires on the latest bar (open a short, or flip
  from long to short).
- ``EXIT`` — the position was closed on the latest bar for any reason
  (hard stop, profit target, EMA trail, signal flip, partial take-profit,
  end-of-session). All exit reasons collapse to a single ``EXIT`` label.
- ``IN_LONG`` / ``IN_SHORT`` — holding from a prior bar with no new action.
- ``HOLD`` — flat with nothing actionable.

For dual-side strategies (``side: both`` — LS-VWAP, IVWAP-ORB, …) both BUY
and SELL can fire on the same instrument over time. The lifecycle walker is
direction-aware so a SELL after a LONG correctly produces an EXIT for the
long *and* a SELL entry on the next bar (or the same bar if the strategy
flips on a single candle).

Lightweight: re-runs only the **indicator engine + compiler** for DSL
strategies, or the **builtin engine's signal columns + simulator trade log**
for Pine-parity engines. No full vectorbt backtest. Safe to call on every
Streamlit refresh / WebSocket tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import StrategyDefinition, TradeSide
from fortuna.strategies.builtin.dispatch import builtin_engine_id
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

_INDICATOR_ENGINE = IndicatorEngine()
_COMPILER = StrategyCompiler()

# TradingView-parity colors so the live panel/banner matches chart markers.
_C_BUY = "#26A69A"
_C_SELL = "#EF5350"
_C_EXIT = "#FF9800"
_C_HOLD = "#9E9E9E"
_C_IN_POS = "#2962FF"


@dataclass
class LiveSignal:
    """Current per-strategy state for the dashboard live panel."""

    strategy_name: str
    action: str
    """One of: BUY, SELL, EXIT_LONG, EXIT_SHORT, IN_LONG, IN_SHORT, HOLD."""
    label: str
    """Short label suitable for badges (BUY / SELL / EXIT / LONG / SHORT / HOLD)."""
    bar_time: pd.Timestamp
    bar_close: float
    enter: bool
    exit: bool
    in_position: bool
    side: Optional[str] = None
    entry_price: Optional[float] = None
    bars_in_trade: int = 0
    color: str = _C_HOLD
    exit_reason: Optional[str] = None
    """For EXIT actions: 'stop', 'target', 'ema_trail', 'tp1', 'tp2', 'flip', …"""

    def is_actionable(self) -> bool:
        return self.action in {"BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"}


# ───────────────────────────── lifecycle walker ──────────────────────────────


def _trade_state(entries: pd.Series, exits: pd.Series) -> tuple[bool, int, int]:
    """Legacy single-direction walker (kept for the DSL fast path and tests).

    Returns ``(in_position, entry_idx_or_-1, bars_in_trade)``. Exits resolve
    any open position; an entry on the same bar as an exit opens a new one.
    """
    in_pos = False
    entry_idx = -1
    e_arr = entries.to_numpy()
    x_arr = exits.to_numpy()
    n = len(entries)
    for i in range(n):
        if in_pos and x_arr[i]:
            in_pos = False
            entry_idx = -1
        if not in_pos and e_arr[i]:
            in_pos = True
            entry_idx = i
    bars_in_trade = (n - 1 - entry_idx) if in_pos and entry_idx >= 0 else 0
    return in_pos, entry_idx, bars_in_trade


def _dual_side_state(
    long_entries: pd.Series,
    short_entries: pd.Series,
    exit_bars: pd.Series,
) -> tuple[str, int, int]:
    """Direction-aware walker for dual-side strategies.

    Returns ``(state, entry_idx, bars_in_trade)`` where ``state`` is one of
    ``"FLAT"``, ``"LONG"``, ``"SHORT"``. Order of operations per bar:

    1. If the simulator closed the position on this bar (``exit_bars[i]``),
       go FLAT first — the closing fill happens before any new entry.
    2. If both long and short entries fire on the same bar, the one matching
       a flip from the existing direction wins; otherwise long_entry wins
       (matches Pine's left-to-right evaluation when ``pyramiding=0``).
    """
    state = "FLAT"
    entry_idx = -1
    le = long_entries.to_numpy()
    se = short_entries.to_numpy()
    xe = exit_bars.to_numpy()
    n = len(long_entries)
    for i in range(n):
        if state != "FLAT" and xe[i]:
            state = "FLAT"
            entry_idx = -1
        # Flip on a single bar: if currently SHORT and long_entry fires, flip
        # (the exit happens implicitly via the new entry's opposite direction).
        if state == "SHORT" and le[i]:
            state = "LONG"
            entry_idx = i
        elif state == "LONG" and se[i]:
            state = "SHORT"
            entry_idx = i
        elif state == "FLAT":
            if le[i]:
                state = "LONG"
                entry_idx = i
            elif se[i]:
                state = "SHORT"
                entry_idx = i
    bars_in_trade = (n - 1 - entry_idx) if state != "FLAT" and entry_idx >= 0 else 0
    return state, entry_idx, bars_in_trade


def _exit_bars_from_trades(
    trades_df: Optional[pd.DataFrame], index: pd.Index
) -> tuple[pd.Series, dict[pd.Timestamp, str]]:
    """Convert a simulator trade log into a per-bar exit boolean + reason map.

    The simulator records every closed leg in ``trades_df`` with ``exit_time``
    and ``exit_reason``. ``pyramid`` rows are *not* real exits (they're
    continuation markers) — we skip them. Partial exits (``tp1`` for LS-VWAP)
    *are* real reductions but the position remains open; we treat every
    non-pyramid ``exit_time`` as an exit bar — the ``_dual_side_state``
    walker will already have re-opened the position from a fresh entry if
    the strategy flipped on the same candle.

    Snap tolerance: an exit_time that falls more than one bar interval away
    from any indexed bar is dropped, not snapped. Without that guard, a
    stale or out-of-range exit_time gets silently mapped to a random bar
    far away (via ``get_indexer(method='nearest')``) and falsely fires an
    EXIT label on the live latest bar.
    """
    exit_bars = pd.Series(False, index=index)
    reasons: dict[pd.Timestamp, str] = {}
    if trades_df is None or trades_df.empty or len(index) == 0:
        return exit_bars, reasons

    # Largest valid snap distance = one bar interval (or 60 s for irregular
    # indexes that have only a single bar).
    if len(index) >= 2:
        deltas = pd.to_datetime(index).to_series().diff().dropna()
        bar_seconds = deltas.dt.total_seconds().median() if not deltas.empty else 60.0
    else:
        bar_seconds = 60.0
    tolerance = pd.Timedelta(seconds=max(bar_seconds, 60.0))

    for _, row in trades_df.iterrows():
        reason = str(row.get("exit_reason", "") or "").lower()
        if reason == "pyramid":
            continue
        xt = pd.Timestamp(row.get("exit_time"))
        if pd.isna(xt):
            continue
        if xt in exit_bars.index:
            exit_bars.loc[xt] = True
            reasons[xt] = reason
            continue
        try:
            pos = exit_bars.index.get_indexer([xt], method="nearest")[0]
        except (KeyError, ValueError):
            continue
        if pos < 0:
            continue
        snapped = exit_bars.index[pos]
        if abs(pd.Timestamp(snapped) - xt) > tolerance:
            continue
        exit_bars.iloc[pos] = True
        reasons[snapped] = reason
    return exit_bars, reasons


# ────────────────────────────── action mapping ───────────────────────────────


def _action_single(
    strategy: StrategyDefinition, enter: bool, exit: bool, in_position: bool
) -> tuple[str, str, str]:
    """Resolve action for a single-direction (DSL) strategy."""
    is_short = strategy.side == TradeSide.SHORT
    if exit and in_position:
        return (
            "EXIT_SHORT" if is_short else "EXIT_LONG",
            "EXIT",
            _C_EXIT,
        )
    if enter and not in_position:
        return (
            "SELL" if is_short else "BUY",
            "SELL" if is_short else "BUY",
            _C_SELL if is_short else _C_BUY,
        )
    if in_position:
        return (
            "IN_SHORT" if is_short else "IN_LONG",
            "SHORT" if is_short else "LONG",
            _C_IN_POS,
        )
    return "HOLD", "HOLD", _C_HOLD


# ────────────────────────────── public surface ───────────────────────────────


def compute_live_signal(
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    *,
    strategy_name: Optional[str] = None,
) -> Optional[LiveSignal]:
    """Compute the current (latest-bar) signal for one strategy."""
    if ohlcv is None or ohlcv.empty:
        return None
    name = strategy_name or strategy.name

    try:
        engine = builtin_engine_id(strategy)
        if engine:
            from fortuna.strategies.builtin.dispatch import run_builtin_backtest

            bt = run_builtin_backtest(strategy, ohlcv)
            enriched = bt.enriched_data
            return _signal_from_builtin(enriched, ohlcv, name, strategy)
        enriched = _INDICATOR_ENGINE.compute(ohlcv, strategy.indicators)
        long_e, long_x, short_e, short_x = _COMPILER.compile_dual(strategy, enriched)
        return _signal_from_dsl(
            strategy, ohlcv, name, long_e, long_x, short_e, short_x
        )
    except Exception as exc:
        logger.debug("live signal failed for %s: %s", name, exc)
        return None


def _signal_from_builtin(
    enriched: pd.DataFrame,
    ohlcv: pd.DataFrame,
    name: str,
    strategy: StrategyDefinition,
) -> Optional[LiveSignal]:
    long_e = enriched.get("long_entry", pd.Series(False, index=enriched.index))
    short_e = enriched.get("short_entry", pd.Series(False, index=enriched.index))
    long_e = long_e.fillna(False).astype(bool).reindex(enriched.index, fill_value=False)
    short_e = short_e.fillna(False).astype(bool).reindex(enriched.index, fill_value=False)

    # Single-side strategies (LONG-only MMTS/ORB, SHORT-only variants) zero
    # out the opposing column so the dual-side walker behaves identically to
    # the single-direction walker.
    if strategy.side == TradeSide.LONG:
        short_e = pd.Series(False, index=enriched.index)
    elif strategy.side == TradeSide.SHORT:
        long_e = pd.Series(False, index=enriched.index)

    trades_df = enriched.attrs.get("trades") if hasattr(enriched, "attrs") else None
    exit_bars, reasons = _exit_bars_from_trades(trades_df, enriched.index)

    state, entry_idx, bars_in_trade = _dual_side_state(long_e, short_e, exit_bars)
    last_idx = enriched.index[-1]
    long_now = bool(long_e.iloc[-1])
    short_now = bool(short_e.iloc[-1])
    exit_now = bool(exit_bars.iloc[-1])

    # Resolve the action *before* the walker already mutated state for this
    # bar — i.e. determine what fired on bar N relative to state at bar N-1.
    # The walker above already produced post-bar state; for the action we
    # need pre-bar state. Re-walk through N-1 bars.
    if len(enriched) >= 2:
        prior_state, _, _ = _dual_side_state(
            long_e.iloc[:-1], short_e.iloc[:-1], exit_bars.iloc[:-1]
        )
    else:
        prior_state = "FLAT"

    if long_now and prior_state != "LONG":
        action, label, color = "BUY", "BUY", _C_BUY
    elif short_now and prior_state != "SHORT":
        action, label, color = "SELL", "SELL", _C_SELL
    elif exit_now and prior_state != "FLAT":
        if prior_state == "LONG":
            action, label, color = "EXIT_LONG", "EXIT", _C_EXIT
        else:
            action, label, color = "EXIT_SHORT", "EXIT", _C_EXIT
    elif state == "LONG":
        action, label, color = "IN_LONG", "LONG", _C_IN_POS
    elif state == "SHORT":
        action, label, color = "IN_SHORT", "SHORT", _C_IN_POS
    else:
        action, label, color = "HOLD", "HOLD", _C_HOLD

    side_label: Optional[str]
    if action in ("BUY", "IN_LONG", "EXIT_LONG"):
        side_label = "LONG"
    elif action in ("SELL", "IN_SHORT", "EXIT_SHORT"):
        side_label = "SHORT"
    else:
        side_label = None

    entry_price: Optional[float] = None
    if state != "FLAT" and 0 <= entry_idx < len(ohlcv):
        entry_price = float(ohlcv["close"].iloc[entry_idx])

    return LiveSignal(
        strategy_name=name,
        action=action,
        label=label,
        bar_time=pd.Timestamp(last_idx),
        bar_close=float(ohlcv["close"].iloc[-1]),
        enter=long_now or short_now,
        exit=exit_now,
        in_position=state != "FLAT",
        side=side_label,
        entry_price=entry_price,
        bars_in_trade=bars_in_trade,
        color=color,
        exit_reason=reasons.get(pd.Timestamp(last_idx)),
    )


def _signal_from_dsl(
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    name: str,
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
) -> Optional[LiveSignal]:
    """Map DSL compiler output to a LiveSignal using the direction-aware
    walker. Treats long_exit/short_exit signals as the simulator's "exit
    bars" since DSL strategies don't have an external trade log."""
    if long_entries.empty and short_entries.empty:
        return None

    if strategy.side == TradeSide.SHORT:
        long_entries = pd.Series(False, index=long_entries.index)
        long_exits = pd.Series(False, index=long_exits.index)
    elif strategy.side == TradeSide.LONG:
        short_entries = pd.Series(False, index=short_entries.index)
        short_exits = pd.Series(False, index=short_exits.index)

    # Direction-aware exit bars: a long_exit only counts when in LONG, a
    # short_exit only counts when in SHORT. We build that with a single
    # forward walk that tracks state.
    n = len(long_entries)
    le = long_entries.fillna(False).to_numpy(dtype=bool)
    lx = long_exits.fillna(False).to_numpy(dtype=bool)
    se = short_entries.fillna(False).to_numpy(dtype=bool)
    sx = short_exits.fillna(False).to_numpy(dtype=bool)

    state = "FLAT"
    entry_idx = -1
    for i in range(n):
        if state == "LONG" and lx[i]:
            state = "FLAT"
            entry_idx = -1
        elif state == "SHORT" and sx[i]:
            state = "FLAT"
            entry_idx = -1
        if state == "SHORT" and le[i]:
            state = "LONG"
            entry_idx = i
        elif state == "LONG" and se[i]:
            state = "SHORT"
            entry_idx = i
        elif state == "FLAT":
            if le[i]:
                state = "LONG"
                entry_idx = i
            elif se[i]:
                state = "SHORT"
                entry_idx = i

    # Re-walk through bar N-1 so we know the action at bar N relative to
    # state at bar N-1 (matches the builtin-engine logic).
    prior_state = "FLAT"
    prior_idx = -1
    for i in range(max(0, n - 1)):
        if prior_state == "LONG" and lx[i]:
            prior_state = "FLAT"
            prior_idx = -1
        elif prior_state == "SHORT" and sx[i]:
            prior_state = "FLAT"
            prior_idx = -1
        if prior_state == "SHORT" and le[i]:
            prior_state = "LONG"
            prior_idx = i
        elif prior_state == "LONG" and se[i]:
            prior_state = "SHORT"
            prior_idx = i
        elif prior_state == "FLAT":
            if le[i]:
                prior_state = "LONG"
                prior_idx = i
            elif se[i]:
                prior_state = "SHORT"
                prior_idx = i

    long_now = bool(le[-1]) if n else False
    short_now = bool(se[-1]) if n else False
    long_exit_now = bool(lx[-1]) if n else False
    short_exit_now = bool(sx[-1]) if n else False

    if long_now and prior_state != "LONG":
        action, label, color = "BUY", "BUY", _C_BUY
    elif short_now and prior_state != "SHORT":
        action, label, color = "SELL", "SELL", _C_SELL
    elif long_exit_now and prior_state == "LONG":
        action, label, color = "EXIT_LONG", "EXIT", _C_EXIT
    elif short_exit_now and prior_state == "SHORT":
        action, label, color = "EXIT_SHORT", "EXIT", _C_EXIT
    elif state == "LONG":
        action, label, color = "IN_LONG", "LONG", _C_IN_POS
    elif state == "SHORT":
        action, label, color = "IN_SHORT", "SHORT", _C_IN_POS
    else:
        action, label, color = "HOLD", "HOLD", _C_HOLD

    side_label: Optional[str]
    if action in ("BUY", "IN_LONG", "EXIT_LONG"):
        side_label = "LONG"
    elif action in ("SELL", "IN_SHORT", "EXIT_SHORT"):
        side_label = "SHORT"
    else:
        side_label = None

    entry_price: Optional[float] = None
    if state != "FLAT" and 0 <= entry_idx < len(ohlcv):
        entry_price = float(ohlcv["close"].iloc[entry_idx])

    bars_in_trade = (n - 1 - entry_idx) if state != "FLAT" and entry_idx >= 0 else 0

    return LiveSignal(
        strategy_name=name,
        action=action,
        label=label,
        bar_time=pd.Timestamp(ohlcv.index[-1]),
        bar_close=float(ohlcv["close"].iloc[-1]),
        enter=long_now or short_now,
        exit=long_exit_now or short_exit_now,
        in_position=state != "FLAT",
        side=side_label,
        entry_price=entry_price,
        bars_in_trade=bars_in_trade,
        color=color,
    )


def compute_live_overlays(
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    overlay_columns: list[str],
    *,
    tail: int = 30,
) -> dict[str, pd.Series]:
    """Recompute indicator overlay values on the *current* OHLCV (incl. the
    forming bar) and return the last ``tail`` non-NaN points per column.

    Cheap path-wise — just runs the same enrichment the static backtest uses
    (builtin engine for MMTS/ORB/LS-VWAP/IVWAP-ORB, indicator engine +
    compiler otherwise) — so we can call it from the bar-stream HTTP poll
    loop every couple of seconds to keep EMAs / Bollinger bands / VWAP etc.
    extending into new live bars."""
    if ohlcv is None or ohlcv.empty or not overlay_columns:
        return {}
    try:
        if builtin_engine_id(strategy):
            from fortuna.strategies.builtin.dispatch import run_builtin_backtest

            bt = run_builtin_backtest(strategy, ohlcv)
            enriched = bt.enriched_data
        else:
            enriched = _INDICATOR_ENGINE.compute(ohlcv, strategy.indicators)
    except Exception as exc:  # noqa: BLE001
        logger.debug("live overlays failed for %s: %s", strategy.name, exc)
        return {}

    out: dict[str, pd.Series] = {}
    for col in overlay_columns:
        if col in enriched.columns:
            s = enriched[col].tail(tail).dropna()
            if not s.empty:
                out[col] = s
    return out


def compute_live_signals(
    strategy_paths: list[Path],
    ohlcv: pd.DataFrame,
) -> dict[str, LiveSignal]:
    """Compute live signals for every strategy on disk; failures are skipped."""
    out: dict[str, LiveSignal] = {}
    if ohlcv is None or ohlcv.empty:
        return out
    for path in strategy_paths:
        try:
            strategy = load_strategy(path)
        except Exception as exc:
            logger.debug("live signal load failed %s: %s", path, exc)
            continue
        sig = compute_live_signal(strategy, ohlcv, strategy_name=path.stem)
        if sig is not None:
            out[path.stem] = sig
    return out
