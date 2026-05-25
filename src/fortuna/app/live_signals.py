"""Live BUY / SELL / HOLD / EXIT signals for the latest (forming) bar.

Mirrors TradingView strategy-tester semantics:

- ``ENTRY``: an entry rule fires on the latest closed/forming bar
- ``EXIT``: an exit rule fires on the latest closed/forming bar
- ``IN POSITION``: previous entry without a matching exit yet
- ``HOLD``: nothing actionable

Light-weight: re-runs only the **indicator engine + compiler** (or the builtin
engine's signal columns) on the current OHLCV — no full vectorbt backtest. Safe
to call on every Streamlit refresh / WebSocket tick.
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


@dataclass
class LiveSignal:
    """Current per-strategy state for the dashboard live panel."""

    strategy_name: str
    action: str
    """One of: BUY, SELL, EXIT_LONG, EXIT_SHORT, IN_LONG, IN_SHORT, HOLD."""
    label: str
    """Short label suitable for badges (BUY / SELL / EXIT / HOLD / LONG / SHORT)."""
    bar_time: pd.Timestamp
    bar_close: float
    enter: bool
    exit: bool
    in_position: bool
    side: Optional[str] = None
    entry_price: Optional[float] = None
    bars_in_trade: int = 0
    color: str = "#9E9E9E"

    def is_actionable(self) -> bool:
        return self.action in {"BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"}


def _builtin_entry_exit(enriched: pd.DataFrame, side: TradeSide) -> tuple[pd.Series, pd.Series]:
    """Return (entries, exits) for a builtin engine using long_entry/short_entry columns."""
    long_e = enriched.get("long_entry", pd.Series(False, index=enriched.index)).fillna(False).astype(bool)
    short_e = enriched.get("short_entry", pd.Series(False, index=enriched.index)).fillna(False).astype(bool)
    if side == TradeSide.SHORT:
        return short_e, long_e
    return long_e, short_e


def _trade_state(entries: pd.Series, exits: pd.Series) -> tuple[bool, int, int]:
    """Walk forward to determine current position state.

    Returns ``(in_position, entry_idx_or_-1, bars_in_trade)``.
    Exits resolve any open position; an entry on the same bar as an exit opens a new one.
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


def _action_for(strategy: StrategyDefinition, enter: bool, exit: bool, in_position: bool) -> tuple[str, str, str]:
    """Map raw entry/exit/position to (action, label, color)."""
    is_short = strategy.side == TradeSide.SHORT
    if exit and in_position:
        return ("EXIT_SHORT" if is_short else "EXIT_LONG", "EXIT", "#FF9800")
    if enter and not in_position:
        return ("SELL" if is_short else "BUY", "SELL" if is_short else "BUY", "#EF5350" if is_short else "#26A69A")
    if in_position:
        return ("IN_SHORT" if is_short else "IN_LONG", "SHORT" if is_short else "LONG", "#2962FF")
    return ("HOLD", "HOLD", "#787B86")


def compute_live_signal(
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    *,
    strategy_name: Optional[str] = None,
) -> Optional[LiveSignal]:
    """Compute the current (latest-bar) signal for one strategy.

    Returns ``None`` if there is not enough data to evaluate.
    """
    if ohlcv is None or ohlcv.empty:
        return None
    name = strategy_name or strategy.name
    try:
        if builtin_engine_id(strategy):
            from fortuna.strategies.builtin.dispatch import run_builtin_backtest

            bt = run_builtin_backtest(strategy, ohlcv)
            enriched = bt.enriched_data
            entries, exits = _builtin_entry_exit(enriched, strategy.side)
        else:
            enriched = _INDICATOR_ENGINE.compute(ohlcv, strategy.indicators)
            entries, exits = _COMPILER.compile(strategy, enriched)
    except Exception as exc:
        logger.debug("live signal failed for %s: %s", name, exc)
        return None

    if entries.empty:
        return None

    in_pos, entry_idx, bars_in_trade = _trade_state(entries, exits)
    enter_now = bool(entries.iloc[-1])
    exit_now = bool(exits.iloc[-1])
    last_idx = ohlcv.index[-1]
    last_close = float(ohlcv["close"].iloc[-1])
    side_label = "SHORT" if strategy.side == TradeSide.SHORT else "LONG"
    action, label, color = _action_for(strategy, enter_now, exit_now, in_pos)

    entry_price: Optional[float] = None
    if in_pos and 0 <= entry_idx < len(ohlcv):
        entry_price = float(ohlcv["close"].iloc[entry_idx])

    return LiveSignal(
        strategy_name=name,
        action=action,
        label=label,
        bar_time=pd.Timestamp(last_idx),
        bar_close=last_close,
        enter=enter_now,
        exit=exit_now,
        in_position=in_pos,
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
    (builtin engine for MMTS/ORB, indicator engine + compiler otherwise) — so
    we can call it from the bar-stream HTTP poll loop every couple of seconds
    to keep EMAs / Bollinger bands / VWAP etc. extending into new live bars."""
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
