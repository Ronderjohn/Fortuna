"""Per-bar BUY / SELL / HOLD and ENTER / EXIT flags for chart overlays."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.search.signals import emit_signal
from fortuna.strategy.schema import StrategyDefinition, TradeSide
from fortuna.strategies.builtin.dispatch import builtin_engine_id


def build_signal_frame(
    strategy: StrategyDefinition,
    ohlcv: pd.DataFrame,
    enriched: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build per-bar signal table aligned to ``ohlcv.index``.

    Columns:
    - ``signal``: BUY | SELL | HOLD (rule-level, TradingView-style)
    - ``enter``: True when an entry rule fires (opening intent)
    - ``exit``: True when an exit rule fires (closing intent)
    """
    frame = enriched if enriched is not None else ohlcv
    if not frame.index.equals(ohlcv.index):
        frame = frame.reindex(ohlcv.index)

    engine = builtin_engine_id(strategy)
    if engine in ("mmts", "orb", "lsvwap", "ivorb"):
        return _builtin_signal_frame(frame, strategy.side)

    return _compiler_signal_frame(strategy, frame)


def _compiler_signal_frame(
    strategy: StrategyDefinition,
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    compiler = StrategyCompiler()
    entries, exits = compiler.compile(strategy, enriched)
    entry_b = entries.fillna(False).to_numpy(dtype=bool)
    exit_b = exits.fillna(False).to_numpy(dtype=bool)
    n = len(enriched)
    signals = np.empty(n, dtype=object)
    for i in range(n):
        signals[i] = emit_signal(strategy, bool(entry_b[i]), bool(exit_b[i]))
    enter = entry_b & ~exit_b
    exit_mask = exit_b & ~entry_b
    return pd.DataFrame(
        {"signal": signals, "enter": enter, "exit": exit_mask},
        index=enriched.index,
    )


def _builtin_signal_frame(enriched: pd.DataFrame, side: TradeSide) -> pd.DataFrame:
    long_e = enriched.get("long_entry", pd.Series(False, index=enriched.index))
    short_e = enriched.get("short_entry", pd.Series(False, index=enriched.index))
    long_e = long_e.fillna(False).to_numpy(dtype=bool)
    short_e = short_e.fillna(False).to_numpy(dtype=bool)
    n = len(enriched)
    signals = np.full(n, "HOLD", dtype=object)
    enter = np.zeros(n, dtype=bool)
    exit_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        le, se = long_e[i], short_e[i]
        if le and not se:
            signals[i] = "BUY"
            enter[i] = True
        elif se and not le:
            signals[i] = "SELL"
            enter[i] = True
        elif le and se:
            signals[i] = "BUY" if side != TradeSide.SHORT else "SELL"
            enter[i] = True

    return pd.DataFrame(
        {"signal": signals, "enter": enter, "exit": exit_mask},
        index=enriched.index,
    )


def suggest_overlay_columns(
    strategy: StrategyDefinition,
    enriched: Optional[pd.DataFrame],
    *,
    max_lines: int = 4,
) -> list[str]:
    """Indicator columns to draw on the price pane."""
    if enriched is None or enriched.empty:
        return []

    engine = builtin_engine_id(strategy)
    builtin_defaults: dict[str, list[str]] = {
        "mmts": ["ema_fast", "ema_slow", "bb_basis", "bb_upper", "bb_lower"],
        "orb": ["or_high", "or_low"],
        "lsvwap": ["ema_fast", "vwap", "supertrend"],
        "ivorb": ["ema_fast", "vwap", "or_high", "or_low"],
    }
    if engine and engine in builtin_defaults:
        cols = [c for c in builtin_defaults[engine] if c in enriched.columns]
        return cols[:max_lines]

    candidates: list[str] = []
    for ind in strategy.indicators:
        iid = ind.id
        if iid in enriched.columns:
            candidates.append(iid)
        if ind.type.value == "macd":
            for suffix in ("_signal", "_hist"):
                col = f"{iid}{suffix}" if not iid.endswith(suffix) else iid
                if col in enriched.columns:
                    candidates.append(col)
        if ind.type.value == "bollinger":
            for suffix in ("_upper", "_lower", "_mid"):
                col = f"{iid}{suffix}"
                if col in enriched.columns:
                    candidates.append(col)

    for name in ("ema_fast", "ema_slow", "vwap", "bb_upper", "bb_lower"):
        if name in enriched.columns and name not in candidates:
            candidates.append(name)

    out: list[str] = []
    for c in candidates:
        if c in enriched.columns and c not in out:
            out.append(c)
        if len(out) >= max_lines:
            break
    return out
