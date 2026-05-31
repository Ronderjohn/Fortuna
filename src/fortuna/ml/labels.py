"""Supervised label generation for strategy signal events."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from fortuna.ml.types import LabelConfig, SignalExample

ACTIONABLE = frozenset({"BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"})
ENTRY_ACTIONS = frozenset({"BUY", "SELL"})
EXIT_ACTIONS = frozenset({"EXIT_LONG", "EXIT_SHORT"})


def directional_multiplier(action: str) -> float:
    """Return sign multiplier for raw forward pct given signal action."""
    act = str(action).upper()
    if act == "BUY":
        return 1.0
    if act == "SELL":
        return -1.0
    if act == "EXIT_LONG":
        return -1.0
    if act == "EXIT_SHORT":
        return 1.0
    return 0.0


def raw_forward_pct(ohlcv: pd.DataFrame, bar_idx: int, horizon_bars: int) -> Optional[float]:
    """Close-to-close forward return in percent from bar_idx to bar_idx+horizon."""
    end_idx = bar_idx + horizon_bars
    if bar_idx < 0 or end_idx >= len(ohlcv):
        return None
    if "close" not in ohlcv.columns:
        return None
    cur = float(ohlcv.iloc[bar_idx]["close"])
    nxt = float(ohlcv.iloc[end_idx]["close"])
    if cur == 0:
        return None
    return (nxt - cur) / cur * 100.0


def forward_return_at_bar(
    ohlcv: pd.DataFrame,
    bar_idx: int,
    action: str,
    horizon_bars: int,
) -> Optional[float]:
    """Direction-adjusted forward return in percent."""
    raw = raw_forward_pct(ohlcv, bar_idx, horizon_bars)
    if raw is None:
        return None
    mult = directional_multiplier(action)
    if mult == 0.0:
        return None
    return raw * mult


def cost_for_action(action: str, label_config: LabelConfig) -> float:
    """Entry labels use round-trip cost; exit labels use one leg."""
    act = str(action).upper()
    total = label_config.cost_model.total_cost_pct()
    if act in ENTRY_ACTIONS:
        return total
    if act in EXIT_ACTIONS:
        return total / 2.0
    return total


def label_from_forward_return(
    forward_return_pct: float,
    action: str,
    *,
    label_config: LabelConfig,
) -> tuple[int, float]:
    """Return binary label and cost-adjusted return."""
    cost = cost_for_action(action, label_config)
    adjusted = forward_return_pct - cost
    label = 1 if adjusted > label_config.min_return_pct else 0
    return label, adjusted


def build_signal_examples(
    ohlcv: pd.DataFrame,
    signal_events: pd.DataFrame,
    *,
    symbol: str = "",
    label_config: LabelConfig | None = None,
) -> list[SignalExample]:
    """Build labeled examples from OHLCV and per-bar signal events.

    ``signal_events`` must contain columns: bar_idx, strategy_name, action.
    Optional: bar_time, symbol.
    """
    cfg = label_config or LabelConfig()
    required = {"bar_idx", "strategy_name", "action"}
    missing = required - set(signal_events.columns)
    if missing:
        raise ValueError(f"signal_events missing columns: {sorted(missing)}")

    examples: list[SignalExample] = []
    for _, row in signal_events.iterrows():
        action = str(row["action"]).upper()
        if action not in ACTIONABLE:
            if not (cfg.include_hold and action == "HOLD"):
                continue
            if cfg.include_hold and cfg.hold_sample_rate < 1.0:
                bar_idx = int(row["bar_idx"])
                if (bar_idx * 9973) % 100 >= int(cfg.hold_sample_rate * 100):
                    continue

        bar_idx = int(row["bar_idx"])
        fwd = forward_return_at_bar(ohlcv, bar_idx, action, cfg.horizon_bars)
        if fwd is None:
            continue

        label, adjusted = label_from_forward_return(fwd, action, label_config=cfg)
        if "bar_time" in signal_events.columns and pd.notna(row.get("bar_time")):
            bar_time = pd.Timestamp(row["bar_time"])
        elif bar_idx < len(ohlcv):
            bar_time = pd.Timestamp(ohlcv.index[bar_idx])
        else:
            bar_time = pd.Timestamp("1970-01-01")

        examples.append(
            SignalExample(
                bar_idx=bar_idx,
                bar_time=bar_time,
                symbol=str(symbol or row.get("symbol", "")),
                strategy_name=str(row["strategy_name"]),
                action=action,
                label=label,
                forward_return_pct=fwd,
                cost_adjusted_return_pct=adjusted,
            )
        )
    return examples


def events_from_entry_exit_columns(
    enriched: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    *,
    strategy_name: str,
    side: str = "LONG",
) -> pd.DataFrame:
    """Build signal event rows from boolean entry/exit columns."""
    rows: list[dict] = []
    side_u = str(side).upper()
    for bar_idx in range(len(enriched)):
        entry = bool(entries.iloc[bar_idx]) if bar_idx < len(entries) else False
        exit_sig = bool(exits.iloc[bar_idx]) if bar_idx < len(exits) else False
        if entry and not exit_sig:
            action = "SELL" if side_u == "SHORT" else "BUY"
            rows.append(
                {
                    "bar_idx": bar_idx,
                    "strategy_name": strategy_name,
                    "action": action,
                    "bar_time": enriched.index[bar_idx],
                }
            )
        elif exit_sig and not entry:
            action = "EXIT_SHORT" if side_u == "SHORT" else "EXIT_LONG"
            rows.append(
                {
                    "bar_idx": bar_idx,
                    "strategy_name": strategy_name,
                    "action": action,
                    "bar_time": enriched.index[bar_idx],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["bar_idx", "strategy_name", "action", "bar_time"])
    return pd.DataFrame(rows)
