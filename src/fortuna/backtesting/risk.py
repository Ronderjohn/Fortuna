"""Shared risk parameter extraction for backtest runners."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fortuna.strategy.schema import StrategyDefinition
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


def _type_value(field: Any) -> str:
    t = getattr(field, "type", None)
    if t is None:
        return ""
    return t.value if hasattr(t, "value") else str(t)


def percent_sl_tp(strategy: StrategyDefinition) -> tuple[Optional[float], Optional[float]]:
    """Return (stop_loss_pct, take_profit_pct) as positive fractions, or None."""
    sl_pct: Optional[float] = None
    tp_pct: Optional[float] = None
    stop_loss = strategy.risk.stop_loss
    if stop_loss and _type_value(stop_loss) == "percent":
        sl_pct = float(stop_loss.value)
    elif stop_loss and _type_value(stop_loss) == "atr":
        logger.debug("ATR stops handled by vectorbt runner only in Phase 1")
    if strategy.risk.take_profit and _type_value(strategy.risk.take_profit) == "percent":
        tp_pct = float(strategy.risk.take_profit.value)
    return sl_pct, tp_pct


def vectorbt_sl_tp(
    strategy: StrategyDefinition,
    enriched: pd.DataFrame,
) -> tuple[Optional[float | np.ndarray], Optional[float]]:
    """Stop/take-profit params for vectorbt Portfolio.from_signals."""
    sl_stop: Optional[float | np.ndarray] = None
    tp_stop: Optional[float] = None
    stop_loss = strategy.risk.stop_loss
    if stop_loss:
        if _type_value(stop_loss) == "percent":
            sl_stop = stop_loss.value
        elif _type_value(stop_loss) == "atr":
            atr_id = stop_loss.atr_indicator_id
            if atr_id and atr_id in enriched.columns:
                close = enriched["close"]
                sl_stop = (stop_loss.atr_multiplier * enriched[atr_id] / close).to_numpy(
                    dtype=float, copy=False
                )
            else:
                logger.warning(
                    "ATR stop configured but indicator %r missing; skipping stop loss",
                    atr_id,
                )
    if strategy.risk.take_profit:
        tp_stop = strategy.risk.take_profit.value
    return sl_stop, tp_stop
