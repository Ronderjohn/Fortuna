"""Compile strategy conditions into entry/exit signals."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from fortuna.strategy.schema import (
    AndCondition,
    CompareCondition,
    Condition,
    CrossoverCondition,
    CrossunderCondition,
    NotCondition,
    OrCondition,
    StrategyDefinition,
    TradeSide,
)
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyCompiler:
    """Evaluate strategy DSL conditions against enriched OHLCV data."""

    def compile(self, strategy: StrategyDefinition, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return (entries, exits) boolean series aligned to df index.

        Single-direction surface kept for backward compat. For ``side: both``
        strategies, prefer :meth:`compile_dual` so the short rules are
        evaluated alongside the long rules.
        """
        entries = self._combine_conditions(strategy.rules.entry_conditions, df)
        exits = self._combine_conditions(strategy.rules.exit_conditions, df)

        if strategy.rules.filters:
            filters = self._combine_conditions(strategy.rules.filters, df)
            entries = entries & filters

        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        if not exits.any() and entries.any():
            logger.debug("No exit signals; using entry offset exits only via vectorbt SL/TP")

        logger.info(
            "Compiled signals: %s entries, %s exits",
            int(entries.sum()),
            int(exits.sum()),
        )
        return entries, exits

    def compile_dual(
        self, strategy: StrategyDefinition, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Return ``(long_entries, long_exits, short_entries, short_exits)``.

        Direction routing:

        - ``side: long``  → primary rules drive longs; shorts are all False.
        - ``side: short`` → primary rules drive shorts; longs are all False.
        - ``side: both``  → primary rules drive longs, ``short_*_conditions``
          drive shorts. If either short block is empty the corresponding
          short series is all False (the strategy stays one-sided in
          practice but can be flagged ``both`` during migration).

        Filters AND against both the long and short entry series so the same
        constraint applies symmetrically.
        """
        primary_entries = self._combine_conditions(strategy.rules.entry_conditions, df)
        primary_exits = self._combine_conditions(strategy.rules.exit_conditions, df)
        short_entries = self._combine_conditions(strategy.rules.short_entry_conditions, df)
        short_exits = self._combine_conditions(strategy.rules.short_exit_conditions, df)

        if strategy.rules.filters:
            filters = self._combine_conditions(strategy.rules.filters, df)
            primary_entries = primary_entries & filters
            short_entries = short_entries & filters

        primary_entries = primary_entries.fillna(False).astype(bool)
        primary_exits = primary_exits.fillna(False).astype(bool)
        short_entries = short_entries.fillna(False).astype(bool)
        short_exits = short_exits.fillna(False).astype(bool)

        false_series = pd.Series(False, index=df.index)
        if strategy.side == TradeSide.LONG:
            long_e, long_x = primary_entries, primary_exits
            short_e, short_x = false_series, false_series
        elif strategy.side == TradeSide.SHORT:
            long_e, long_x = false_series, false_series
            short_e, short_x = primary_entries, primary_exits
        else:  # BOTH
            long_e, long_x = primary_entries, primary_exits
            short_e, short_x = short_entries, short_exits

        logger.debug(
            "compile_dual %s: long=%s/%s short=%s/%s",
            strategy.name,
            int(long_e.sum()),
            int(long_x.sum()),
            int(short_e.sum()),
            int(short_x.sum()),
        )
        return long_e, long_x, short_e, short_x

    def _combine_conditions(self, conditions: list[Condition], df: pd.DataFrame) -> pd.Series:
        if not conditions:
            return pd.Series(False, index=df.index)
        if len(conditions) == 1:
            return self._eval_condition(conditions[0], df)
        result = self._eval_condition(conditions[0], df)
        for cond in conditions[1:]:
            result = result | self._eval_condition(cond, df)
        return result

    def _eval_condition(self, condition: Condition, df: pd.DataFrame) -> pd.Series:
        if isinstance(condition, CompareCondition):
            return self._eval_compare(condition, df)
        if isinstance(condition, CrossoverCondition):
            return self._eval_crossover(condition, df)
        if isinstance(condition, CrossunderCondition):
            return self._eval_crossunder(condition, df)
        if isinstance(condition, AndCondition):
            parts = [self._eval_condition(c, df) for c in condition.conditions]
            out = parts[0]
            for p in parts[1:]:
                out = out & p
            return out
        if isinstance(condition, OrCondition):
            parts = [self._eval_condition(c, df) for c in condition.conditions]
            out = parts[0]
            for p in parts[1:]:
                out = out | p
            return out
        if isinstance(condition, NotCondition):
            return ~self._eval_condition(condition.condition, df)
        raise TypeError(f"Unknown condition type: {type(condition)}")

    def _resolve_operand(self, operand: Union[str, float, int], df: pd.DataFrame) -> pd.Series:
        if isinstance(operand, (int, float)):
            return pd.Series(float(operand), index=df.index, dtype=float)
        key = str(operand).lower()
        if key in df.columns:
            return df[key]
        raise KeyError(f"Operand '{operand}' not found in data columns: {list(df.columns)}")

    def _eval_compare(self, cond: CompareCondition, df: pd.DataFrame) -> pd.Series:
        left = self._resolve_operand(cond.left, df)
        right = self._resolve_operand(cond.right, df)
        ops = {
            "gt": left > right,
            "gte": left >= right,
            "lt": left < right,
            "lte": left <= right,
            "eq": left == right,
            "neq": left != right,
        }
        return ops[cond.operator].fillna(False)

    def _eval_crossover(self, cond: CrossoverCondition, df: pd.DataFrame) -> pd.Series:
        left = np.asarray(self._resolve_operand(cond.left, df), dtype=np.float64)
        right = np.asarray(self._resolve_operand(cond.right, df), dtype=np.float64)
        prev_left = np.roll(left, 1)
        prev_right = np.roll(right, 1)
        prev_left[0] = left[0]
        prev_right[0] = right[0]
        crossed = (left > right) & (prev_left <= prev_right)
        return pd.Series(crossed, index=df.index, dtype=bool)

    def _eval_crossunder(self, cond: CrossunderCondition, df: pd.DataFrame) -> pd.Series:
        left = np.asarray(self._resolve_operand(cond.left, df), dtype=np.float64)
        right = np.asarray(self._resolve_operand(cond.right, df), dtype=np.float64)
        prev_left = np.roll(left, 1)
        prev_right = np.roll(right, 1)
        prev_left[0] = left[0]
        prev_right[0] = right[0]
        crossed = (left < right) & (prev_left >= prev_right)
        return pd.Series(crossed, index=df.index, dtype=bool)
