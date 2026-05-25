"""Tests for strategy DSL."""

from pathlib import Path

import pytest

from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import CompareCondition


def test_load_ema_crossover(ema_crossover_path: Path) -> None:
    strategy = load_strategy(ema_crossover_path)
    assert strategy.name == "ema_crossover"
    assert len(strategy.indicators) == 2
    assert strategy.rules.entry_conditions[0].type == "crossover"


def test_compare_condition_validation() -> None:
    cond = CompareCondition(left="rsi", operator="lt", right=30)
    assert cond.right == 30


def test_invalid_strategy_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"name": "x", "indicators": [{"id": "bad", "type": "not_a_real_indicator"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid strategy"):
        load_strategy(bad)
