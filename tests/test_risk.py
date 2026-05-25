"""Risk parameter extraction tests."""

from pathlib import Path

import pytest

from fortuna.backtesting.risk import percent_sl_tp
from fortuna.strategy.loader import load_strategy


def test_percent_sl_tp_from_strategy(ema_crossover_path: Path) -> None:
    strategy = load_strategy(ema_crossover_path)
    sl, tp = percent_sl_tp(strategy)
    assert sl == pytest.approx(0.015)
    assert tp == pytest.approx(0.03)
