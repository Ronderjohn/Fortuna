"""Param grid expansion tests."""

from pathlib import Path

from fortuna.search.param_expander import ParamExpander
from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import StrategyDefinition


def test_param_expander_cartesian(ema_crossover_path: Path) -> None:
    base = load_strategy(ema_crossover_path)
    data = base.model_dump()
    data["param_grid"] = {
        "indicators.ema_fast.params.window": [8, 12],
        "indicators.ema_slow.params.window": [21, 26],
    }
    base = StrategyDefinition.model_validate(data)
    variants = ParamExpander().expand(base, max_candidates=100)
    assert len(variants) == 4
    fast_windows = {
        ind.params["window"]
        for v in variants
        for ind in v.indicators
        if ind.id == "ema_fast"
    }
    assert fast_windows == {8, 12}
