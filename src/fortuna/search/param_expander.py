"""Expand strategy parameter grids into candidate variants."""

from __future__ import annotations

import copy
import itertools
from typing import Any

from fortuna.strategy.schema import StrategyDefinition


def filter_param_grid(base: StrategyDefinition, grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Keep only grid keys that resolve on this strategy (e.g. skip EMA grid on RSI)."""
    if not grid:
        return {}
    base_dict = base.model_dump()
    out: dict[str, list[Any]] = {}
    for key, values in grid.items():
        if not values:
            continue
        probe = copy.deepcopy(base_dict)
        probe.pop("param_grid", None)
        try:
            _set_path(probe, key, values[0])
        except KeyError:
            continue
        out[key] = values
    return out


class ParamExpander:
    """Cartesian expansion of param_grid paths on StrategyDefinition."""

    def expand(self, base: StrategyDefinition, max_candidates: int = 500) -> list[StrategyDefinition]:
        grid = filter_param_grid(base, base.param_grid or {})
        if not grid:
            return [base.model_copy(deep=True)]

        keys = list(grid.keys())
        value_lists = [grid[k] for k in keys]
        combos = list(itertools.product(*value_lists))
        if len(combos) > max_candidates:
            combos = combos[:max_candidates]

        results: list[StrategyDefinition] = []
        base_dict = base.model_dump()
        for combo in combos:
            data = copy.deepcopy(base_dict)
            data.pop("param_grid", None)
            for key, value in zip(keys, combo):
                _set_path(data, key, value)
            variant = StrategyDefinition.model_validate(data)
            variant.name = f"{base.name}_{len(results)}"
            results.append(variant)
        return results


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set nested dict value; supports indicators.{id}.params.{key}."""
    if path.startswith("indicators."):
        parts = path.split(".")
        if len(parts) >= 4 and parts[2] == "params":
            ind_id = parts[1]
            param_key = parts[3]
            for ind in data.get("indicators", []):
                if ind.get("id") == ind_id:
                    ind.setdefault("params", {})[param_key] = value
                    return
        raise KeyError(f"Cannot resolve indicator path: {path}")

    parts = path.split(".")
    cur: Any = data
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
