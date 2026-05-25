"""Load and save strategy JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from fortuna.strategy.schema import StrategyDefinition


def load_strategy(path: str | Path) -> StrategyDefinition:
    """Load and validate a strategy from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return StrategyDefinition.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid strategy schema in {path}:\n{e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def save_strategy(strategy: StrategyDefinition, path: str | Path) -> None:
    """Serialize strategy to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        strategy.model_dump_json(indent=2),
        encoding="utf-8",
    )
