"""Strategy candidate with parameter binding."""

from __future__ import annotations

from dataclasses import dataclass

from fortuna.strategy.schema import StrategyDefinition


@dataclass(frozen=True)
class CandidateStrategy:
    """A strategy variant for tournament evaluation."""

    strategy: StrategyDefinition
    candidate_id: str
    source_path: str = ""

    @property
    def name(self) -> str:
        return self.strategy.name
