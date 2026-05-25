"""Tournament result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TournamentBarResult:
    timestamp: str
    signal: str
    winner_strategy: str
    winner_candidate_id: str
    score: float
    candidates_evaluated: int
    elapsed_ms: float
    timed_out: bool
