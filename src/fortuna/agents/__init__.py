"""Agent interfaces for future LLM integration."""

from fortuna.agents.base import (
    CritiqueResult,
    CriticAgent,
    OptimizerAgent,
    ResearchAgent,
    ResearchContext,
)
from fortuna.agents.stubs import NullResearchAgent, RuleBasedCriticAgent, PassThroughOptimizerAgent

__all__ = [
    "CritiqueResult",
    "CriticAgent",
    "NullResearchAgent",
    "OptimizerAgent",
    "PassThroughOptimizerAgent",
    "ResearchAgent",
    "ResearchContext",
    "RuleBasedCriticAgent",
]
