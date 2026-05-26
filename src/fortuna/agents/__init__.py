"""Agent interfaces and concrete Phase 1 + Phase 2 implementations."""

from fortuna.agents.base import (
    AgentProposal,
    CritiqueResult,
    CriticAgent,
    MarketContext,
    OptimizerAgent,
    ResearchAgent,
    ResearchContext,
)
from fortuna.agents.stubs import NullResearchAgent, RuleBasedCriticAgent, PassThroughOptimizerAgent

# Phase 2 concrete implementations are imported lazily — they pull in torch /
# stable-baselines3, which is the ``rl`` dependency group and should not be a
# hard requirement for callers that only need the Phase 1 ABCs.
__all__ = [
    "AgentProposal",
    "CritiqueResult",
    "CriticAgent",
    "MarketContext",
    "NullResearchAgent",
    "OptimizerAgent",
    "PassThroughOptimizerAgent",
    "ResearchAgent",
    "ResearchContext",
    "RuleBasedCriticAgent",
    # Phase 2 implementations — re-exported for convenience.
    "RLResearchAgent",
    "BacktestCriticAgent",
    "AdaptiveOptimizerAgent",
]


def __getattr__(name: str):  # noqa: D401 - module-level lazy import
    if name == "RLResearchAgent":
        from fortuna.agents.research_agent import RLResearchAgent

        return RLResearchAgent
    if name == "BacktestCriticAgent":
        from fortuna.agents.critic_agent import BacktestCriticAgent

        return BacktestCriticAgent
    if name == "AdaptiveOptimizerAgent":
        from fortuna.agents.optimizer_agent import AdaptiveOptimizerAgent

        return AdaptiveOptimizerAgent
    raise AttributeError(name)
