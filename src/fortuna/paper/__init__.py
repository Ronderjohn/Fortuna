"""Paper trading: equal-capital competition and walk-forward league."""

from fortuna.paper.account import PaperAccount
from fortuna.paper.competition import PaperCompetition, PaperCompetitionConfig, PaperCompetitionResult
from fortuna.paper.config import PaperLeagueConfig
from fortuna.paper.engine import PaperTradeEngine, PaperSessionResult
from fortuna.paper.league import PaperLeague, PaperLeagueResult

__all__ = [
    "PaperAccount",
    "PaperCompetition",
    "PaperCompetitionConfig",
    "PaperCompetitionResult",
    "PaperLeague",
    "PaperLeagueConfig",
    "PaperLeagueResult",
    "PaperSessionResult",
    "PaperTradeEngine",
]
