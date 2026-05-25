"""Paper competition: equal capital, independent accounts."""

from pathlib import Path

import pandas as pd

from fortuna.paper.account import PaperAccount
from fortuna.paper.competition import PaperCompetition, PaperCompetitionConfig


def test_paper_account_counts_wins_losses() -> None:
    import numpy as np

    returns = [0.02, -0.01, 0.0, -0.03, 0.01]
    equity = np.array([100_000, 102_000, 100_980, 100_980, 97_950, 98_930], dtype=float)
    acc = PaperAccount.from_backtest(
        strategy_name="test",
        symbol="X.NS",
        init_cash=100_000,
        trade_returns=returns,
        equity_curve=equity[: len(returns) + 1],
    )
    assert acc.wins == 2
    assert acc.losses == 2
    assert acc.breakeven == 1
    assert acc.total_trades == 5


def test_two_accounts_same_start_different_pnl() -> None:
    import numpy as np

    a = PaperAccount.from_backtest(
        strategy_name="a",
        symbol="X",
        init_cash=100_000,
        trade_returns=[0.05, 0.02],
        equity_curve=np.array([100_000, 105_000, 107_100]),
    )
    b = PaperAccount.from_backtest(
        strategy_name="b",
        symbol="X",
        init_cash=100_000,
        trade_returns=[-0.02, -0.01],
        equity_curve=np.array([100_000, 98_000, 97_020]),
    )
    assert a.init_cash == b.init_cash == 100_000
    assert a.final_equity != b.final_equity
    assert a.wins == 2 and b.losses == 2
