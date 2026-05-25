"""Paper trade session: one strategy, one isolated account, same capital as peers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.config.settings import Settings, get_settings
from fortuna.indicators.engine import IndicatorEngine
from fortuna.paper.account import PaperAccount
from fortuna.strategy.schema import StrategyDefinition, TradeSide
from fortuna.strategies.builtin.dispatch import is_builtin_strategy, run_builtin_backtest
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PaperSessionResult:
    strategy_name: str
    symbol: str
    account: PaperAccount
    fold_id: Optional[int] = None
    phase: str = "paper"

    @property
    def metrics(self):
        return self.account.metrics


class PaperTradeEngine:
    """
    Run one strategy on OHLCV in an isolated paper account.

    Each call uses ``settings.init_cash`` (or override) — competitors never share cash.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        indicator_engine: Optional[IndicatorEngine] = None,
        compiler: Optional[StrategyCompiler] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._runner = NumPyBacktestRunner(
            self.settings,
            indicator_engine or IndicatorEngine(),
            compiler or StrategyCompiler(),
        )

    def run(
        self,
        strategy: StrategyDefinition,
        ohlcv: pd.DataFrame,
        *,
        symbol: Optional[str] = None,
        init_cash: Optional[float] = None,
        fold_id: Optional[int] = None,
        phase: str = "paper",
    ) -> PaperSessionResult:
        symbol = symbol or strategy.symbol
        cash = float(init_cash if init_cash is not None else self.settings.init_cash)

        if is_builtin_strategy(strategy):
            bt = run_builtin_backtest(strategy, ohlcv, symbol=symbol, init_cash=cash)
        else:
            if strategy.side != TradeSide.LONG:
                raise NotImplementedError(
                    f"Paper engine supports long-only JSON strategies ({strategy.name}); "
                    "use MMTS for long/short."
                )
            bt = self._runner.run(strategy, ohlcv, symbol=symbol)

        trade_returns = list(bt.trade_returns or [])
        equity = bt.equity_curve
        if equity is None:
            equity = np.full(len(ohlcv), cash, dtype=np.float64)

        account = PaperAccount.from_backtest(
            strategy_name=strategy.name,
            symbol=symbol,
            init_cash=cash,
            trade_returns=trade_returns,
            equity_curve=np.asarray(equity, dtype=np.float64),
        )

        logger.info(
            "Paper %s: start=%.0f end=%.0f P&L=%.2f%% W/L=%d/%d trades=%d",
            strategy.name,
            account.init_cash,
            account.final_equity,
            account.profit_pct,
            account.wins,
            account.losses,
            account.total_trades,
        )

        return PaperSessionResult(
            strategy_name=strategy.name,
            symbol=symbol,
            account=account,
            fold_id=fold_id,
            phase=phase,
        )
