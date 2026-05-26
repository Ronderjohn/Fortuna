"""``BacktestCriticAgent`` — judges a strategy or RL policy via recent OOS bars."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from fortuna.agents.base import CriticAgent, CritiqueResult
from fortuna.backtesting.metrics import BacktestMetrics
from fortuna.backtesting.standard.config import MarketCostModel
from fortuna.backtesting.standard.filters import FilterVerdict
from fortuna.features.position import PositionState
from fortuna.features.rl_indicators import enrich_for_rl
from fortuna.reporting.strategy_tester.metrics import compute_all_metrics
from fortuna.reporting.strategy_tester.trade import TradeRecord, TradeSide
from fortuna.rl.env.actions import (
    ACTION_ENTER_LONG,
    ACTION_ENTER_SHORT,
    ACTION_EXIT_LONG,
    ACTION_EXIT_SHORT,
)
from fortuna.rl.env.session import is_squareoff_time
from fortuna.rl.inference.signal_generator import RLSignalGenerator
from fortuna.strategy.schema import StrategyDefinition


class BacktestCriticAgent(CriticAgent):
    """Runs a candidate policy over recent OHLCV and judges via FilterVerdict."""

    def __init__(
        self,
        signal_generator: RLSignalGenerator,
        *,
        cost_model: Optional[MarketCostModel] = None,
        recent_n_bars: int = 600,
    ) -> None:
        self._gen = signal_generator
        self._cost = cost_model or MarketCostModel()
        self._n = int(recent_n_bars)

    def critique(
        self,
        strategy: StrategyDefinition,
        metrics: BacktestMetrics,
    ) -> CritiqueResult:
        """Phase 1 contract — delegate to a simple Sharpe threshold."""
        sharpe = float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0)
        approved = sharpe > 0.5
        issues: list[str] = [] if approved else [f"Sharpe {sharpe:.2f} below 0.5"]
        return CritiqueResult(
            approved=approved,
            score_adjustment=float(sharpe),
            issues=issues,
            suggestions=[],
        )

    def critique_policy(self, ohlcv: pd.DataFrame) -> CritiqueResult:
        """Phase 2 entry point — run RL policy over the last N bars."""
        if not self._gen.is_available:
            return CritiqueResult(
                approved=False,
                score_adjustment=0.0,
                issues=["No RL policy loaded"],
                suggestions=["Train a policy via scripts/run_rl_train.py"],
            )
        if ohlcv is None or ohlcv.empty:
            return CritiqueResult(
                approved=False,
                issues=["Empty OHLCV"],
            )

        tail = ohlcv.iloc[-self._n:] if len(ohlcv) > self._n else ohlcv
        enriched = enrich_for_rl(tail)
        if len(enriched) <= 20:
            return CritiqueResult(approved=False, issues=["Insufficient bars"])

        trades = self._simulate(enriched)
        metrics, _ = compute_all_metrics(trades, initial_capital=100_000.0)
        period_days = max(1, len(enriched) // 75)
        verdict = FilterVerdict.evaluate(metrics, period_days=period_days)

        return CritiqueResult(
            approved=bool(verdict.passed),
            score_adjustment=float(verdict.score),
            issues=list(verdict.reasons),
            suggestions=self._build_suggestions(metrics, verdict),
        )

    # ----------------------------------------------------------- internals
    def _simulate(self, enriched: pd.DataFrame) -> list[TradeRecord]:
        position = PositionState()
        trades: list[TradeRecord] = []

        for i in range(len(enriched)):
            row = enriched.iloc[i]
            ts = row.name
            price = float(row["close"])

            proposal = self._gen.predict(row, position, ts)
            if proposal is None:
                continue
            action_id = proposal.action_id

            if is_squareoff_time(ts) and position.is_open:
                action_id = ACTION_EXIT_LONG if position.is_long else ACTION_EXIT_SHORT

            if action_id == ACTION_ENTER_LONG and position.is_flat:
                position.open_long(price, ts)
            elif action_id == ACTION_ENTER_SHORT and position.is_flat:
                position.open_short(price, ts)
            elif action_id == ACTION_EXIT_LONG and position.is_long:
                self._close(trades, position, price, ts)
            elif action_id == ACTION_EXIT_SHORT and position.is_short:
                self._close(trades, position, price, ts)

            position.step()

        if position.is_open:
            last = enriched.iloc[-1]
            self._close(trades, position, float(last["close"]), last.name)

        return trades

    def _close(
        self,
        trades: list[TradeRecord],
        position: PositionState,
        price: float,
        ts: pd.Timestamp,
    ) -> None:
        side = position.side
        entry_price = position.entry_price
        entry_time = position.entry_time
        size = position.size
        realized_pct = position.close(price, ts)

        gross = (price - entry_price) if side == "LONG" else (entry_price - price)
        pnl_currency = gross * size
        cost_currency = self._cost.round_trip_rate * abs(entry_price) * size
        holding = pd.Timestamp(ts) - pd.Timestamp(entry_time)
        if not isinstance(holding, pd.Timedelta):
            holding = pd.Timedelta(0)

        trades.append(
            TradeRecord(
                entry_time=pd.Timestamp(entry_time),
                exit_time=pd.Timestamp(ts),
                entry_price=float(entry_price),
                exit_price=float(price),
                side=TradeSide.LONG if side == "LONG" else TradeSide.SHORT,
                qty=float(size),
                pnl=float(pnl_currency - cost_currency),
                pnl_percent=float(realized_pct),
                holding_time=holding,
                commission=float(cost_currency),
            )
        )

    @staticmethod
    def _build_suggestions(metrics, verdict: FilterVerdict) -> list[str]:
        suggestions: list[str] = []
        if metrics.performance.total_trades == 0:
            suggestions.append("Policy is HOLD-locked — lower holding_penalty or extend training")
        if metrics.risk.max_drawdown_pct > 0.25:
            suggestions.append("Drawdown too high — raise drawdown_penalty in RewardConfig")
        if metrics.performance.profit_factor < 1.0:
            suggestions.append("Negative expectancy — review reward shaping / increase total_timesteps")
        if not suggestions and not verdict.passed:
            suggestions.append("Verdict failed without obvious cause — retrain with a different seed")
        return suggestions
