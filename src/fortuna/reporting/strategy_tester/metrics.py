"""TradingView Strategy Tester metric calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from fortuna.reporting.strategy_tester.trade import TradeRecord


@dataclass
class PerformanceMetrics:
    """Core performance block."""

    total_net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate_pct: float = 0.0
    average_trade: float = 0.0
    average_winning_trade: float = 0.0
    average_losing_trade: float = 0.0
    largest_winning_trade: float = 0.0
    largest_losing_trade: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0


@dataclass
class RiskMetrics:
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_hold_time: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    expectancy: float = 0.0
    risk_reward_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0


@dataclass
class IntradayMetrics:
    trades_per_day: float = 0.0
    average_pnl_per_day: float = 0.0
    best_trading_hour: int = -1
    worst_trading_hour: int = -1
    best_hour_pnl: float = 0.0
    worst_hour_pnl: float = 0.0
    session_opening_pnl: float = 0.0
    session_midday_pnl: float = 0.0
    session_closing_pnl: float = 0.0
    trading_days: int = 0


@dataclass
class StrategyMetrics:
    performance: PerformanceMetrics
    risk: RiskMetrics
    intraday: IntradayMetrics

    def comparison_row(self, strategy_name: str) -> dict[str, Any]:
        p, r = self.performance, self.risk
        return {
            "strategy_name": strategy_name,
            "total_profit": round(p.total_net_profit, 2),
            "total_profit_pct": 0.0,
            "max_drawdown": round(r.max_drawdown, 2),
            "max_drawdown_pct": round(r.max_drawdown_pct * 100, 4),
            "sharpe": round(r.sharpe_ratio, 4),
            "sortino": round(r.sortino_ratio, 4),
            "profit_factor": round(p.profit_factor, 4),
            "expectancy": round(r.expectancy, 4),
            "win_rate": round(p.win_rate_pct, 2),
            "total_trades": p.total_trades,
        }

    @classmethod
    def zero(cls) -> "StrategyMetrics":
        """Sentinel "no-trade" metrics. Sharpe is -1e9 so FilterVerdict.FAIL is guaranteed.

        Used by the RL training loop when an early-stage policy produces zero trades
        on a fold (common when the policy learns to HOLD everything). Downstream
        consumers can treat these as the canonical "this strategy didn't trade" reply.
        """
        risk = RiskMetrics()
        risk.sharpe_ratio = -1e9
        risk.sortino_ratio = -1e9
        risk.calmar_ratio = -1e9
        return cls(
            performance=PerformanceMetrics(),
            risk=risk,
            intraday=IntradayMetrics(),
        )


def _max_streak(pnls: np.ndarray, positive: bool) -> int:
    best = cur = 0
    for p in pnls:
        hit = (p > 0) if positive else (p < 0)
        if hit:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def equity_curve_from_trades(
    trades: list[TradeRecord],
    initial_capital: float,
) -> pd.DataFrame:
    """Post-trade equity (TradingView: equity updates when trades close)."""
    if not trades:
        return pd.DataFrame(
            {"timestamp": [], "equity": [], "drawdown": [], "drawdown_pct": []}
        )
    rows = []
    equity = float(initial_capital)
    peak = equity
    for t in sorted(trades, key=lambda x: x.exit_time):
        equity += t.pnl
        peak = max(peak, equity)
        dd = peak - equity
        dd_pct = dd / peak if peak > 0 else 0.0
        rows.append(
            {
                "timestamp": t.exit_time,
                "equity": equity,
                "drawdown": dd,
                "drawdown_pct": dd_pct,
            }
        )
    return pd.DataFrame(rows)


def drawdown_series(equity: pd.Series) -> tuple[pd.Series, pd.Series]:
    peak = equity.cummax()
    dd = peak - equity
    dd_pct = dd / peak.replace(0, np.nan)
    return dd, dd_pct.fillna(0.0)


def compute_performance(trades: list[TradeRecord]) -> PerformanceMetrics:
    m = PerformanceMetrics()
    if not trades:
        return m

    pnls = np.array([t.pnl for t in trades], dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    m.total_commission = float(sum(t.commission for t in trades))
    m.total_slippage = float(sum(t.slippage for t in trades))
    m.gross_profit = float(wins.sum()) if len(wins) else 0.0
    m.gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    m.total_net_profit = float(pnls.sum())
    m.total_trades = len(trades)
    m.winning_trades = int(len(wins))
    m.losing_trades = int(len(losses))
    m.breakeven_trades = int(np.sum(pnls == 0))
    m.win_rate_pct = (m.winning_trades / m.total_trades * 100.0) if m.total_trades else 0.0

    if m.gross_loss > 0:
        m.profit_factor = m.gross_profit / m.gross_loss
    else:
        m.profit_factor = float("inf") if m.gross_profit > 0 else 0.0
    m.profit_factor = float(min(m.profit_factor, 999.0))

    m.average_trade = m.total_net_profit / m.total_trades
    m.average_winning_trade = float(wins.mean()) if len(wins) else 0.0
    m.average_losing_trade = float(losses.mean()) if len(losses) else 0.0
    m.largest_winning_trade = float(wins.max()) if len(wins) else 0.0
    m.largest_losing_trade = float(losses.min()) if len(losses) else 0.0
    return m


def compute_risk(
    trades: list[TradeRecord],
    trade_equity: pd.DataFrame,
    initial_capital: float,
    bar_equity: Optional[pd.Series] = None,
) -> RiskMetrics:
    r = RiskMetrics()
    if not trades:
        return r

    pnls = np.array([t.pnl for t in trades], dtype=np.float64)
    r.max_consecutive_wins = _max_streak(pnls, positive=True)
    r.max_consecutive_losses = _max_streak(pnls, positive=False)

    holds = [t.holding_time for t in trades if t.holding_time is not None]
    if holds:
        r.average_hold_time = pd.Timedelta(np.mean([h.total_seconds() for h in holds]), unit="s")

    win_rate = (pnls > 0).mean()
    loss_rate = (pnls < 0).mean()
    avg_win = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0.0
    avg_loss = float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0.0
    r.expectancy = win_rate * avg_win - loss_rate * avg_loss
    r.risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)

    if not trade_equity.empty:
        eq = trade_equity["equity"]
        dd, dd_pct = drawdown_series(eq)
        r.max_drawdown = float(dd.max())
        r.max_drawdown_pct = float(dd_pct.max())
    elif bar_equity is not None and len(bar_equity) > 1:
        dd, dd_pct = drawdown_series(bar_equity)
        r.max_drawdown = float(dd.max())
        r.max_drawdown_pct = float(dd_pct.max())

    returns = _daily_returns(trades, trade_equity, bar_equity, initial_capital)
    r.sharpe_ratio = _sharpe(returns)
    r.sortino_ratio = _sortino(returns)
    r.calmar_ratio = _calmar(returns, r.max_drawdown_pct, initial_capital, trade_equity)

    return r


def _daily_returns(
    trades: list[TradeRecord],
    trade_equity: pd.DataFrame,
    bar_equity: Optional[pd.Series],
    initial_capital: float,
) -> pd.Series:
    if bar_equity is not None and len(bar_equity) > 2:
        daily = bar_equity.resample("1D").last().dropna()
        return daily.pct_change().dropna()
    if not trade_equity.empty:
        s = trade_equity.set_index("timestamp")["equity"]
        daily = s.resample("1D").last().dropna()
        if len(daily) > 1:
            return daily.pct_change().dropna()
    if trades:
        df = pd.DataFrame(
            {"exit_time": [t.exit_time for t in trades], "pnl": [t.pnl for t in trades]}
        ).set_index("exit_time")
        daily_pnl = df["pnl"].resample("1D").sum()
        return daily_pnl / initial_capital
    return pd.Series(dtype=float)


def _sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(returns) < 2 or returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 1 or downside.std() < 1e-12:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def _calmar(
    returns: pd.Series,
    max_dd_pct: float,
    initial_capital: float,
    trade_equity: pd.DataFrame,
) -> float:
    if max_dd_pct < 1e-12:
        return 0.0
    if not trade_equity.empty and len(trade_equity) >= 2:
        start = trade_equity["timestamp"].iloc[0]
        end = trade_equity["timestamp"].iloc[-1]
        years = max((end - start).days / 365.25, 1 / 365.25)
        final = trade_equity["equity"].iloc[-1]
        annual = (final / initial_capital) ** (1 / years) - 1
        return float(annual / max_dd_pct)
    if len(returns) > 0:
        annual = float(returns.mean() * 252)
        return annual / max_dd_pct
    return 0.0


def _session_bucket(ts: pd.Timestamp) -> str:
    """NSE cash session buckets (IST assumed on index)."""
    h, m = ts.hour, ts.minute
    minutes = h * 60 + m
    open_start = 9 * 60 + 15
    mid_start = 10 * 60 + 30
    close_start = 14 * 60
    if minutes < open_start:
        return "pre"
    if minutes < mid_start:
        return "opening"
    if minutes < close_start:
        return "midday"
    return "closing"


def compute_intraday(trades: list[TradeRecord]) -> IntradayMetrics:
    im = IntradayMetrics()
    if not trades:
        return im

    df = pd.DataFrame(
        {
            "exit_time": [t.exit_time for t in trades],
            "pnl": [t.pnl for t in trades],
        }
    )
    df["date"] = df["exit_time"].dt.date
    df["hour"] = df["exit_time"].dt.hour

    days = df["date"].nunique()
    im.trading_days = int(days)
    im.trades_per_day = len(trades) / days if days else 0.0
    daily_pnl = df.groupby("date")["pnl"].sum()
    im.average_pnl_per_day = float(daily_pnl.mean()) if len(daily_pnl) else 0.0

    hourly = df.groupby("hour")["pnl"].sum()
    if len(hourly):
        im.best_trading_hour = int(hourly.idxmax())
        im.worst_trading_hour = int(hourly.idxmin())
        im.best_hour_pnl = float(hourly.max())
        im.worst_hour_pnl = float(hourly.min())

    df["session"] = df["exit_time"].apply(_session_bucket)
    sess = df.groupby("session")["pnl"].sum()
    im.session_opening_pnl = float(sess.get("opening", 0.0))
    im.session_midday_pnl = float(sess.get("midday", 0.0))
    im.session_closing_pnl = float(sess.get("closing", 0.0))
    return im


def compute_all_metrics(
    trades: list[TradeRecord],
    initial_capital: float,
    bar_equity: Optional[pd.Series] = None,
) -> tuple[StrategyMetrics, pd.DataFrame]:
    trade_eq = equity_curve_from_trades(trades, initial_capital)
    perf = compute_performance(trades)
    risk = compute_risk(trades, trade_eq, initial_capital, bar_equity)
    intraday = compute_intraday(trades)
    return StrategyMetrics(performance=perf, risk=risk, intraday=intraday), trade_eq
