"""Format backtest results for Streamlit dashboard (presentable reports)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from fortuna.app.parallel_runner import BatchRunResult
from fortuna.reporting.strategy_tester.report import StrategyTesterReport

__all__ = [
    "leaderboard_dataframe",
    "render_leaderboard",
    "render_ohlcv_ticker",
    "render_strategy_report_card",
    "render_strategy_comparison_grid",
    "export_leaderboard_csv",
    "render_live_signal_banner",
    "render_live_signal_panel",
]


def _pct_color(val: float) -> str:
    if val > 0:
        return "color: #26A69A; font-weight: 600"
    if val < 0:
        return "color: #EF5350; font-weight: 600"
    return "color: #787B86"


def leaderboard_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "strategy": "Strategy",
            "profit_pct": "Return %",
            "net_profit": "Net P&L (₹)",
            "trades": "Trades",
            "wins": "Wins",
            "losses": "Losses",
            "win_rate_pct": "Win rate %",
            "profit_factor": "Profit factor",
            "max_dd_pct": "Max DD %",
            "sharpe": "Sharpe",
        }
    )
    return df


def render_leaderboard(batch: BatchRunResult, winner: Optional[str]) -> None:
    import streamlit as st

    rows = batch.leaderboard_rows()
    if not rows:
        st.warning("No strategy results to display.")
        return

    if winner:
        w = next(r for r in rows if r["strategy"] == winner)
        st.markdown(
            f"""
            <div style="background: linear-gradient(90deg, #1e3a5f 0%, #2962FF 100%);
                padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1rem;">
              <span style="color: #B2B5BE; font-size: 0.85rem;">CHAMPION STRATEGY</span><br/>
              <span style="color: white; font-size: 1.5rem; font-weight: 700;">{winner}</span>
              <span style="color: {'#26A69A' if w['profit_pct'] >= 0 else '#EF5350'};
                font-size: 1.25rem; margin-left: 0.75rem;">
                {w['profit_pct']:+.2f}%
              </span>
              <span style="color: #B2B5BE; font-size: 0.9rem; margin-left: 1rem;">
                {w['trades']} trades · PF {w['profit_factor']:.2f} · Sharpe {w['sharpe']:.2f}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    df = leaderboard_dataframe(rows)
    try:
        styled = df.style.map(
            lambda v: _pct_color(v) if isinstance(v, (int, float)) else "",
            subset=["Return %"],
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=min(420, 38 * len(df) + 38))
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(420, 38 * len(df) + 38))


def render_ohlcv_ticker(ohlcv: pd.DataFrame, symbol: str, timeframe: str) -> None:
    import streamlit as st

    if ohlcv.empty:
        return
    last = ohlcv.iloc[-1]
    prev = ohlcv.iloc[-2] if len(ohlcv) > 1 else last
    chg = float(last["close"]) - float(prev["close"])
    chg_pct = (chg / float(prev["close"]) * 100) if float(prev["close"]) else 0
    color = "#26A69A" if chg >= 0 else "#EF5350"

    st.markdown(
        f"""
        <div style="display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: baseline;
            padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #e0e0e0;">
          <span style="font-size: 1.35rem; font-weight: 700;">{symbol}</span>
          <span style="color: #787B86;">{timeframe}</span>
          <span>O <b>{last['open']:.2f}</b></span>
          <span>H <b>{last['high']:.2f}</b></span>
          <span>L <b>{last['low']:.2f}</b></span>
          <span>C <b>{last['close']:.2f}</b></span>
          <span style="color: {color}; font-weight: 600;">
            {chg:+.2f} ({chg_pct:+.2f}%)
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_strategy_report_card(report: StrategyTesterReport) -> None:
    import streamlit as st

    p, r, i = report.performance, report.risk, report.intraday
    profit_pct = p.total_net_profit / report.initial_capital * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net profit", f"₹{p.total_net_profit:,.0f}", f"{profit_pct:+.2f}%")
    c2.metric("Profit factor", f"{p.profit_factor:.2f}", f"{p.total_trades} trades")
    c3.metric("Win rate", f"{p.win_rate_pct:.1f}%", f"{p.winning_trades}W / {p.losing_trades}L")
    c4.metric("Max drawdown", f"{r.max_drawdown_pct * 100:.2f}%", f"Sharpe {r.sharpe_ratio:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Expectancy", f"₹{r.expectancy:,.2f}")
    c6.metric("Avg trade", f"₹{p.average_trade:,.2f}")
    c7.metric("Trades / day", f"{i.trades_per_day:.1f}")
    c8.metric("Best hour", f"{i.best_trading_hour}h", f"₹{i.best_hour_pnl:+,.0f}")

    if not report.trades_df.empty:
        st.subheader("Trade log")
        show = report.trades_df[
            ["entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl", "pnl_percent"]
        ].copy()
        show.columns = [
            "Entry",
            "Exit",
            "Side",
            "Entry ₹",
            "Exit ₹",
            "P&L ₹",
            "Return %",
        ]
        st.dataframe(show, use_container_width=True, hide_index=True, height=280)


def render_strategy_comparison_grid(batch: BatchRunResult) -> None:
    """Compact metric cards for top 3 strategies."""
    import streamlit as st

    rows = batch.leaderboard_rows()[:3]
    if not rows:
        return
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows, strict=True):
        with col:
            sign = "+" if row["profit_pct"] >= 0 else ""
            col.markdown(
                f"""
                <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem;
                    background: #fafafa;">
                  <div style="font-weight: 600; margin-bottom: 0.5rem;">{row['strategy']}</div>
                  <div style="font-size: 1.4rem; font-weight: 700;
                    color: {'#26A69A' if row['profit_pct'] >= 0 else '#EF5350'};">
                    {sign}{row['profit_pct']:.2f}%
                  </div>
                  <div style="color: #787B86; font-size: 0.85rem; margin-top: 0.35rem;">
                    {row['trades']} trades · PF {row['profit_factor']:.2f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def export_leaderboard_csv(batch: BatchRunResult) -> bytes:
    df = leaderboard_dataframe(batch.leaderboard_rows())
    return df.to_csv(index=False).encode("utf-8")


def _html(component) -> None:
    """Render raw HTML — uses ``st.html`` when available, falls back to markdown."""
    import streamlit as st

    if hasattr(st, "html"):
        st.html(component)
    else:
        st.markdown(component, unsafe_allow_html=True)


def render_live_signal_banner(signal: Optional[Any], strategy_name: str, timeframe: str) -> None:
    """Big colored banner for the currently selected strategy's latest signal."""
    import streamlit as st

    if signal is None:
        st.info(f"No live signal yet for **{strategy_name}**.")
        return

    label = getattr(signal, "label", "HOLD")
    action = getattr(signal, "action", "HOLD")
    color = getattr(signal, "color", "#787B86")
    bar_close = getattr(signal, "bar_close", 0.0)
    bar_time = getattr(signal, "bar_time", None)
    in_pos = getattr(signal, "in_position", False)
    entry_price = getattr(signal, "entry_price", None)
    bars_in_trade = getattr(signal, "bars_in_trade", 0)
    side = getattr(signal, "side", "LONG")

    pulse = "" if action == "HOLD" else "animation: fortuna-pulse 1.6s ease-in-out infinite;"
    sub_parts: list[str] = []
    if in_pos and entry_price is not None:
        pct = (bar_close - entry_price) / entry_price * 100.0
        if side == "SHORT":
            pct = -pct
        pct_color = "#26A69A" if pct >= 0 else "#EF5350"
        sub_parts.append(
            f"In {side} from ₹{entry_price:.2f} · {bars_in_trade} bars · "
            f'<span style="color:{pct_color}">{pct:+.2f}%</span>'
        )
    if bar_time is not None:
        sub_parts.append(f"Last bar {bar_time:%d %b %H:%M} · close ₹{bar_close:.2f}")
    sub = "<br>".join(sub_parts)

    html = (
        f'<style>@keyframes fortuna-pulse{{0%{{box-shadow:0 0 0 0 {color}40;}}'
        f'70%{{box-shadow:0 0 0 14px {color}00;}}100%{{box-shadow:0 0 0 0 {color}00;}}}}</style>'
        f'<div style="display:flex;align-items:center;gap:1rem;padding:0.85rem 1.1rem;'
        f'border-radius:10px;background:{color}15;border:1px solid {color}40;{pulse}'
        f'margin-bottom:0.6rem;">'
        f'<div style="font-size:1.05rem;font-weight:800;padding:0.35rem 0.85rem;'
        f'border-radius:6px;background:{color};color:white;letter-spacing:0.06em;">{label}</div>'
        f'<div style="line-height:1.35;">'
        f'<div style="font-weight:600;color:#1a1f2c;">{strategy_name} '
        f'<span style="color:#787B86;font-weight:400;">· {timeframe}</span></div>'
        f'<div style="color:#5d6068;font-size:0.88rem;">{sub}</div>'
        f'</div></div>'
    )
    _html(html)


def render_live_signal_panel(signals: dict[str, Any], winner: Optional[str] = None) -> None:
    """Compact grid showing every strategy's current live signal."""
    import streamlit as st

    if not signals:
        return

    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        name, sig = item
        order = {
            "BUY": 0,
            "SELL": 1,
            "EXIT_LONG": 2,
            "EXIT_SHORT": 2,
            "IN_LONG": 3,
            "IN_SHORT": 3,
            "HOLD": 4,
        }
        return (order.get(getattr(sig, "action", "HOLD"), 9), name)

    st.markdown("**Live signals (all strategies)**")
    items = sorted(signals.items(), key=sort_key)

    cards: list[str] = []
    for name, sig in items:
        label = getattr(sig, "label", "HOLD")
        color = getattr(sig, "color", "#787B86")
        close = getattr(sig, "bar_close", 0.0)
        in_pos = getattr(sig, "in_position", False)
        entry_price = getattr(sig, "entry_price", None)
        side = getattr(sig, "side", "LONG")
        crown = " 👑" if name == winner else ""
        sub_html = ""
        if in_pos and entry_price is not None:
            pct = (close - entry_price) / entry_price * 100.0
            if side == "SHORT":
                pct = -pct
            pct_color = "#26A69A" if pct >= 0 else "#EF5350"
            sub_html = (
                f'<div style="font-size:0.75rem;color:{pct_color};margin-top:0.15rem;">'
                f"open {pct:+.2f}%</div>"
            )
        card = (
            f'<div style="border:1px solid #e0e0e0;border-left:4px solid {color};'
            f'border-radius:6px;padding:0.55rem 0.75rem;background:#ffffff;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:0.4rem;">'
            f'<div style="font-weight:600;font-size:0.9rem;color:#1a1f2c;">{name}{crown}</div>'
            f'<div style="background:{color};color:white;font-size:0.7rem;font-weight:700;'
            f'padding:0.12rem 0.5rem;border-radius:4px;letter-spacing:0.04em;">{label}</div>'
            f"</div>"
            f'<div style="color:#787B86;font-size:0.78rem;margin-top:0.2rem;">₹{close:.2f}</div>'
            f"{sub_html}"
            f"</div>"
        )
        cards.append(card)

    grid_html = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));'
        'gap:0.6rem;margin-bottom:0.6rem;">'
        + "".join(cards)
        + "</div>"
    )
    _html(grid_html)
