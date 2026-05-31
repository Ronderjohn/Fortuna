#!/usr/bin/env python
"""Fortuna Streamlit dashboard — TradingView-style symbol search and strategy reports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.environ.setdefault("FORTUNA_CONFIG", "configs/intraday.yaml")

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

# Apply SSL bypass BEFORE any requests/urllib3/websocket import — needed on
# Windows boxes with TLS-intercepting AV (Avast etc.). Toggle via .env.
from fortuna.utils.insecure_ssl import maybe_disable_ssl_verification

maybe_disable_ssl_verification()

import streamlit as st

from fortuna.app.config import AppConfig
from fortuna.app.presentation import (
    export_leaderboard_csv,
    render_leaderboard,
    render_live_signal_banner,
    render_live_signal_panel,
    render_ohlcv_ticker,
    render_strategy_comparison_grid,
    render_strategy_report_card,
)
from fortuna.app.session_engine import FortunaSessionEngine
from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.config.settings import load_settings
from fortuna.config.smartapi_settings import get_smartapi_settings

_TV_CSS = """
<style>
  .fortuna-header {
    background: #131722;
    color: #d1d4dc;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
  }
  .fortuna-symbol-btn {
    font-size: 1.25rem;
    font-weight: 700;
    color: #2962FF;
    cursor: pointer;
  }
  div[data-testid="stDialog"] div[role="dialog"] {
    max-width: 720px;
  }
  .symbol-result {
    padding: 0.35rem 0.5rem;
    border-radius: 4px;
  }
  .symbol-result:hover {
    background: #f0f3fa;
  }
</style>
"""


def _smartapi_ready() -> tuple[bool, str]:
    try:
        s = get_smartapi_settings()
        if not s.api_key or not s.client_code:
            return False, "Missing SMARTAPI_API_KEY or SMARTAPI_CLIENT_CODE in .env"
        if not s.password or not s.totp_secret:
            return False, "Missing SMARTAPI_PASSWORD or SMARTAPI_TOTP_SECRET in .env"
        return True, "Connected (credentials present)"
    except Exception as e:
        return False, str(e)


def _is_nse_session() -> bool:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= t <= 15 * 60 + 30


@st.cache_resource(show_spinner="Loading NSE market universe from Angel One…")
def get_catalog() -> SymbolCatalog:
    """Background: download/cache OpenAPIScripMaster and index all NSE equities."""
    catalog = SymbolCatalog()
    n = catalog.ensure_loaded()
    return catalog


@st.cache_resource
def get_engine() -> FortunaSessionEngine:
    settings = load_settings()
    return FortunaSessionEngine(settings, AppConfig.from_settings(settings))


def _build_stream_markers(state, strategy_name: str) -> list[dict]:
    """Build Lightweight Charts marker dicts for the *currently loaded*
    strategy, using the same conventions as the static renderer so the
    look stays consistent between initial paint and live updates."""
    import pandas as pd

    from fortuna.app.lightweight_chart import ist_unix_seconds

    if not state or not getattr(state, "batch", None):
        return []
    run = state.batch.results.get(strategy_name) if state.batch.results else None
    if not run or not run.report or run.report.trades_df is None or run.report.trades_df.empty:
        return []

    bull, bear, exit_c = "#26a69a", "#ef5350", "#9c27b0"
    markers: list[dict] = []
    for _, t in run.report.trades_df.iterrows():
        is_long = str(t["side"]).upper() == "LONG"
        ep = float(t["entry_price"])
        xp = float(t["exit_price"])
        pnl = float(t.get("pnl", 0.0))
        reason = str(t.get("exit_reason", "") or "").lower()
        e_t = ist_unix_seconds(pd.Timestamp(t["entry_time"]))
        x_t = ist_unix_seconds(pd.Timestamp(t["exit_time"]))
        if reason == "pyramid":
            markers.append({
                "time": e_t,
                "position": "belowBar" if is_long else "aboveBar",
                "color": bull if is_long else bear,
                "shape": "arrowUp" if is_long else "arrowDown",
                "text": "BUY" if is_long else "SELL",
                "size": 0.9,
            })
            continue
        markers.append({
            "time": e_t,
            "position": "belowBar" if is_long else "aboveBar",
            "color": bull if is_long else bear,
            "shape": "arrowUp" if is_long else "arrowDown",
            "text": f"{'BUY' if is_long else 'SELL'} @ {ep:.2f}",
            "size": 1.3,
        })
        markers.append({
            "time": x_t,
            "position": "aboveBar" if is_long else "belowBar",
            "color": exit_c,
            "shape": "circle",
            "text": f"EXIT {xp:.2f} ({pnl:+.0f})",
            "size": 1.0,
        })
    markers.sort(key=lambda m: m["time"])
    return markers


@st.cache_resource
def get_stream_port() -> int:
    """Spawn (once per Streamlit process) a local HTTP server that the chart
    iframe polls for live bar / marker updates. Returns the bound port so the
    iframe can build its fetch URL.

    Running it as a separate HTTP endpoint instead of Streamlit's normal rerun
    cycle is what lets the chart update *in place* via Lightweight Charts'
    ``series.update()`` API — the iframe is never reloaded, so the user's
    zoom / pan state is preserved (TradingView-style smooth streaming).
    """
    import pandas as pd

    from fortuna.app import stream_server
    from fortuna.app.lightweight_chart import ist_unix_seconds
    from fortuna.app.live_signals import compute_live_overlays
    from fortuna.app.strategy_paths import list_strategy_paths
    from fortuna.strategy.loader import load_strategy

    def provider(*, symbol: str, timeframe: str, strategy: str, since: float) -> dict:
        eng = get_engine()
        state = eng.state
        if not state or state.symbol != symbol or state.timeframe != timeframe:
            return {"bars": [], "markers": [], "overlays": [], "last": 0}
        df = eng.get_ohlcv_for_chart()
        if df is None or df.empty:
            return {"bars": [], "markers": [], "overlays": [], "last": 0}

        closed = state.ohlcv
        max_closed_unix = (
            ist_unix_seconds(pd.Timestamp(closed.index[-1]))
            if closed is not None and not closed.empty
            else 0
        )

        tail = df.tail(60)
        cutoff = int(since)
        bars: list[dict] = []
        for ts, row in tail.iterrows():
            t = ist_unix_seconds(pd.Timestamp(ts))
            if t < cutoff:
                continue
            # Never stream a bar older than the last closed candle (stale forming bucket).
            if max_closed_unix and t < max_closed_unix:
                continue
            bars.append({
                "time": t,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })

        markers = _build_stream_markers(state, strategy)

        # Recompute indicator overlays so EMA / Bollinger / VWAP lines keep
        # extending with the new live bars. The work is the same as a single
        # signal-refresh pass, so it stays cheap.
        overlays: list[dict] = []
        run = state.batch.results.get(strategy) if state.batch and state.batch.results else None
        overlay_cols = list(run.report.overlay_columns) if (run and run.report) else []
        if overlay_cols:
            try:
                strat_path = next(
                    (p for p in list_strategy_paths(eng.settings, eng.app_config) if p.stem == strategy),
                    None,
                )
                if strat_path is not None:
                    strat_def = load_strategy(strat_path)
                    series_map = compute_live_overlays(strat_def, df, overlay_cols, tail=20)
                    for name, series in series_map.items():
                        points = []
                        for ts, val in series.items():
                            t = ist_unix_seconds(pd.Timestamp(ts))
                            if t < cutoff:
                                continue
                            points.append({"time": t, "value": float(val)})
                        if points:
                            overlays.append({"name": name, "points": points})
            except Exception:
                pass

        last_time = bars[-1]["time"] if bars else 0
        return {"bars": bars, "markers": markers, "overlays": overlays, "last": last_time}

    return stream_server.spawn(provider, port=8765)


def _init_session_state(settings) -> None:
    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = settings.default_symbol
    if "symbol_dialog_open" not in st.session_state:
        st.session_state.symbol_dialog_open = False
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""


@st.dialog("Symbol search — NSE + NFO", width="large")
def _symbol_search_dialog(catalog: SymbolCatalog, settings) -> None:
    """TradingView-style symbol picker spanning NSE cash + NFO futures."""
    st.caption(
        f"**{catalog.equity_count:,}** NSE equities · "
        f"**{catalog.futures_count:,}** NFO futures · "
        "SmartAPI · type to filter (try `CROMPTON` to see both)"
    )

    q = st.text_input(
        "Search",
        value=st.session_state.get("dialog_search", ""),
        placeholder="Symbol or name — e.g. RELIANCE, CROMPTON, NIFTY",
        label_visibility="collapsed",
        key="dialog_search_input",
    )
    st.session_state.dialog_search = q

    hits = catalog.search(q, limit=60)
    if not hits:
        st.warning("No matches. Try a shorter symbol or company name.")
        return

    options = {h.symbol: h.display for h in hits}
    labels = list(options.keys())
    current = st.session_state.selected_symbol
    default_idx = labels.index(current) if current in labels else 0

    picked = st.radio(
        "Results",
        labels,
        index=default_idx,
        format_func=lambda s: options[s],
        key="dialog_symbol_pick",
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load chart & strategies", type="primary", use_container_width=True):
            st.session_state.selected_symbol = picked
            st.session_state.symbol_dialog_open = False
            st.session_state.trigger_load = True
            st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.symbol_dialog_open = False
            st.rerun()


def _run_load(
    engine: FortunaSessionEngine,
    symbol: str,
    timeframe: str,
    days: int,
    force: bool,
    live_on: bool,
) -> None:
    with st.status(f"Fetching {symbol} via SmartAPI and running all strategies…", expanded=True) as status:
        st.write("Downloading OHLCV history…")
        engine.load_symbol(symbol, timeframe=timeframe, days=days, force_refresh=force)
        state = engine.state
        if state.load_error:
            status.update(label="Load failed", state="error")
            st.error(state.load_error)
            return
        st.write(f"Backtesting {len(state.batch.results) if state.batch else 0} strategies in parallel…")
        status.update(label="Ready", state="complete")
    st.session_state.loaded = True
    st.session_state.trigger_load = False
    if state.batch and state.batch.winner:
        st.session_state.chart_strat = state.batch.winner
    st.session_state.chart_context = f"{symbol}_{timeframe}"
    if live_on:
        try:
            engine.start_live()
        except Exception as e:
            st.sidebar.error(f"Live feed: {e}")


def main() -> None:
    st.set_page_config(
        page_title="Fortuna",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_TV_CSS, unsafe_allow_html=True)

    ok, status_msg = _smartapi_ready()
    if not ok:
        st.error(f"SmartAPI required: {status_msg}")
        st.info("Copy `.env.example` to `.env` and set Angel One SmartAPI credentials.")
        st.stop()

    settings = load_settings()
    app_cfg = AppConfig.from_settings()
    _init_session_state(settings)

    catalog = get_catalog()
    engine = get_engine()
    # Spawn the bar-stream HTTP server eagerly so it's ready by the time the
    # chart iframe wants to poll it. Cached @st.cache_resource keeps this to
    # a single server per process.
    try:
        get_stream_port()
    except Exception:
        pass

    # Header bar — TradingView-like symbol entry
    h1, h2, h3 = st.columns([2, 2, 1])
    with h1:
        display = catalog.resolve_display(st.session_state.selected_symbol)
        if st.button(f"🔍  {display}", use_container_width=False, help="Search NSE stocks"):
            st.session_state.symbol_dialog_open = True
    with h2:
        st.caption(f"Market universe: **{catalog.count:,}** stocks · {status_msg}")
    with h3:
        st.caption("Fortuna intraday")

    if st.session_state.get("symbol_dialog_open"):
        _symbol_search_dialog(catalog, settings)

    with st.container():
        c_tf, c_days, c_force, c_live, c_load = st.columns([1, 1, 1, 1, 1])
        with c_tf:
            from fortuna.data.timeframes import TRADINGVIEW_INTERVALS, interval_label

            timeframe = st.selectbox(
                "Interval",
                TRADINGVIEW_INTERVALS,
                index=TRADINGVIEW_INTERVALS.index("5m"),
                format_func=interval_label,
                label_visibility="visible",
            )
        with c_days:
            days = st.number_input("History (days)", min_value=7, max_value=90, value=settings.default_days)
        with c_force:
            force = st.checkbox("Refresh data", value=False)
        with c_live:
            live_on = st.checkbox("Live", value=_is_nse_session())
            if live_on and not get_smartapi_settings().use_live_feed:
                st.caption("Set SMARTAPI_USE_LIVE_FEED=true")
        with c_load:
            st.write("")
            load_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if load_clicked or st.session_state.get("trigger_load"):
        _run_load(
            engine,
            st.session_state.selected_symbol,
            timeframe,
            int(days),
            force,
            live_on,
        )

    if engine.is_live():
        # Do an initial poll so the page paints with fresh state immediately.
        try:
            engine.poll_live()
            engine.refresh_live_signals()
        except Exception:
            pass

        pulse_seconds = max(2, int(app_cfg.live_refresh_seconds))

        @st.fragment(run_every=f"{pulse_seconds}s")
        def _live_pulse() -> None:
            """Lightweight live tick: pulls SmartAPI ticks into the engine and
            refreshes signals every few seconds.

            We deliberately do NOT trigger any Streamlit reruns from here. The
            chart iframe polls the bar-stream HTTP endpoint directly and
            applies new bars / markers in-place via Lightweight Charts'
            ``series.update()`` — so streaming feels smooth and the user's
            pan / zoom state is never disturbed (matches TradingView)."""
            try:
                engine.poll_live()
                engine.refresh_live_signals()
            except Exception as e:  # noqa: BLE001
                st.caption(f"Live poll error: {e}")
                return

            new_ts = engine.state.last_bar_time
            sig_age = engine.state.last_signal_refresh

            label = f"{new_ts:%H:%M}" if new_ts is not None else "waiting"
            sig_suffix = (
                f" · signals {sig_age:%H:%M:%S}" if sig_age is not None else ""
            )
            st.markdown(
                "<div style='background:#0e8345;color:#fff;padding:4px 12px;"
                "border-radius:6px;display:inline-block;font-weight:600;"
                f"font-size:0.85rem;'>📡 LIVE · last bar {label}{sig_suffix}</div>",
                unsafe_allow_html=True,
            )

        _live_pulse()
    elif live_on and engine.state.ohlcv is not None and _is_nse_session():
        try:
            engine.start_live()
            st.toast("Live feed auto-started — market is open.", icon="📡")
        except Exception as e:
            st.sidebar.warning(f"Live feed unavailable: {e}")

    state = engine.state
    if state.ohlcv is None or state.batch is None:
        st.info(
            "Click **🔍 symbol** to search the full NSE market (like TradingView), "
            "then **Analyze** to fetch SmartAPI data, run strategies, and open charts."
        )
        with st.expander("Browse popular symbols", expanded=True):
            popular = catalog.search("", limit=12)
            cols = st.columns(4)
            for i, hit in enumerate(popular):
                with cols[i % 4]:
                    if st.button(hit.display.split(" — ")[0], key=f"pop_{hit.symbol}"):
                        st.session_state.selected_symbol = hit.symbol
                        st.session_state.trigger_load = True
                        st.rerun()
        st.stop()

    ohlcv = state.ohlcv
    batch = state.batch
    winner = batch.winner
    rows = batch.leaderboard_rows()
    live_signals = engine.live_signals()

    render_ohlcv_ticker(ohlcv, state.symbol, state.timeframe)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Bars", f"{len(ohlcv):,}")
    m2.metric("From", str(ohlcv.index.min().date()))
    m3.metric("To", str(ohlcv.index.max().date()))
    m4.metric("Strategies", len(batch.results))
    live_label = "ON" if engine.is_live() else "OFF"
    if engine.is_live() and state.last_bar_time is not None:
        live_label = f"ON · {state.last_bar_time:%H:%M}"
    m5.metric("Live", live_label)

    render_strategy_comparison_grid(batch)
    render_live_signal_panel(live_signals, winner=winner)

    tab_chart, tab_board, tab_report, tab_detail, tab_rl, tab_assist, tab_exec, tab_monitor = st.tabs(
        [
            "📈 Chart",
            "🏆 Leaderboard",
            "📊 Performance",
            "🔬 Strategy detail",
            "🤖 Models",
            "🧠 Assistant",
            "💼 Execution",
            "🛰️ Monitor",
        ]
    )

    strategy_names = sorted(batch.results.keys())
    default_idx = strategy_names.index(winner) if winner in strategy_names else 0
    chart_ctx = f"{state.symbol}_{state.timeframe}"
    if st.session_state.get("chart_context") != chart_ctx:
        st.session_state.chart_context = chart_ctx
        if winner in strategy_names:
            st.session_state.chart_strat = winner
    elif winner in strategy_names and st.session_state.get("chart_strat") not in strategy_names:
        st.session_state.chart_strat = winner

    with tab_chart:
        if st.session_state.get("chart_strat") not in strategy_names:
            st.session_state.chart_strat = (
                winner if winner in strategy_names else strategy_names[0]
            )
        strat = st.selectbox(
            "Strategy",
            strategy_names,
            key="chart_strat",
            label_visibility="collapsed",
        )
        import streamlit.components.v1 as components

        from fortuna.app.lightweight_chart import (
            build_lightweight_charts_spec,
            render_lightweight_charts_html,
        )
        from fortuna.reporting.strategy_tester.chart_viewport import (
            visible_bars_for_timeframe,
        )

        render_live_signal_banner(live_signals.get(strat), strat, state.timeframe)
        vb = visible_bars_for_timeframe(state.timeframe)

        run = batch.results.get(strat)
        report = run.report if run else None
        chart_ohlcv = engine.get_ohlcv_for_chart()
        if report is None or chart_ohlcv is None or chart_ohlcv.empty:
            st.info("No chart data available yet.")
        else:
            try:
                spec = build_lightweight_charts_spec(
                    ohlcv=chart_ohlcv,
                    enriched=report.enriched_data,
                    overlay_columns=report.overlay_columns,
                    trades_df=report.trades_df,
                    live_signal=live_signals.get(strat),
                    title=strat,
                    height=560,
                    visible_bars=vb,
                )
                if spec:
                    # Hand the iframe a polling URL. The JS inside the iframe
                    # fetches incremental bars + markers from this endpoint
                    # every couple of seconds and applies them in-place via
                    # series.update() — preserving zoom/pan (TV-style stream).
                    stream_url = None
                    try:
                        port = get_stream_port()
                        from urllib.parse import urlencode

                        qs = urlencode({
                            "symbol": state.symbol,
                            "tf": state.timeframe,
                            "strat": strat,
                        })
                        stream_url = f"http://localhost:{port}/fortuna/bars?{qs}"
                    except Exception:
                        stream_url = None

                    components.html(
                        render_lightweight_charts_html(
                            spec,
                            height=580,
                            stream_url=stream_url,
                            stream_poll_ms=2500,
                        ),
                        height=600,
                        scrolling=False,
                    )
                else:
                    st.info("No bars to plot.")
            except Exception as e:
                st.error(f"Chart error: {e}")
        live_note = (
            f" · last refresh {state.last_signal_refresh:%H:%M:%S} UTC"
            if state.last_signal_refresh is not None and engine.is_live()
            else ""
        )
        st.caption(
            f"**{state.timeframe}** · NSE session 09:15–15:30 IST · candles plotted "
            f"by trading-bar sequence (no weekend / off-hour gaps). "
            f"▲ BUY below bar · ▼ SELL above bar · ◆ EXIT. "
            f"Live forming-bar signal flashes on the latest candle.{live_note}"
        )

    with tab_board:
        render_leaderboard(batch, winner)
        st.download_button(
            "Download leaderboard CSV",
            data=export_leaderboard_csv(batch),
            file_name=f"{state.symbol}_{state.timeframe}_leaderboard.csv",
            mime="text/csv",
        )

    with tab_report:
        st.subheader("Strategy comparison")
        render_leaderboard(batch, winner)
        st.subheader("Selected strategy deep-dive")
        pick = st.selectbox("Strategy", strategy_names, index=default_idx, key="report_pick")
        run = batch.results.get(pick)
        if run and run.report:
            render_strategy_report_card(run.report)

    with tab_detail:
        pick = st.selectbox("Strategy", strategy_names, index=default_idx, key="detail_pick")
        run = batch.results.get(pick)
        if run is None:
            st.stop()
        if run.error:
            st.error(run.error)
        elif run.report:
            render_strategy_report_card(run.report)
            with st.expander("Raw JSON summary"):
                st.json(run.report.summary_dict())

    with tab_rl:
        _render_models_panel(engine, live_signals)

    with tab_assist:
        _render_assistant_panel(engine)

    with tab_exec:
        _render_execution_panel(engine)

    with tab_monitor:
        _render_monitor_panel(engine)


def _render_assistant_panel(engine) -> None:
    st.subheader("Agentic assistant")
    adapter_enabled = bool(getattr(engine.settings, "conversational_adapter_enabled", False))
    if adapter_enabled:
        st.caption(
            "Conversational adapter is enabled. Free-form prompts are normalized into "
            "typed advisory tool calls with safe fallback behavior."
        )
    else:
        st.caption(
            "Conversational adapter is disabled. The assistant still supports explicit "
            "commands such as /search and /analyze."
        )

    examples = (
        "/search RELIANCE",
        "/analyze RELIANCE",
        "/analyze RELIANCE FUT",
        "How is Reliance looking on 15m for 20d?",
        "Should I enter NIFTY CE 25000 28MAY2026?",
    )
    st.write("Examples:")
    for ex in examples:
        st.code(ex, language="text")

    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = []

    assistant = engine.conversational_assistant()
    prompt = st.chat_input("Ask Fortuna about a stock, future, or option contract")
    if prompt:
        result = assistant.handle_interaction(prompt)
        st.session_state.assistant_history.append({"role": "user", "content": prompt})
        st.session_state.assistant_history.append(
            {
                "role": "assistant",
                "content": result.reply,
                "meta": {
                    "source": result.source,
                    "confidence": result.source_confidence,
                    "tool": result.tool or "",
                    "rationale": result.source_rationale,
                },
            }
        )
        max_pairs = max(2, int(getattr(engine.settings, "conversational_max_history", 12)))
        st.session_state.assistant_history = st.session_state.assistant_history[-(max_pairs * 2) :]

    for item in st.session_state.assistant_history:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            meta = item.get("meta") or {}
            if meta:
                st.caption(
                    f"source={meta.get('source', '')} · "
                    f"confidence={float(meta.get('confidence', 0.0)):.2f} · "
                    f"tool={meta.get('tool', '')}"
                )


def _render_execution_panel(engine) -> None:
    """Live positions table + intraday equity curve + risk-budget gauges."""
    if not getattr(engine, "execution_enabled", False):
        st.info(
            "Live execution is **disabled**. Set `FORTUNA_EXECUTION_ENABLED=1` "
            "in `.env` (or `execution.enabled: true` in `configs/default.yaml`) "
            "and reload the dashboard to activate the paper broker."
        )
        return

    router = engine.execution_router
    account = engine.live_account
    if router is None or account is None:
        st.warning("Execution router failed to initialize. Check logs/execution/.")
        return

    summary = account.to_summary()
    cfg = router.config

    cols = st.columns(5)
    cols[0].metric("Equity", f"{summary['equity']:,.0f}", f"{summary['realized_pnl']:+,.2f}")
    cols[1].metric("Unrealized", f"{summary['unrealized_pnl']:+,.2f}")
    cols[2].metric("Open positions", summary["open_positions"])
    cols[3].metric("Trades", f"{summary['total_trades']} ({summary['win_rate_pct']:.1f}%)")
    halted = router.risk_gate.is_halted
    cols[4].metric(
        "Status",
        "HALTED" if halted else "LIVE",
        delta=f"DD {summary['max_drawdown_pct']:.2f}%",
        delta_color="inverse" if halted else "normal",
    )
    if halted:
        st.error(f"Circuit breaker: {router.risk_gate.halted_reason}")

    _render_broker_sync_block(engine)
    _render_what_to_do_now(engine)

    pos_cols = st.columns([1, 1])
    pos_cols[0].subheader("Open positions")
    if not account.positions:
        pos_cols[0].caption("No open positions.")
    else:
        rows = []
        for sym, pos in account.positions.items():
            mark = pos.last_mtm_price or pos.avg_price
            rows.append({
                "symbol": sym,
                "side": pos.side.value,
                "qty": pos.qty,
                "avg": round(pos.avg_price, 2),
                "mark": round(mark, 2),
                "MTM": round(pos.unrealized_pnl(mark), 2),
                "tag": pos.entry_tag or "",
            })
        pos_cols[0].dataframe(rows, use_container_width=True, hide_index=True)

    pos_cols[1].subheader("Risk budget")
    cap = cfg.gross_exposure_cap_pct / 100.0 * cfg.init_cash
    used = summary["gross_exposure"]
    pos_cols[1].progress(min(1.0, used / cap if cap > 0 else 0.0), text=f"Gross exposure {used:,.0f} / {cap:,.0f}")
    loss_used_pct = summary["max_drawdown_pct"]
    pos_cols[1].progress(
        min(1.0, loss_used_pct / cfg.daily_loss_halt_pct if cfg.daily_loss_halt_pct > 0 else 0.0),
        text=f"Daily DD {loss_used_pct:.2f}% / {cfg.daily_loss_halt_pct:.2f}%",
    )

    st.subheader("Intraday equity curve")
    eq = account.equity_curve
    if not eq:
        st.caption("No equity samples yet (router hasn't seen a bar close).")
    else:
        import pandas as pd_local
        eq_df = pd_local.DataFrame(
            [{"ts": p.ts, "equity": p.equity, "drawdown_pct": p.drawdown_pct} for p in eq]
        ).set_index("ts")
        st.line_chart(eq_df["equity"])

    st.subheader("Today's trade ledger")
    trades = list(account.trades)
    if not trades:
        st.caption("No closed trades yet.")
    else:
        trade_rows = []
        for t in trades:
            trade_rows.append({
                "symbol": t.symbol,
                "side": t.side.value,
                "qty": t.qty,
                "entry": round(t.entry_price, 2),
                "exit": round(t.exit_price, 2),
                "net P&L": round(t.net_pnl, 2),
                "return %": round(t.return_pct, 2),
                "tag": t.tag or "",
            })
        st.dataframe(trade_rows, use_container_width=True, hide_index=True)

    _render_holdings_block(engine)
    _render_reconciliation_block(engine)


def _render_broker_sync_block(engine) -> None:
    """Show broker-sync status + a Sync now button (Phase 1)."""
    if not getattr(engine, "account_sync_enabled", False):
        with st.expander("🔌 Sync portfolio from Angel One (disabled)", expanded=False):
            err = getattr(engine, "account_sync_error", None)
            if err:
                st.warning(f"Account sync disabled: {err}")
            else:
                st.caption(
                    "Set `FORTUNA_EXECUTION_ACCOUNT_SYNC=1` in `.env` to mirror "
                    "your real Angel One cash + positions + holdings into the paper "
                    "simulator. Read-only — no orders are placed."
                )
        return

    snap = engine.broker_snapshot
    cols = st.columns([3, 1])
    if snap is None:
        cols[0].caption("Broker sync enabled — no snapshot yet.")
    else:
        funds = snap.funds
        cols[0].markdown(
            f"**Broker:** Angel One &nbsp;|&nbsp; **Net:** ₹{funds.net:,.0f} &nbsp;|&nbsp; "
            f"**Available:** ₹{funds.available_cash:,.0f} &nbsp;|&nbsp; "
            f"**Used margin:** ₹{funds.used_margin:,.0f} &nbsp;|&nbsp; "
            f"**Synced:** {snap.taken_at:%Y-%m-%d %H:%M:%S}"
        )
        if snap.errors:
            cols[0].caption(f"Partial sync (errors: {', '.join(snap.errors)})")
    if cols[1].button("Sync now"):
        applied = engine.sync_account_from_broker()
        if applied is None:
            st.warning(engine.account_sync_error or "Account sync failed.")
        else:
            st.success(
                f"Synced — cash ₹{applied['init_cash_after']:,.0f}, "
                f"{applied['positions_added']} positions, "
                f"{applied['holdings_added']} holdings."
            )
            st.rerun()


def _render_what_to_do_now(engine) -> None:
    """Join real positions + holdings with agentic Fortuna decisions."""
    account = engine.live_account
    if account is None:
        return
    state = engine.state if hasattr(engine, "state") else None
    focus_symbol = state.symbol if state else ""

    held_symbols = list(account.positions.keys()) + [
        s for s in account.holdings if s not in account.positions
    ]
    signals_by_symbol: dict[str, dict] = {}
    if held_symbols and hasattr(engine, "live_signals_for_symbols"):
        with st.spinner("Computing signals for your holdings…"):
            signals_by_symbol = engine.live_signals_for_symbols(held_symbols)
    decisions_by_symbol: dict[str, object] = {}
    if held_symbols and hasattr(engine, "agent_decisions_for_symbols"):
        with st.spinner("Running advisory agents for your holdings…"):
            decisions_by_symbol = engine.agent_decisions_for_symbols(held_symbols)
    focus_signals = engine.live_signals() if hasattr(engine, "live_signals") else {}
    if focus_symbol:
        signals_by_symbol.setdefault(focus_symbol, focus_signals)
        if hasattr(engine, "agent_decisions"):
            decisions_by_symbol.update(engine.agent_decisions())

    rows: list[dict] = []
    seen_symbols: set[str] = set()

    # Real intraday positions (broker-seeded OR paper-acquired).
    for sym, pos in account.positions.items():
        seen_symbols.add(sym)
        rec, drivers, confidence, risk = _decision_or_legacy_recommendation(
            sym, "position", pos.side.value, signals_by_symbol, focus_symbol,
            decisions_by_symbol,
        )
        rows.append({
            "symbol": sym,
            "kind": "Position",
            "side": pos.side.value,
            "qty": pos.qty,
            "avg": round(pos.avg_price, 2),
            "mark": round(pos.last_mtm_price or pos.avg_price, 2),
            "MTM": round(pos.unrealized_pnl(pos.last_mtm_price or pos.avg_price), 2),
            "recommendation": rec,
            "confidence": confidence,
            "drivers": drivers,
            "risk": risk,
        })

    # Long-term DEMAT holdings (broker-mirrored).
    for sym, h in account.holdings.items():
        if sym in seen_symbols:
            continue
        seen_symbols.add(sym)
        rec, drivers, confidence, risk = _decision_or_legacy_recommendation(
            sym, "holding", "LONG", signals_by_symbol, focus_symbol,
            decisions_by_symbol,
        )
        rows.append({
            "symbol": sym,
            "kind": "Holding",
            "side": "LONG",
            "qty": h.qty,
            "avg": round(h.avg_price, 2),
            "mark": round(h.last_price or h.avg_price, 2),
            "MTM": round((h.last_price - h.avg_price) * h.qty, 2)
                if h.last_price is not None else 0.0,
            "recommendation": rec,
            "confidence": confidence,
            "drivers": drivers,
            "risk": risk,
        })

    # Currently-focused symbol with no holding but with actionable signals:
    # surface as a "Fortuna is suggesting an entry" row.
    if focus_symbol and focus_symbol not in seen_symbols and focus_signals:
        rec, drivers, confidence, risk = _decision_or_legacy_recommendation(
            focus_symbol, "flat", None, signals_by_symbol, focus_symbol,
            decisions_by_symbol,
        )
        if rec != "HOLD":
            rows.append({
                "symbol": focus_symbol,
                "kind": "Flat",
                "side": "—",
                "qty": 0,
                "avg": None,
                "mark": None,
                "MTM": 0.0,
                "recommendation": rec,
                "confidence": confidence,
                "drivers": drivers,
                "risk": risk,
            })

    st.subheader("🧭 What to do now")
    if not rows:
        st.caption(
            "Nothing to recommend yet — open positions / holdings will appear here "
            "with the latest Fortuna recommendation joined alongside."
        )
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "**recommendation** is produced by Fortuna's advisory agents when enabled; "
        "otherwise the table falls back to legacy strategy-signal aggregation. "
        "Paper learning tracks agentic decisions without placing live broker orders."
    )


def _decision_or_legacy_recommendation(
    symbol,
    kind,
    current_side,
    signals_by_symbol,
    focus_symbol,
    decisions_by_symbol,
):
    sym_key = symbol.upper().strip()
    if "." not in sym_key and not sym_key.endswith("-EQ"):
        sym_key = f"{sym_key}.NS"
    decision = decisions_by_symbol.get(sym_key) or decisions_by_symbol.get(symbol)
    if decision is not None:
        action_obj = getattr(decision, "action", "HOLD")
        action = getattr(action_obj, "value", None) or str(action_obj)
        confidence = getattr(decision, "confidence", 0.0)
        rationale = getattr(decision, "rationale", None)
        reasons = getattr(rationale, "reasons", []) if rationale is not None else []
        risk_notes = getattr(rationale, "risk_notes", []) if rationale is not None else []
        drivers = "; ".join(reasons[:3])
        risk = "; ".join(risk_notes[:2])
        return (action, drivers, f"{confidence * 100:.0f}%", risk)
    rec, drivers = _recommendation_for(
        symbol, kind, current_side, signals_by_symbol, focus_symbol,
    )
    return (rec, drivers, "", "")


def _recommendation_for(symbol, kind, current_side, signals_by_symbol, focus_symbol):
    """Aggregate per-strategy signals into a single recommendation for a symbol."""
    sym_key = symbol.upper().strip()
    if "." not in sym_key and not sym_key.endswith("-EQ"):
        sym_key = f"{sym_key}.NS"
    signals = signals_by_symbol.get(sym_key) or signals_by_symbol.get(symbol) or {}
    if not signals:
        return ("HOLD (loading…)", "")

    actions: list[tuple[str, str]] = []
    for strat_name, sig in signals.items():
        action = sig.action
        if action in {"BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"}:
            actions.append((action, strat_name))

    if not actions:
        return ("HOLD", "")

    if current_side in {"LONG", "long"} or kind == "holding":
        exits = [s for a, s in actions if a == "EXIT_LONG"]
        if exits:
            return ("EXIT_LONG", ", ".join(sorted(set(exits))))
        return ("HOLD", "no exit yet")
    if current_side in {"SHORT", "short"}:
        exits = [s for a, s in actions if a == "EXIT_SHORT"]
        if exits:
            return ("EXIT_SHORT", ", ".join(sorted(set(exits))))
        return ("HOLD", "no exit yet")

    # Flat → look for BUY / SELL agreement
    buys = [s for a, s in actions if a == "BUY"]
    sells = [s for a, s in actions if a == "SELL"]
    if len(buys) >= len(sells) and buys:
        return ("BUY", ", ".join(sorted(set(buys))))
    if sells:
        return ("SELL_SHORT", ", ".join(sorted(set(sells))))
    return ("HOLD", "")


def _render_holdings_block(engine) -> None:
    """Phase 1: list long-term DEMAT holdings mirrored from the broker."""
    account = engine.live_account
    if account is None or not account.holdings:
        return
    st.subheader("Long-term holdings (mirrored from Angel One)")
    rows = []
    for sym, h in account.holdings.items():
        mv = h.market_value()
        rows.append({
            "symbol": sym,
            "tradingsymbol": h.tradingsymbol,
            "qty": h.qty,
            "avg": round(h.avg_price, 2),
            "ltp": round(h.last_price, 2) if h.last_price is not None else None,
            "market value": round(mv, 2),
            "P&L": round(h.pnl, 2) if h.pnl is not None else None,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_reconciliation_block(engine) -> None:
    """Phase 2: paper-vs-real reconciliation rendered inline."""
    if not getattr(engine, "account_sync_enabled", False):
        return
    st.subheader("🪞 Reconciliation (paper vs broker)")
    if st.button("Run reconciliation now"):
        report = engine.run_reconciliation()
        if report is None:
            st.warning("Reconciliation unavailable — check sync status.")
            return
        st.session_state["__fortuna_reco"] = report
    report = st.session_state.get("__fortuna_reco")
    if report is None:
        st.caption("Click **Run reconciliation now** to compare Fortuna's paper journal against your real Angel One trade book for today.")
        return
    s = report.to_dict()
    cols = st.columns(5)
    cols[0].metric("Paper fills", s["paper_fills"])
    cols[1].metric("Real fills", s["real_fills"])
    cols[2].metric("Matched", s["matched"])
    cols[3].metric("Coverage %", f"{s['coverage_pct']:.2f}")
    cols[4].metric("Avg slippage %", f"{s['avg_slippage_pct']:+.4f}")

    if report.matched:
        st.markdown("**Matched**")
        st.dataframe([
            {
                "symbol": m.symbol,
                "side": m.side.value,
                "qty": m.qty,
                "paper px": round(m.paper_price, 2),
                "real px": round(m.real_price, 2),
                "slippage %": round(m.slippage_pct, 4),
                "strategy": m.strategy or "",
            }
            for m in report.matched
        ], use_container_width=True, hide_index=True)
    if report.paper_only:
        st.markdown("**Paper-only (you missed these)**")
        st.dataframe([
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "qty": p.qty,
                "paper px": round(p.fill_price, 2),
                "strategy": p.strategy or "",
                "ts": p.fill_ts.strftime("%H:%M:%S") if p.fill_ts else "",
            }
            for p in report.paper_only
        ], use_container_width=True, hide_index=True)
    if report.real_only:
        st.markdown("**Real-only (manual, no Fortuna recommendation)**")
        st.dataframe([
            {
                "symbol": rt.fortuna_symbol or rt.tradingsymbol,
                "txn": rt.transaction_type,
                "qty": rt.quantity,
                "real px": round(rt.fill_price, 2),
                "product": rt.producttype or "",
                "ts": rt.fill_time.strftime("%H:%M:%S") if rt.fill_time else "",
            }
            for rt in report.real_only
        ], use_container_width=True, hide_index=True)


def _render_monitor_panel(engine) -> None:
    """Live event feed; filter by severity / event_type / symbol / strategy."""
    if not getattr(engine, "execution_enabled", False):
        st.info("Enable execution to populate the monitor feed (see Execution tab).")
        return
    monitor = engine.execution_monitor
    if monitor is None:
        st.warning("Execution monitor unavailable.")
        return

    events = monitor.snapshot()
    if not events:
        st.caption("No events yet.")
        return

    severities = sorted({e.severity.value for e in events})
    event_types = sorted({e.event_type for e in events})

    cols = st.columns(3)
    sel_sev = cols[0].multiselect("Severity", severities, default=severities)
    sel_type = cols[1].multiselect("Event type", event_types, default=event_types)
    limit = cols[2].selectbox("Show", [50, 100, 250, 500, len(events)], index=1)

    rows = []
    for evt in reversed(events):
        if evt.severity.value not in sel_sev:
            continue
        if evt.event_type not in sel_type:
            continue
        rows.append({
            "ts": evt.ts.strftime("%H:%M:%S"),
            "severity": evt.severity.value,
            "event": evt.event_type,
            "symbol": evt.symbol or "",
            "strategy": evt.strategy or "",
            "message": evt.message,
        })
        if len(rows) >= int(limit):
            break

    if not rows:
        st.caption("No events match the current filter.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)

    stats = getattr(engine.execution_router, "stats", None)
    if stats is not None:
        st.divider()
        cols = st.columns(4)
        cols[0].metric("Bars processed", stats.bars_processed)
        cols[1].metric("Signals seen", stats.signals_seen)
        cols[2].metric("Orders placed", stats.orders_placed)
        cols[3].metric("Risk blocks", stats.risk_blocks)
        cols = st.columns(4)
        cols[0].metric("Fills", stats.fills)
        cols[1].metric("RL suppressed", stats.rl_suppressed)
        cols[2].metric("Cooldown skips", stats.cooldown_skips)
        cols[3].metric("Breaker trips", stats.circuit_breaker_trips)


def _render_models_panel(engine, live_signals) -> None:
    """RL, ML, regime, and recent agentic decision status."""
    status = {}
    try:
        status = engine.model_status().to_dict()
    except Exception:  # noqa: BLE001
        status = {}

    _render_registry_panel(status)
    st.divider()
    _render_rl_panel(engine, live_signals, rl_status=status.get("rl") or {})
    st.divider()
    _render_ml_panel(engine, ml_status=status.get("ml") or {})
    st.divider()
    _render_regime_panel(status.get("regime") or {})
    st.divider()
    _render_agentic_log_panel(status.get("agentic") or {})


def _render_registry_panel(status: dict) -> None:
    st.subheader("Promotion registry")
    registry_enabled = bool(status.get("registry_enabled"))
    promotion_required = bool(status.get("promotion_required"))
    cols = st.columns(3)
    cols[0].metric("Registry enabled", "yes" if registry_enabled else "no")
    cols[1].metric("Promotion required", "yes" if promotion_required else "no")
    cols[2].metric(
        "Fallback mode",
        "pointer-only" if (registry_enabled and promotion_required) else "legacy/dev",
    )
    if registry_enabled and promotion_required:
        st.caption(
            "Missing live pointers keep the affected ML/RL path out of advisory, while deterministic signals continue."
        )
    else:
        st.caption(
            "Registry is relaxed or disabled, so dev-time fallback loading may still be used."
        )


def _render_ml_panel(engine, *, ml_status: dict) -> None:
    cols = st.columns([3, 1])
    cols[0].subheader("ML signal scorer")
    if cols[1].button("Reload ML scorer", key="reload_ml_scorer"):
        ok = engine.reload_ml_scorer()
        if ok:
            st.success("ML scorer reloaded.")
            st.rerun()
        else:
            st.warning("No ML scorer available — agentic runs without ML votes.")

    if not ml_status.get("enabled"):
        st.info("ML scorer disabled (`FORTUNA_AGENTIC_ML_SCORER_ENABLED=0`).")
        return
    if not ml_status.get("available"):
        st.info(
            "No promoted ML scorer loaded. Train and promote via "
            "`uv run python scripts/promote_model.py --kind ml_scorer --run-id <id>`."
        )
        if ml_status.get("live_pointer"):
            st.caption(f"Live pointer: `{ml_status['live_pointer']}`")
        if ml_status.get("artifact_dir"):
            st.caption(f"Artifact dir: `{ml_status['artifact_dir']}`")
        if ml_status.get("last_promotion"):
            _render_promotion_caption(ml_status["last_promotion"])
        return

    st.markdown(
        f"**Run ID:** `{ml_status.get('run_id', '—')}` &nbsp;|&nbsp; "
        f"**Verdict:** `{'PASS' if ml_status.get('verdict_passed') else 'FAIL'}` &nbsp;|&nbsp; "
        f"**Advisory ready:** `{ml_status.get('advisory_ready', False)}`"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("OOS precision", f"{float(ml_status.get('oos_precision', 0)):.3f}")
    m2.metric("OOS ROC-AUC", f"{float(ml_status.get('oos_roc_auc', 0)):.3f}")
    m3.metric("Schema hash", (ml_status.get("feature_schema_hash") or "—")[:12])
    if ml_status.get("live_pointer"):
        st.caption(f"Live pointer: `{ml_status['live_pointer']}`")
    if ml_status.get("artifact_dir"):
        st.caption(f"Artifact dir: `{ml_status['artifact_dir']}`")
    if ml_status.get("last_promotion"):
        _render_promotion_caption(ml_status["last_promotion"])


def _render_regime_panel(regime_status: dict) -> None:
    st.subheader("Regime detector")
    if regime_status.get("available"):
        st.success(f"Available at `{regime_status.get('path', '—')}`")
    else:
        st.info("Regime classifier not found — RL uses default policy routing.")


def _render_agentic_log_panel(agentic_status: dict) -> None:
    st.subheader("Recent agentic decisions")
    rows = agentic_status.get("recent_decisions") or []
    if not rows:
        st.info("No agentic decisions logged yet.")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    _render_learning_summary(agentic_status.get("learning_summary") or {})


def _render_rl_panel(engine, live_signals, *, rl_status: dict | None = None) -> None:
    """Render the RL policy status + metadata card.

    Shows a friendly empty-state when no checkpoint exists, otherwise
    renders the OOS metrics + FilterVerdict + raw metadata.json.
    """
    rl_gen = getattr(engine, "rl_generator", None)
    available = bool(rl_gen and getattr(rl_gen, "is_available", False))

    cols = st.columns([3, 1])
    cols[0].subheader("RL policy status")
    if cols[1].button("Reload policy", key="reload_rl_policy"):
        ok = engine.reload_rl_generator()
        if ok:
            st.success("RL policy reloaded.")
            st.rerun()
        else:
            st.warning("No RL policy available — deterministic fallback is active.")

    if rl_status and rl_status.get("live_pointer"):
        st.caption(f"Live pointer: `{rl_status['live_pointer']}`")
    if rl_status and rl_status.get("checkpoint_dir"):
        st.caption(f"Checkpoint dir: `{rl_status['checkpoint_dir']}`")

    if not available:
        st.info(
            "No validated RL checkpoint is loaded. Train one via "
            "`uv run python scripts/run_rl_train.py` and promote it via "
            "`uv run python scripts/promote_policy.py --run-id <id>`. "
            "The dashboard continues to render deterministic signals."
        )
        if rl_status and rl_status.get("last_promotion"):
            _render_promotion_caption(rl_status["last_promotion"])
        return

    meta = getattr(rl_gen, "metadata", None)
    if meta is None:
        st.warning("Policy loaded but metadata.json missing.")
        return

    badge = "PASS" if meta.verdict_passed else "FAIL"
    ready = "yes" if getattr(meta, "advisory_ready", False) else "no"
    st.markdown(
        f"**Run ID:** `{meta.run_id}` &nbsp;|&nbsp; **Verdict:** `{badge}` &nbsp;|&nbsp; "
        f"**Advisory ready:** `{ready}` &nbsp;|&nbsp; **Policy:** `{meta.policy_type}`"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OOS Sharpe", f"{meta.oos_metrics.sharpe_ratio:.2f}")
    m2.metric("OOS PF", f"{meta.oos_metrics.profit_factor:.2f}")
    m3.metric("OOS Trades", meta.oos_metrics.total_trades)
    m4.metric("OOS MaxDD %", f"{meta.oos_metrics.max_drawdown_pct * 100:.2f}")
    if rl_status and rl_status.get("last_promotion"):
        _render_promotion_caption(rl_status["last_promotion"])

    rl_keys = [k for k in live_signals if k.startswith("RL:")]
    if rl_keys:
        latest = live_signals[rl_keys[0]]
        st.markdown(
            f"**Latest RL signal:** `{latest.action}` at "
            f"`{latest.bar_time}` close={latest.bar_close:.2f}"
        )

    with st.expander("metadata.json"):
        st.json(meta.to_dict())


def _render_learning_summary(summary: dict) -> None:
    st.subheader("Paper-learning outcomes")
    total = int(summary.get("total_rows", 0) or 0)
    if total == 0:
        st.caption("No learning rows recorded yet.")
        return
    cols = st.columns(4)
    cols[0].metric("Rows", total)
    cols[1].metric("Resolved", int(summary.get("resolved_rows", 0) or 0))
    cols[2].metric("Paper closed", int(summary.get("paper_closed_rows", 0) or 0))
    avg_pnl = summary.get("avg_realized_pnl_pct")
    cols[3].metric("Avg realized %", "—" if avg_pnl is None else f"{float(avg_pnl):+.2f}")
    recent = summary.get("recent_rows") or []
    if recent:
        st.dataframe(recent, use_container_width=True, hide_index=True)


def _render_promotion_caption(promotion: dict) -> None:
    run_id = promotion.get("run_id") or "—"
    promoted_at = promotion.get("promoted_at") or promotion.get("ts") or "—"
    promoted_by = promotion.get("promoted_by") or "unknown"
    st.caption(
        f"Last promotion: run `{run_id}` by `{promoted_by}` at `{promoted_at}`"
    )


if __name__ == "__main__":
    main()
