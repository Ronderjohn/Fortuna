#!/usr/bin/env python
"""Fortuna Streamlit dashboard — TradingView-style symbol search and strategy reports."""

from __future__ import annotations

import os
import sys
import time
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

from fortuna.app.assistant_presets import (
    build_dashboard_assistant_actions,
    build_dashboard_assistant_examples,
)
from fortuna.app.acceptance_bundle import build_and_export_acceptance_bundle
from fortuna.app.config import AppConfig
from fortuna.app.operator_workflow import (
    annotate_artifact_recovery_posture,
    annotate_discovery_refresh,
    export_workflow_snapshot,
    replace_workflow_snapshot_sections,
    summarize_market_universe,
    summarize_multi_agent_workflow,
    summarize_portfolio_allocation,
    summarize_shortlist_analysis,
    summarize_shortlist_briefing,
    summarize_training_candidates,
    summarize_training_research,
    WorkflowSnapshot,
)
from fortuna.app.presentation import (
    export_leaderboard_csv,
    render_leaderboard,
    render_live_signal_banner,
    render_live_signal_panel,
    render_ohlcv_ticker,
    render_strategy_comparison_grid,
    render_strategy_report_card,
)
from fortuna.app.promotion_review import build_and_export_promotion_review
from fortuna.app.multi_agent_team import replace_workflow_universe
from fortuna.app.nightly_alignment_view import (
    format_latest_nightly_discovery_context,
    format_latest_nightly_execution_posture,
)
from fortuna.app.session_engine import FortunaSessionEngine
from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.app.workflow_artifacts import (
    artifact_follow_up_prefers_discovery,
    artifact_follow_up_refresh_target,
    artifact_follow_up_team_posture,
    build_workflow_artifact_follow_up,
    build_workflow_artifact_rows,
)
from fortuna.config.settings import load_settings
from fortuna.config.smartapi_settings import get_smartapi_settings
from fortuna.models.metadata import ModelKind

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
        if eng.is_live():
            try:
                eng.poll_live()
            except Exception:
                pass
        state = eng.state
        if not state or state.symbol != symbol or state.timeframe != timeframe:
            return {"bars": [], "markers": [], "overlays": [], "last": 0}
        df = eng.get_ohlcv_for_chart()
        if df is None or df.empty:
            return {"bars": [], "markers": [], "overlays": [], "last": 0}

        tail = df.tail(60)
        cutoff = int(since)
        bars_by_time: dict[int, dict] = {}
        for ts, row in tail.iterrows():
            t = ist_unix_seconds(pd.Timestamp(ts))
            if t < cutoff:
                continue
            bars_by_time[t] = {
                "time": t,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        # Always include the latest candle (forming bar in-place updates use the
        # same or a newer timestamp) so live ticks reach the chart even when
        # ``since`` would otherwise filter them out.
        if not df.empty:
            ts = df.index[-1]
            t = ist_unix_seconds(pd.Timestamp(ts))
            row = df.iloc[-1]
            bars_by_time[t] = {
                "time": t,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        bars = [bars_by_time[k] for k in sorted(bars_by_time)]

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
    if "_live_on_prev" not in st.session_state:
        st.session_state._live_on_prev = None


def _try_start_live(engine: FortunaSessionEngine, *, show_error: bool = True) -> bool:
    """Start SmartAPI WebSocket feed; return True when running."""
    if engine.is_live():
        return True
    if engine.state.ohlcv is None:
        return False
    try:
        engine.start_live()
        return engine.is_live()
    except Exception as e:
        if show_error:
            st.error(f"Live feed failed: {e}")
        return False


def _try_stop_live(engine: FortunaSessionEngine) -> None:
    if engine.is_live():
        engine.stop_live()


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
    st.session_state.load_params = (symbol, timeframe, int(days))
    if live_on:
        if not _try_start_live(engine):
            st.warning(
                "Live is checked but the WebSocket feed did not start. "
                "Confirm **SMARTAPI_USE_LIVE_FEED=true** in `.env`, then re-run **Analyze**."
            )


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

    get_smartapi_settings.cache_clear()
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

            loaded_tf = engine.state.timeframe if engine.state.ohlcv is not None else settings.default_timeframe
            if loaded_tf not in TRADINGVIEW_INTERVALS:
                loaded_tf = "5m"
            timeframe = st.selectbox(
                "Interval",
                TRADINGVIEW_INTERVALS,
                index=TRADINGVIEW_INTERVALS.index(loaded_tf),
                format_func=interval_label,
                label_visibility="visible",
            )
        with c_days:
            loaded_days = int(engine.state.days) if engine.state.ohlcv is not None else int(settings.default_days)
            days = st.number_input(
                "History (days)",
                min_value=7,
                max_value=90,
                value=loaded_days,
            )
        with c_force:
            force = st.checkbox("Refresh data", value=False)
        with c_live:
            live_on = st.checkbox("Live", value=_is_nse_session())
            smart = get_smartapi_settings()
            if live_on and not smart.use_live_feed:
                st.caption("Set SMARTAPI_USE_LIVE_FEED=true in `.env` and refresh")
            elif live_on and smart.use_live_feed:
                st.caption("WebSocket snap quotes · NSE session")
        prev_live = st.session_state.get("_live_on_prev")
        if prev_live is not None and prev_live != live_on:
            st.session_state._live_on_prev = live_on
            if live_on and engine.state.ohlcv is not None and _is_nse_session():
                if _try_start_live(engine):
                    st.toast("Live feed started.", icon="📡")
            elif not live_on:
                _try_stop_live(engine)
            st.rerun()
        st.session_state._live_on_prev = live_on
        with c_load:
            st.write("")
            load_clicked = st.button("Analyze", type="primary", use_container_width=True)

    symbol = st.session_state.selected_symbol
    load_params = (symbol, timeframe, int(days))
    prior_params = st.session_state.get("load_params")
    params_changed = (
        st.session_state.get("loaded")
        and prior_params is not None
        and prior_params != load_params
    )
    if params_changed:
        st.info(
            f"Interval or history changed — reloading **{symbol}** at **{timeframe}** "
            f"({int(days)} days)…"
        )

    if load_clicked or st.session_state.get("trigger_load") or params_changed:
        reload_force = force or params_changed
        _run_load(
            engine,
            symbol,
            timeframe,
            int(days),
            reload_force,
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
            stats = engine.live_stats()
            tick_n = int(stats.get("tick_count") or 0)
            last_tick = stats.get("last_tick_at")
            tick_age = ""
            if isinstance(last_tick, (int, float)):
                tick_age = f" · {max(0, int(time.monotonic() - last_tick))}s since tick"

            label = f"{new_ts:%H:%M}" if new_ts is not None else "waiting"
            sig_suffix = (
                f" · signals {sig_age:%H:%M:%S}" if sig_age is not None else ""
            )
            st.markdown(
                "<div style='background:#0e8345;color:#fff;padding:4px 12px;"
                "border-radius:6px;display:inline-block;font-weight:600;"
                f"font-size:0.85rem;'>📡 LIVE · {tick_n} ticks · last bar {label}"
                f"{tick_age}{sig_suffix}</div>",
                unsafe_allow_html=True,
            )

        _live_pulse()
    elif live_on and engine.state.ohlcv is not None and _is_nse_session():
        if get_smartapi_settings().use_live_feed and _try_start_live(engine, show_error=False):
            st.toast("Live feed auto-started — market is open.", icon="📡")
            st.rerun()
        else:
            st.warning(
                "Live is enabled but the feed is not connected. "
                "Check **SMARTAPI_USE_LIVE_FEED=true** and click **Analyze** again."
            )

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

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Bars", f"{len(ohlcv):,}")
    m2.metric("Interval", state.timeframe)
    m3.metric("From", str(ohlcv.index.min().date()))
    m4.metric("To", str(ohlcv.index.max().date()))
    m5.metric("Strategies", len(batch.results))
    live_label = "ON" if engine.is_live() else "OFF"
    if engine.is_live():
        stats = engine.live_stats()
        tick_n = int(stats.get("tick_count") or 0)
        if state.last_bar_time is not None:
            live_label = f"ON · {state.last_bar_time:%H:%M} · {tick_n} ticks"
        else:
            live_label = f"ON · {tick_n} ticks"
    m6.metric("Live", live_label)

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
                    timeframe=state.timeframe,
                )
                if spec:
                    # Hand the iframe a polling URL. The JS inside the iframe
                    # fetches incremental bars + markers from this endpoint
                    # every couple of seconds and applies them in-place via
                    # series.update() — preserving zoom/pan (TV-style stream).
                    stream_url = None
                    if engine.is_live() and _is_nse_session():
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
                    elif live_on and not engine.is_live():
                        st.caption(
                            "Chart streaming starts once the live WebSocket feed connects. "
                            "Ensure **Live** is checked and click **Analyze**."
                        )

                    components.html(
                        render_lightweight_charts_html(
                            spec,
                            height=580,
                            stream_url=stream_url,
                            stream_poll_ms=1500,
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
            f"**{state.timeframe}** · NSE session 09:15–15:30 IST · "
            f"initial view anchors on today's session (pan left for prior days). "
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
    st.subheader("Signal assistant")
    adapter_enabled = bool(getattr(engine.settings, "conversational_adapter_enabled", False))
    if adapter_enabled:
        st.caption(
            "Telegram and chat-style signal requests are the primary product path. "
            "This console is the operator-side view of the same typed assistant."
        )
    else:
        st.caption(
            "Conversational mode is disabled, but explicit signal commands still work."
        )

    expert_visible = bool(getattr(engine.settings, "signal_expert_commands_visible", False))
    actions = build_dashboard_assistant_actions(include_expert=expert_visible)
    examples = build_dashboard_assistant_examples(include_expert=expert_visible)
    st.write("Examples:")
    for ex in examples:
        st.code(ex, language="text")

    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = []
    if "assistant_workflow" not in st.session_state:
        st.session_state.assistant_workflow = None

    assistant = engine.conversational_assistant()
    st.write("Quick actions:")
    queued_prompt = None
    action_columns = st.columns(3)
    for idx, action in enumerate(actions):
        column = action_columns[idx % len(action_columns)]
        if column.button(
            action.label,
            key=f"assistant-action-{idx}",
            help=action.caption,
            use_container_width=True,
        ):
            queued_prompt = action.prompt

    prompt = st.chat_input(
        "Ask for a signal on a stock, future, or option"
    )
    chosen_prompt = queued_prompt or prompt
    if chosen_prompt:
        _run_assistant_prompt(engine, assistant, chosen_prompt)

    _render_assistant_workflow_console(engine)

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


def _run_assistant_prompt(engine, assistant, prompt: str) -> None:
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


def _infer_research_refresh_target_from_command(command: str) -> str:
    text = str(command or "").strip()
    if not text:
        return "rl"
    parts = text.split()
    for idx, part in enumerate(parts):
        token = str(part or "").strip().lower()
        if token == "--refresh-target" and idx + 1 < len(parts):
            value = str(parts[idx + 1] or "").strip().lower()
            if value in {"rl", "ml", "all"}:
                return value
        if token.startswith("--refresh-target="):
            value = token.split("=", 1)[1].strip().lower()
            if value in {"rl", "ml", "all"}:
                return value
    return "rl"


def _sync_research_refresh_target_selection(recommended_command: str) -> str:
    options = ("rl", "ml", "all")
    suggested = _infer_research_refresh_target_from_command(recommended_command)
    selection_key = "assistant-workflow-research-refresh-target"
    suggested_key = "assistant-workflow-research-refresh-target-suggested"
    current = str(st.session_state.get(selection_key, "") or "").strip().lower()
    previous_suggested = str(st.session_state.get(suggested_key, "") or "").strip().lower()
    if current not in options or not current or current == previous_suggested:
        st.session_state[selection_key] = suggested
    st.session_state[suggested_key] = suggested
    return str(st.session_state.get(selection_key, suggested) or suggested)


def _sync_research_refresh_target_preference(
    summary: dict,
    recommended_command: str,
    *,
    artifact_target: str | None = None,
) -> str:
    target = str(summary.get("recommended_refresh_target") or "").strip().lower()
    if target not in {"rl", "ml", "all"}:
        artifact = str(artifact_target or "").strip().lower()
        if artifact in {"rl", "ml", "all"}:
            target = artifact
    if target not in {"rl", "ml", "all"}:
        target = _sync_research_refresh_target_selection(recommended_command)
    else:
        options = ("rl", "ml", "all")
        selection_key = "assistant-workflow-research-refresh-target"
        suggested_key = "assistant-workflow-research-refresh-target-suggested"
        current = str(st.session_state.get(selection_key, "") or "").strip().lower()
        previous_suggested = str(st.session_state.get(suggested_key, "") or "").strip().lower()
        if current not in options or not current or current == previous_suggested:
            st.session_state[selection_key] = target
        st.session_state[suggested_key] = target
    return str(st.session_state.get("assistant-workflow-research-refresh-target", target) or target)


def _nightly_alignment_prefill_state(summary: dict) -> tuple[bool, bool]:
    if not isinstance(summary, dict):
        return False, False
    recommended_action = str(summary.get("recommended_action") or "").strip()
    if not recommended_action:
        return False, False
    typed_force_refresh = summary.get("recommended_force_refresh")
    if isinstance(typed_force_refresh, bool):
        return True, typed_force_refresh
    enabled_reports = int(summary.get("enabled_reports", 0) or 0)
    refreshed_aligned_reports = int(summary.get("refreshed_aligned_reports", 0) or 0)
    refreshed_ratio = (
        float(refreshed_aligned_reports) / float(enabled_reports)
        if enabled_reports > 0
        else 0.0
    )
    severe = enabled_reports >= 2 and refreshed_ratio <= 0.2
    return True, severe


def _format_recent_nightly_alignment_trend(summary: dict) -> str:
    if not isinstance(summary, dict):
        return ""
    recent_window = int(summary.get("recent_trend_window", 0) or 0)
    recent_enabled = int(summary.get("recent_trend_enabled", 0) or 0)
    recent_aligned = int(summary.get("recent_trend_aligned", 0) or 0)
    latest_status = str(summary.get("recent_trend_latest_status") or "").strip() or "—"
    latest_basket = int(summary.get("recent_trend_latest_basket_size", 0) or 0)
    if recent_window <= 0:
        recent = summary.get("recent_rows") or ()
        if not recent:
            return ""
        recent_window_rows = tuple(row for row in recent[:3] if isinstance(row, dict))
        if not recent_window_rows:
            return ""
        recent_window = len(recent_window_rows)
        recent_enabled = sum(
            1 for row in recent_window_rows if bool(row.get("alignment_enabled"))
        )
        recent_aligned = sum(
            1
            for row in recent_window_rows
            if bool(row.get("alignment_enabled")) and str(row.get("target_mix") or "").strip()
        )
        latest = recent_window_rows[0]
        latest_status = str(latest.get("overall_status") or "").strip() or "—"
        latest_basket = int(latest.get("basket_size", 0) or 0)
    return (
        f"Recent trend: {recent_aligned}/{recent_enabled} aligned over last "
        f"{recent_window} run(s); latest={latest_status} basket={latest_basket}"
    )


def _target_from_alignment_mix(mix: str) -> str:
    text = str(mix or "").strip().lower()
    if not text:
        return ""
    keys = set()
    for part in text.split(","):
        chunk = part.strip()
        if not chunk or "=" not in chunk:
            continue
        key = chunk.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    if not keys:
        return ""
    if "none" in keys or "both" in keys:
        return "all"
    if "ml" in keys and "rl" not in keys:
        return "rl"
    if "rl" in keys and "ml" not in keys:
        return "ml"
    if "ml" in keys and "rl" in keys:
        return "all"
    return ""


def _sync_research_refresh_toggle(
    *,
    key: str,
    suggested_key: str,
    suggested_value: bool,
) -> bool:
    current = st.session_state.get(key)
    previous_suggested = st.session_state.get(suggested_key)
    if current is None or current == previous_suggested:
        st.session_state[key] = suggested_value
    st.session_state[suggested_key] = suggested_value
    return bool(st.session_state.get(key, suggested_value))


def _workflow_snapshot_from_state(
    workflow: dict,
    *,
    source: str,
    timeframe: str,
    days: int,
) -> WorkflowSnapshot | None:
    if not isinstance(workflow, dict):
        return None
    snapshot = workflow.get("workflow_snapshot")
    if isinstance(snapshot, WorkflowSnapshot):
        return snapshot
    required = ("team", "universe", "shortlist", "briefing", "candidates", "research", "allocation")
    if any(workflow.get(key) is None for key in required):
        return None
    return WorkflowSnapshot(
        source=source,
        timeframe=timeframe,
        lookback_days=int(days),
        team=workflow["team"],
        universe=workflow["universe"],
        shortlist=workflow["shortlist"],
        briefing=workflow["briefing"],
        candidates=workflow["candidates"],
        research=workflow["research"],
        allocation=workflow["allocation"],
    )


def _annotate_snapshot_artifact_follow_up(
    snapshot: WorkflowSnapshot | None,
    workflow: dict,
) -> WorkflowSnapshot | None:
    if snapshot is None or not isinstance(workflow, dict):
        return snapshot
    follow_up = build_workflow_artifact_follow_up(workflow)
    if follow_up is None:
        return snapshot
    return annotate_artifact_recovery_posture(
        snapshot,
        action_label=follow_up.action_label,
        target=follow_up.target,
        posture_text=artifact_follow_up_team_posture(follow_up),
    )


def _render_assistant_workflow_console(engine) -> None:
    st.divider()
    st.subheader("Workflow console")
    st.caption(
        "Refresh the liquid universe, ranked shortlist, allocation view, and shortlist-driven "
        "ML/RL training candidates from one place."
    )
    try:
        model_health = engine.model_status().to_dict()
    except Exception:
        model_health = {}
    current_workflow = st.session_state.assistant_workflow or {}
    artifact_follow_up = build_workflow_artifact_follow_up(current_workflow)
    nightly_alignment = (
        ((model_health.get("agentic") or {}).get("nightly_alignment") or {})
        if isinstance(model_health, dict)
        else {}
    )
    recommended_action = str(nightly_alignment.get("recommended_action") or "").strip()
    recommended_command = str(nightly_alignment.get("recommended_cli_command") or "").strip()
    suggested_refresh_target = _sync_research_refresh_target_preference(
        nightly_alignment,
        recommended_command,
        artifact_target=artifact_follow_up_refresh_target(artifact_follow_up),
    )
    suggested_refresh_data, suggested_force_refresh = _nightly_alignment_prefill_state(
        nightly_alignment
    )
    research_refresh_data_default = _sync_research_refresh_toggle(
        key="assistant-workflow-research-refresh-data",
        suggested_key="assistant-workflow-research-refresh-data-suggested",
        suggested_value=suggested_refresh_data,
    )
    research_force_refresh_default = _sync_research_refresh_toggle(
        key="assistant-workflow-research-force-refresh",
        suggested_key="assistant-workflow-research-force-refresh-suggested",
        suggested_value=suggested_force_refresh,
    )
    if recommended_action:
        st.warning(recommended_action)
        if recommended_command:
            st.caption(f"Suggested command: `{recommended_command}`")
            st.caption(f"Suggested in-app refresh target: `{suggested_refresh_target}`")
            if suggested_refresh_data:
                st.caption(
                    "Suggested in-app refresh mode: `Refresh research data`"
                    + (" + `Force research refresh`" if suggested_force_refresh else "")
                )
    elif artifact_follow_up is not None:
        target_note = (
            f" target=`{artifact_follow_up.target}`" if artifact_follow_up.target else ""
        )
        st.caption(
            f"Artifact-driven follow-up: **{artifact_follow_up.action_label}**"
            f"{target_note} - {artifact_follow_up.rationale}"
        )

    controls = st.columns(5)
    timeframe = controls[0].selectbox(
        "Timeframe",
        ("5m", "15m", "30m", "1h", "1d"),
        index=0,
        key="assistant-workflow-timeframe",
    )
    days = int(
        controls[1].number_input(
            "Lookback days",
            min_value=5,
            max_value=120,
            value=20,
            step=5,
            key="assistant-workflow-days",
        )
    )
    source = controls[2].selectbox(
        "Universe source",
        ("auto", "screener", "registry"),
        index=0,
        key="assistant-workflow-source",
    )
    universe_limit = int(
        controls[3].number_input(
            "Universe limit",
            min_value=5,
            max_value=40,
            value=10,
            step=1,
            key="assistant-workflow-universe-limit",
        )
    )
    candidate_limit = int(
        controls[4].number_input(
            "Shortlist limit",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
            key="assistant-workflow-candidate-limit",
        )
    )
    refresh_controls = st.columns(3)
    research_refresh_data = refresh_controls[0].checkbox(
        "Refresh research data",
        value=research_refresh_data_default,
        key="assistant-workflow-research-refresh-data",
    )
    research_refresh_target = refresh_controls[1].selectbox(
        "Research refresh target",
        ("rl", "ml", "all"),
        index=("rl", "ml", "all").index(suggested_refresh_target),
        key="assistant-workflow-research-refresh-target",
    )
    research_force_refresh = refresh_controls[2].checkbox(
        "Force research refresh",
        value=research_force_refresh_default,
        key="assistant-workflow-research-force-refresh",
    )
    if artifact_follow_up_prefers_discovery(artifact_follow_up):
        st.caption(
            "Latest saved review artifacts are leaning toward **Refresh market universe** "
            "before another research refresh."
        )

    action_cols = st.columns(6)
    if action_cols[0].button(
        "Refresh workflow", key="assistant-workflow-refresh", use_container_width=True
    ):
        briefing_limit = max(3, min(candidate_limit, 10))
        team = engine.multi_agent_workflow(
            universe_limit=max(universe_limit, candidate_limit * 2),
            analysis_limit=candidate_limit,
            timeframe=timeframe,
            days=days,
            source=source,
            selection_policy="diversified",
            refresh_research_data=research_refresh_data,
            research_refresh_target=research_refresh_target,
            research_refresh_timeframe=timeframe,
            research_refresh_days=days,
            force_refresh=research_force_refresh,
        )
        workflow_snapshot = summarize_multi_agent_workflow(team)
        st.session_state.assistant_workflow = {
            "team_response": team,
            "workflow_snapshot": workflow_snapshot,
            "team": workflow_snapshot.team,
            "universe": workflow_snapshot.universe,
            "shortlist": workflow_snapshot.shortlist,
            "briefing": workflow_snapshot.briefing,
            "candidates": workflow_snapshot.candidates,
            "research": workflow_snapshot.research,
            "allocation": workflow_snapshot.allocation,
            "team_headline": team.headline,
            "team_roles": tuple(role.to_dict() for role in team.roles),
            "team_nightly_alignment": (
                team.nightly_alignment.to_dict() if team.nightly_alignment is not None else {}
            ),
        }
    if action_cols[1].button(
        "Refresh market universe",
        key="assistant-workflow-universe-refresh",
        use_container_width=True,
        disabled=not bool(st.session_state.assistant_workflow),
    ):
        workflow = st.session_state.assistant_workflow or {}
        universe_response = engine.market_universe(
            limit=max(universe_limit, candidate_limit * 2),
            timeframe="1d",
            days=max(days, 20),
            source=source,
        )
        workflow["universe"] = summarize_market_universe(universe_response)
        team_response = workflow.get("team_response")
        if team_response is not None:
            refreshed_team = replace_workflow_universe(
                team_response,
                universe=universe_response,
            )
            workflow_snapshot = annotate_discovery_refresh(
                summarize_multi_agent_workflow(refreshed_team),
                source=source,
                timeframe="1d",
                days=max(days, 20),
                refreshed_count=len(getattr(universe_response, "candidates", ())),
            )
            workflow["team_response"] = refreshed_team
            workflow["workflow_snapshot"] = workflow_snapshot
            workflow["team"] = workflow_snapshot.team
            workflow["universe"] = workflow_snapshot.universe
            workflow["team_headline"] = refreshed_team.headline
            workflow["team_roles"] = tuple(role.to_dict() for role in refreshed_team.roles)
            workflow["team_nightly_alignment"] = (
                refreshed_team.nightly_alignment.to_dict()
                if refreshed_team.nightly_alignment is not None
                else {}
            )
        st.session_state.assistant_workflow = workflow
    if action_cols[2].button(
        "Refresh training research",
        key="assistant-workflow-research-refresh",
        use_container_width=True,
    ):
        briefing_limit = max(3, min(candidate_limit, 10))
        workflow = st.session_state.assistant_workflow or {}
        workflow["research"] = summarize_training_research(
            engine.training_research_plan(
                universe_limit=max(universe_limit, candidate_limit * 2),
                analysis_limit=candidate_limit,
                timeframe=timeframe,
                days=days,
                source=source,
                selection_policy="diversified",
                refresh_data=research_refresh_data,
                refresh_target=research_refresh_target,
                refresh_timeframe=timeframe,
                refresh_days=days,
                force_refresh=research_force_refresh,
            )
        )
        if "candidates" not in workflow:
            workflow["candidates"] = summarize_training_candidates(
                engine.training_candidates(
                    universe_limit=max(universe_limit, candidate_limit * 2),
                    analysis_limit=candidate_limit,
                    timeframe=timeframe,
                    days=days,
                    source=source,
                )
            )
        if "allocation" not in workflow:
            workflow["allocation"] = summarize_portfolio_allocation(
                engine.portfolio_allocation(
                    universe_limit=max(universe_limit, briefing_limit * 2),
                    analysis_limit=briefing_limit,
                    timeframe=timeframe,
                    days=days,
                    source=source,
                )
            )
        workflow_source = (
            str(workflow.get("team_response").source)
            if workflow.get("team_response") is not None
            else source
        )
        snapshot = _workflow_snapshot_from_state(
            workflow,
            source=workflow_source,
            timeframe=timeframe,
            days=days,
        )
        if snapshot is not None:
            workflow["workflow_snapshot"] = replace_workflow_snapshot_sections(
                snapshot,
                research=workflow["research"],
                candidates=workflow["candidates"],
                allocation=workflow["allocation"],
            )
        st.session_state.assistant_workflow = workflow
    if action_cols[3].button(
        "Save workflow snapshot",
        key="assistant-workflow-save-snapshot",
        use_container_width=True,
        disabled=not bool(st.session_state.assistant_workflow),
    ):
        workflow = st.session_state.assistant_workflow or {}
        workflow_source = (
            str(workflow.get("team_response").source)
            if workflow.get("team_response") is not None
            else source
        )
        snapshot = _workflow_snapshot_from_state(
            workflow,
            source=workflow_source,
            timeframe=timeframe,
            days=days,
        )
        snapshot = _annotate_snapshot_artifact_follow_up(snapshot, workflow)
        if snapshot is None:
            st.warning("Load the workflow first so Fortuna has a full snapshot to export.")
        else:
            project_root = Path(getattr(engine.settings, "project_root", Path.cwd()))
            out_path = (
                project_root
                / "reports"
                / "dashboard"
                / f"workflow_snapshot_{time.strftime('%Y%m%d-%H%M%S')}.json"
            )
            written = export_workflow_snapshot(snapshot, out_path)
            workflow["workflow_snapshot"] = snapshot
            workflow["saved_workflow_snapshot_path"] = str(written)
            st.session_state.assistant_workflow = workflow
    if action_cols[4].button(
        "Build acceptance review",
        key="assistant-workflow-build-acceptance",
        use_container_width=True,
        disabled=not bool(st.session_state.assistant_workflow),
    ):
        workflow = st.session_state.assistant_workflow or {}
        workflow_source = (
            str(workflow.get("team_response").source)
            if workflow.get("team_response") is not None
            else source
        )
        snapshot = _workflow_snapshot_from_state(
            workflow,
            source=workflow_source,
            timeframe=timeframe,
            days=days,
        )
        snapshot = _annotate_snapshot_artifact_follow_up(snapshot, workflow)
        if snapshot is None:
            st.warning("Load the workflow first so Fortuna has a snapshot to review.")
        else:
            project_root = Path(getattr(engine.settings, "project_root", Path.cwd()))
            reports_dir = project_root / "reports" / "dashboard"
            snapshot_path = str(workflow.get("saved_workflow_snapshot_path") or "").strip()
            if not snapshot_path:
                snapshot_path = str(
                    export_workflow_snapshot(
                        snapshot,
                        reports_dir / f"workflow_snapshot_{time.strftime('%Y%m%d-%H%M%S')}.json",
                    )
                )
                workflow["saved_workflow_snapshot_path"] = snapshot_path
            stamp = time.strftime("%Y%m%d-%H%M%S")
            bundle, md_path, json_path = build_and_export_acceptance_bundle(
                settings=engine.settings,
                engine=engine,
                models_root=project_root / "models",
                ml_base=project_root / "models" / "ml_signal_scorer",
                workflow_snapshot_path=snapshot_path,
                out_dir=reports_dir,
                basename=f"acceptance_bundle_{stamp}",
                include_model_health=True,
            )
            workflow["workflow_snapshot"] = snapshot
            workflow["acceptance_bundle_status"] = bundle.overall_status
            workflow["acceptance_bundle_markdown_path"] = str(md_path)
            workflow["acceptance_bundle_json_path"] = str(json_path)
            st.session_state.assistant_workflow = workflow
    if action_cols[5].button(
        "Build promotion review",
        key="assistant-workflow-build-promotion",
        use_container_width=True,
        disabled=not bool(st.session_state.assistant_workflow),
    ):
        workflow = st.session_state.assistant_workflow or {}
        workflow_source = (
            str(workflow.get("team_response").source)
            if workflow.get("team_response") is not None
            else source
        )
        snapshot = _workflow_snapshot_from_state(
            workflow,
            source=workflow_source,
            timeframe=timeframe,
            days=days,
        )
        snapshot = _annotate_snapshot_artifact_follow_up(snapshot, workflow)
        if snapshot is None:
            st.warning("Load the workflow first so Fortuna has a snapshot to review.")
        else:
            project_root = Path(getattr(engine.settings, "project_root", Path.cwd()))
            reports_dir = project_root / "reports" / "dashboard"
            snapshot_path = str(workflow.get("saved_workflow_snapshot_path") or "").strip()
            if not snapshot_path:
                snapshot_path = str(
                    export_workflow_snapshot(
                        snapshot,
                        reports_dir / f"workflow_snapshot_{time.strftime('%Y%m%d-%H%M%S')}.json",
                    )
                )
                workflow["saved_workflow_snapshot_path"] = snapshot_path
            stamp = time.strftime("%Y%m%d-%H%M%S")
            review_result = build_and_export_promotion_review(
                kind=ModelKind.RL_POLICY,
                models_root=project_root / "models",
                ml_base=project_root / "models" / "ml_signal_scorer",
                symbol=str(getattr(engine.state, "symbol", "") or "").strip() or None,
                workflow_snapshot_path=snapshot_path,
                out_dir=reports_dir,
                basename=f"promotion_review_rl_{stamp}",
                settings=engine.settings,
            )
            review_kind = "rl"
            if review_result is None:
                review_result = build_and_export_promotion_review(
                    kind=ModelKind.ML_SCORER,
                    models_root=project_root / "models",
                    ml_base=project_root / "models" / "ml_signal_scorer",
                    workflow_snapshot_path=snapshot_path,
                    out_dir=reports_dir,
                    basename=f"promotion_review_ml_{stamp}",
                    settings=engine.settings,
                )
                review_kind = "ml"
            if review_result is None:
                st.warning(
                    "No promoted ML/RL artifact is available yet, so Fortuna could not build a promotion review."
                )
            else:
                review, md_path, json_path = review_result
                workflow["workflow_snapshot"] = snapshot
                workflow["promotion_review_kind"] = review_kind
                workflow["promotion_review_markdown_path"] = str(md_path)
                workflow["promotion_review_json_path"] = str(json_path)
                workflow["promotion_review_run_id"] = str(review.record.run_id or "")
                st.session_state.assistant_workflow = workflow

    workflow = st.session_state.assistant_workflow
    if not workflow:
        st.info("Click **Refresh workflow** to load the current universe, setups, and candidates.")
        return

    artifact_rows = build_workflow_artifact_rows(workflow)
    if artifact_rows:
        st.caption("Workflow artifacts")
        st.dataframe(
            [row.to_dict() for row in artifact_rows],
            use_container_width=True,
            hide_index=True,
        )
        artifact_follow_up = build_workflow_artifact_follow_up(workflow)
        if artifact_follow_up is not None:
            target_note = (
                f" target=`{artifact_follow_up.target}`" if artifact_follow_up.target else ""
            )
            st.caption(
                f"Suggested follow-up: **{artifact_follow_up.action_label}**"
                f"{target_note} - {artifact_follow_up.rationale}"
            )

    team = workflow.get("team")
    universe = workflow["universe"]
    shortlist = workflow["shortlist"]
    briefing = workflow["briefing"]
    candidates = workflow["candidates"]
    research = workflow.get("research")
    allocation = workflow["allocation"]
    team_headline = str(workflow.get("team_headline") or "").strip()
    team_roles = workflow.get("team_roles") or ()
    team_nightly_alignment = workflow.get("team_nightly_alignment") or {}

    if team_headline:
        st.caption(team_headline)
    team_recovery_posture = artifact_follow_up_team_posture(artifact_follow_up)
    if team_recovery_posture:
        st.caption(f"Team recovery posture: {team_recovery_posture}")
    if team_roles:
        preferred_roles = {
            "universe_scout",
            "briefing_agent",
            "portfolio_allocator",
            "research_planner",
        }
        role_line = " | ".join(
            f"{str(row.get('name', 'agent'))}: {str(row.get('headline', '')).strip()}"
            for row in team_roles
            if isinstance(row, dict)
            and str(row.get("name", "")).strip() in preferred_roles
        )
        if role_line:
            st.caption(role_line)
    if team is not None:
        team_metrics = getattr(team, "metrics", {}) or {}
        research_policy = str(team_metrics.get("research_selection_policy") or "").strip()
        research_target = str(team_metrics.get("research_refresh_target") or "").strip()
        research_ml_count = int(team_metrics.get("research_ml_count", 0) or 0)
        research_rl_count = int(team_metrics.get("research_rl_count", 0) or 0)
        research_refreshed_count = int(team_metrics.get("research_refreshed_count", 0) or 0)
        research_refresh_requested = bool(team_metrics.get("research_refresh_requested", False))
        research_headline = str(team_metrics.get("research_headline") or "").strip()
        discovery_alignment_summary = str(
            team_metrics.get("discovery_alignment_summary") or ""
        ).strip()
        discovery_overlap_symbols = str(team_metrics.get("discovery_overlap_symbols") or "").strip()
        research_alignment_summary = str(
            team_metrics.get("research_alignment_summary") or ""
        ).strip()
        research_alignment_mix = str(team_metrics.get("research_alignment_target_mix") or "").strip()
        research_alignment_target = _target_from_alignment_mix(research_alignment_mix)
        if research_headline:
            research_line = f"Research planner: {research_headline}"
            details: list[str] = []
            if research_policy:
                details.append(f"policy `{research_policy}`")
            if research_target:
                details.append(f"target `{research_target}`")
            if research_ml_count or research_rl_count:
                details.append(f"ML `{research_ml_count}` / RL `{research_rl_count}`")
            if research_refresh_requested:
                details.append(f"refreshed `{research_refreshed_count}`")
            if research_alignment_summary:
                details.append(research_alignment_summary)
            if research_alignment_mix:
                details.append(f"mix `{research_alignment_mix}`")
            if research_alignment_target:
                details.append(f"suggested target `{research_alignment_target}`")
            if details:
                research_line += " | " + " | ".join(details)
            st.caption(research_line)
        if discovery_alignment_summary:
            discovery_line = f"Discovery posture: {discovery_alignment_summary}"
            if discovery_overlap_symbols:
                discovery_line += f" | overlap `{discovery_overlap_symbols}`"
            st.caption(discovery_line)
    if isinstance(team_nightly_alignment, dict):
        latest_execution_text = format_latest_nightly_execution_posture(
            team_nightly_alignment
        )
        if latest_execution_text:
            st.caption(latest_execution_text)
        latest_discovery_text = format_latest_nightly_discovery_context(
            team_nightly_alignment
        )
        if latest_discovery_text:
            st.caption(latest_discovery_text)
        team_recommended_action = str(
            team_nightly_alignment.get("recommended_action") or ""
        ).strip()
        if team_recommended_action:
            st.info(f"Team posture: {team_recommended_action}")
        team_discovery_action = str(
            team_nightly_alignment.get("recommended_discovery_action") or ""
        ).strip()
        if team_discovery_action:
            st.info(f"Team discovery follow-up: {team_discovery_action}")
        trend_text = _format_recent_nightly_alignment_trend(team_nightly_alignment)
        if trend_text:
            st.caption(trend_text)
        latest_mix = str(
            team_nightly_alignment.get("latest_target_mix") or ""
        ).strip()
        if latest_mix:
            st.caption(f"Latest team target mix: `{latest_mix}`")
        latest_refreshed_mix = str(
            team_nightly_alignment.get("latest_refreshed_target_mix") or ""
        ).strip()
        if latest_refreshed_mix:
            st.caption(f"Latest refreshed team mix: `{latest_refreshed_mix}`")
        discovery_command = str(
            team_nightly_alignment.get("recommended_discovery_cli_command") or ""
        ).strip()
        if discovery_command:
            st.caption(f"Suggested discovery command: `{discovery_command}`")
        if research_alignment_target and suggested_refresh_target != research_alignment_target:
            st.caption(
                "Current basket differs from nightly posture: "
                f"nightly `{suggested_refresh_target}` vs current `{research_alignment_target}`"
            )

    metric_cols = st.columns(8)
    metric_cols[0].metric("Universe", universe.metrics.get("count", 0))
    metric_cols[1].metric("Top liquidity", universe.metrics.get("top_liquidity_score", 0.0))
    metric_cols[2].metric("Shortlist", shortlist.metrics.get("count", 0))
    metric_cols[3].metric("Setups", briefing.metrics.get("candidates", 0))
    metric_cols[4].metric("Allocated", allocation.metrics.get("selected_count", 0))
    metric_cols[5].metric("ML picks", candidates.metrics.get("ml_count", 0))
    metric_cols[6].metric("RL picks", candidates.metrics.get("rl_count", 0))
    metric_cols[7].metric("Research rows", 0 if research is None else research.metrics.get("count", 0))

    tab_universe, tab_shortlist, tab_brief, tab_allocate, tab_candidates, tab_research = st.tabs(
        ["Universe", "Shortlist", "Top setups", "Allocation", "Training candidates", "Training research"]
    )

    with tab_universe:
        if universe.error:
            st.error(universe.error)
        else:
            st.caption(f"Source: {universe.metrics.get('source', 'auto')}")
            st.dataframe(list(universe.rows), use_container_width=True, hide_index=True)

    with tab_shortlist:
        if shortlist.error:
            st.error(shortlist.error)
        else:
            st.caption(f"Source: {shortlist.metrics.get('source', 'auto')}")
            st.dataframe(list(shortlist.rows), use_container_width=True, hide_index=True)

    with tab_brief:
        if briefing.error:
            st.error(briefing.error)
        else:
            headline = briefing.metrics.get("headline")
            if headline:
                st.caption(str(headline))
            st.dataframe(list(briefing.rows), use_container_width=True, hide_index=True)
            for note in briefing.notes:
                st.warning(note)

    with tab_allocate:
        if allocation.error:
            st.error(allocation.error)
        else:
            headline = allocation.metrics.get("headline")
            if headline:
                st.caption(str(headline))
            st.dataframe(list(allocation.rows), use_container_width=True, hide_index=True)
            for note in allocation.notes:
                st.info(note)

    with tab_candidates:
        if candidates.error:
            st.error(candidates.error)
        else:
            st.caption(f"Source: {candidates.metrics.get('source', 'auto')}")
            st.dataframe(list(candidates.rows), use_container_width=True, hide_index=True)

    with tab_research:
        if research is None:
            st.info("Click **Refresh training research** to load the current ML/RL prep plan.")
        elif research.error:
            st.error(research.error)
        else:
            st.caption(
                "Source: "
                f"{research.metrics.get('source', 'auto')} | "
                f"Policy: {research.metrics.get('selection_policy', 'diversified')} | "
                f"Refresh target: {research.metrics.get('refresh_target', 'all')} | "
                f"Refreshed: {research.metrics.get('refreshed_count', 0)}"
            )
            if research.metrics.get("refresh_requested"):
                st.success(
                    f"Targeted research refresh requested for "
                    f"{research.metrics.get('refresh_target', 'all')} "
                    f"and refreshed {research.metrics.get('refreshed_count', 0)} symbol(s)."
                )
            st.dataframe(list(research.rows), use_container_width=True, hide_index=True)


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

    readiness = status.get("runtime_readiness") or {}
    if readiness:
        posture = readiness.get("overall_posture", "unknown")
        nightly = readiness.get("nightly_posture", "unknown")
        summary = readiness.get("summary", "")
        if posture == "blocked":
            st.error(f"Runtime posture: **{posture}** | Nightly: **{nightly}** — {summary}")
        elif posture == "advisory_ready":
            st.success(f"Runtime posture: **{posture}** | Nightly: **{nightly}** — {summary}")
        elif posture == "advisory_partial":
            st.warning(f"Runtime posture: **{posture}** | Nightly: **{nightly}** — {summary}")
        else:
            st.info(f"Runtime posture: **{posture}** | Nightly: **{nightly}** — {summary}")
        blockers = readiness.get("blockers") or []
        if blockers:
            st.caption(
                "Blockers: "
                + "; ".join(f"{row.get('name')}: {row.get('detail')}" for row in blockers[:3])
            )
    scaling = status.get("scaling") or {}
    if scaling:
        st.caption(
            "Scaling: "
            f"**{scaling.get('basket_posture', 'unknown')}** "
            f"(recommended max {scaling.get('recommended_max_symbols', '—')} symbols) — "
            f"{scaling.get('summary', '')}"
        )

    _render_registry_panel(status)
    st.divider()
    activation = status.get("activation") or {}
    _render_rl_panel(
        engine,
        live_signals,
        rl_status=status.get("rl") or {},
        activation=activation.get("rl") or {},
    )
    st.divider()
    _render_ml_panel(
        engine,
        ml_status=status.get("ml") or {},
        activation=activation.get("ml") or {},
    )
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


def _render_activation_caption(activation: dict) -> None:
    if not activation:
        return
    stage = activation.get("stage", "unknown")
    summary = activation.get("summary", "")
    cmd = activation.get("recommended_command")
    blockers = activation.get("blockers") or []
    top_blocker = blockers[0].get("detail") if blockers else None
    detail = cmd or top_blocker or summary
    st.caption(f"Activation: **{stage}** — {detail}")


def _render_ml_panel(engine, *, ml_status: dict, activation: dict | None = None) -> None:
    cols = st.columns([3, 1])
    cols[0].subheader("ML signal scorer")
    if cols[1].button("Reload ML scorer", key="reload_ml_scorer"):
        ok = engine.reload_ml_scorer()
        if ok:
            st.success("ML scorer reloaded.")
            st.rerun()
        else:
            st.warning("No ML scorer available — agentic runs without ML votes.")

    _render_activation_caption(activation or {})

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
    st.divider()
    _render_nightly_alignment_summary(agentic_status.get("nightly_alignment") or {})


def _render_rl_panel(
    engine,
    live_signals,
    *,
    rl_status: dict | None = None,
    activation: dict | None = None,
) -> None:
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

    _render_activation_caption(activation or {})

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


def _render_nightly_alignment_summary(summary: dict) -> None:
    st.subheader("Recent nightly alignment")
    report_count = int(summary.get("report_count", 0) or 0)
    if report_count == 0:
        st.caption("No recent nightly alignment evidence found yet.")
        return
    enabled_reports = int(summary.get("enabled_reports", 0) or 0)
    aligned_reports = int(summary.get("aligned_reports", 0) or 0)
    ratio = (
        float(aligned_reports) / float(enabled_reports)
        if enabled_reports > 0
        else 0.0
    )
    latest_mix = str(summary.get("latest_target_mix") or "").strip()
    latest_refreshed_mix = str(summary.get("latest_refreshed_target_mix") or "").strip()
    latest_workflow_alignment = str(
        summary.get("latest_workflow_research_alignment_summary") or ""
    ).strip()
    latest_workflow_mix = str(
        summary.get("latest_workflow_research_alignment_target_mix") or ""
    ).strip()
    latest_workflow_target = str(
        summary.get("latest_workflow_research_recommended_target") or ""
    ).strip()
    latest_workflow_discovery = str(
        summary.get("latest_workflow_discovery_alignment_summary") or ""
    ).strip()
    latest_workflow_discovery_overlap = str(
        summary.get("latest_workflow_discovery_overlap_symbols") or ""
    ).strip()
    latest_workflow_discovery_warning = bool(
        summary.get("latest_workflow_discovery_warning", False)
    )
    latest_execution_text = format_latest_nightly_execution_posture(summary)
    latest_discovery_text = format_latest_nightly_discovery_context(summary)
    effective_target = str(summary.get("effective_refresh_target") or "").strip()
    nightly_target = str(summary.get("recommended_refresh_target") or "").strip()
    workflow_target_mismatch = bool(summary.get("workflow_target_mismatch", False))
    cols = st.columns(4)
    cols[0].metric("Reports", report_count)
    cols[1].metric("Enabled", enabled_reports)
    cols[2].metric("Aligned", aligned_reports)
    cols[3].metric("Ratio", f"{ratio:.2f}")
    if latest_mix:
        st.caption(f"Latest target mix: `{latest_mix}`")
    if latest_refreshed_mix:
        st.caption(f"Latest refreshed target mix: `{latest_refreshed_mix}`")
    if latest_workflow_alignment:
        text = f"Latest workflow basket/research: `{latest_workflow_alignment}`"
        if latest_workflow_mix:
            text += f" | mix `{latest_workflow_mix}`"
        st.caption(text)
    if latest_workflow_target:
        st.caption(f"Latest workflow target hint: `{latest_workflow_target}`")
    if latest_workflow_discovery:
        discovery_text = latest_workflow_discovery
        if latest_workflow_discovery_overlap:
            discovery_text += f" | overlap={latest_workflow_discovery_overlap}"
        st.caption(f"Latest workflow discovery: `{discovery_text}`")
    if latest_workflow_discovery_warning:
        st.caption("Latest workflow discovery needs follow-up.")
    if latest_execution_text:
        st.caption(latest_execution_text)
    if latest_discovery_text:
        st.caption(latest_discovery_text)
    if enabled_reports > 0 and aligned_reports < enabled_reports:
        st.warning(
            "Recent nightly basket/research alignment is drifting. "
            f"Only {aligned_reports}/{enabled_reports} enabled reports carried target-mix evidence."
        )
    recommended_action = str(summary.get("recommended_action") or "").strip()
    if recommended_action:
        st.info(f"Suggested next step: {recommended_action}")
    if effective_target:
        st.caption(f"Suggested refresh target: `{effective_target}`")
    if nightly_target:
        st.caption(f"Nightly target hint: `{nightly_target}`")
    recommended_command = str(summary.get("recommended_cli_command") or "").strip()
    if recommended_command:
        st.caption(f"Suggested command: `{recommended_command}`")
    recommended_discovery_action = str(summary.get("recommended_discovery_action") or "").strip()
    if recommended_discovery_action:
        st.info(f"Suggested discovery follow-up: {recommended_discovery_action}")
    recommended_discovery_command = str(
        summary.get("recommended_discovery_cli_command") or ""
    ).strip()
    if recommended_discovery_command:
        st.caption(f"Suggested discovery command: `{recommended_discovery_command}`")
    if workflow_target_mismatch:
        st.caption(
            "Current workflow differs from nightly posture: "
            f"nightly `{nightly_target}` vs current `{latest_workflow_target}`"
        )
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
    workflow_snapshot = str(promotion.get("workflow_snapshot_path") or "").strip()
    if workflow_snapshot:
        st.caption(f"Workflow snapshot: `{workflow_snapshot}`")
    snapshot = promotion.get("workflow_snapshot") or {}
    if snapshot:
        source = snapshot.get("source") or "—"
        timeframe = snapshot.get("timeframe") or "—"
        days = snapshot.get("lookback_days")
        lookback = f"{days}d" if days is not None else "—"
        st.caption(
            "Workflow summary: "
            f"source=`{source}` timeframe=`{timeframe}` lookback=`{lookback}` "
            f"universe={int(snapshot.get('universe_count', 0) or 0)} "
            f"shortlist={int(snapshot.get('shortlist_count', 0) or 0)} "
            f"briefing={int(snapshot.get('briefing_candidates', 0) or 0)} "
            f"ml={int(snapshot.get('ml_count', 0) or 0)} "
            f"rl={int(snapshot.get('rl_count', 0) or 0)}"
        )


if __name__ == "__main__":
    main()
