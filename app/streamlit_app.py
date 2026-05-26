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

        tail = df.tail(60)
        cutoff = int(since)
        bars: list[dict] = []
        for ts, row in tail.iterrows():
            t = ist_unix_seconds(pd.Timestamp(ts))
            if t < cutoff:
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

    tab_chart, tab_board, tab_report, tab_detail, tab_rl = st.tabs(
        ["📈 Chart", "🏆 Leaderboard", "📊 Performance", "🔬 Strategy detail", "🤖 RL Policy"]
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
        from fortuna.app.lightweight_chart import (
            build_lightweight_charts_spec,
            render_lightweight_charts_html,
        )
        from fortuna.reporting.strategy_tester.chart_viewport import (
            visible_bars_for_timeframe,
        )
        import streamlit.components.v1 as components

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
        _render_rl_panel(engine, live_signals)


def _render_rl_panel(engine, live_signals) -> None:
    """Render the RL policy status + metadata card.

    Shows a friendly empty-state when no checkpoint exists, otherwise
    renders the OOS metrics + FilterVerdict + raw metadata.json.
    """
    rl_gen = getattr(engine, "rl_generator", None)
    available = bool(rl_gen and getattr(rl_gen, "is_available", False))

    cols = st.columns([3, 1])
    cols[0].subheader("RL policy status")
    if cols[1].button("Reload policy"):
        ok = engine.reload_rl_generator()
        if ok:
            st.success("RL policy reloaded.")
            st.rerun()
        else:
            st.warning("No RL policy available — deterministic fallback is active.")

    if not available:
        st.info(
            "No validated RL checkpoint is loaded. Train one via "
            "`uv run python scripts/run_rl_train.py` and promote it via "
            "`uv run python scripts/promote_policy.py --run-id <id>`. "
            "The dashboard continues to render deterministic signals."
        )
        return

    meta = getattr(rl_gen, "metadata", None)
    if meta is None:
        st.warning("Policy loaded but metadata.json missing.")
        return

    badge = "PASS" if meta.verdict_passed else "FAIL"
    st.markdown(f"**Run ID:** `{meta.run_id}` &nbsp;|&nbsp; **Verdict:** `{badge}` &nbsp;|&nbsp; "
                f"**Policy:** `{meta.policy_type}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OOS Sharpe", f"{meta.oos_metrics.sharpe_ratio:.2f}")
    m2.metric("OOS PF", f"{meta.oos_metrics.profit_factor:.2f}")
    m3.metric("OOS Trades", meta.oos_metrics.total_trades)
    m4.metric("OOS MaxDD %", f"{meta.oos_metrics.max_drawdown_pct * 100:.2f}")

    rl_keys = [k for k in live_signals if k.startswith("RL:")]
    if rl_keys:
        latest = live_signals[rl_keys[0]]
        st.markdown(
            f"**Latest RL signal:** `{latest.action}` at "
            f"`{latest.bar_time}` close={latest.bar_close:.2f}"
        )

    with st.expander("metadata.json"):
        st.json(meta.to_dict())


if __name__ == "__main__":
    main()
