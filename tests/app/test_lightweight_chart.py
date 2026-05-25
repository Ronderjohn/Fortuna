"""Lightweight Charts spec builder — bar-sequence indexing + signal markers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fortuna.app.live_signals import LiveSignal
from fortuna.app.lightweight_chart import (
    build_lightweight_charts_spec,
    render_lightweight_charts_html,
)


def _ohlcv_two_sessions() -> pd.DataFrame:
    """Two NSE sessions with a weekend gap (Fri close → Mon open)."""
    fri = pd.date_range("2026-05-22 09:15", periods=10, freq="5min")
    mon = pd.date_range("2026-05-25 09:15", periods=10, freq="5min")
    idx = fri.append(mon)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.2, len(idx)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, len(idx)),
        },
        index=idx,
    )


def test_time_axis_uses_real_unix_seconds() -> None:
    """Bars must be monotonic unix seconds, with Fri→Mon gap of ~2.5 days."""
    ohlcv = _ohlcv_two_sessions()
    spec = build_lightweight_charts_spec(ohlcv=ohlcv)
    candles = spec[0]["series"][0]["data"]
    times = [c["time"] for c in candles]
    assert times == sorted(times)
    assert all(t > 1_700_000_000 for t in times)  # > 2023-11-14 — sanity
    fri_last = times[9]
    mon_first = times[10]
    gap_hours = (mon_first - fri_last) / 3600
    assert 60 < gap_hours < 80


def test_time_axis_renders_in_ist_not_utc() -> None:
    """The displayed clock-time on the axis must match IST.

    Lightweight Charts always renders unix seconds in UTC, so we shift IST
    wall-clock seconds to UTC equivalents. A 09:15 IST bar must therefore
    map to a unix timestamp whose UTC ``%H:%M`` is 09:15.
    """
    ohlcv = _ohlcv_two_sessions()
    spec = build_lightweight_charts_spec(ohlcv=ohlcv)
    first_ts = spec[0]["series"][0]["data"][0]["time"]
    as_utc = pd.Timestamp(first_ts, unit="s", tz="UTC")
    assert as_utc.strftime("%H:%M") == "09:15"
    assert as_utc.strftime("%Y-%m-%d") == "2026-05-22"


def test_buy_sell_exit_markers_are_emitted_for_trades() -> None:
    ohlcv = _ohlcv_two_sessions()
    trades_df = pd.DataFrame(
        [
            {
                "entry_time": ohlcv.index[2],
                "exit_time": ohlcv.index[6],
                "side": "LONG",
                "entry_price": 100.0,
                "exit_price": 101.5,
                "pnl": 150.0,
            },
            {
                "entry_time": ohlcv.index[12],
                "exit_time": ohlcv.index[16],
                "side": "SHORT",
                "entry_price": 102.0,
                "exit_price": 100.0,
                "pnl": 200.0,
            },
        ]
    )
    spec = build_lightweight_charts_spec(ohlcv=ohlcv, trades_df=trades_df)
    markers = spec[0]["series"][0]["markers"]
    texts = [m["text"] for m in markers]

    assert any(t.startswith("BUY") for t in texts)
    assert any(t.startswith("SELL") for t in texts)
    assert sum(1 for t in texts if t.startswith("EXIT")) == 2

    buy_marker = next(m for m in markers if m["text"].startswith("BUY"))
    assert buy_marker["shape"] == "arrowUp"
    assert buy_marker["position"] == "belowBar"

    sell_marker = next(m for m in markers if m["text"].startswith("SELL"))
    assert sell_marker["shape"] == "arrowDown"
    assert sell_marker["position"] == "aboveBar"


def test_live_signal_appends_pulsing_marker_on_last_bar() -> None:
    ohlcv = _ohlcv_two_sessions()
    sig = LiveSignal(
        strategy_name="test",
        action="BUY",
        label="BUY",
        bar_time=ohlcv.index[-1],
        bar_close=float(ohlcv["close"].iloc[-1]),
        enter=True,
        exit=False,
        in_position=False,
        side="LONG",
        entry_price=None,
        bars_in_trade=0,
    )
    spec = build_lightweight_charts_spec(ohlcv=ohlcv, live_signal=sig)
    markers = spec[0]["series"][0]["markers"]
    live = [m for m in markers if "LIVE" in m["text"]]
    assert len(live) == 1
    assert live[0]["shape"] == "arrowUp"


def test_oscillator_columns_go_to_separate_pane() -> None:
    ohlcv = _ohlcv_two_sessions()
    enriched = ohlcv.copy()
    enriched["ema_9"] = ohlcv["close"] * 0.99
    enriched["rsi"] = 30 + (np.arange(len(ohlcv)) % 40)
    spec = build_lightweight_charts_spec(
        ohlcv=ohlcv,
        enriched=enriched,
        overlay_columns=["ema_9", "rsi"],
    )
    series = spec[0]["series"]
    pane_by_title = {s["title"]: s["options"]["pane"] for s in series}
    assert pane_by_title["Price"] == 0
    assert pane_by_title["ema_9"] == 0
    assert pane_by_title["rsi"] >= 1


def test_volume_or_huge_series_never_squashes_price_pane() -> None:
    """Regression: a volume-magnitude series must NOT land on the price pane.

    Symptom (before fix): y-axis ranged into the millions, candles squashed
    flat at the bottom.
    """
    ohlcv = _ohlcv_two_sessions()
    enriched = ohlcv.copy()
    enriched["volume"] = np.random.default_rng(0).integers(1_000_000, 3_000_000, len(ohlcv))
    enriched["ema_9"] = ohlcv["close"] * 0.99

    spec = build_lightweight_charts_spec(
        ohlcv=ohlcv,
        enriched=enriched,
        overlay_columns=["volume", "ema_9"],
    )
    series = spec[0]["series"]
    titles = [s["title"] for s in series]
    panes = {s["title"]: s["options"]["pane"] for s in series}

    assert "volume" not in titles  # excluded from overlays altogether
    assert panes["ema_9"] == 0  # EMA still on price pane


def test_columns_with_extreme_magnitude_are_dropped_from_price_pane() -> None:
    """A custom indicator at 1e6 scale must not appear on the price pane."""
    ohlcv = _ohlcv_two_sessions()
    enriched = ohlcv.copy()
    enriched["foo_huge"] = ohlcv["close"] * 5000.0  # ~5000x price
    enriched["bar_normal"] = ohlcv["close"] * 1.01  # near price

    spec = build_lightweight_charts_spec(
        ohlcv=ohlcv,
        enriched=enriched,
        overlay_columns=["foo_huge", "bar_normal"],
    )
    series = spec[0]["series"]
    price_pane_titles = [s["title"] for s in series if s["options"]["pane"] == 0]
    assert "foo_huge" not in price_pane_titles
    assert "bar_normal" in price_pane_titles


def test_initial_view_is_anchored_on_recent_bars() -> None:
    """Long history must be trimmed so the chart focuses on current data.

    Symptom before this fix: Lightweight Charts received ~1437 bars and
    rendered them all compressed into the viewport, hiding recent action
    behind ancient history.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2026-01-01 09:15", periods=2000, freq="5min")
    close = 1000 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=idx,
    )
    spec = build_lightweight_charts_spec(ohlcv=ohlcv, visible_bars=75)
    candles = spec[0]["series"][0]["data"]
    assert len(candles) <= 75 * 12 + 1  # history window honored
    assert len(candles) >= 75  # but visible_bars is comfortably renderable
    # The last bar in the chart must be the last bar of the input.
    last_t = candles[-1]["time"]
    expected_last = pd.Timestamp(idx[-1]).tz_localize("UTC").timestamp()
    assert int(last_t) == int(expected_last)


def test_chart_options_anchor_on_right_edge() -> None:
    """timeScale options must keep the right edge in view (not fit-content)."""
    ohlcv = _ohlcv_two_sessions()
    spec = build_lightweight_charts_spec(ohlcv=ohlcv, visible_bars=75)
    ts_opts = spec[0]["chart"]["timeScale"]
    assert ts_opts["barSpacing"] >= 10
    assert ts_opts["minBarSpacing"] >= 4
    assert ts_opts["rightOffset"] >= 4
    assert ts_opts.get("shiftVisibleRangeOnNewBar") is True


def test_initial_view_focuses_on_todays_session_for_intraday() -> None:
    """For 5m data the default barSpacing must show roughly today's bars only.

    Today=21 bars at 5m → target ~25 bars in view → barSpacing ~ 1200/25 ~ 48.
    The chart can't be allowed to default to a "fit all 1500 bars" layout.
    """
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2026-05-04", "2026-05-25")
    rows = []
    for d in days:
        bars = 75 if d.date() != pd.Timestamp("2026-05-25").date() else 21
        for i in range(bars):
            rows.append(d + pd.Timedelta(minutes=5 * i + 9 * 60 + 15))
    idx = pd.DatetimeIndex(rows)
    close = 1000 + np.cumsum(rng.normal(0, 0.4, len(idx)))
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 0.4, "low": close - 0.4, "close": close},
        index=idx,
    )
    spec = build_lightweight_charts_spec(
        ohlcv=ohlcv, visible_bars=75, estimated_width_px=1200
    )
    ts_opts = spec[0]["chart"]["timeScale"]
    bar_spacing = ts_opts["barSpacing"]
    right_offset = ts_opts["rightOffset"]
    visible = 1200 // bar_spacing - right_offset
    # Today has 21 bars — the visible count should be in the same ballpark
    # (today + a small history slice), NOT hundreds.
    assert 15 <= visible <= 60


def test_spec_carries_initial_focus_metadata_for_js_consumer() -> None:
    """The HTML wrapper reads ``_initialFocus`` to pin setVisibleLogicalRange."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2026-05-04", "2026-05-25")
    rows = []
    for d in days:
        bars = 75 if d.date() != pd.Timestamp("2026-05-25").date() else 21
        for i in range(bars):
            rows.append(d + pd.Timedelta(minutes=5 * i + 9 * 60 + 15))
    idx = pd.DatetimeIndex(rows)
    close = 1000 + np.cumsum(rng.normal(0, 0.4, len(idx)))
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 0.4, "low": close - 0.4, "close": close},
        index=idx,
    )
    spec = build_lightweight_charts_spec(ohlcv=ohlcv, visible_bars=75)
    focus = spec[0]["chart"]["_initialFocus"]
    assert focus["bars"] == 21  # today's bars only
    assert focus["rightPadding"] >= 6


def test_html_renderer_calls_set_visible_logical_range() -> None:
    """Regression: the HTML embed must override the lib's default fit-content."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-05-22 09:15", periods=300, freq="5min")
    close = 1000 + np.cumsum(rng.normal(0, 0.3, len(idx)))
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 0.4, "low": close - 0.4, "close": close},
        index=idx,
    )
    spec = build_lightweight_charts_spec(ohlcv=ohlcv, visible_bars=75)
    html = render_lightweight_charts_html(spec, height=580)
    assert "setVisibleLogicalRange" in html
    assert "_initialFocus" in html
    assert "lightweight-charts@5" in html
    assert "createSeriesMarkers" in html
    assert "<div id=\"fortuna-chart\">" in html


def test_right_padding_anchors_at_nse_session_close_for_intraday() -> None:
    """When the last loaded bar is mid-session (live feed lagging), the chart's
    right edge must still extend to 15:30 IST so the full trading day is on
    screen — otherwise the chart visually "stops" before market close.
    """
    idx = pd.date_range("2026-05-25 09:15", periods=60, freq="5min")
    assert idx[-1].strftime("%H:%M") == "14:10"  # well before 15:30
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.2, len(idx)))
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 0.4, "low": close - 0.4, "close": close},
        index=idx,
    )
    spec = build_lightweight_charts_spec(ohlcv=ohlcv, visible_bars=75)
    focus = spec[0]["chart"]["_initialFocus"]
    # 14:10 → 15:30 = 80 minutes = 16 bars of 5m. With the +2 cushion the
    # right padding must cover at least 16 slots.
    assert focus["rightPadding"] >= 16
    # rightOffset in timeScale options mirrors _initialFocus.rightPadding.
    assert spec[0]["chart"]["timeScale"]["rightOffset"] == focus["rightPadding"]


def test_right_padding_uses_fallback_for_daily_timeframe() -> None:
    """Daily / weekly / monthly bars must NOT trigger the 15:30 IST anchor —
    they have no intra-session granularity so the fallback padding applies."""
    idx = pd.bdate_range("2024-01-02", "2026-05-22")
    rng = np.random.default_rng(0)
    close = 1000 + np.cumsum(rng.normal(0, 1.0, len(idx)))
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close},
        index=idx,
    )
    spec = build_lightweight_charts_spec(ohlcv=ohlcv, visible_bars=75)
    # Fallback bound = max(6, min(20, focus_bars // 5 + 4)). focus_bars = 30
    # for daily, so fallback = min(20, 30 // 5 + 4) = min(20, 10) = 10.
    assert spec[0]["chart"]["_initialFocus"]["rightPadding"] <= 20


def test_initial_view_for_daily_timeframe_targets_about_thirty_bars() -> None:
    """1-day bars: focus on the last ~30 candles, not multi-year history."""
    idx = pd.bdate_range("2024-01-02", "2026-05-22")
    rng = np.random.default_rng(0)
    close = 1000 + np.cumsum(rng.normal(0, 1.0, len(idx)))
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close},
        index=idx,
    )
    spec = build_lightweight_charts_spec(
        ohlcv=ohlcv, visible_bars=75, estimated_width_px=1200
    )
    ts_opts = spec[0]["chart"]["timeScale"]
    visible = 1200 // ts_opts["barSpacing"] - ts_opts["rightOffset"]
    assert 20 <= visible <= 50
