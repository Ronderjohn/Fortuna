"""TradingView Lightweight Charts builder for the Streamlit dashboard.

Two public entry points:

- :func:`build_lightweight_charts_spec` – produces the JSON config (chart
  options + series + markers) for a strategy chart.
- :func:`render_lightweight_charts_html` – wraps that spec in a self-contained
  HTML/JS page that loads ``lightweight-charts`` v5 from CDN and explicitly
  calls ``timeScale().setVisibleLogicalRange(...)`` after the data is loaded.

Why a custom HTML wrapper instead of a published Streamlit component?
None of the off-the-shelf Streamlit wrappers around Lightweight Charts call
``setVisibleLogicalRange`` after ``setData``, so the chart auto-fits *all*
historical bars on first paint and the user always sees the entire dataset
compressed. The JSON config has no static option to disable that behaviour,
so the only correct fix is to embed the chart ourselves and run the focus
call in JS — which also lets us drive in-place streaming updates via a
custom polling loop without ever reloading the iframe.

Design notes:

- Bars carry their **IST wall-clock as UTC** unix seconds so axis labels
  render in IST (the lib has no timezone option in v4/v5).
- Native ▲ BUY / ▼ SELL / ◆ EXIT markers via ``createSeriesMarkers``.
- Multi-pane layout (price + oscillators) via the v5 ``addSeries`` paneIndex.
- Defensive overlay routing: only series whose magnitude is comparable to
  price land on the price pane; oscillators get their own pane and large-
  scale series (volume etc.) are dropped entirely.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from fortuna.app.live_signals import LiveSignal
from fortuna.reporting.strategy_tester.chart_viewport import normalize_to_naive_ist

_TV_BULL = "#26A69A"
_TV_BEAR = "#EF5350"
_TV_EXIT = "#9C27B0"
_TV_LIVE_EXIT = "#FF9800"

_OVERLAY_COLORS: dict[str, str] = {
    "ema_fast": "#26A69A",
    "ema_slow": "#FF9800",
    "ema_9": "#26A69A",
    "ema_21": "#FF9800",
    "ema_50": "#7B1FA2",
    "ema_200": "#1976D2",
    "vwap": "#7B1FA2",
    "bb_basis": "#787B86",
    "bb_upper": "#2962FF",
    "bb_lower": "#2962FF",
    "or_high": "#2962FF",
    "or_low": "#F23645",
    "supertrend": "#43A047",
    "rsi": "#7E57C2",
    "macd": "#2962FF",
    "macd_signal": "#FF9800",
    "macd_hist": "#787B86",
    "atr": "#FF9800",
}


_EXCLUDE_COLUMNS = frozenset(
    {"open", "high", "low", "close", "volume", "symbol", "long_entry", "short_entry"}
)


def _overlay_color(col: str) -> str:
    if col in _OVERLAY_COLORS:
        return _OVERLAY_COLORS[col]
    key = col.lower()
    for hint, color in _OVERLAY_COLORS.items():
        if hint in key:
            return color
    return "#787B86"


def ist_unix_seconds(ts: pd.Timestamp) -> int:
    """Public alias for :func:`_to_seconds` — exported so other modules
    (e.g. the bar-stream HTTP server) can emit chart-compatible timestamps."""
    return _to_seconds(ts)


def _to_seconds(ts: pd.Timestamp) -> int:
    """IST wall-clock timestamp → unix seconds *as if* the clock were UTC.

    Lightweight Charts always renders unix epoch seconds in UTC. To make the
    axis labels show IST (e.g. "09:15", "15:30") we feed the chart timestamps
    that, when interpreted as UTC, equal the IST wall-clock value. This is a
    standard trick for forcing a fixed timezone on a chart that doesn't expose
    a timezone option.
    """
    naive = pd.Timestamp(ts)
    if naive.tzinfo is not None:
        naive = naive.tz_convert("Asia/Kolkata").tz_localize(None)
    return int(naive.tz_localize("UTC").timestamp())


def _build_time_lookup(ohlcv: pd.DataFrame) -> dict[pd.Timestamp, int]:
    """Map each bar timestamp to its (UTC-interpreted IST) unix-seconds value."""
    return {pd.Timestamp(ts): _to_seconds(pd.Timestamp(ts)) for ts in ohlcv.index}


def _candle_rows(
    ohlcv: pd.DataFrame, ts_to_unix: dict[pd.Timestamp, int]
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for ts, row in ohlcv.iterrows():
        t = ts_to_unix.get(pd.Timestamp(ts))
        if t is None:
            continue
        rows.append(
            {
                "time": t,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    return rows


def _line_rows(
    series: pd.Series, ts_to_unix: dict[pd.Timestamp, int]
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for ts, val in series.dropna().items():
        t = ts_to_unix.get(pd.Timestamp(ts))
        if t is None:
            continue
        out.append({"time": t, "value": float(val)})
    return out


def _classify_overlays(
    overlay_columns: list[str],
    enriched: pd.DataFrame,
    close: pd.Series,
) -> tuple[list[str], list[str], list[str]]:
    """Route each overlay column to one of three groups.

    Returns ``(price_cols, oscillator_cols, dropped)``. Routing rules:

    1. OHLCV/internal columns ("volume", "long_entry", …) → dropped.
    2. Series whose median sits within 0.25× to 4× of the price median → price.
    3. Series in the 0–150 envelope (RSI, stoch, MFI etc.) → oscillator.
    4. Anything else (very large or very negative magnitudes) → dropped, so it
       can't squash the candles by living on the price scale.
    """
    price_cols: list[str] = []
    osc_cols: list[str] = []
    dropped: list[str] = []
    c = close.dropna()
    med_c = float(c.median()) if not c.empty else 0.0
    for col in overlay_columns:
        if not col or col.lower() in _EXCLUDE_COLUMNS:
            dropped.append(col)
            continue
        if col not in enriched.columns:
            dropped.append(col)
            continue
        s = enriched[col].dropna()
        if s.empty:
            dropped.append(col)
            continue
        med_s = float(s.median())
        max_s, min_s = float(s.max()), float(s.min())
        key = col.lower()

        # Explicit oscillator hints
        if any(h in key for h in ("rsi", "stoch", "mfi", "williams", "cci")):
            osc_cols.append(col)
            continue
        if "macd_hist" in key or "macd_signal" in key:
            osc_cols.append(col)
            continue

        if med_c > 0:
            ratio = abs(med_s) / med_c
        else:
            ratio = 0.0

        # Price-scale: within a band around price.
        if med_c > 0 and 0.25 <= ratio <= 4.0 and min_s >= -abs(med_c):
            price_cols.append(col)
            continue

        # Bounded oscillator (RSI-like)
        if 0 <= max_s <= 150 and min_s >= -50:
            osc_cols.append(col)
            continue

        dropped.append(col)
    return price_cols, osc_cols, dropped


def _trade_markers(
    trades_df: pd.DataFrame, ts_to_unix: dict[pd.Timestamp, int]
) -> list[dict[str, Any]]:
    if trades_df is None or trades_df.empty:
        return []
    markers: list[dict[str, Any]] = []
    for _, t in trades_df.iterrows():
        et = pd.Timestamp(t["entry_time"])
        xt = pd.Timestamp(t["exit_time"])
        e_t = _nearest(ts_to_unix, et)
        x_t = _nearest(ts_to_unix, xt)
        is_long = str(t["side"]).upper() == "LONG"
        ep = float(t["entry_price"])
        xp = float(t["exit_price"])
        pnl = float(t.get("pnl", 0.0))
        reason = str(t.get("exit_reason", "") or "").lower()
        # Pyramid (continuation) rows have entry_time == exit_time and
        # exit_reason == "pyramid". Render them as smaller continuation
        # markers without an EXIT dot — they're not real round-trips, just
        # visual signals on each bar where the same-direction entry rule
        # kept firing (mirrors TradingView's MMTS pyramiding markers).
        if reason == "pyramid":
            markers.append(
                {
                    "time": e_t if e_t is not None else x_t,
                    "position": "belowBar" if is_long else "aboveBar",
                    "color": _TV_BULL if is_long else _TV_BEAR,
                    "shape": "arrowUp" if is_long else "arrowDown",
                    "text": "BUY" if is_long else "SELL",
                    "size": 0.9,
                }
            )
            continue
        if e_t is not None:
            markers.append(
                {
                    "time": e_t,
                    "position": "belowBar" if is_long else "aboveBar",
                    "color": _TV_BULL if is_long else _TV_BEAR,
                    "shape": "arrowUp" if is_long else "arrowDown",
                    "text": f"{'BUY' if is_long else 'SELL'} @ {ep:.2f}",
                    "size": 1.3,
                }
            )
        if x_t is not None:
            markers.append(
                {
                    "time": x_t,
                    "position": "aboveBar" if is_long else "belowBar",
                    "color": _TV_EXIT,
                    "shape": "circle",
                    "text": f"EXIT {xp:.2f} ({pnl:+.0f})",
                    "size": 1.0,
                }
            )
    markers.sort(key=lambda m: m["time"])
    return markers


def _live_marker(
    live_signal: Optional[LiveSignal],
    ts_to_unix: dict[pd.Timestamp, int],
) -> Optional[dict[str, Any]]:
    if live_signal is None or live_signal.action == "HOLD":
        return None
    t = _nearest(ts_to_unix, pd.Timestamp(live_signal.bar_time))
    if t is None:
        return None
    action = live_signal.action
    if action == "BUY":
        return {
            "time": t,
            "position": "belowBar",
            "color": _TV_BULL,
            "shape": "arrowUp",
            "text": "● LIVE BUY",
            "size": 2.0,
        }
    if action == "SELL":
        return {
            "time": t,
            "position": "aboveBar",
            "color": _TV_BEAR,
            "shape": "arrowDown",
            "text": "● LIVE SELL",
            "size": 2.0,
        }
    if action.startswith("EXIT"):
        return {
            "time": t,
            "position": "aboveBar" if action == "EXIT_LONG" else "belowBar",
            "color": _TV_LIVE_EXIT,
            "shape": "square",
            "text": "● LIVE EXIT",
            "size": 2.0,
        }
    return None


def _nearest(ts_to_unix: dict[pd.Timestamp, int], ts: pd.Timestamp) -> Optional[int]:
    if not ts_to_unix:
        return None
    if ts in ts_to_unix:
        return ts_to_unix[ts]
    keys = list(ts_to_unix.keys())
    nearest = min(keys, key=lambda k: abs((k - ts).total_seconds()))
    return ts_to_unix[nearest]


def _focus_bar_count(ohlcv: pd.DataFrame) -> tuple[int, bool]:
    """Compute how many of the rightmost bars must fit in the initial view.

    For intraday timeframes the focus is *today's session* (or the latest
    session present in the data, e.g. on a Sunday it falls back to Friday).
    For daily/weekly/monthly timeframes there is no "today" granularity, so
    we focus on the last 30 bars.

    Returns ``(target_bars, is_intraday)``.
    """
    if len(ohlcv) < 2:
        return max(len(ohlcv), 30), False
    deltas = ohlcv.index.to_series().diff().dropna()
    median_delta = deltas.median()
    interval_seconds = float(median_delta.total_seconds()) if median_delta else 300.0
    is_intraday = interval_seconds < 86_400  # < 1 day

    if not is_intraday:
        return 30, False

    last_date = pd.Timestamp(ohlcv.index[-1]).normalize()
    same_day_mask = pd.Series(ohlcv.index).map(
        lambda ts: pd.Timestamp(ts).normalize() == last_date
    )
    today_count = int(same_day_mask.sum())
    return max(today_count, 12), True


# NSE cash session ends at 15:30 IST. Used as the chart's right-edge anchor so
# the full trading day is always visible regardless of how many live bars have
# streamed in yet (otherwise the chart visually "stops" wherever the last bar
# is — typically a few minutes behind wall-clock during live trading).
_NSE_CLOSE_MINUTES = 15 * 60 + 30


def _bars_to_session_close(ohlcv: pd.DataFrame, fallback: int) -> int:
    """Empty-bar slot count needed past the last bar so the visible right edge
    reaches 15:30 IST (NSE session close) of the latest session.

    Intraday-only: daily / weekly / monthly timeframes return ``fallback``
    unchanged. This pins the chart's right edge to market close so the user
    can always see the *end of the session* — even before live bars catch up
    to the wall-clock minute (or when the live feed lags).
    """
    if ohlcv.empty or len(ohlcv) < 2:
        return fallback
    deltas = ohlcv.index.to_series().diff().dropna()
    if deltas.empty:
        return fallback
    interval_seconds = float(deltas.median().total_seconds())
    if interval_seconds <= 0 or interval_seconds >= 86_400:
        return fallback
    interval_min = interval_seconds / 60.0
    last_ts = pd.Timestamp(ohlcv.index[-1])
    last_minutes = last_ts.hour * 60 + last_ts.minute
    if last_minutes >= _NSE_CLOSE_MINUTES:
        return fallback
    bars_needed = int(round((_NSE_CLOSE_MINUTES - last_minutes) / interval_min))
    # +2 cushion so the 15:30 label is fully visible (not clipped at the edge).
    return max(fallback, bars_needed + 2)


def build_lightweight_charts_spec(
    *,
    ohlcv: pd.DataFrame,
    enriched: Optional[pd.DataFrame] = None,
    overlay_columns: Optional[list[str]] = None,
    trades_df: Optional[pd.DataFrame] = None,
    live_signal: Optional[LiveSignal] = None,
    title: str = "",
    height: int = 560,
    visible_bars: int = 75,
    history_multiplier: int = 12,
    estimated_width_px: int = 1200,
) -> list[dict[str, Any]]:
    """Build the ``renderLightweightCharts`` config for one strategy chart.

    Initial-viewport behaviour:

    - The chart is painted with a ``barSpacing`` chosen so that *today's*
      session (or the last 30 bars on daily+) sits in the visible area with a
      small right-side padding. Older bars are still **available** in the
      dataset; the user pans/zooms left to inspect history.
    - We trim the dataset to the last ``visible_bars * history_multiplier``
      bars (~9 trading days at 5m, several months at 1D) so panning has room
      without dumping irrelevant ancient data into the renderer.
    """
    if ohlcv is None or ohlcv.empty:
        return []

    ohlcv = normalize_to_naive_ist(ohlcv)
    history_window = max(visible_bars * max(1, history_multiplier), 200)
    if len(ohlcv) > history_window:
        ohlcv = ohlcv.iloc[-history_window:]

    ts_to_unix = _build_time_lookup(ohlcv)
    candles = _candle_rows(ohlcv, ts_to_unix)

    focus_bars, _is_intraday = _focus_bar_count(ohlcv)
    right_padding = max(6, min(20, focus_bars // 5 + 4))
    # Stretch the right edge to 15:30 IST so the full NSE session is visible
    # (matters when the live feed is a few bars behind wall-clock).
    right_padding = _bars_to_session_close(ohlcv, right_padding)
    target_bars_in_view = focus_bars + right_padding
    bar_spacing = int(round(estimated_width_px / max(target_bars_in_view, 12)))
    bar_spacing = max(6, min(64, bar_spacing))

    price_cols: list[str] = []
    osc_cols: list[str] = []
    enr_aligned: Optional[pd.DataFrame] = None
    if enriched is not None and overlay_columns:
        enr_aligned = enriched.reindex(ohlcv.index)
        price_cols, osc_cols, _dropped = _classify_overlays(
            overlay_columns, enr_aligned, ohlcv["close"]
        )

    markers = _trade_markers(trades_df, ts_to_unix) if trades_df is not None else []
    live_m = _live_marker(live_signal, ts_to_unix)
    if live_m is not None:
        markers.append(live_m)
    markers.sort(key=lambda m: m["time"])

    series: list[dict[str, Any]] = [
        {
            "type": "Candlestick",
            "title": title or "Price",
            "data": candles,
            "options": {
                "upColor": _TV_BULL,
                "downColor": _TV_BEAR,
                "borderVisible": False,
                "wickUpColor": _TV_BULL,
                "wickDownColor": _TV_BEAR,
                "pane": 0,
            },
            "markers": markers,
        }
    ]

    if enr_aligned is not None:
        for col in price_cols:
            data = _line_rows(enr_aligned[col], ts_to_unix)
            if not data:
                continue
            series.append(
                {
                    "type": "Line",
                    "title": col,
                    "data": data,
                    "options": {
                        "color": _overlay_color(col),
                        "lineWidth": 2,
                        "priceLineVisible": False,
                        "lastValueVisible": False,
                        "pane": 0,
                    },
                }
            )
        for i, col in enumerate(osc_cols, start=1):
            data = _line_rows(enr_aligned[col], ts_to_unix)
            if not data:
                continue
            series.append(
                {
                    "type": "Line",
                    "title": col,
                    "data": data,
                    "options": {
                        "color": _overlay_color(col),
                        "lineWidth": 2,
                        "priceLineVisible": False,
                        "lastValueVisible": False,
                        "pane": i,
                    },
                }
            )

    n_panes = 1 + len(osc_cols)
    pane_height = max(80, (height - 32) // n_panes)
    chart_options = {
        "height": pane_height * n_panes + 32,
        "layout": {
            "background": {"type": "solid", "color": "#ffffff"},
            "textColor": "#1a1f2c",
            "fontFamily": "Inter, system-ui, sans-serif",
        },
        "grid": {
            "vertLines": {"color": "#f0f3fa"},
            "horzLines": {"color": "#f0f3fa"},
        },
        "rightPriceScale": {
            "borderColor": "#dde0e8",
            "scaleMargins": {"top": 0.08, "bottom": 0.08},
        },
        "timeScale": {
            "borderColor": "#dde0e8",
            "timeVisible": True,
            "secondsVisible": False,
            "barSpacing": bar_spacing,
            "minBarSpacing": 4,
            "rightOffset": right_padding,
            "shiftVisibleRangeOnNewBar": True,
            "fixLeftEdge": False,
            "fixRightEdge": False,
        },
        "crosshair": {"mode": 1},
        # Custom hint consumed by ``render_lightweight_charts_html``. Tells the
        # JS wrapper to call ``setVisibleLogicalRange`` on the rightmost
        # ``focus_bars`` so the user sees today's session by default.
        "_initialFocus": {
            "bars": int(focus_bars),
            "rightPadding": int(right_padding),
        },
    }

    return [{"chart": chart_options, "series": series}]


_LWC_CDN = "https://unpkg.com/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js"


def render_lightweight_charts_html(
    spec: list[dict[str, Any]],
    *,
    height: int = 580,
    background: str = "#ffffff",
    stream_url: Optional[str] = None,
    stream_poll_ms: int = 2500,
) -> str:
    """Wrap a chart spec in a self-contained HTML page.

    The returned string is intended for ``streamlit.components.v1.html(...)``.
    It loads lightweight-charts v5 from a CDN, builds the chart, populates
    series + markers, and calls ``setVisibleLogicalRange`` so the initial view
    is anchored on the most recent ``_initialFocus.bars`` (today's session by
    default). The user can pan/zoom freely after that.

    If ``stream_url`` is provided, the iframe starts a JavaScript polling loop
    that fetches ``{"bars": [...], "markers": [...], "last": <unix>}`` every
    ``stream_poll_ms`` milliseconds and applies the deltas in-place via
    Lightweight Charts' ``series.update()`` API. Because the iframe is never
    reloaded, the user's pan / zoom state is preserved across updates — this
    is what gives the chart a smooth, TradingView-like streaming feel.
    """
    spec_json = json.dumps(spec, separators=(",", ":"), default=str)
    stream_url_js = json.dumps(stream_url or "", separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  html,body{{margin:0;padding:0;background:{background};font-family:Inter,system-ui,sans-serif;}}
  #fortuna-chart{{width:100%;height:{height - 12}px;}}
</style></head><body>
<div id="fortuna-chart"></div>
<script src="{_LWC_CDN}"></script>
<script>
(function() {{
  const SPEC = {spec_json};
  if (!SPEC || !SPEC.length) return;
  const chartCfg = SPEC[0].chart || {{}};
  const seriesCfg = SPEC[0].series || [];
  const focus = chartCfg._initialFocus || {{bars: 75, rightPadding: 8}};
  const container = document.getElementById('fortuna-chart');

  const lwcOptions = JSON.parse(JSON.stringify(chartCfg));
  delete lwcOptions._initialFocus;
  lwcOptions.width = container.clientWidth;
  lwcOptions.height = container.clientHeight;
  lwcOptions.autoSize = false;

  const chart = LightweightCharts.createChart(container, lwcOptions);
  const seriesTypes = {{
    Candlestick: LightweightCharts.CandlestickSeries,
    Line: LightweightCharts.LineSeries,
    Area: LightweightCharts.AreaSeries,
    Bar: LightweightCharts.BarSeries,
    Histogram: LightweightCharts.HistogramSeries,
    Baseline: LightweightCharts.BaselineSeries,
  }};

  let candleSeries = null;
  let markerPrim = null;
  let lastBarTime = 0;
  let maxLen = 0;
  // Title → series mapping so the streaming endpoint can extend indicator
  // lines (EMA, Bollinger bands, VWAP, …) in-place without re-rendering.
  const seriesByTitle = {{}};
  seriesCfg.forEach(s => {{
    const ctor = seriesTypes[s.type];
    if (!ctor) return;
    const opts = Object.assign({{}}, s.options || {{}});
    const paneIdx = opts.pane || 0;
    delete opts.pane;
    const series = chart.addSeries(ctor, opts, paneIdx);
    if (s.priceScale) series.priceScale().applyOptions(s.priceScale);
    const data = s.data || [];
    series.setData(data);
    if (s.title) seriesByTitle[s.title] = series;
    if (s.type === 'Candlestick') {{
      candleSeries = series;
      if (data.length) lastBarTime = data[data.length - 1].time || 0;
    }}
    if (s.markers && s.markers.length && LightweightCharts.createSeriesMarkers) {{
      const prim = LightweightCharts.createSeriesMarkers(series, s.markers);
      if (s.type === 'Candlestick') markerPrim = prim;
    }}
    if (data.length > maxLen) maxLen = data.length;
  }});

  if (maxLen > 0) {{
    const fromIdx = Math.max(0, maxLen - focus.bars - 1);
    const toIdx = maxLen - 1 + (focus.rightPadding || 6);
    chart.timeScale().setVisibleLogicalRange({{ from: fromIdx, to: toIdx }});
  }}

  const ro = new ResizeObserver(() => {{
    chart.applyOptions({{ width: container.clientWidth }});
  }});
  ro.observe(container);

  // ---------------------------------------------------------------
  // Live streaming: poll the backend for incremental bars + markers
  // and apply them in place. NEVER reload the iframe.
  // ---------------------------------------------------------------
  const STREAM_URL = {stream_url_js};
  const POLL_MS = {stream_poll_ms};
  if (STREAM_URL && candleSeries) {{
    let inFlight = false;
    let lastMarkersHash = "";

    function hashMarkers(ms) {{
      if (!ms || !ms.length) return "0";
      let h = ms.length + ":";
      for (let i = 0; i < ms.length; i++) {{
        h += (ms[i].time || 0) + "/" + (ms[i].text || "") + ",";
      }}
      return h;
    }}

    async function poll() {{
      if (inFlight) return;
      inFlight = true;
      try {{
        const sep = STREAM_URL.indexOf('?') >= 0 ? '&' : '?';
        const url = STREAM_URL + sep + 'since=' + lastBarTime;
        const r = await fetch(url, {{cache: 'no-store'}});
        if (!r.ok) return;
        const data = await r.json();
        const bars = data.bars || [];
        for (const b of bars) {{
          if (!b || typeof b.time !== 'number') continue;
          // series.update() in lightweight-charts performs an in-place mutation
          // when b.time === existing bar (forming-bar tick), and appends when
          // b.time > last bar — no full data re-set, so zoom is preserved.
          candleSeries.update(b);
          if (b.time > lastBarTime) lastBarTime = b.time;
        }}
        // Indicator overlay extensions — each entry is {{name, points}}.
        // We find the matching series by its title (e.g. "ema_fast",
        // "bb_upper") and call .update() per point, which appends new bars
        // or updates the forming bar's value without redrawing the line.
        const overlays = data.overlays || [];
        for (const ov of overlays) {{
          const s = seriesByTitle[ov.name];
          if (!s || !ov.points) continue;
          for (const pt of ov.points) {{
            if (!pt || typeof pt.time !== 'number') continue;
            s.update(pt);
          }}
        }}
        const markers = data.markers || [];
        const h = hashMarkers(markers);
        if (h !== lastMarkersHash) {{
          if (markerPrim && markerPrim.setMarkers) {{
            markerPrim.setMarkers(markers);
          }} else if (LightweightCharts.createSeriesMarkers) {{
            markerPrim = LightweightCharts.createSeriesMarkers(candleSeries, markers);
          }}
          lastMarkersHash = h;
        }}
      }} catch (e) {{
        // Silent — server may be momentarily unavailable between reruns.
      }} finally {{
        inFlight = false;
      }}
    }}
    // Stagger the first poll so the iframe paints first.
    setTimeout(poll, 500);
    setInterval(poll, POLL_MS);
  }}
}})();
</script></body></html>"""
