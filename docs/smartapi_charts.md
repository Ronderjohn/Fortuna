# SmartAPI and charts

Angel One **SmartAPI does not expose TradingView chart images or embed URLs**.

Available market data endpoints used by Fortuna:

| API | Purpose |
|-----|---------|
| `getCandleData` | OHLCV history (1m, 5m, 15m, 30m, 1h, 1d) |
| WebSocket | Live ticks → aggregated bars |

The Angel One **trading terminal** uses TradingView as its UI, but that rendering is not available through the REST/WebSocket API.

Fortuna builds TradingView-style charts locally from:

1. SmartAPI (or cache) OHLCV
2. Strategy indicator columns (`enriched_data`)
3. Simulated trade ledger (`trades_df`)

## Streamlit dashboard

The dashboard (`app/streamlit_app.py`) renders charts in-browser via
**Lightweight Charts v5** (`src/fortuna/app/lightweight_chart.py`), embedded
through a custom HTML wrapper that:

- Loads `lightweight-charts@5` from a CDN
- Calls `setVisibleLogicalRange` so the initial view focuses on today's NSE
  session with the right edge anchored at 15:30 IST (market close)
- Streams incremental bars / markers / indicator overlays from a lightweight
  stdlib HTTP server (`src/fortuna/app/stream_server.py`) via JS polling, so
  zoom/pan state is preserved across updates

Static CLI report bundles (`logs/strategy_tester/.../charts/`) are
matplotlib PNGs only — equity curve, drawdown, P&L distribution, etc.
