# Phase 1 — Deterministic Research Foundation

> **Long-term vision.** Integrate powerful ML / DL models with **reinforcement
> learning** to generate trading strategies that produce the best indicators
> and signals, then let **autonomous agents** consume those signals to make
> decisions and execute orders — all driven by adaptive learning on live and
> historical market data.
>
> **Phase 1 mandate.** Build the deterministic, offline-first research
> foundation that the ML / DL / RL stack will sit on top of: strategy DSL,
> data layer, indicator engine, backtest engines, institutional walk-forward
> validation, TradingView-style reporting, and an operator dashboard with
> live read-only feed. Without this layer, RL training has no honest reward
> signal and agents have nothing to validate their decisions against.
>
> **Out of scope for Phase 1 (planned for later phases).** ML / DL feature
> learning, reinforcement-learning strategy generation, autonomous decision
> agents, live broker execution, multi-symbol portfolio orchestration.
>
> **Status.** Implemented and validated against NSE intraday equities (default
> symbol `ICICIBANK.NS`, default timeframe `5m`) under the `fortuna` package at
> `c:\dev\Fortuna`. Last updated **2026-05-25**.

---

## 1. Purpose

The long-term Fortuna system is an **ML/DL + RL-driven strategy generator
feeding autonomous trading agents**. None of that works unless the layer
underneath is honest — RL needs a reward signal that isn't a fantasy, and
agents need a validated strategy space they can search inside. Most
retail-grade backtesters either (a) overfit to a single curve with no
walk-forward control, or (b) trade signals through a black-box engine you
can't audit. Phase 1 fixes both, and exposes its surface in the exact
shapes the ML / RL / agent phases will consume:

1. **Deterministic, auditable signals.** Every strategy is a JSON file or a
   Pine-parity Python engine (MMTS, ORB). Indicators are computed by a single
   `IndicatorEngine`; entries/exits are compiled by a single `StrategyCompiler`;
   trade simulation is one of two reference runners (NumPy or vectorbt). The
   same code path runs in the backtest, the institutional walk-forward, the
   paper league, and the dashboard's live signal panel — which is exactly the
   contract an RL policy or agent will later step through.
2. **Honest evaluation = honest reward.** Realistic Indian-equity intraday
   costs (brokerage, exchange fees, slippage, spread), NSE session rules
   (09:15–15:30 IST with 15:15 square-off), train/validation/test
   chronological splits, Monte Carlo trade-shuffle, and composite ranking
   that **never** optimizes for win rate alone. The same metrics that score
   a strategy today will score an RL policy tomorrow.
3. **TradingView-style operator experience.** A Streamlit dashboard with
   Lightweight Charts v5 that opens on today's session, displays the winning
   strategy by default, streams live bars and BUY/SELL/EXIT markers in-place,
   and lets you compare every strategy on the same OHLCV in seconds. This
   is the same console an agent's decisions will later be rendered through.
4. **Pre-wired for the ML / RL phase.** Indicator batching, GPU policy,
   indicator cache, walk-forward folds, adaptive parameter learner, and the
   strategy DSL are all built so the next phases can plug in ML / DL
   feature learning, RL policy training, and decision agents **without
   rewriting the core**.

---

## 2. High-level architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            Streamlit dashboard                           │
│  app/streamlit_app.py · session engine · parallel runner · chart iframe  │
└──────────┬──────────────────────────────────────────────┬────────────────┘
           │                                              │
           │ poll @ N s                  HTTP poll @ 2.5s │
           ▼                                              ▼
┌─────────────────────┐                ┌────────────────────────────────────┐
│  Live session       │                │  Stream HTTP server (stdlib)       │
│  bridge (WebSocket  │   bars + ticks │  /fortuna/bars  → JSON            │
│  → bar aggregator)  │ ─────────────► │  bars · markers · overlays         │
└─────────┬───────────┘                └────────────────────────────────────┘
          │
          │ append closed bars
          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          MarketDataManager                               │
│   cache (Parquet)  ◄──  data source (SmartAPI / OpenChart / yfinance)    │
│        ▲                                                                 │
│        └── DuckDB analytical views (zero-copy on Parquet)                │
└──────────────────────────────────────────────────────────────────────────┘
          ▲
          │ get_ohlcv()
          │
┌─────────┴────────────────────────────────────────────────────────────────┐
│   Strategy DSL (JSON / Pydantic) ─► IndicatorEngine ─► StrategyCompiler  │
│                                                              │           │
│                                                              ▼           │
│      ┌────────────────┬────────────────┬───────────────────────┐        │
│      │ NumPyRunner    │ vectorbt       │ Builtin engines        │        │
│      │  (default)     │ (--group vbt)  │  (MMTS, ORB, pyramid)  │        │
│      └────────┬───────┴────────┬───────┴───────────┬───────────┘        │
│               ▼                ▼                   ▼                     │
│           BacktestResult ─► Strategy Tester report ─► Charts / CSV / JSON│
└──────────────────────────────────────────────────────────────────────────┘
          │
          │ orchestration
          ▼
┌────────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  Institutional         │   │  Arena             │   │  Paper league      │
│  pipeline (train/val/  │   │  (multi-strategy   │   │  (walk-forward     │
│  test, MC, filters)    │   │  + param sweep)    │   │  + adaptive learn) │
└────────────────────────┘   └────────────────────┘   └────────────────────┘
```

---

## 3. Module map

### 3.1 `strategy/` — DSL + loading
- **`schema.py`** — Pydantic models for `StrategyDefinition`, `IndicatorSpec`,
  `Condition` (`compare` / `crossover` / `crossunder` / `and` / `or` / `not`),
  `RiskManagement`, `StrategyMetadata`. `param_grid` field carries the param
  expansion grid consumed by `search/`.
- **`loader.py`** — JSON → `StrategyDefinition` with validation.
- **`warmup.py`** — Computes warmup-bar requirements (`required_warmup_bars`)
  so the arena slices enough history before evaluating a candidate.

### 3.2 `data/` — Cache-first market data
- **`manager.py`** — `MarketDataManager.get_ohlcv()` is cache-first: it
  consults the Parquet cache, only re-fetches if stale (5m: hours; daily: days),
  resamples non-native timeframes (45m, 2h, 3h, 4h, 1W, 1M), filters NSE
  sessions, and registers Parquet files into DuckDB for analytical queries.
- **`cache.py`** — Parquet read/write keyed by `(symbol, timeframe)`.
- **`duckdb_store.py`** — Registers Parquet files as DuckDB views; zero-copy.
- **`timeframes.py`** — `TradingView-style intervals`: native SmartAPI
  (`5m / 15m / 30m / 1h / 1d`) and locally resampled (`45m, 2h, 3h, 4h, 1W, 1M`).
- **`instruments.py`** — Angel One `OpenAPIScripMaster` instrument registry
  (used for SmartAPI symbol → token lookups and the dashboard's NSE search).
- **`sources/`**
  - `smartapi_session.py` — Login (`loginByPassword` + TOTP) and session reuse.
    Strips the `Bearer ` prefix the SDK sometimes prepends (caused "Invalid Token"
    failures otherwise).
  - `smartapi_rest.py` — Direct REST `getCandleData`; emits a synthetic
    `errorCode=RATE_LIMIT` for HTTP 403 (the SDK hides this as "Invalid Token").
  - `smartapi_historical.py` — REST-preferred candle fetch with chunking +
    exponential backoff on rate limits.
  - `smartapi_live.py` + `smartapi_bar_aggregator.py` + `smartapi_ws_decode.py`
    — WebSocket 2.0 snap-quote stream → bar aggregator → Parquet append +
    in-memory tick callback.
  - `smartapi_throttle.py` — Token-bucket rate limiter for the ~3 req/s ceiling.
  - `openchart_source.py`, `yfinance_source.py` — Free fallback sources.

### 3.3 `indicators/` — Numerically-stable, vectorized
- **`engine.py`** — Applies a list of `IndicatorSpec` to OHLCV and returns the
  enriched DataFrame (single concat at the end — important for pandas
  block-management performance).
- **`registry.py`** — Pure functions per indicator: EMA, SMA, RSI, ATR, VWAP,
  MACD, Bollinger Bands, Volume SMA, rolling high/low.
- **`kernels.py`** — Numba-compatible numpy kernels reused by builtin engines
  (`ema_array`, `atr_array`, `sma_array`).

### 3.4 `backtesting/` — Two reference runners + one institutional pipeline
- **`compiler.py`** — Turns the strategy DSL into boolean entry/exit Series.
- **`engine.py`** — `BacktestEngine` (vectorbt-based, long/short/both, fees,
  slippage, SL/TP).
- **`numpy_runner.py`** — `NumPyBacktestRunner` — **the default**, long-only,
  percent SL/TP, deterministic, no vectorbt dependency.
- **`risk.py`** — `percent_sl_tp` / `vectorbt_sl_tp` helpers.
- **`metrics.py`** — `BacktestMetrics` (return, Sharpe, drawdown, win rate,
  expectancy, profit factor) + `metrics_from_equity`.
- **`standard/`** — Institutional walk-forward pipeline:
  - `config.py` — `InstitutionalBacktestConfig`, `MarketCostModel` (brokerage
    + exchange + slippage + spread), `SessionRules` (09:15–15:30 + 15:15
    square-off), `SplitRatios`, `FilterThresholds`, `BenchmarkTargets`.
  - `calendar.py` — `filter_session_bars`, `trading_days_count`.
  - `splits.py` — Chronological train/val/test split with calendar-day estimate.
  - `runner.py` — Per-strategy fixed-param or grid-search execution on each
    split.
  - `filters.py` — `evaluate_strategy(...)` → `FilterVerdict` (PASS/FAIL with
    reasons) + composite ranking score that weights Sharpe, profit factor,
    drawdown penalty, expectancy, and trade count — **not** raw profit.
  - `robustness.py` — `monte_carlo_trade_shuffle` (default 500 iterations),
    walk-forward summary helpers.
  - `comparison.py` — Cross-strategy comparison tables for the test phase.
  - `pipeline.py` — End-to-end `InstitutionalPipeline.run()` that produces a
    full per-strategy report bundle under `logs/institutional/`.

### 3.5 `reporting/` — TradingView Strategy Tester clone
- **`strategy_tester/metrics.py`** — `PerformanceMetrics`, `RiskMetrics`,
  `IntradayMetrics`. `compute_all_metrics(trades, equity, ...)` is the single
  source of truth for everything the dashboard shows.
- **`strategy_tester/trade.py`** — `TradeRecord` + `trades_to_dataframe`.
- **`strategy_tester/report.py`** — `StrategyTesterReport` dataclass + JSON
  summary + CSV exports. Pyramid trades are filtered before metric
  computation so MMTS pyramiding doesn't skew the win-rate stats.
- **`strategy_tester/backtest_bridge.py`** + **`chart_context.py`** — Adapt
  any `BacktestResult` (NumPy / vectorbt / builtin) into a `StrategyTesterReport`.
- **`strategy_tester/visualize.py`** — Matplotlib PNG charts (equity,
  drawdown, P&L distribution, monthly P&L heatmap).
- **`strategy_tester/chart_viewport.py`** — Default visible-bars per timeframe
  and `normalize_to_naive_ist` (coerces mixed tz-naive/tz-aware indices to a
  single tz-naive IST index — historical bars are tz-naive, live bars are
  tz-aware; concatenating them used to break `sort_index`).
- **`strategy_tester/signal_overlay.py`** + **`costs.py`** — Cost model and
  signal-frame builder used by both the chart and the report card.

### 3.6 `evaluation/` — Composite scoring + routing
- **`scorer.py`** — `StrategyScorer.score(metrics, metadata)` → 0–100
  composite weighted by Sharpe + return + drawdown penalty + profit factor +
  consistency, with robustness penalties for too-few trades, suspiciously high
  win rates, and high parameter counts.
- **`ranker.py`** — Routes scored strategies to `strategies/validated/` or
  `strategies/rejected/` with a JSON sidecar.

### 3.7 `search/` — Param sweep + batch evaluation
- **`param_expander.py`** — Materializes a `param_grid` into candidates.
- **`candidate.py`** — `CandidateStrategy` (strategy + concrete params).
- **`indicator_cache.py`** — Memoizes enriched DataFrames so candidates that
  share indicator specs don't recompute them.
- **`evaluator.py`** + **`worker.py`** — Single-candidate evaluation paths.
- **`batch_evaluator.py`** — Time-budgeted parallel evaluation. On Windows
  the default is **threads-not-processes** (see §6) to keep RAM low.

### 3.8 `compute/` — CPU/GPU policy
- **`device.py`** — Detects CuPy / CUDA and queries VRAM.
- **`policy.py`** — `ComputePolicy` decides when to push EMA/SMA/RSI/ATR
  jobs to the GPU based on bar count, unique series, and estimated VRAM.
  Tuned to 4 GB cards.
- **`memory.py`** — Estimates per-job VRAM cost.
- **`gpu_kernels.py`** — Stubbed CuPy paths; `compute_series_cpu` /
  `compute_series_gpu`.
- **`scheduler.py`** — Throttles `effective_workers(requested)` based on
  whether the GPU is active.
- **`batch_indicators.py`** — Deduplicates indicator jobs across candidates.

### 3.9 `arena/` — Parallel multi-strategy competitions
- **`arena.py`** — `StrategyArena.run_full_sample()` (one window, best params
  per strategy) and `run_streamed()` (rolling windows).
- **`param_search.py`** — `ParamSearchEngine` (drives `BatchEvaluator`).
- **`leaderboard.py`** — `StrategyLeaderboard` (rank by `profit_pct` |
  `win_ratio_pct` | `composite_score`).

### 3.10 `tournament/` — Per-bar winner selection
- **`runner.py`** + **`winner.py`** — Pick the bar-by-bar winner across many
  strategies/variants. Output is a CSV the next phase can use to drive a
  live BUY/SELL signal stream.

### 3.11 `paper/` — Walk-forward paper league with adaptive learning
- **`engine.py`** — `PaperTradeEngine` runs one strategy in an isolated
  `PaperAccount` (init cash, fees, slippage).
- **`account.py`** — Per-strategy cash + equity ledger (`PaperAccount.from_backtest`).
- **`competition.py`** — Single-fold competition.
- **`blackbox.py`** — `walk_forward_folds` — strict train/test isolation
  (calibrate-in-sample, validate-out-of-sample, the same principle as
  calibrating an option-pricing model on history and measuring hedge PnL on
  unseen dates).
- **`league.py`** — Multi-strategy walk-forward league: `train → test → learn`
  per fold.
- **`learner.py`** — `AdaptiveLearner` narrows each strategy's `param_grid`
  toward winners between folds; persistent state under
  `logs/paper_league/{run_id}/learning/*_learning.json`.

### 3.12 `strategies/builtin/` — Pine-parity engines
- **`mmts.py`** — Measured Move Trend Strategy (TradingView Pine parity):
  EMA stack trend filter (close vs `ema_fast`), optional Bollinger Band
  confluence filter, ATR-based stops, **pyramiding** (3 additions of 0.5×
  size by default — matches TV's consecutive bear/bull markers).
- **`orb.py`** — Opening Range Breakout: first N×5m bars set the OR, long
  on close above OR-high / short on close below OR-low, ATR stop, R-multiple
  target, square-off at 15:15 IST.
- **`dispatch.py`** — `is_builtin_strategy(strategy)` /
  `run_builtin_backtest(strategy, ohlcv, ...)` — routed automatically when
  `metadata.engine` is set.

### 3.13 `app/` — Streamlit dashboard
- **`streamlit_app.py`** — Single-page UI: header (symbol picker), controls
  (timeframe / days / refresh / live), live pulse fragment, tabs
  (Chart / Leaderboard / Performance / Strategy detail).
- **`session_engine.py`** — `FortunaSessionEngine`: thread-safe per-symbol
  load → parallel backtest → live-signal refresh. Holds a `SessionState`
  (`ohlcv`, `batch`, `live_signals`, `last_bar_time`, `last_signal_refresh`).
  Normalizes all indices to tz-naive IST via `_normalize_index_to_naive_ist`.
- **`parallel_runner.py`** — `ParallelStrategyRunner.run_parallel(...)`:
  thread-pool over every strategy on the **same** in-memory OHLCV; each
  result becomes a `StrategyTesterReport`; leaderboard rows are sorted by
  net profit by default.
- **`live_session.py`** — `LiveSessionBridge`: WebSocket feed → bar
  aggregator → in-memory append + Parquet append (`MarketDataManager.append_bar`).
  Coerces every ts via `_to_naive_ist` to keep one tz-coherent index.
- **`live_signals.py`** — `compute_live_signals(...)` recomputes BUY / SELL
  / EXIT / HOLD / IN_LONG / IN_SHORT on the latest (possibly forming) bar
  without running a full backtest. `compute_live_overlays(...)` returns the
  last N indicator points for the chart's smooth line extension.
- **`lightweight_chart.py`** — Builds the JSON spec consumed by Lightweight
  Charts v5 + ships a self-contained HTML wrapper that:
  - Loads `lightweight-charts` v5 from CDN.
  - Calls `timeScale().setVisibleLogicalRange(...)` after `setData()` —
    every off-the-shelf Streamlit wrapper auto-fits all history on first
    paint; we explicitly pin the viewport to today's session.
  - Polls the stream HTTP server every ~2.5 s and applies updates with
    `candleSeries.update(b)` / `markerPrim.setMarkers(markers)` /
    `seriesByTitle[name].update(pt)` — the iframe is **never reloaded**,
    so the user's zoom/pan state is preserved.
  - Anchors the right edge at **15:30 IST** (`_bars_to_session_close`) so
    the full session is always visible even when live bars lag.
- **`stream_server.py`** — Stdlib `ThreadingHTTPServer` exposing
  `/fortuna/bars` (live JSON) and `/fortuna/health`. CORS-permissive, no
  access-log noise, one server per Streamlit process via
  `@st.cache_resource`.
- **`presentation.py`** — Streamlit widgets (leaderboard table, live signal
  panel, OHLCV ticker, strategy report card).
- **`symbol_catalog.py`** — Wraps `InstrumentRegistry` for the TradingView-
  style symbol picker.
- **`strategy_paths.py`** — Lists strategy JSON files from the configured
  `strategy_dirs`.
- **`config.py`** — `AppConfig` (parallel workers, live refresh seconds,
  strategy dirs) loaded from `configs/intraday.yaml` under the `app:` key.

### 3.14 `agents/` — Decision-agent interfaces (stubs only in Phase 1)
- **`base.py`** — Abstract `ResearchAgent` (propose strategies),
  `CriticAgent` (judge a strategy + its backtest metrics), `OptimizerAgent`
  (suggest parameter improvements). These are the seats the later
  **ML / DL / RL agents** will sit in: a research agent that an RL policy
  drives, a critic that consumes the same `BacktestMetrics` the
  deterministic stack already produces, and an optimizer that consumes
  `AdaptiveLearner` fold history. The interfaces are intentionally
  framework-agnostic (no PyTorch / TF / RLlib imports at the contract
  layer) so the runtime can be swapped without churning the rest of the
  codebase.
- **`stubs.py`** — Trivial dummy implementations used by tests.

### 3.15 `quant/` — Theoretical reference
- **`black_scholes.py`** — European call/put pricing + implied vol sanity
  check. Used as a benchmark when comparing strategy Sharpe to a
  delta-hedged option book (not used to generate equity signals).

### 3.16 `utils/`
- **`logging.py`** — `get_logger(__name__)` everywhere; unified format.
- **`runtime_env.py`** — `apply_low_spec_gpu_defaults()` (GPU on,
  ProcessPool off on Windows by default).
- **`insecure_ssl.py`** — `maybe_disable_ssl_verification()` honoring
  `FORTUNA_INSECURE_SSL=true` for Avast/AVG/corporate TLS-MITM boxes.
  Must run **before** `requests`/`urllib3`/`websocket` import.
- **`timing.py`** — `timed_step("section")` context manager emitting
  `[TIMING]` log lines (used everywhere for cache/fetch/eval breakdown).

---

## 4. End-to-end flows

### 4.1 Dashboard load (the happy path)

```
User clicks "Analyze" with symbol=ICICIBANK.NS, tf=5m, days=30, live=on
   │
   ▼
FortunaSessionEngine.load_symbol()
   │
   ├── MarketDataManager.get_ohlcv("ICICIBANK.NS", "5m", days=30)
   │     │
   │     ├── Parquet cache check (4h stale window for 5m)
   │     ├── if miss → SmartAPI getCandleData (chunked, throttled, backoff)
   │     ├── materialize_timeframe (resample if needed)
   │     ├── filter_session_bars (09:15–15:30, Mon–Fri only)
   │     ├── cache.write(parquet) + duckdb.register_parquet()
   │     └── normalize_to_naive_ist
   │
   ├── ParallelStrategyRunner.run_parallel(ohlcv, strategy_paths)
   │     │
   │     └── ThreadPoolExecutor(max_workers=6)
   │           │
   │           ├── for each strategy:
   │           │     ├── load_strategy(path)
   │           │     ├── if builtin: run_builtin_backtest(...)
   │           │     ├── else      : NumPyBacktestRunner.run(...)
   │           │     └── build_report_from_backtest(...) → StrategyTesterReport
   │           │
   │           └── BatchRunResult.leaderboard_rows() (sorted by net profit)
   │
   ├── compute_live_signals(...) for every strategy
   ├── if live=on → LiveSessionBridge.start() (WebSocket subscribe)
   └── set chart_strat = batch.winner  (chart opens on the winner)

   ▼
Dashboard renders:
   ├── Header (symbol picker, market universe count)
   ├── Controls (timeframe, days, refresh, live, Analyze)
   ├── OHLCV ticker
   ├── 5 KPI metrics (bars, from, to, strategies, live)
   ├── Strategy comparison grid + live signal panel
   ├── Tabs: 📈 Chart  🏆 Leaderboard  📊 Performance  🔬 Strategy detail
   │
   └── Chart tab embeds the Lightweight Charts iframe
         │
         └── iframe polls http://localhost:{port}/fortuna/bars every 2.5s
               │
               └── stream_server.provider(symbol, tf, strategy, since=last)
                     │
                     ├── eng.get_ohlcv_for_chart() (in-memory, normalized)
                     ├── _build_stream_markers(state, strategy) (BUY/SELL/EXIT)
                     ├── compute_live_overlays(strategy_def, df, overlays, tail=20)
                     └── returns { bars[], markers[], overlays[], last }
```

### 4.2 Live tick path (no Streamlit reruns)

```
SmartAPI WebSocket 2.0 (snap quote @ 1Hz / SmartAPI cadence)
    │
    ▼
decode_ws_message → tick
    │
    ▼
BarAggregator.on_tick(tick)
    ├── update forming bar (in-memory only)
    ├── if bar boundary crossed:
    │     ├── close current bar
    │     ├── LiveSessionBridge._on_bar_closed()
    │     │     ├── MarketDataManager.append_bar() → Parquet + DuckDB
    │     │     └── enqueue LiveEvent(BAR_CLOSED)
    │     └── start new forming bar
    └── enqueue LiveEvent(TICK)

@st.fragment(run_every="5s") _live_pulse():
    ├── engine.poll_live()             # drain LiveEvent queue
    ├── engine.refresh_live_signals()  # compute_live_signals on tail
    └── render small "📡 LIVE · last bar HH:MM" pill
    (no st.rerun — only the fragment re-renders)

Independently, chart iframe polls /fortuna/bars every 2.5s:
    └── candleSeries.update(latest_bar)         # smooth update, preserves zoom
        markerPrim.setMarkers(new_markers)
        seriesByTitle["ema_fast"].update(point) # indicator line extends
```

### 4.3 Institutional walk-forward run

```
InstitutionalPipeline.run()
   │
   ├── MarketDataManager.get_ohlcv (≥ min_history_days, default 20)
   ├── filter_session_bars
   ├── validate_history_length
   ├── chronological_split → SplitResult(train 70% / val 15% / test 15%)
   │
   ├── for each strategy in [intraday/, generated/, builtin/]:
   │     ├── if builtin or no grid → run_strategy_fixed
   │     ├── else                  → run_strategy with grid_override
   │     │
   │     ├── per phase (train / validation / test):
   │     │     ├── compute_all_metrics → StrategyMetrics
   │     │     ├── export PNG bundle (equity / drawdown / heatmap)
   │     │     └── write trades.csv + signals.csv + report.json
   │     │
   │     ├── evaluate_strategy(test) → FilterVerdict (PASS/FAIL + reasons)
   │     ├── monte_carlo_trade_shuffle(test.trades, 500 iters)
   │     └── persist filter_verdict.json + monte_carlo.json
   │
   ├── build_comparison_table(results, phase="test")
   └── print_comparison_table + CSV export
```

### 4.4 Paper league with adaptive learning

```
PaperLeague.run()
   │
   ├── walk_forward_folds(ohlcv, train_bars, test_bars, step_bars)
   │
   └── for each fold:
         ├── for each competitor:
         │     ├── TRAIN PHASE
         │     │     ParamSearchEngine.search(strategy, fold.train, grid)
         │     │     → best_candidate
         │     │
         │     ├── TEST PHASE (out-of-sample, paper)
         │     │     PaperTradeEngine.run(best_candidate, fold.test)
         │     │     → PaperSessionResult (account + metrics)
         │     │
         │     └── LEARN PHASE
         │           AdaptiveLearner.update(strategy, fold_record)
         │           → narrowed param_grid for next fold
         │
         └── persist oos_leaderboard.csv + all_oos_folds.csv +
             learning/{strategy}_learning.json
```

---

## 5. System design concepts

### 5.1 Cache-first data layer
Everything reads through `MarketDataManager.get_ohlcv`. The Parquet cache is
keyed by `(symbol, timeframe)`; staleness windows differ by timeframe (hours
for intraday, days for daily). DuckDB registers each Parquet file as a view
so queries like "filter 09:15–15:30 between dates X and Y" are pushed down
zero-copy. Result: the dashboard re-loading a symbol is ~free; the
institutional pipeline + arena + paper league all share the same files.

### 5.2 Strategy DSL with builtin escape hatches
JSON is the default form (declarative, agent-friendly), but realistic intraday
strategies need things the DSL can't express — pyramiding, ATR-stop chaining,
session-aware square-off, multi-leg trades. Those live as **builtin engines**
(`MMTS`, `ORB`) and are dispatched transparently via `metadata.engine`. The
DSL remains the public surface; the engines stay swappable.

### 5.3 Two backtest runners, one report
The default runner is `NumPyBacktestRunner` (long-only, percent SL/TP, no
vectorbt dependency, deterministic, ~10× cheaper to import). `vectorbt` is an
opt-in (`uv sync --group vbt`) used for long/short and advanced sizing tests.
Both produce a `BacktestResult`, and both flow into the **same**
`StrategyTesterReport` via `build_report_from_backtest`. The dashboard, the
arena, the paper league, and the institutional pipeline all consume the
same report shape.

### 5.4 Composite scoring — never win-rate-alone
`StrategyScorer` weights Sharpe (35%), drawdown penalty (25%), profit factor
(15%), total return (15%), consistency (10%). It applies robustness penalties
for (a) too few trades, (b) suspiciously high win rate with few trades, (c)
high parameter count (overfit proxy). The institutional `_composite_rank_score`
extends this with explicit trade-count and expectancy components, and
benchmarks every strategy against fixed targets (Sharpe ≥ 1.2, PF ≥ 1.5,
max-DD ≤ 15%).

### 5.5 Black-box walk-forward
`paper/blackbox.py:walk_forward_folds` enforces **strict train/test
isolation**. The same philosophy as calibrating an option-pricing model on
in-sample dates and measuring hedge PnL on unseen dates — Phase 1's
acceptance criterion for any strategy is its cumulative OOS PnL across
folds, not the in-sample fit.

### 5.6 Adaptive learning
`AdaptiveLearner` narrows each strategy's `param_grid` toward the winners
of the previous fold. State persists per strategy (`generation`, `best_params`,
`fold_history`, `cumulative_oos_profit_pct`) under
`logs/paper_league/{run_id}/learning/`. This is the seed for Phase 2's
online optimization — already running end-to-end, just without LLM-driven
hypothesis generation.

### 5.7 In-place chart streaming
The dashboard chart was the biggest UX battle of Phase 1. The final
architecture:

1. The chart is **not** a Streamlit component — it's a self-contained HTML
   page (Lightweight Charts v5 from CDN) embedded with `st.components.v1.html`.
2. Initial paint pins the viewport to today's session via
   `timeScale().setVisibleLogicalRange(...)` and stretches the right edge to
   15:30 IST via `_bars_to_session_close`.
3. The iframe **polls a stdlib HTTP server** running in a daemon thread inside
   the Streamlit process. The server's `provider(symbol, tf, strategy, since)`
   returns only what's new since the last `since` timestamp: a list of bars,
   a list of markers, and a list of indicator-overlay points.
4. The JS applies updates with `series.update()` / `setMarkers()` — the
   iframe is never reloaded, so the user's zoom/pan/crosshair state is
   preserved (matches TradingView's native streaming feel).
5. A separate `@st.fragment(run_every="5s")` pulse drains the WebSocket
   event queue and updates a tiny "📡 LIVE" status pill — this fragment
   never triggers a full Streamlit rerun.

### 5.8 Tz handling
Mixing cache (tz-naive IST) with live bars (tz-aware `Asia/Kolkata`) used to
break `sort_index` with `TypeError: Cannot compare tz-naive and tz-aware`.
We now have one canonical normalization (`_normalize_index_to_naive_ist` /
`_to_naive_ist` / `normalize_to_naive_ist`) applied at **every boundary**
(session engine, live bridge, chart viewport).

### 5.9 ML/RL-agent-ready interfaces
`agents/base.py` defines pure abstract base classes for the three roles
the later ML / RL phase will fill (Research, Critic, Optimizer). They
consume the same `StrategyDefinition`, `BacktestMetrics`, and
`StrategyTesterReport` types used by the deterministic stack — there is no
separate "agent data model" to maintain, and no schema migration when the
RL runtime lands. The adaptive learner in `paper/learner.py` is the
proto-optimizer: its fold history is already the shape an offline RL
trainer would consume.

---

## 6. Optimizations performed

### 6.1 Indicator engine
- **Single concat at the end.** `IndicatorEngine.compute` accumulates new
  columns in a `dict[str, pd.Series]` and concatenates once instead of
  `df.assign(...)` in a loop — avoids pandas' fragmented-block warning and
  O(n²) DataFrame rebuilds across long indicator chains.
- **Numba kernels** in `indicators/kernels.py` for inner loops (`ema_array`,
  `atr_array`, `sma_array`) used by builtin engines.
- **Indicator cache** (`search/indicator_cache.py`) memoizes enriched
  DataFrames so candidates that share indicator specs in an arena param
  sweep don't recompute them.
- **Batch deduplication** (`compute/batch_indicators.py`) collects unique
  `IndicatorJob`s across all candidates so a 50-variant EMA sweep only
  computes 50 unique window EMAs once.

### 6.2 Backtesting
- **NumPy runner default.** Reduces the cold-import cost and removes
  vectorbt as a hard dependency for the dashboard. vectorbt is opt-in for
  long/short and advanced sizing tests.
- **Pyramid filtering at metric time.** `build_strategy_report` strips
  `exit_reason == "pyramid"` trades before calling `compute_all_metrics` —
  pyramid additions are visual continuation markers, not standalone trades,
  so including them would skew the win rate.

### 6.3 Data layer
- **Cache-first with timeframe-aware staleness.** Intraday (5m / 15m / ...)
  uses hours; daily/weekly/monthly uses days. Configurable via
  `cache_max_age_hours` / `cache_max_age_days`.
- **DuckDB views** registered per Parquet file — analytical queries push
  down filters without copying data into Python.
- **Chunked SmartAPI fetches** (20-day chunks for 1m/5m/15m, 90-day for
  higher TFs) + token-bucket throttle + exponential backoff on rate limits.
- **REST-preferred candle fetch** — the REST path returns clearer error
  codes (`RATE_LIMIT` for HTTP 403) than the SDK, which masquerades them as
  `"Invalid Token"`. The session module strips the SDK's stray `"Bearer "`
  prefix on JWTs to prevent double-prefixing on retry.

### 6.4 Compute policy (4 GB GPU friendly)
- **Hybrid policy.** `ComputePolicy.should_gpu_indicators(n_bars,
  n_unique_series)` only routes to GPU when (a) enough bars / series exist
  to amortize transfer cost and (b) the estimated VRAM fits inside
  `gpu_mem_fraction × total_vram`.
- **Per-series batch size** clipped to VRAM (`series_batch_size`).
- **CPU workers gated by GPU.** When the GPU is active, CPU workers are
  capped at `cpu_max_workers_when_gpu` (default 1) to prevent system-RAM
  thrash.

### 6.5 Concurrency
- **Threads by default on Windows.** `apply_low_spec_gpu_defaults()` sets
  `FORTUNA_LOW_MEMORY=1` so `BatchEvaluator` uses threads instead of
  processes (every Windows process re-imports NumPy/Numba/CuPy → 2× memory
  for two workers). `FORTUNA_ALLOW_PROCESS_POOL=1` restores the process
  pool on 16 GB+ machines.
- **Dashboard parallel runner** is a `ThreadPoolExecutor` — the NumPy
  runner releases the GIL inside its kernels enough that 6 workers
  consistently beat 1 on 8-core CPUs without doubling memory.

### 6.6 Dashboard streaming
- **No Streamlit reruns for live updates.** The chart polls a dedicated
  HTTP server; only the tiny "LIVE" pill is inside an `@st.fragment`. The
  Streamlit main script effectively runs once per user action.
- **15:30 IST right-edge anchor.** Both the chart spec
  (`_bars_to_session_close`) and the viewport helper
  (`_stretch_to_nse_close`) always pad the visible range to the session
  close so the chart never visually "stops" wherever the latest bar lags.
- **Live overlays.** `compute_live_overlays` re-runs only the indicator
  engine + compiler on the latest tail, returning the last ~20 non-NaN
  points per overlay — cheap enough to call inside every `/fortuna/bars`
  response.

### 6.7 Reporting
- **Single source of truth** (`compute_all_metrics`) shared by the
  dashboard, the institutional pipeline, the paper league, and the CLI
  scripts.
- **PNG-only static bundle.** After removing Plotly the report bundle is
  matplotlib PNGs (equity, drawdown, P&L distribution, monthly P&L heatmap)
  plus CSVs — small, fast, and easy to attach to emails / Slack.

---

## 7. Key engineering decisions

| Decision | Why |
|---|---|
| **Single Pydantic schema** for strategies | Validation, autocomplete, agent-friendly serialization, one obvious place to add new condition / indicator types. |
| **DSL + builtin engines (not DSL-only)** | Realistic intraday logic (pyramiding, ATR stops, square-off) doesn't fit a declarative tree; forcing it would either limit the strategies or balloon the DSL. |
| **NumPy runner is the default** | vectorbt is heavy (matplotlib, scipy, plotly transitively) and slow to import. The dashboard imports the runner on every cold start. |
| **Parquet + DuckDB, no Postgres** | Offline-first; the same files work across machines, no daemon to run, DuckDB pushes filters into Parquet metadata. |
| **TradingView chart over Streamlit-native** | `st.line_chart` / Plotly both auto-fit on every rerun and lose zoom state. Lightweight Charts v5 + custom HTML + HTTP poll preserves operator state across updates. |
| **Stream HTTP server (stdlib)** | Zero new dependencies, runs in-process as a daemon thread, CORS-permissive, one server per Streamlit process. |
| **GPU is optional, opt-in** | CuPy + CUDA is the heaviest dep in the stack. Phase 1 must run on a 4 GB GPU **or** CPU-only — neither path is privileged. The same GPU policy will later host RL training batches. |
| **Agent contract now, runtime later** | The agent interfaces (Research / Critic / Optimizer) are stable and consume the same data types as the deterministic stack. The ML / DL / RL runtime is intentionally not pinned in Phase 1 — it lands when training, replay buffers, and policy evaluation are ready to be wired against the existing backtest reward signal. |

---

## 8. Out of scope for Phase 1

These are **planned for later phases**, not permanently excluded. They are
held back from Phase 1 specifically because they need an honest reward
signal and a validated strategy space first — which is what Phase 1
delivers.

- **ML / DL feature learning.** No deep feature extractors or learned
  embeddings yet. Phase 1 uses hand-coded indicators so the reward signal
  fed to the next phase is interpretable.
- **Reinforcement-learning strategy generation.** The current adaptive
  learner (`paper/learner.py`) is a deterministic, bandit-style param
  refiner — it's the *seat* an RL policy will later occupy, not RL itself.
  No PPO / DQN / Actor-Critic, no replay buffers, no policy networks in
  Phase 1.
- **Autonomous decision / execution agents.** The `agents/` package
  defines the interface (`ResearchAgent`, `CriticAgent`, `OptimizerAgent`)
  but ships only stub implementations. No live order placement, no order
  management, no risk-throttled execution loop. SmartAPI usage stays
  **read-only** (historical + WebSocket quotes).
- **Multi-symbol portfolio orchestration.** The dashboard and engines work
  one symbol at a time. Portfolio-level position sizing, correlation, and
  capital allocation come later.
- **Distributed infrastructure.** No Redis, no Postgres, no Kafka. DuckDB
  + Parquet + threads is enough for single-node research; the RL phase may
  add a job queue, but Phase 1 does not.

---

## 9. Recent architectural changelog

### 2026-05-25 — Plotly removal & docs rewrite
- Dashboard switched fully to **Lightweight Charts v5** (`lightweight_chart.py`)
  with the stdlib HTTP stream server (`stream_server.py`).
- Removed Plotly figure builder (`build_tradingview_figure`), `ChartService`,
  `plot_tradingview_chart`, `plot_equity_interactive`, and every Plotly-only
  viewport helper (`plotly_nse_rangebreaks`, `viewport_x_range`,
  `viewport_y_range`, `oscillator_y_range`, `split_overlay_columns`,
  `filter_nse_session_ohlcv`, `chart_history_cap`, `is_oscillator_series`).
- Static report bundle is now **matplotlib PNGs only**.
- Dropped dependencies: `plotly`, `streamlit-autorefresh`,
  `streamlit-lightweight-charts-ntf`.
- Chart's right edge now always extends to **15:30 IST** regardless of how
  far the live feed has caught up (`_bars_to_session_close`,
  `_stretch_to_nse_close`).
- README + Phase_1.md fully rewritten against the current code.

### 2026-05-24 — Streamlit-first cleanup + intraday MMTS / RSI fixes
- Dashboard is the primary research UI; batch chart export to `logs/` is off
  by default.
- `rsi_scalp_5m` fix: entry was `RSI < 30 AND close > EMA9` (almost never
  true → 0 trades). Now `RSI < 30` with exit `RSI > 55`.
- MMTS now matches TradingView's published variants: trend filter uses
  `close vs ema_fast` by default, Bollinger Band confluence is optional,
  pyramiding (3× 0.5 size) is on by default for sell-side signal density.
- `compute_live_overlays` added — EMA / Bollinger / VWAP lines extend with
  the live feed instead of freezing at the last historical bar.

### 2026-05-23 — Live feed reliability
- SmartAPI session strips stray `Bearer ` JWT prefix (was causing
  "Invalid Token" mid-run).
- HTTP 403 from SmartAPI is now surfaced as `errorCode=RATE_LIMIT` rather
  than "Invalid Token"; historical fetcher does exponential backoff.
- `_normalize_index_to_naive_ist` applied at every cache/live boundary to
  prevent tz-naive ↔ tz-aware sort errors.

---

## 10. Verification

```powershell
cd c:\dev\Fortuna
uv sync --group dev --group dashboard --group charts --group smartapi
uv run pytest -q
```

Dashboard:

```powershell
$env:FORTUNA_CONFIG = "configs/intraday.yaml"
uv run python scripts/run_dashboard.py
```

Institutional walk-forward:

```powershell
uv run python scripts/run_institutional_backtest.py
```

Paper league (walk-forward + learning):

```powershell
uv run python scripts/run_paper_league.py `
  --symbol RELIANCE.NS --timeframe 5m `
  --strategies strategies/generated/ `
  --param-grid strategies/grids/ema_grid.json `
  --train-bars 156 --test-bars 78 --fold-step 78
```

All artifacts land under `logs/` (gitignored — see `logs/README.md`).

---

## 11. Where this is heading — ML / DL / RL + agent execution

Phase 1 is the foundation. The system we are actually building is an
**ML/DL + RL-driven strategy generator feeding autonomous trading
agents**, and every Phase 1 contract was shaped so that next layer can
plug in without rewriting the core.

Concretely, the deterministic surface shipped here maps onto the ML / RL
phase like this:

| Phase 1 contract | What the ML / RL phase plugs into it |
|---|---|
| `StrategyDefinition` (Pydantic) | RL **action space** — a policy emits a strategy/param tuple; the same JSON shape is consumed unchanged by the backtest. |
| `IndicatorEngine` + `compute/batch_indicators.py` | **Feature pipeline** for ML / DL models. Indicator cache means a policy that re-uses overlapping features pays for them once per training epoch. |
| `NumPyBacktestRunner` / builtin engines | **Environment step** for RL. Deterministic, fast, no vectorbt cold-import cost — drop-in `env.step(action) → (state, reward)`. |
| `StrategyMetrics` (perf + risk + intraday) | **Reward shaping inputs.** Sharpe, profit factor, drawdown penalty, expectancy — composed today by `StrategyScorer`, re-composable into any RL reward function tomorrow. |
| `paper/blackbox.py` walk-forward folds | **Train / eval split for RL.** Strict OOS isolation is already enforced; the same fold layout becomes train/eval episodes. |
| `paper/learner.py` (`AdaptiveLearner`) | **Bandit-style proto-optimizer.** The fold-history record (`FoldRecord`) is the shape an offline RL trainer / Bayesian optimizer will consume. Swap the narrowing heuristic for a learned policy with no schema change. |
| `agents/base.py` (`ResearchAgent` / `CriticAgent` / `OptimizerAgent`) | **Decision-agent seats.** RL or supervised inference agents implement these interfaces; the dashboard renders their decisions through the existing live-signal panel. |
| `LiveSessionBridge` + `stream_server.py` | **Real-time execution loop.** Today: read-only ticks → signals on the chart. Tomorrow: same pipeline, but an agent consumes the signal and emits an order intent. The Streamlit UI is already wired to render BUY / SELL / EXIT actions; only the broker adapter is missing. |
| `MarketCostModel` (brokerage + slippage + spread) | **Honest reward.** Costs are applied at the same `compute_all_metrics` boundary the RL reward will read — no "looks great in training, broke in live" because frictions were ignored. |

The deterministic core stays the **source of truth and the reward signal**.
ML / DL models learn features; RL agents learn policies; decision agents
execute. None of them replace the validation layer — they are validated
*by* it.
