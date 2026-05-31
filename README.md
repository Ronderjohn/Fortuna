# Fortuna

Institutional-grade autonomous quant research system for **NSE intraday equities**.
A deterministic, offline-first research and advisory stack built around a
Streamlit dashboard that mirrors TradingView's chart + Strategy Tester workflow,
with reproducible backtests, walk-forward validation, model promotion, optional
agentic advisory, paper learning, and downstream Telegram notifications.

Fortuna is **advisory-first**:
- deterministic strategy signals remain the transparent baseline,
- ML and RL participate only when artifacts are validated and loaded,
- agentic orchestration produces one final advisory decision per symbol,
- paper learning and notifications remain downstream of that decision,
- real-money auto-execution is still out of scope.

---

## What's inside

| Layer | Purpose | Key modules |
|---|---|---|
| **Strategy DSL** | Pydantic-validated JSON strategies + builtin engines (MMTS, ORB) | `strategy/`, `strategies/builtin/` |
| **Data layer** | SmartAPI / OpenChart / yfinance → Parquet cache → DuckDB analytics | `data/`, `data/sources/` |
| **Indicator engine** | EMA / SMA / RSI / ATR / VWAP / MACD / Bollinger / Volume SMA / rolling | `indicators/` |
| **Backtest engines** | NumPy fast path (default) + optional vectorbt + builtin Pine-parity engines | `backtesting/` |
| **Institutional pipeline** | Train/val/test split, NSE session rules, realistic costs, Monte Carlo | `backtesting/standard/` |
| **Strategy reporting** | TradingView-style report bundle (perf / risk / intraday / trades CSV) | `reporting/strategy_tester/` |
| **Search & arena** | Param-grid expansion + batch evaluator + leaderboards (CPU/GPU) | `search/`, `arena/`, `compute/` |
| **Paper league** | Walk-forward black-box folds + adaptive parameter learning | `paper/` |
| **Dashboard** | Streamlit + Lightweight Charts v5 (CDN embed) + stdlib HTTP stream server | `app/`, `app/streamlit_app.py` |
| **Live read-only feed** | SmartAPI WebSocket 2.0 → bar aggregator → in-memory + Parquet append | `data/sources/smartapi_live.py`, `app/live_session.py` |

The dashboard is the **primary** way to use Fortuna. Everything else (CLI scripts,
arena, institutional pipeline, paper league) writes the same artifact format the
dashboard already understands.

### Advisory stack

Fortuna now includes an optional **agentic advisory pipeline** on top of the
deterministic research stack:

- **ML signal scorer** — scores deterministic signal quality ([docs](docs/ml_signal_scorer.md))
- **RL policies** — per-symbol advisory votes after OOS validation ([docs](docs/rl_policy_workflow.md))
- **Agentic orchestrator** — combines evidence into one `AgentDecision` ([docs](docs/agentic_advisory.md))
- **Native conversational adapter** — free-form operator prompts normalized into typed advisory tools
- **Model promotion registry** — auditable live pointers for ML/RL artifacts
- **Telegram alerts** — deduped, throttled, downstream of final decisions only
- **Operator runbook** — [docs/operator_runbook.md](docs/operator_runbook.md)
- **Current system map** — [docs/current_system_flow.md](docs/current_system_flow.md)

Default mode remains **advisory**. Deterministic dashboard signals continue when
ML, RL, agentic, or Telegram subsystems are disabled or unavailable.

---

## Requirements

- Python **3.11+**
- Windows 11 / Linux / macOS (Windows-first; PowerShell examples)
- ~8 GB RAM is fine — Fortuna defaults turn off ProcessPool on Windows
- Optional GPU: GTX 1650 / 4 GB VRAM works (CuPy CUDA 12 by default; switch to `cupy-cuda11x` for CUDA 11)
- [uv](https://docs.astral.sh/uv/) for environment management

---

## Setup

```powershell
cd c:\dev\Fortuna
uv lock
uv sync
uv sync --group dev
copy .env.example .env
```

Optional groups (install only when needed):

| Group | What it adds | When |
|---|---|---|
| `dashboard` | `streamlit` | Always for the UI |
| `charts` | `matplotlib` | Static PNG bundles (equity, drawdown, monthly P&L heatmap) |
| `smartapi` | `smartapi-python`, `pyotp`, `websocket-client`, `logzero` | NSE historical + live feed |
| `nse` | `openchart` | Alternative free NSE 5m source |
| `vbt` | `vectorbt` | Slower vectorbt backtests (NumPy runner is default) |
| `gpu` | `cupy-cuda12x` | GPU batch indicators (param sweeps) |
| `notebook` | `jupyter` | Notebooks under `notebooks/` |

```powershell
uv sync --group smartapi --group charts --group dashboard
```

### `.env`

```env
SMARTAPI_API_KEY=...
SMARTAPI_CLIENT_CODE=...
SMARTAPI_PASSWORD=...           # your trading PIN
SMARTAPI_TOTP_SECRET=...        # base32 secret from Angel One TOTP setup
SMARTAPI_USE_LIVE_FEED=true     # set to true to enable WebSocket live bars
FORTUNA_INSECURE_SSL=false      # only true on Avast/AVG/corporate TLS-MITM boxes
```

---

## Operator quick start

```powershell
$env:FORTUNA_CONFIG = "configs/intraday.yaml"
uv run python scripts/run_operator_preflight.py
uv run python scripts/run_dashboard.py
```

Helpful next steps:
- Validate SmartAPI env more deeply: `uv run python scripts/test_smartapi_env.py`
- Run the Telegram assistant: `uv run python scripts/run_telegram_bot.py`
- Train RL: `uv run --group rl python scripts/run_rl_train.py --help`
- Train ML scorer: `uv run python scripts/train_ml_signal_scorer.py --help`
- Promote validated artifacts: `uv run python scripts/promote_model.py --help`

Minimal advisory + Telegram flags:

```env
FORTUNA_AGENTIC_ENABLED=1
FORTUNA_TELEGRAM_ENABLED=1
FORTUNA_TELEGRAM_BOT_TOKEN=...
FORTUNA_TELEGRAM_CHAT_ID=...
```

Optional native framework layer:

```env
FORTUNA_CONVERSATIONAL_ADAPTER_ENABLED=true
FORTUNA_CONVERSATIONAL_ADAPTER_MODE=heuristic
```

See [docs/telegram_trading_assistant.md](docs/telegram_trading_assistant.md)
for the Telegram bot workflow, supported commands, and examples for equities,
futures, and options.

---

## Quick start — the dashboard

```powershell
$env:FORTUNA_CONFIG = "configs/intraday.yaml"
uv run python scripts/run_dashboard.py
```

Or:

```powershell
streamlit run app/streamlit_app.py
```

What you get:

- **Full NSE symbol search** (Angel One `OpenAPIScripMaster`, ~2 000+ tickers, cached locally)
- **Timeframes:** 5m, 15m, 30m, 45m, 1h, 2h, 3h, 4h, 1D, 1W, 1M
  (resampled locally when SmartAPI has no native bar — 45m / 2h / 3h / 4h / 1W / 1M)
- **Parallel backtest** of every strategy under `strategies/intraday/`, `strategies/generated/`, `strategies/builtin/`
- **Lightweight Charts v5** TradingView-style chart, opening on the **winning strategy**
- **Live mode** (when SmartAPI is open): bars, BUY/SELL/EXIT markers, and indicator overlays
  update **in place** with no Streamlit reruns (via a stdlib HTTP stream server polled
  by the chart iframe at ~2.5 s)
- Tabs: **Chart**, **Leaderboard**, **Performance**, **Strategy detail**, **Models**,
  **Assistant**, and optional paper/execution monitoring surfaces

See [docs/smartapi_charts.md](docs/smartapi_charts.md) for the chart architecture.

---

## Strategy DSL

Strategies are JSON files validated by Pydantic (`strategy/schema.py`):

```jsonc
{
  "name": "ema_pullback_5m",
  "symbol": "ICICIBANK.NS",
  "timeframe": "5m",
  "side": "long",
  "indicators": [
    { "id": "ema_fast", "type": "ema", "params": { "window": 9 } },
    { "id": "ema_slow", "type": "ema", "params": { "window": 21 } }
  ],
  "rules": {
    "entry_conditions": [{
      "type": "and",
      "conditions": [
        { "type": "compare",   "left": "ema_fast", "operator": "gt", "right": "ema_slow" },
        { "type": "crossover", "left": "close",    "right": "ema_fast" }
      ]
    }],
    "exit_conditions": [
      { "type": "crossunder", "left": "close", "right": "ema_fast" }
    ]
  },
  "risk": {
    "stop_loss":      { "type": "percent", "value": 0.01 },
    "take_profit":    { "type": "percent", "value": 0.02 },
    "position_sizing":{ "type": "fixed_fraction", "value": 0.95 }
  }
}
```

Supported condition types: `compare`, `crossover`, `crossunder`, `and`, `or`, `not`.
Indicators supported: `ema`, `sma`, `rsi`, `atr`, `vwap`, `macd`, `bollinger`,
`volume_sma`, `rolling_high`, `rolling_low`.

Pine-parity engines for harder logic (pyramiding, ATR stops, square-off, opening range)
live under `strategies/builtin/` and are dispatched automatically when
`metadata.engine` is set:

- **MMTS** (`mmts`) — Measured Move Trend Strategy, long + short, pyramiding
- **ORB** (`orb`) — Opening Range Breakout, NSE intraday with square-off

---

## CLI scripts

All scripts live in `scripts/` and import via `uv run` or
`.\.venv\Scripts\python.exe`. Each apply the same low-spec defaults
(`apply_low_spec_gpu_defaults()`) — GPU on, ProcessPool off on Windows.

### One-off backtest

```powershell
uv run python scripts/run_backtest.py `
  --strategy strategies/generated/ema_crossover.json `
  --symbol AAPL --timeframe 1d
```

Routes the strategy to `strategies/validated/` or `strategies/rejected/` based on the
composite score (`evaluation/scorer.py`).

### Strategy arena (multi-strategy + param search)

```powershell
uv run python scripts/run_arena.py `
  --symbol RELIANCE.NS --timeframe 5m `
  --strategies strategies/generated/ `
  --param-grid strategies/grids/ema_grid.json `
  --max-candidates 50 --max-workers 2 `
  --rank-by profit_pct --mode full
```

- `--mode full`   — one window, best params per strategy
- `--mode stream` — rolling bar windows
- Output: `logs/arena/{run_id}/leaderboard.csv` + `all_variants.csv` + per-strategy JSON

### Per-bar tournament

```powershell
uv run python scripts/run_tournament.py `
  --symbol RELIANCE.NS --timeframe 5m `
  --strategies strategies/generated/ `
  --param-grid strategies/grids/ema_grid.json `
  --window-bars 78 --max-workers 2 --time-budget 30
```

Output: `logs/tournament/{run_id}/signals.csv` (timestamp, signal, winner, score).

### Institutional walk-forward pipeline

6+ months of data, train/val/test split, NSE session rules, realistic costs,
Monte Carlo trade-shuffle, automatic filter verdicts.

```powershell
uv run python scripts/run_institutional_backtest.py
```

Output: `logs/institutional/{run_id}/{strategy}/(train|validation|test)/` with
`report.json`, `trades.csv`, `signals.csv`, `filter_verdict.json`, `monte_carlo.json`,
and matplotlib PNG charts.

### Paper-trading league (walk-forward + learning)

Strategies paper-trade on **held-out OOS bars only** (black-box). Each fold:
**train** (search params) → **test** (paper trade) → **learn** (narrow grid).

```powershell
uv run python scripts/run_paper_league.py `
  --symbol RELIANCE.NS --timeframe 5m `
  --strategies strategies/generated/ `
  --param-grid strategies/grids/ema_grid.json `
  --train-bars 156 --test-bars 78 --fold-step 78 `
  --max-candidates 36 --rank-by profit_pct
```

Output: `logs/paper_league/{run_id}/oos_leaderboard.csv`, `all_oos_folds.csv`,
`learning/*_learning.json` (per-strategy adaptive state).
Use `--no-learning` to freeze grids.

### SmartAPI utilities

```powershell
.\.venv\Scripts\python.exe scripts\test_smartapi_env.py       # verbose .env check + sample candle fetch
.\.venv\Scripts\python.exe scripts\smartapi_check.py          # short login + instrument-master check
.\.venv\Scripts\python.exe scripts\smartapi_backfill.py --symbol RELIANCE.NS --timeframe 5m --days 30
.\.venv\Scripts\python.exe scripts\run_smartapi_live.py --symbol RELIANCE.NS --timeframe 5m
```

---

## Tests

```powershell
uv sync --group dev
uv run pytest -q
```

| Suite | Path |
|---|---|
| Strategy DSL / compiler / scorer / risk | `tests/test_*.py` |
| Data sources (SmartAPI, symbols, timeframes) | `tests/data/` |
| App / dashboard helpers (chart, stream server, live signals) | `tests/app/` |
| Reporting (strategy tester, TradingView charts) | `tests/reporting/` |
| Search / param expander / batch evaluator / indicator cache | `tests/search/` |
| Compute (batch indicators, GPU policy) | `tests/compute/` |
| Backtest (NumPy + institutional) | `tests/backtest/`, `tests/backtesting/` |
| Tournament / arena / paper | `tests/tournament/`, `tests/arena/`, `tests/paper/` |
| vectorbt (slow — opt-in) | `tests/backtest/test_vectorbt_engine.py` |
| Integration (network, real SmartAPI) | `tests/data/test_integration.py` |

```powershell
uv run pytest -q -m slow            # vectorbt
uv run pytest -q -m integration     # SmartAPI live
```

Or use `.\scripts\verify_env.ps1` for a clean run.

---

## Configuration

YAML overrides + environment variables (Pydantic `BaseSettings`):

- `configs/default.yaml`       — baseline (yfinance, daily)
- `configs/intraday.yaml`      — **NSE 5m intraday default** (SmartAPI, ICICIBANK.NS)
- `configs/institutional.yaml` — train/val/test pipeline defaults

Switch profiles with:

```powershell
$env:FORTUNA_CONFIG = "configs/intraday.yaml"
```

Any setting can also be overridden via `FORTUNA_*` env vars (e.g. `FORTUNA_INIT_CASH=200000`).
SmartAPI credentials live under `SMARTAPI_*` in `.env`.

---

## GPU compute (optional)

Optional CuPy CUDA path batches EMA / SMA / RSI / ATR across param-sweep candidates
(arena, tournament, paper league). Tuned for 4 GB VRAM by default.

```powershell
uv sync --group gpu
uv run python scripts/check_gpu.py
```

| Variable | Default | Purpose |
|---|---|---|
| `FORTUNA_USE_GPU`            | `1`    | Master switch |
| `FORTUNA_GPU_MEM_FRACTION`   | `0.75` | Cap VRAM usage (~3 GB on a 4 GB card) |
| `FORTUNA_GPU_MIN_BARS`       | `32`   | Min bars to bother with GPU |
| `FORTUNA_GPU_MIN_SERIES`     | `2`    | Min unique indicator series |
| `FORTUNA_LOW_MEMORY`         | `1`    | Disable ProcessPool on Windows (set automatically by scripts) |
| `FORTUNA_CPU_WORKERS_GPU`    | `1`    | CPU processes while GPU is busy |

CUDA 11.x: switch `cupy-cuda12x` to `cupy-cuda11x` in `pyproject.toml`.

---

## Troubleshooting (8 GB Windows)

| Symptom | Likely cause | Fix |
|---|---|---|
| Exit code `4294967295` / silent kill | OOM (ProcessPool spawning multiple CuPy/NumPy/Numba processes) | Scripts default to **no ProcessPool on Windows**; only enable with `FORTUNA_ALLOW_PROCESS_POOL=1` on 16 GB+ machines |
| SmartAPI "Invalid Token" mid-run | Rate limit — SDK returns HTTP 403 as "Invalid Token" | Already handled: `smartapi_historical.py` does exponential backoff |
| Chart axis stops at 15:00 | Live feed lagged a few bars | Right edge is anchored to 15:30 IST regardless (`_bars_to_session_close`) |
| Chart re-renders / loses zoom | `streamlit_autorefresh` would force reruns | Removed — chart polls a stdlib HTTP server and applies `series.update()` in-place |
| Tz-naive vs tz-aware error | Mixing cache (naive IST) with live bars (Asia/Kolkata) | Normalized via `_normalize_index_to_naive_ist` everywhere |
| SSL UnknownIssuer | Avast/AVG/corporate proxy doing TLS MITM | Set `FORTUNA_INSECURE_SSL=true` (loaded *before* any HTTP client) |

`[TIMING]` log lines show whether time is spent in `data.fetch` vs `eval.*`.

---

## Project layout

```
fortuna/
├── app/
│   └── streamlit_app.py                 # Dashboard entry point
├── configs/
│   ├── default.yaml
│   ├── intraday.yaml                    # NSE 5m default (FORTUNA_CONFIG points here)
│   └── institutional.yaml
├── docs/
│   ├── architecture.md  smartapi_charts.md  institutional_backtest.md
│   └── first_run.md  strategy_walkthrough.md  agents.md  workspace.md
├── scripts/                             # run_dashboard / run_arena / run_paper_league / ...
├── src/fortuna/
│   ├── agents/                          # ResearchAgent / CriticAgent / OptimizerAgent stubs
│   ├── app/                             # Streamlit session engine, chart, stream server
│   ├── arena/                           # Multi-strategy parallel competitions
│   ├── backtesting/                     # NumPy + vectorbt + institutional pipeline
│   ├── compute/                         # CPU/GPU policy, batch indicator scheduler
│   ├── config/                          # Pydantic settings + SmartAPI settings
│   ├── data/                            # Manager, cache, DuckDB, sources/, timeframes
│   ├── evaluation/                      # Composite scorer + ranker
│   ├── indicators/                      # Registry + Numba kernels
│   ├── paper/                           # Paper league, walk-forward, adaptive learner
│   ├── quant/                           # Black-Scholes (theoretical reference only)
│   ├── reporting/                       # TradingView Strategy Tester analytics
│   ├── search/                          # Param expander + batch evaluator
│   ├── strategies/builtin/              # MMTS, ORB engines + dispatcher
│   ├── strategy/                        # Pydantic schema + loader + warmup helper
│   ├── tournament/                      # Per-bar winner selection
│   └── utils/                           # Logging, runtime env, SSL, timing
├── strategies/
│   ├── builtin/        mmts.json
│   ├── intraday/       ema_pullback_5m.json  macd_trend_5m.json  rsi_scalp_5m.json  ...
│   ├── generated/      ema_crossover.json  vwap_reclaim.json  rsi_mean_reversion.json
│   └── grids/          ema_grid.json
└── tests/                               # pytest suites mirroring src/
```

---

## Phase 1 scope

**Included.** Strategy DSL · data layer · indicator engine · NumPy + vectorbt backtests ·
institutional walk-forward + Monte Carlo · TradingView-style reporting · Streamlit
dashboard with Lightweight Charts v5 + in-place streaming · adaptive paper-league ·
agentic advisory orchestrator · ML signal scorer · RL training/inference · model
promotion registry · Telegram notification hardening (advisory only).

**Still planned / not built.** Live broker order placement · multi-symbol portfolio
orchestration · distributed training infrastructure · dedicated ML training CLI.

The Phase 1 deliverable is the deterministic foundation; Phases 1–6 add the advisory
ensemble that operators use via the dashboard and optional Telegram.

See [Phase_1.md](Phase_1.md) for the original architecture vision and
[docs/operator_runbook.md](docs/operator_runbook.md) for day-to-day operations.

---

## License

MIT
