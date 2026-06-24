# Fortuna

Institutional-grade autonomous quant research system for **NSE intraday equities**.
A deterministic, advisory-first stack built around typed analysis services,
Telegram-first signal delivery, and a slim Streamlit operator console, with
reproducible backtests, walk-forward validation, model promotion, optional
agentic advisory, paper learning, and auditable downstream notifications.

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
| **Operator console** | Streamlit + Lightweight Charts v5 (CDN embed) + stdlib HTTP stream server | `app/`, `app/streamlit_app.py` |
| **Live read-only feed** | SmartAPI WebSocket 2.0 → bar aggregator → in-memory + Parquet append | `data/sources/smartapi_live.py`, `app/live_session.py` |

Telegram is now the **primary** end-user surface for Fortuna. The dashboard
remains as an operator/admin console over the same typed runtime and artifacts.

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
- **How to use Fortuna** — [docs/how_to_use_fortuna.md](docs/how_to_use_fortuna.md)
- **Current system map** — [docs/current_system_flow.md](docs/current_system_flow.md)
- **Telegram quickstart** — [docs/telegram_bot_quickstart.md](docs/telegram_bot_quickstart.md)
- **Operator runbook** — [docs/operator_runbook.md](docs/operator_runbook.md)
- **How to use Fortuna** — [docs/how_to_use_fortuna.md](docs/how_to_use_fortuna.md)

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
- Start with the practical user guide: [docs/how_to_use_fortuna.md](docs/how_to_use_fortuna.md)
- Validate SmartAPI env more deeply: `uv run python scripts/test_smartapi_env.py`
- Run the Telegram assistant: `uv run python scripts/run_telegram_bot.py`
- **Operator loop (advisory):** universe → shortlist → briefing → allocation →
  training research → nightly → promotion → acceptance review — see
  [docs/operator_runbook.md](docs/operator_runbook.md) and
  [docs/current_system_flow.md](docs/current_system_flow.md)
- Build a liquid market universe: `uv run python scripts/build_market_universe.py --help`
- Analyze the top ranked shortlist: `uv run python scripts/analyze_market_shortlist.py --help`
- Prepare shortlist-driven training candidates: `uv run python scripts/prepare_training_candidates.py --help`
- Build a typed ML/RL training research plan: `uv run python scripts/build_training_research_plan.py --help`
- Brief the top shortlisted setups: `uv run python scripts/brief_market_shortlist.py --help`
- Allocate shortlisted setups under simple portfolio limits: `uv run python scripts/allocate_market_shortlist.py --help`
- Export a full workflow snapshot artifact: `uv run python scripts/build_workflow_snapshot.py --help`
- Export a higher-level acceptance bundle artifact:
  `uv run python scripts/build_acceptance_bundle.py --help`
- Rehearse the candidate-style nightly acceptance path locally:
  `uv run python scripts/run_nightly_acceptance_dry_run.py --help`
- Train RL: `uv run --group rl python scripts/run_rl_train.py --help`
- Train ML scorer: `uv run --group ml python scripts/train_ml_signal_scorer.py --help`
- Promote validated artifacts: `uv run python scripts/promote_model.py --help`
- Promotion CLIs can optionally carry `--workflow-snapshot <path>` so the live
  pointer and registry audit reference the same shortlist/briefing/candidate
  artifact used during review
- Review the promoted artifact and linked workflow context:
  `uv run python scripts/review_promotion.py --help`
- Promotion review can also export markdown or JSON artifacts for later audit:
  `uv run python scripts/review_promotion.py --kind rl_policy --out reports/promotion_review.md`

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

Optional OpenAI planner mode above the same typed tools:

```env
FORTUNA_CONVERSATIONAL_ADAPTER_ENABLED=true
FORTUNA_CONVERSATIONAL_ADAPTER_MODE=openai
FORTUNA_OPENAI_API_KEY=...
FORTUNA_OPENAI_MODEL=gpt-5.4-mini
```

See [docs/telegram_trading_assistant.md](docs/telegram_trading_assistant.md)
for the Telegram bot workflow, supported commands, and examples for equities,
futures, options, market-universe scans, shortlist briefings, allocation views,
and direct training-research inspection.

For the current operator flow and product posture, use:

- [docs/how_to_use_fortuna.md](docs/how_to_use_fortuna.md)
- [docs/operator_runbook.md](docs/operator_runbook.md)
- [docs/telegram_bot_quickstart.md](docs/telegram_bot_quickstart.md)
- [docs/current_system_flow.md](docs/current_system_flow.md)

For future OpenAI-backed agent work, Fortuna now also exposes a typed
market-universe surface that can shortlist liquid, trend-relevant symbols from
an optional Screener CSV export plus local OHLCV ranking, a shortlist
analysis surface that can critique top candidates before deeper review, and a
training-candidate surface that can suggest which symbols deserve ML/RL data
refresh and model-prep attention. The same typed layer also includes shortlist
briefings so Telegram or an OpenAI planner can summarize the top setups
without bypassing the advisory core. It now also includes an explicit typed
multi-agent workflow surface, so local callers can request one composed
universe/critic/briefing/allocation/research pass instead of stitching those
roles together by hand. Telegram `/workflow` can now use that same explicit
team surface directly instead of manually composing each stage itself. The
dashboard workflow console now also refreshes through that same typed team
response, so the operator surfaces share one multi-agent backbone.

That market-universe ranking can now also apply a light adaptive weighting from
recent paper-closed outcomes, so symbols with stronger recent realized advisory
quality can be nudged up while recently weak exposures are nudged down. This is
kept intentionally small and fail-soft; liquidity and trend remain the primary drivers.
The ranked universe now also carries a typed market-context snapshot
(`regime`, `volume_ratio`, `activity_score`) so shortlist and training flows can
see more than raw liquidity alone.
The adaptive universe policy is now a bit richer too: it blends symbol memory
with smaller action-level, regime-level, and global recency-weighted outcome
biases, so the watchlist can reflect both "this name has worked lately" and
"this directional/context regime has worked lately" without becoming opaque.
The universe stage can now also absorb light recurring evidence from recent
candidate-driven nightly reports, so symbols that keep surviving nightly
training-candidate selection and promotion can earn a bounded discovery boost.
Decision-time learning rows now also preserve a compact signal-context bundle
(`primary_signal`, signal counts, regime confidence), so downstream training
candidate ranking can start learning from which setup family has actually been
working, not only from symbol-level outcomes.
Training-candidate manifests now also preserve setup-family/adaptive-score
context such as `winning_strategy` and `adaptive_score_adjustment`, which makes
the ML/RL refresh queue easier to audit and compare across nightly runs.
Training-candidate ranking can now also learn lightly from recent successful
candidate-driven nightly reports, so symbols that keep reappearing in recurring
ML/RL prep evidence can be nudged upward in a bounded, auditable way.
There is now also a typed training-research plan layer that can turn those
shortlist-driven ML/RL candidates into a concrete refresh list, optionally
backfilling the selected symbols immediately for model-prep workflows.
That research-plan artifact is no longer only for review: recent nightly
`training_research_plan.json` evidence can now lightly bias both universe
ranking and shortlist-derived training-candidate ranking, so recurring refresh
intent starts feeding back into future discovery and model-prep selection.
The shortlist allocator is also a little more portfolio-shaped now: beyond
simple caps, it can apply a bounded portfolio critic that penalizes or skips
redundant weak setups when the basket is getting too same-side, too
same-regime, too watch-grade, or too high-risk. There is now also an optional
research-alignment pass that can compare the shortlist basket to the current
typed ML/RL training-research plan and push back on setups that are drifting
away from the names the research loop is currently reinforcing. That alignment
logic can now also distinguish refreshed ML/RL prep evidence from plan-only
support, so same-side baskets can lightly prefer setups backed by actually
refreshed training-research rows.

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
- The **Assistant** tab now includes quick actions for liquid-universe scans,
  shortlist briefings, allocation views, ML/RL training-candidate selection,
  direct training-research inspection, and direct symbol analysis
- The **Assistant** tab also includes a small workflow console with current
  universe, shortlist, allocation, training-candidate, and training-research
  tables for faster operator review
- A typed portfolio allocator now exists above shortlist briefing so the stack
  can turn ranked setups into a constrained basket under simple exposure limits
- That allocator now also considers current open positions and recent paper-closed
  learning outcomes, so basket selection is less likely to ignore live overlap
  or keep favoring recently weak exposures
- The allocator now also carries typed regime/risk metadata and applies a small,
  flag-gated context weighting pass, so selected baskets can expose
  `allocation_score`, `risk_bucket`, and `sizing_hint` instead of only raw weight
- The allocator now also supports bounded regime and high-risk caps, which helps
  the shortlist-to-basket step avoid quietly concentrating in one market regime
  or stacking too many weak-quality setups
- Shortlist briefings now warn on directional crowding, overlapping underlyings,
  and overlap with already-open positions when account state is available
- Training-candidate ranking now also applies light exposure penalties so ML/RL
  candidate manifests prefer less crowded names when multiple setups overlap
- Training-candidate ranking can now also apply a light recent-outcome bias, so
  symbols with stronger recent paper-closed advisory quality can move up the
  ML/RL prep queue while recently weak exposures are nudged down
- Training candidates now retain typed market-context fields from the universe
  stage, so ML/RL prep can distinguish trending, ranging, and more active names
- Candidate-driven nightly runs can now opt into a diversified training basket
  policy with `--candidate-selection-policy diversified`, which rotates across
  action/regime buckets instead of only taking the top-ranked names in order
- Shortlist briefings now share the same exposure-aware prioritization idea, so
  the top-setups view and the training-candidate view are more consistent
- Shortlist analysis itself now carries selection rank / exposure-penalty
  metadata, so the ranking story is consistent from shortlist to briefing to training
- The dashboard workflow console now exposes the pipeline stage by stage:
  liquid universe, ranked shortlist, curated briefing, constrained allocation,
  and final training candidates
- Candidate-driven nightly runs can now emit the same workflow snapshot artifact
  automatically, and they now emit the resolved `training_candidates.json`
  artifact too, so training review and operator review can reference one report
- That workflow snapshot now carries allocation-stage evidence too, so review
  can see not just the shortlist and candidates, but also which setups were
  actually selected under portfolio constraints
- Workflow snapshots can now also preserve allocation research-alignment notes,
  so repeated reviews can see whether the shortlist basket stayed aligned with
  the current typed ML/RL training-research plan
- Workflow snapshots now also preserve training-research refresh evidence, so
  in-app remediation runs can show how many ML/RL prep rows were refreshed
  instead of leaving that action only in transient dashboard state
- Workflow snapshots and promotion/model-review summaries can now also surface
  the top adaptive universe note, so ranking shifts from recent paper outcomes
  are visible instead of implicit
- Nightly promotions now carry that workflow snapshot path into the live pointer
  payload and `models/registry/promotions.jsonl` audit trail
- The dashboard **Models** review surface now shows that linked workflow
  snapshot path and a compact workflow summary alongside the latest promotion
  metadata
- Nightly can also emit archived promotion-review artifacts under the report
  directory, so training, promotion, and review stay bundled together
- There is now a higher-level acceptance bundle path that can tie workflow
  snapshot, promotion review, and model-health evidence into one reviewable
  markdown or JSON artifact for operator sign-off
- That acceptance bundle can now also read the latest nightly JSON report
  directly, so recurring training/promotion artifacts can flow into one
  acceptance review without manually restitching the evidence
- For candidate-driven nightly runs, the acceptance bundle now also checks for
  expected basket/training-candidate/workflow/promotion step coverage, and it
  validates that the referenced `training_candidates.json` artifact exists, so
  the report structure itself becomes part of the audit trail
- The acceptance bundle can now also surface whether allocation
  research-alignment evidence was visible in the workflow snapshot, which makes
  the portfolio basket / ML-RL prep relationship easier to audit over time
- The acceptance bundle can now also warn when a workflow snapshot requested a
  training-research refresh but did not actually refresh any rows, which makes
  remediation attempts part of the audit trail too
- Candidate-driven nightly markdown reports can now surface that same
  allocation research-alignment summary directly in the workflow section, so
  repeated runs expose basket / research coherence without needing a separate
  acceptance pass first
- Model-health summaries can now also scan recent nightly reports and show a
  compact allocation-research alignment trend, so operators can spot recent
  coherence or drift without opening each nightly artifact by hand
- Acceptance bundles can now warn when that recent nightly alignment trend
  drops below a simple configured ratio, which makes drift show up during
  higher-level review instead of only in raw model-health data
- Promotion review output can now surface that same recent nightly alignment
  warning, so model review shows both the linked workflow snapshot and recent
  basket/research drift context in one place
- Acceptance and promotion review can now also warn when the current workflow
  snapshot shows weak basket/research overlap, so a selected basket drifting
  away from the current ML/RL prep plan shows up during review
- Acceptance-bundle and promotion-review JSON exports now also persist the typed
  remediation posture directly: recommended refresh target, force-refresh mode,
  and the suggested CLI command
- The dashboard **Models** surface can now show the same recent nightly
  alignment trend and drift warning, so operators can spot coherence issues
  during normal runtime review without exporting reports
- When that recent nightly alignment drops below a configurable threshold, the
  model-health surface can now suggest a concrete next step: refresh the
  training research plan and review candidate-selection policy before the next
  promotion or nightly cycle
- That recommendation now also carries a concrete typed remediation command,
  centered on `scripts/build_training_research_plan.py`, so the operator can
  move from warning to action with less guesswork
- The dashboard workflow console can now load the typed training-research plan
  directly, including a dedicated refresh action and research tab, so drift
  remediation is closer to the normal operator loop
- That in-app training-research remediation path can now optionally refresh
  targeted ML/RL data too, so the operator can move from drift warning to
  updated prep evidence in one bounded workflow
- Promotion review and nightly markdown reports now also surface that
  training-research remediation evidence more explicitly, so refresh attempts
  stay visible in recurring review paths and not only in acceptance exports
- Recent nightly feedback now also distinguishes between planned
  training-research rows and rows that actually refreshed ML/RL prep data, so
  future universe ranking and training-candidate selection can weight executed
  refresh evidence more strongly than paper plans alone
- Workflow snapshots, nightly summaries, and model-health alignment trends can
  now also carry refreshed allocator research-target mix, so operators can see
  whether the actual selected basket is staying aligned with recently refreshed
  ML/RL prep evidence over time
- Acceptance and promotion review can now warn not just on general
  basket/research drift, but also on weak refreshed-alignment coverage, so
  repeated divergence from recently refreshed prep evidence shows up more
  directly in the multi-agent review loop
- The allocator itself can now also lean on recent nightly training-research
  execution evidence, so same-side baskets can penalize weaker setups when the
  current side already has stronger recurring refreshed/promoted research support
- Model-health remediation suggestions can now bias the recommended
  training-research refresh target toward `ml`, `rl`, or `all` from the recent
  refreshed-alignment mix, so the next-step command is a little more specific
  to the actual drift pattern
- The dashboard workflow console now follows that same guidance by preselecting
  the in-app training-research refresh target from the suggested remediation
  command until the operator deliberately overrides it
- When refreshed-alignment drift is severe, the same workflow console can now
  also prefill `Refresh research data` and, for the harshest cases,
  `Force research refresh`, so the in-app remediation path is closer to the
  suggested recovery flow by default
- There is now a bounded local dry-run harness for that same path, so we can
  generate a synthetic candidate-nightly evidence stack and verify the
  acceptance flow without touching live data or real training jobs
- That dry-run now also emits its nightly JSON/markdown report through the
  actual nightly report writer, so the rehearsal exercises more of the real
  reporting path than a hand-written fixture alone
- Candidate-driven nightly runs now also emit a typed
  `training_research_plan.json` artifact, so recurring ML/RL refresh intent is
  preserved alongside `training_candidates.json` and `workflow_snapshot.json`
- That dry-run now also includes model-health visibility through the same
  acceptance bundle, so the rehearsal covers registry/promotion health too

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

See [docs/operator_runbook.md](docs/operator_runbook.md) for day-to-day
operations.

---

## License

MIT
