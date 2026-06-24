# How To Use Fortuna

Practical user guide for running Fortuna as a **Telegram-first NSE intraday
signal application** with a slim operator console behind it.

This guide is for the normal operator workflow:

- set up the environment
- start the Telegram bot
- get compact signal replies for symbols and contracts
- use the dashboard as the operator/admin console
- run deeper research and workflow commands only when needed

Fortuna is **not** a live auto-trading application. Real-money execution is out
of scope.

## 1. What Fortuna does

Fortuna combines several layers into one operator workflow:

- deterministic strategy signals
- optional ML signal scoring
- optional RL advisory votes
- multi-agent advisory orchestration
- market-universe ranking and shortlist analysis
- training candidate and research planning
- nightly review, promotion review, and acceptance artifacts

The main user interface is the **Telegram bot**. The Streamlit dashboard is the
operator/admin console.

## 2. Before you start

Requirements:

- Python 3.11+
- `uv`
- local project checkout at `C:\dev\Fortuna`

Initial setup:

```powershell
cd C:\dev\Fortuna
uv lock
uv sync
uv sync --group dev
copy .env.example .env
```

Recommended optional groups for normal usage:

```powershell
uv sync --group dashboard --group charts --group smartapi
```

## 3. Configure `.env`

For SmartAPI-backed cache/live usage, fill the SmartAPI fields in `.env`:

```env
SMARTAPI_API_KEY=...
SMARTAPI_CLIENT_CODE=...
SMARTAPI_PASSWORD=...
SMARTAPI_TOTP_SECRET=...
SMARTAPI_USE_LIVE_FEED=true
FORTUNA_INSECURE_SSL=false
```

Useful safe defaults:

```env
FORTUNA_AGENTIC_ENABLED=1
FORTUNA_EXECUTION_ENABLED=0
FORTUNA_MODEL_REGISTRY_ENABLED=1
FORTUNA_MODEL_PROMOTION_REQUIRED=1
```

Telegram-first setup:

```env
FORTUNA_TELEGRAM_ENABLED=1
FORTUNA_TELEGRAM_BOT_TOKEN=...
FORTUNA_TELEGRAM_ALLOWED_CHAT_IDS=123456789,987654321
FORTUNA_TELEGRAM_ADMIN_CHAT_IDS=123456789
FORTUNA_SIGNAL_REPLY_STYLE=compact
FORTUNA_SIGNAL_RL_MODE=warm
```

## 4. First run

Run the operator preflight first:

```powershell
uv run python scripts/run_operator_preflight.py
```

Then start the Telegram bot:

```powershell
uv run python scripts/run_telegram_bot.py
```

Start the dashboard separately when you want the operator console:

```powershell
uv run python scripts/run_dashboard.py
```

If SmartAPI credentials need a deeper check:

```powershell
uv run python scripts/test_smartapi_env.py
```

On a fresh setup, warnings like `ml_scorer: disabled`, `no per-symbol live
pointer`, or typed lines such as `Runtime posture: deterministic_only` and
`Nightly posture: manual_only` are acceptable. They mean Fortuna will still run,
but optional model-backed advisory layers are not fully promoted yet.

## 5. How to use the application day to day

### A. Telegram signal workflow

Use Telegram for the normal user flow:

- `/search RELIANCE`
- `/analyze RELIANCE`
- `/analyze RELIANCE 15m 20d`
- `/analyze RELIANCE FUT`
- `/analyze NIFTY CE 25000 28MAY2026`
- natural-language follow-up such as `What about 15m?`

Default replies are intentionally short:

- verdict
- confidence
- short why
- risk notes when present
- next action

### B. Dashboard session

Use the dashboard for operator/admin work:

- request audit and health checks
- promoted model readiness
- universe, shortlist, allocation, and workflow review
- artifact inspection and maintenance
- troubleshooting when Telegram surfaces degraded replies

### C. Build the research universe

Rank a liquid market universe:

```powershell
uv run python scripts/build_market_universe.py --source auto --limit 15
```

SmartAPI-native bounded discovery:

```powershell
uv run python scripts/build_market_universe.py --source smartapi --limit 15
```

Use this when you want a liquid shortlist seed for the rest of the workflow.

Optional fundamentals/reference overlay (local CSV, no runtime scraping):

1. Copy `data/universe/fundamentals_overlay.example.csv` to `data/universe/fundamentals_overlay.csv`
2. Set `FORTUNA_MARKET_UNIVERSE_FUNDAMENTALS_OVERLAY_ENABLED=1`
3. Edit rows with `symbol`, optional `sector`/`industry`, `market_cap_bucket`, `operator_quality_score` (0–1), `operator_exclude`, and `note`

The overlay applies a bounded additive liquidity bump (default max ±0.05) and surfaces metadata on universe and shortlist responses. When disabled or missing, SmartAPI/screener behavior is unchanged.

### Scaling posture and nightly basket size

Fortuna surfaces machine compute budget separately from ML/RL advisory readiness:

| `basket_posture` | Meaning |
|------------------|---------|
| `medium_ok` | Basket within recommended tier for this machine |
| `small_only` | Low-RAM machine — keep basket ≤ `FORTUNA_SCALING_BASKET_SMALL_MAX` (default 8) |
| `large_not_recommended` | Basket above medium tier max (`FORTUNA_SCALING_BASKET_MEDIUM_MAX`, default 15) |

Check scaling on operator preflight, the dashboard **Models** tab, and nightly reports (`## Scaling posture`).

| Setting | Default | Purpose |
|---------|---------|---------|
| `FORTUNA_SCALING_BASKET_SMALL_MAX` | `8` | Upper bound for small basket tier |
| `FORTUNA_SCALING_BASKET_MEDIUM_MAX` | `15` | Upper bound for medium tier |
| `FORTUNA_SCALING_ENFORCE_BASKET_CAP` | `false` | Block nightly when basket exceeds recommended max |
| `FORTUNA_NIGHTLY_REPORT_RETENTION_ENABLED` | `false` | Prune old `reports/nightly/*.json` (+ paired `.md`) |
| `FORTUNA_NIGHTLY_REPORT_RETENTION_COUNT` | `30` | Keep newest N reports when retention enabled |

### Workflow observability (optional)

Fortuna can emit typed workflow boundary events to a local JSONL file for cross-module debugging:

| Setting | Default | Purpose |
|---------|---------|---------|
| `FORTUNA_OBSERVABILITY_ENABLED` | `false` | Master gate for workflow event capture |
| `FORTUNA_OBSERVABILITY_LOG_DIR` | `logs/observability` | Local JSONL directory |
| `FORTUNA_OBSERVABILITY_EXPORTER` | `local` | `local`, `otlp`, `both`, or `none` |
| `FORTUNA_OBSERVABILITY_OTLP_ENDPOINT` | empty | Optional OTLP HTTP endpoint (`uv sync --group otel`) |

When enabled, review `logs/observability/workflow_events.jsonl` for operator preflight, universe/candidate/research, nightly train steps, promotion review, and Telegram routing events. Observability failures are fail-soft and never change advisory behavior.

### D. Analyze and brief the shortlist

Analyze the top-ranked names:

```powershell
uv run python scripts/analyze_market_shortlist.py --analysis-limit 5
```

Generate a briefing:

```powershell
uv run python scripts/brief_market_shortlist.py --analysis-limit 5
```

Allocate the shortlist under simple portfolio constraints:

```powershell
uv run python scripts/allocate_market_shortlist.py --max-positions 3 --max-per-exposure 1
```

This is the normal research flow before deeper review or nightly prep.

### E. Prepare training and workflow artifacts

Export shortlist-driven training candidates:

```powershell
uv run python scripts/prepare_training_candidates.py --backfill --force-refresh
```

Build the ML/RL training research plan:

```powershell
uv run python scripts/build_training_research_plan.py --source auto --refresh-target rl
```

Export a full workflow snapshot:

```powershell
uv run python scripts/build_workflow_snapshot.py --source auto --timeframe 5m --days 20
```

These artifacts are useful for nightly review, promotion review, and audit.

## 6. Nightly and review workflow

### Local acceptance rehearsal

Run the local rehearsal path:

```powershell
uv run python scripts/run_nightly_acceptance_dry_run.py
```

Bounded emit-replay path:

```powershell
uv run python scripts/run_nightly_acceptance_dry_run.py --mode emit-replay
```

Verify an existing nightly report directory:

```powershell
uv run python scripts/run_nightly_acceptance_dry_run.py --mode verify --report-dir reports/nightly
```

Recommended sequence:

1. `candidate`
2. `emit-replay`
3. `verify`

The replay output now includes typed diagnostics such as replay status, run id,
missing artifacts, broken links, and summary text.

If `verify` is pointed at an older nightly directory, a broken replay-linkage
result can be expected when that nightly output predates the newer
training-candidate, training-research, and workflow-snapshot artifact chain.

### Acceptance bundle

Build the higher-level acceptance review:

```powershell
uv run python scripts/build_acceptance_bundle.py --workflow-snapshot reports/workflow_snapshot.json
```

Or point it at nightly output:

```powershell
uv run python scripts/build_acceptance_bundle.py --nightly-report-dir reports/nightly
```

Use this to review whether workflow, nightly, promotion, and model-health
artifacts agree with each other.

### Promotion review

Review the promoted model and linked workflow context:

```powershell
uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS
```

Export a promotion review artifact:

```powershell
uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS --out reports/promotion_review.md --format md
```

Check ML/RL activation stage without a promoted audit row:

```powershell
uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS --activation-only
uv run python scripts/review_promotion.py --kind ml_scorer --activation-only
```

### ML/RL activation stages

Each lane (ML scorer, RL policy) reports one typed stage:

| Stage | Meaning | Typical next step |
|-------|---------|-------------------|
| `disabled` | Subsystem off | Enable `FORTUNA_AGENTIC_*` flags |
| `missing_artifact` | No validated candidate on disk | Run train script |
| `unpromoted` | Validated artifact exists; live pointer missing | `promote_model.py` / `promote_policy.py` |
| `promoted_not_advisory_ready` | Pointer exists but metadata not advisory-ready | Re-train or review rejected metrics |
| `not_loaded` | Advisory-ready artifact exists; session not loaded | Reload Models tab / restart session |
| `active` | Loaded and advisory-ready | Monitor only |

Stages appear in operator preflight, the **Models** tab, `ModelHealthResponse.activation`,
and promotion review output.

## 7. Telegram usage

Start the Telegram assistant:

```powershell
uv run python scripts/run_telegram_bot.py
```

Telegram is downstream of final advisory decisions. It is useful for:

- search and instrument lookups
- briefings and workflow inspection
- operator-friendly summaries

Telegram should be treated as an operator interface, not an execution engine.

## 8. Paper-learning and models

Paper-learning is optional. Enable it only when you want paper router +
learning rows:

```env
FORTUNA_AGENTIC_PAPER_LEARNING_ENABLED=1
```

Nightly training help:

```powershell
uv run --group rl python scripts/nightly_train.py --help
```

If the Windows scheduled nightly task has been disabled after repeated partial
or failed runs, leave it disabled until ML/RL runtime readiness is verified and
nightly output is consistently useful. Nightly reports now include a typed
`lane_readiness` step with per-lane disposition (`run`, `skip`, `block`) and
`skip_class` (`policy`, `missing_prerequisites`, `global_blocker`). Overall
status may be `ok`, `skipped`, `blocked`, or `partial (...)` — not only `ok`
when expensive lanes were intentionally skipped.

ML scorer training help:

```powershell
uv run --group ml python scripts/train_ml_signal_scorer.py --help
```

Promotion help:

```powershell
uv run python scripts/promote_model.py --help
```

## 9. Safe operating posture

Recommended normal posture:

- keep execution disabled
- use advisory decisions as decision support
- treat deterministic signals as the baseline
- treat ML/RL/agentic layers as additive, not mandatory
- use workflow snapshots, acceptance bundles, and promotion reviews for audit

## 10. Troubleshooting

### Dashboard starts but no ML/RL advice appears

Usually means:

- no promoted live pointer exists
- registry is enabled and artifacts are not promoted
- model is not advisory-ready

Deterministic signals should still work.

### SmartAPI issues

If credentials are missing or invalid:

- cached/offline data may still work
- live feed and some SmartAPI-backed flows will not

Check:

```powershell
uv run python scripts/test_smartapi_env.py
```

### Telegram not sending

Check:

- `FORTUNA_TELEGRAM_ENABLED=1`
- bot token and chat id are present
- notification logs under `logs/agentic/notifications.jsonl`

### Replay or acceptance warnings

Use:

```powershell
uv run python scripts/run_nightly_acceptance_dry_run.py --mode verify --report-dir reports/nightly
```

Look at:

- missing artifacts
- broken linkage
- replay summary

## 11. Recommended daily workflow

For a normal operator day:

1. Run preflight
2. Start dashboard
3. Review live signals and advisory decisions
4. Refresh market universe when needed
5. Analyze shortlist and read briefing
6. Export workflow snapshot if doing a formal review
7. Run nightly and acceptance review after market close
8. Review promotion artifacts before trusting new model pointers

## 12. Related docs

- [README.md](../README.md)
- [operator_runbook.md](operator_runbook.md)
- [telegram_trading_assistant.md](telegram_trading_assistant.md)
- [agentic_advisory.md](agentic_advisory.md)
- [rl_policy_workflow.md](rl_policy_workflow.md)
- [ml_signal_scorer.md](ml_signal_scorer.md)
