# Operator Runbook

Practical checklist for running Fortuna's **advisory + optional paper learning**
stack on NSE intraday equities. This is not a live auto-trading runbook — real-money
execution is out of scope.

## Before market open

1. **Environment**
   - Copy `.env.example` → `.env` and fill SmartAPI credentials if using live feed.
   - Confirm `FORTUNA_AGENTIC_ENABLED=1` only when you want ensemble advisory.
   - Keep `FORTUNA_EXECUTION_ENABLED=0` for normal advisory-only operation.
   - Turn on `FORTUNA_AGENTIC_PAPER_LEARNING_ENABLED=1` only when you want the
     paper router + learning dataset. This remains paper-only, never live order placement.
   - Run:
     ```powershell
     uv run python scripts/run_operator_preflight.py
     ```
     before market open. Read the typed readiness lines at the bottom:
     `Runtime posture`, `Nightly posture`, `Scaling posture`, and `Readiness summary`.

### Before scaling basket size

Check scaling posture separately from model readiness:

1. Run operator preflight and read `Scaling posture` (also on the dashboard **Models** tab).
2. Compare your planned nightly basket count to `recommended_max_symbols`.
3. On low-RAM machines (`small_only`), keep baskets at or below `FORTUNA_SCALING_BASKET_SMALL_MAX` (default 8).
4. Only enable `FORTUNA_SCALING_ENFORCE_BASKET_CAP=1` when you want nightly lane gating to block oversized baskets.
5. Optional report cleanup: `FORTUNA_NIGHTLY_REPORT_RETENTION_ENABLED=1` keeps the newest N JSON/MD pairs under `reports/nightly/` (default 30).

### Workflow observability (optional)

Enable typed cross-module event capture when debugging long workflows:

1. Set `FORTUNA_OBSERVABILITY_ENABLED=1` (default off).
2. Re-run the workflow (preflight, universe build, nightly train, promotion review, Telegram command).
3. Inspect `logs/observability/workflow_events.jsonl` for `event_name`, `workflow_id`, `run_id`, `status`, and `duration_ms`.
4. Optional OTLP export: install `uv sync --group otel`, set `FORTUNA_OBSERVABILITY_EXPORTER=otlp`, and point `FORTUNA_OBSERVABILITY_OTLP_ENDPOINT` at your collector.

Observability is additive: existing human-readable logs and typed artifacts remain authoritative.

### Runtime posture vocabulary

| Posture | Meaning |
|---------|---------|
| `deterministic_only` | Agentic advisory off, or only deterministic signals are healthy |
| `advisory_partial` | Agentic on but ML/RL lanes are not fully promoted or advisory-ready |
| `advisory_ready` | All enabled advisory lanes have live pointers and advisory-ready status |
| `blocked` | Strict preflight failures or hard blockers — fix before relying on advisory |

| Nightly posture | Meaning |
|-----------------|---------|
| `manual_only` | Default — run nightly scripts manually (`FORTUNA_NIGHTLY_AUTOMATION_ENABLED=0`) |
| `enabled` | Automation flag on and no nightly blockers |
| `blocked` | Automation flag on but runtime blockers would make unattended runs useless |

2. **Data cache**
   - Ensure historical OHLCV exists for watchlist symbols (backfill if needed).
   - Dashboard works offline on cache; live feed requires SmartAPI auth.

3. **Models (optional)**
   - **Models** tab: check the **Activation** caption on ML/RL panels for stage,
     top blocker, and recommended command.
   - Verify RL/ML show `advisory_ready` and live pointers when registry on.
   - With `FORTUNA_MODEL_REGISTRY_ENABLED=1` and `FORTUNA_MODEL_PROMOTION_REQUIRED=1`,
     missing live pointers mean deterministic-only advisory for that model path (normal).
   - Quick activation check: `uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS --activation-only`
   - Run promotion after nightly training if new checkpoints passed gates.

4. **Telegram (optional)**
   - `FORTUNA_TELEGRAM_ENABLED=1` + bot token / chat id in `.env`.
   - Throttle defaults: 10 msgs/symbol/session, 300s min interval, quiet hours on.
   - Audit log: `logs/agentic/notifications.jsonl`.

## During session (09:15–15:30 IST)

1. **Start dashboard**
   ```powershell
   uv run python scripts/run_dashboard.py
   ```

2. **Watch**
   - Live signal panel: deterministic + optional RL overlay.
   - Agentic decision (when enabled): final action, confidence, rationale.
   - Execution/Monitor tabs if paper learning enabled.

3. **Telegram behavior**
   - Alerts fire on **live bar close** only (`notify=True`), not on Streamlit refresh.
   - Same symbol/action/bar/decision hash is deduped.
   - Outside NSE hours → suppressed when quiet hours enabled.

## After market close

1. **Review logs**
   - `logs/agentic/decisions.jsonl` — what the system advised.
   - `logs/agentic/notifications.jsonl` — what was sent, deduped, or throttled.
   - `logs/agentic/learning_rows.jsonl` — paper-learning outcomes (if enabled).

2. **Nightly training (optional)**
   ```powershell
   uv run --group rl python scripts/nightly_train.py --help
   ```
   Promotes best per-symbol RL checkpoints when `advisory_ready`.
   Do not re-enable the Windows scheduled nightly task until runtime readiness
   is verified end to end. If recent nightly reports are repeatedly partial or
   failing, keep the task disabled and use the local acceptance rehearsal
   instead.

   Nightly lane gating (SR2): before backfill/training, the orchestrator emits
   a `lane_readiness` step. Global blockers (for example missing SmartAPI
   credentials with `--data-source smartapi`) produce one typed skip instead of
   per-symbol failure spam. Report overall status meanings:

   | Status | Meaning |
   |--------|---------|
   | `ok` | At least one backfill/training/regime step completed successfully |
   | `skipped` | Only policy/intentional skips; no unexpected failures |
   | `blocked` | Global prerequisites failed; expensive lanes did not run |
   | `partial (...)` | One or more steps failed unexpectedly |

3. **ML scorer training (manual)**
   ```powershell
   uv run python scripts/train_ml_signal_scorer.py --help
   ```

4. **Promotion (manual)**
   ```powershell
   uv run python scripts/promote_model.py --kind rl_policy --run-id <id>
   uv run python scripts/promote_model.py --kind ml_scorer --run-id <id>
   ```

5. **Acceptance bundle (optional review)**
   ```powershell
   uv run python scripts/build_acceptance_bundle.py --workflow-snapshot reports/workflow_snapshot.json
   uv run python scripts/build_acceptance_bundle.py --nightly-report-dir reports/nightly
   ```
   The bundle ties together nightly report evidence, workflow snapshot, promotion
   review, and model-health context. Markdown/JSON output includes:
   - `refresh_context=` — effective refresh target, source lane, split
     discovery/research/execution follow-up targets, and scout support
   - `recommended_follow_up_lane=` — which recovery lane (`discovery`,
     `research`, or `execution`) best matches the effective target
   - `cross_artifact_alignment=` — whether discovery, research, execution, and
     promotion guidance agree across the linked artifacts

   Typed bundle fields (JSON export): `effective_refresh_target`,
   `refresh_target_source`, `recommended_follow_up_lane`, `scout_support_target`,
   `scout_support_summary`, and `cross_artifact_alignment`.

5b. **Market universe discovery (optional refresh)**
   ```powershell
   # Default auto: Screener CSV when present, else SmartAPI instrument seeds
   uv run python scripts/build_market_universe.py --source auto --limit 15

   # SmartAPI-native bounded discovery (instrument registry or tradable CSV)
   uv run python scripts/build_market_universe.py --source smartapi --limit 15
   ```
   CLI prints `provider_summary=` when available (seed source, OHLCV scoring
   source, and any fallback). Optional bounded symbol list:
   `FORTUNA_MARKET_UNIVERSE_TRADABLE_UNIVERSE_CSV`. Set
   `FORTUNA_MARKET_UNIVERSE_AUTO_PREFER_SMARTAPI=true` to prefer SmartAPI over
   Screener CSV in `--source auto`.

6. **Nightly acceptance dry-run rehearsal (local, advisory-only)**
   ```powershell
   # Default: candidate-style synthetic fixtures
   uv run python scripts/run_nightly_acceptance_dry_run.py --out-dir reports/acceptance/dry_run

   # Emit-based replay: same nightly_basket emit_* entrypoints as production
   uv run python scripts/run_nightly_acceptance_dry_run.py --mode emit-replay --out-dir reports/acceptance/emit_replay

   # Verify linkage for an existing nightly report directory
   uv run python scripts/run_nightly_acceptance_dry_run.py --mode verify --report-dir reports/nightly --out-dir reports/acceptance/dry_run
   ```
   Builds a fully local evidence stack: training candidates, training-research
   plan, workflow snapshot, nightly report, promotion review, and acceptance
   bundle. `--mode emit-replay` routes through real emit/report contracts with
   deterministic stub builders (stops before backfill/training). `--mode verify`
   checks artifact linkage on historical nightly output. No live training or
   network calls. Recommended local sequence is: `candidate` for the synthetic
   baseline, `emit-replay` for the bounded real contract path, then `verify`
   against an existing nightly report directory. Exit code is non-zero when the
   bundle overall status is `fail`. CLI prints typed
   `replay_linkage=status=... run=... missing=... broken=... summary=...`
   diagnostics when linkage is available.

7. **Promotion review with alignment (optional)**
   ```powershell
   uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS --out reports/promotion_review.md --format md
   ```
   Promotion review now carries the same `cross_artifact_alignment` summary as
   acceptance bundles, so model review shows whether discovery, research,
   execution, and promotion targets stay coherent.

## Enabling / disabling subsystems

| Subsystem | Enable | Disable |
|---|---|---|
| Agentic advisory | `FORTUNA_AGENTIC_ENABLED=1` | `=0` (default) |
| ML scorer votes | `FORTUNA_AGENTIC_ML_SCORER_ENABLED=1` | `=0` |
| RL inference | Per-symbol promoted checkpoint | Remove pointer or disable registry |
| Model registry | `FORTUNA_MODEL_REGISTRY_ENABLED=1` + `FORTUNA_MODEL_PROMOTION_REQUIRED=1` | `=0` (mtime fallback) |
| Paper learning | `FORTUNA_AGENTIC_PAPER_LEARNING_ENABLED=1` | `=0` |
| Paper execution router | Implied by paper learning | Keep execution off |
| Telegram | `FORTUNA_TELEGRAM_ENABLED=1` + bot token/chat id | `=0` (zero side effects) |

## Telegram setup

1. Create a Telegram bot and obtain its bot token.
2. Identify the target chat or user chat id.
2. Set in `.env`:
   ```
   FORTUNA_TELEGRAM_ENABLED=1
   FORTUNA_TELEGRAM_BOT_TOKEN=...
   FORTUNA_TELEGRAM_CHAT_ID=...
   ```
3. Messages include: action, symbol, price, confidence, reasons, risk notes, paper status,
   and **"Advisory only — not an execution instruction."**

## Troubleshooting

### SmartAPI credentials missing

- Live feed and account sync fail; dashboard may still render cached OHLCV and backtests.
- Check `.env` and `scripts/test_smartapi_env.py`.

### Missing model artifacts

- RL panel shows empty state; deterministic signals continue.
- With `FORTUNA_MODEL_REGISTRY_ENABLED=1`, unpromoted models are not loaded (by design).
- Promote a validated run or disable registry for dev mtime loading.

### Telegram not sending

- Confirm `FORTUNA_TELEGRAM_ENABLED=1` and Telegram fields populated.
- Read `logs/agentic/notifications.jsonl` for `policy_reason`:
  - `deduped` — same decision already notified
  - `throttled_interval` / `throttled_count` — rate limits
  - `quiet_hours` — outside NSE session
  - `incomplete_credentials` — missing Telegram config

### Streamlit refresh spam

- Should not occur: prefetch uses `notify=False`; restart dedupe reloads previously
  sent or already-deduped keys from the notification audit log.

## Useful commands

```powershell
# Dashboard
uv run python scripts/run_dashboard.py

# Operator preflight
uv run python scripts/run_operator_preflight.py

# Focused tests
uv run pytest tests/agentic tests/ml tests/models tests/rl -q

# RL train
uv run --group rl python scripts/run_rl_train.py --help

# ML scorer train
uv run python scripts/train_ml_signal_scorer.py --help

# Promote model
uv run python scripts/promote_model.py --help

# Acceptance bundle
uv run python scripts/build_acceptance_bundle.py --help

# Local nightly acceptance rehearsal
uv run python scripts/run_nightly_acceptance_dry_run.py --help

# Promotion review with cross-artifact alignment
uv run python scripts/review_promotion.py --help
```

## Related docs

- [Agentic advisory](agentic_advisory.md)
- [ML signal scorer](ml_signal_scorer.md)
- [RL policy workflow](rl_policy_workflow.md)
