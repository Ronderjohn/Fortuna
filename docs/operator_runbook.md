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
     before market open.

2. **Data cache**
   - Ensure historical OHLCV exists for watchlist symbols (backfill if needed).
   - Dashboard works offline on cache; live feed requires SmartAPI auth.

3. **Models (optional)**
   - **Models** tab: verify RL/ML show `advisory_ready` and live pointers when registry on.
   - With `FORTUNA_MODEL_REGISTRY_ENABLED=1` and `FORTUNA_MODEL_PROMOTION_REQUIRED=1`,
     missing live pointers mean deterministic-only advisory for that model path (normal).
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

3. **ML scorer training (manual)**
   ```powershell
   uv run python scripts/train_ml_signal_scorer.py --help
   ```

4. **Promotion (manual)**
   ```powershell
   uv run python scripts/promote_model.py --kind rl_policy --run-id <id>
   uv run python scripts/promote_model.py --kind ml_scorer --run-id <id>
   ```

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
```

## Related docs

- [Agentic advisory](agentic_advisory.md)
- [ML signal scorer](ml_signal_scorer.md)
- [RL policy workflow](rl_policy_workflow.md)
