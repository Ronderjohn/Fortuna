# Fortuna Current System Flow

Practical architecture map of what is implemented today in Fortuna. This is the
current-state view for operators and engineers, not a roadmap.

## 1. Runtime advisory pipeline

```text
market data / cache / live bars
  -> deterministic strategy signals
  -> optional RL policy vote
  -> optional ML signal-quality vote
  -> agentic orchestrator
  -> final AgentDecision
  -> decisions.jsonl
  -> optional paper router + learning rows
  -> optional Telegram notification
  -> optional conversational adapter surface (Telegram + dashboard Assistant tab)
```

Key properties:
- Deterministic strategy output is the baseline and survives subsystem failure.
- RL and ML are advisory inputs, not hard overrides.
- Telegram is downstream of final `AgentDecision` only.
- The conversational adapter, when enabled, sits above typed tools and does not
  replace the advisory core.
- Paper learning remains paper-only and does not place live broker orders.

## 2. Training and promotion pipeline

### ML scorer

```text
learning_rows.jsonl
  -> resolved signal examples
  -> enriched OHLCV features
  -> SignalScorer artifact
  -> models/ml_signal_scorer/validated/<run_id>/
  -> promote_model.py
  -> models/ml_signal_scorer/live/live.json
```

### RL policy

```text
historical OHLCV
  -> walk-forward PPO training
  -> validated/rejected checkpoint
  -> promote_model.py or promote_policy.py
  -> models/live/by_symbol/<SYMBOL_KEY>/live.json
```

### Runtime loading

- With `FORTUNA_MODEL_REGISTRY_ENABLED=1` and `FORTUNA_MODEL_PROMOTION_REQUIRED=1`,
  runtime loads only explicitly promoted live pointers.
- If a promoted artifact is missing, Fortuna falls back to deterministic signals.
- Dashboard **Models** surfaces pointer state, advisory readiness, promotions,
  and recent paper-learning outcomes.

## 3. Key artifacts and logs

### Models

- `models/validated/<run_id>/` — RL validated checkpoints
- `models/rejected/<run_id>/` — RL rejected checkpoints
- `models/live/by_symbol/<SYMBOL_KEY>/live.json` — RL live pointers
- `models/ml_signal_scorer/validated/<run_id>/` — ML scorer artifacts
- `models/ml_signal_scorer/live/live.json` — ML scorer live pointer
- `models/registry/promotions.jsonl` — append-only promotion audit log

### Agentic runtime

- `logs/agentic/decisions.jsonl` — final advisory decisions
- `logs/agentic/notifications.jsonl` — sent/deduped/throttled Telegram audit
- `logs/agentic/learning_rows.jsonl` — durable paper-learning rows and outcomes
- `logs/agentic/learning.jsonl` — paper-learning event stream

## 4. Repo structure map

| Area | Purpose |
|---|---|
| `src/fortuna/data` | Data sources, cache, live feed, symbol registry |
| `src/fortuna/indicators` | Canonical indicators |
| `src/fortuna/backtesting` | Backtest engines and validation |
| `src/fortuna/features` | Shared feature/observation transforms |
| `src/fortuna/ml` | Signal scorer training, artifacts, feature rows |
| `src/fortuna/rl` | RL env, training, checkpoints, inference |
| `src/fortuna/agentic` | Orchestrator, decisions, learning rows, notifications |
| `src/fortuna/execution` | Paper broker, router, monitor, reconciliation |
| `src/fortuna/app` | Session engine, model status, dashboard helpers |
| `scripts` | Operator, training, promotion, diagnostics entrypoints |

## 5. Operator mental model

- **Deterministic baseline:** always available if OHLCV and strategies load.
- **Optional advisory stack:** ML, RL, and agentic orchestration layer on top.
- **Fail-soft behavior:** missing or invalid ML/RL/Telegram components should not
  take down deterministic dashboard usage.
- **Promotion gate:** promoted artifacts are what matter in registry mode, not
  merely the latest trained run.
- **Paper-only downstream actions:** paper router, learning outcomes, and
  notifications are implemented; live order placement is not the normal target.

## Related docs

- [Operator runbook](operator_runbook.md)
- [Agentic advisory](agentic_advisory.md)
- [ML signal scorer](ml_signal_scorer.md)
- [RL policy workflow](rl_policy_workflow.md)
