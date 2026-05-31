---
description: 
alwaysApply: true
---

# Fortuna Agent Operating Guide

This file is the shared brain for Codex agents, Cursor agents, and human
engineering work on Fortuna. Read it before making architectural or code changes.

## Mission

Fortuna is an institutional-grade autonomous quant research and advisory system
for NSE intraday equities.

The target architecture is not "one model predicts trades." The target is an
auditable multi-agent trading-decision system:

1. Deterministic strategies generate transparent candidate signals.
2. ML models score signal quality, strategy fit, and market regime.
3. RL policies propose sequential actions from market and position state.
4. Agentic orchestration combines evidence into one advisory decision.
5. Paper learning records outcomes and improves future models.
6. Telegram/dashboard output tells the operator what to do next.
7. A native conversational adapter may normalize free-form operator text into
   typed tool calls, but must remain above the advisory core.

Default mode is advisory. Real-money execution is not part of the current target.

## Current Architecture

Primary source tree: `src/fortuna/`

Important layers:

- `data/`: SmartAPI/OpenChart/yfinance data, cache, symbols, live feeds.
- `indicators/`: canonical indicator calculations.
- `strategy/` and `strategies/`: DSL and builtin strategy engines.
- `backtesting/`: signal compilation, backtest execution, risk/cost modelling.
- `backtesting/standard/`: walk-forward validation, filters, robustness.
- `reporting/strategy_tester/`: canonical metrics and report surfaces.
- `search/`, `arena/`, `tournament/`: candidate strategy evaluation.
- `features/`: ML/RL observation construction and normalization.
- `rl/`: Gymnasium env, PPO training, checkpointing, live inference.
- `paper/`: walk-forward adaptive learning and paper-feedback state.
- `execution/`: paper broker, risk gate, account mirror, monitor, journal.
- `agentic/`: advisory decision models, orchestrator, logs, notifications.
- `app/session_engine.py`: runtime hub for dashboard data, signals, execution,
  agentic decisions, and live refresh.
- `app/streamlit_app.py`: primary operator UI.

## Non-Negotiable Design Rules

- Preserve deterministic fallback. If ML, RL, Telegram, or agentic code fails,
  existing deterministic dashboard signals must continue working.
- Do not place live SmartAPI orders unless a future explicit phase enables and
  tests it. Current work is advisory plus optional paper learning.
- No future leakage. Fit scalers, labelers, normalizers, thresholds, and models
  only on data available at that point in time.
- Use walk-forward/OOS validation for every model that influences advice.
- Use realistic costs from `MarketCostModel` whenever simulating trades.
- Use `compute_all_metrics()` as the canonical metrics source.
- Keep NSE session rules intact: Mon-Fri, 09:15-15:30 IST, hard square-off.
- Keep artifacts auditable: checkpoints, feature schema hashes, labels, model
  metadata, agent votes, final decisions, and paper outcomes must be persisted.
- Add behavior behind settings flags when it can affect runtime, notifications,
  execution, or dashboard state.
- Prefer small typed interfaces and deterministic tests before adding LLM or
  graph orchestration frameworks.
- If a conversational layer exists, it must map into typed advisory tools and
  must not own execution, promotion, or core decision logic.

## Agent Roles

Codex is the systems architect and integration reviewer:

- Reads broad repo context.
- Designs cross-module phases.
- Implements or reviews integrations.
- Checks architecture consistency, tests, and failure modes.
- Reviews Cursor-agent output after each phase.

Cursor is the engineering cockpit:

- Implements bounded tasks in the IDE.
- Works from phase prompts in `docs/cursor_phase_prompts.md`.
- Produces focused diffs for one phase at a time.
- Does not silently broaden scope.

Human operator:

- Chooses phases and approves plans.
- Runs Cursor agents in Plan mode first.
- Sends completed diffs back to Codex for review/refactor.

## Cursor Agent Working Protocol

When a Cursor agent receives a phase prompt:

1. Stay in Plan mode first.
2. Inspect the referenced files and current implementation.
3. Return a concise plan with files, APIs, tests, and risks.
4. Do not implement until the human approves.
5. Once approved, implement only that phase.
6. Run the requested focused tests.
7. Report changed files, test results, and any unresolved risks.

Do not combine phases unless the human explicitly asks.

## Codex Review Protocol

After each Cursor phase:

1. Inspect `git diff` and affected modules.
2. Verify the phase acceptance criteria directly.
3. Run focused tests and style checks.
4. Refactor only where needed to preserve architecture or correctness.
5. Update prompts/docs if the roadmap changes.

## ML/RL Strategy

The strongest Fortuna design is an ensemble pipeline:

- Deterministic strategies remain the transparent baseline.
- ML scores whether a signal is worth acting on.
- ML/regime models decide which strategy family is currently trustworthy.
- RL proposes stateful actions, especially entry/exit timing.
- The agentic orchestrator makes the final advisory decision from votes.
- A conversational adapter may help the operator reach the right tool call, but
  it does not replace the orchestrator.

Avoid these anti-patterns:

- Replacing deterministic strategies with an opaque model too early.
- Training on all history and validating on the same periods.
- Optimizing only total return.
- Using one symbol's RL policy across unrelated symbols without evidence.
- Sending Telegram alerts directly from low-level signal functions.
- Letting dashboard UI own business logic that belongs in `agentic/`.

## Current Phase Roadmap

Phase 1: ML Signal Scorer

- Build supervised labels from historical strategy signals and forward outcomes.
- Train/calibrate a signal-quality model.
- Expose inference as an agent vote, not as a hard override.

Phase 2: Paper Learning Dataset

- Turn agentic decisions and later outcomes into durable labeled examples.
- Feed labels into ML scorer and adaptive learning.

Phase 3: Agentic Ensemble Upgrade

- Extend `agentic/` to combine deterministic, ML, RL, regime, portfolio, and
  risk evidence with explainable confidence.

Phase 4: RL Policy Improvement

- Improve reward shaping, evaluation, per-symbol checkpoints, and promotion.
- Keep deterministic fallback and checkpoint metadata strict.

Phase 5: Promotion and Monitoring

- Create gates for model promotion into live advisory.
- Add dashboard/reporting views for model health, drift, and paper performance.

Phase 6: Telegram Advisory Hardening

- Improve notification templates, dedupe, throttling, and audit trail.
- Keep Telegram downstream of final `AgentDecision` only.

Phase 7: Documentation

- Operator and contributor docs under `docs/` — see
  [agentic_advisory.md](docs/agentic_advisory.md),
  [operator_runbook.md](docs/operator_runbook.md),
  [ml_signal_scorer.md](docs/ml_signal_scorer.md),
  [rl_policy_workflow.md](docs/rl_policy_workflow.md).

Pre-Framework Readiness:

- Before introducing LangGraph/LangChain or similar orchestration layers,
  prefer the readiness plans under `docs/agentic_framework_readiness/`.
- Sequence the work as:
  1. typed advisory contracts
  2. shared tool surfaces
  3. conversational evaluation suite
  4. conversation boundary/router
  5. traceability and request audit
  6. Telegram interface hardening
  7. framework adoption gate — see [framework_adoption_gate.md](docs/framework_adoption_gate.md)
- Default adoption outcome is **`not_yet`** until documented product sign-off (criterion 5).
- Do not treat "framework readiness" as permission to move core trading logic
  into an opaque conversational runtime.
- The repo now contains implemented readiness artifacts: typed advisory
  contracts, shared advisory tools, a conversation eval suite, conversation
  routing boundaries, Telegram request audit, and a framework adoption gate.
- The repo now also includes a native, flag-gated conversational adapter for
  Telegram and dashboard assistant flows. Treat it as a layer above typed tools,
  not as a replacement for the advisory core.
- Verify technical criteria: `uv run python scripts/check_framework_readiness.py`

## Expected Test Discipline

Use `uv run pytest ...`.

Minimum per phase:

- Unit tests beside new modules.
- Session-engine tests for wiring and disabled defaults.
- No network calls in tests.
- No SmartAPI credential dependency in tests.
- For ML/RL: deterministic tiny datasets, fixed seeds, and artifact tests.

Useful focused checks:

```powershell
uv run pytest tests/agentic -q
uv run pytest tests/execution/test_session_wiring.py -q
uv run pytest tests/rl -q
uv run pytest tests/features -q
uv run ruff check src/fortuna/agentic tests/agentic
```

## Settings Principles

New runtime-impacting behavior must be controlled by `Settings`.

Existing agentic flags:

- `FORTUNA_AGENTIC_ENABLED`
- `FORTUNA_AGENTIC_PAPER_LEARNING_ENABLED`
- `FORTUNA_AGENTIC_ML_SCORER_ENABLED`
- `FORTUNA_AGENTIC_LOG_DIR`
- `FORTUNA_MODEL_REGISTRY_ENABLED`
- `FORTUNA_MODEL_PROMOTION_REQUIRED`
- `FORTUNA_TELEGRAM_ENABLED`
- `FORTUNA_TELEGRAM_PROVIDER`
- `FORTUNA_TELEGRAM_DEDUPE_MEMORY`
- `FORTUNA_TELEGRAM_MAX_PER_SYMBOL_PER_SESSION`
- `FORTUNA_TELEGRAM_MIN_INTERVAL_SECONDS`
- `FORTUNA_TELEGRAM_QUIET_HOURS_ENABLED`
- `FORTUNA_TELEGRAM_BOT_TOKEN`
- `FORTUNA_TELEGRAM_CHAT_ID`

Add new flags only when they are needed to control runtime behavior, model
artifact locations, or production-like risk.
