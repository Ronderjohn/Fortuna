# Fortuna Codebase Guide For Cursor

This document is the current-state engineering map for Fortuna. It is meant to
help a Cursor agent understand how the system is actually wired today, where
the important boundaries are, and how to make changes without breaking the
deterministic advisory baseline.

Use this together with:

- `AGENTS.md` for operating rules and review protocol
- `docs/current_system_flow.md` for the compact architecture view
- `docs/operator_runbook.md` for runtime/operator workflow
- `docs/telegram_trading_assistant.md` for the Telegram interface

## 1. What Fortuna is now

Fortuna is an advisory-first quant system for NSE intraday markets.

It is not a generic chatbot, and it is not a pure model-serving system.
The current implemented design is:

1. market data and cache
2. deterministic strategies
3. optional ML scoring
4. optional RL policy inference
5. agentic orchestration into one final advisory decision
6. optional paper-learning capture
7. optional Telegram delivery

The key product surface today is still the dashboard plus the shared runtime
behind it. The Telegram bot is an interface into that same runtime, not a
separate analysis engine.

Fortuna now also includes a native conversational adapter above typed advisory
tools. That adapter powers freer-form Telegram and dashboard-assistant prompts
without moving core market/advisory logic into a chat runtime.

## 2. What is authoritative today

When there is tension between older planning text and current implementation,
trust the current code and current docs.

Current authoritative runtime surfaces:

- `src/fortuna/app/session_engine.py`
- `src/fortuna/agentic/`
- `src/fortuna/rl/`
- `src/fortuna/ml/`
- `src/fortuna/data/instruments.py`
- `app/streamlit_app.py`
- `scripts/run_operator_preflight.py`
- `scripts/run_telegram_bot.py`

Important note:

- some older project text still says "WhatsApp"
- the current implemented notification/runtime interface is Telegram-oriented

So for present-day work, treat Telegram as the live downstream notification
path unless the human explicitly asks to restore or add another provider.

## 3. Non-negotiable design rules

These are the architectural guardrails that matter most in practice:

- deterministic strategy output must remain usable when ML, RL, agentic, or
  Telegram subsystems fail
- no future leakage in labels, feature fitting, normalizers, thresholds, or
  training windows
- real-money broker execution is not the default product target
- runtime-impacting behavior should be behind `Settings`
- artifacts and decisions must be auditable on disk
- dashboard UI should consume business logic, not own it
- Telegram should remain downstream of final advisory decisions only

If Cursor needs to choose between a clever abstraction and a smaller typed,
testable integration, the smaller typed integration is usually more aligned
with the codebase.

## 4. Top-level module map

### `src/fortuna/data`

Purpose:
- historical data access
- SmartAPI integration
- live-feed aggregation
- cache management
- symbol and derivatives registry

Files to know:
- `data/instruments.py`
  - resolves equities, futures, and explicit options
  - powers dashboard search and Telegram instrument resolution
- `data/sources/smartapi_historical.py`
  - historical OHLCV fetch path
- `data/sources/smartapi_live.py`
  - live-feed transport
- `data/sources/smartapi_bar_aggregator.py`
  - bar aggregation for live runtime

### `src/fortuna/indicators`

Purpose:
- canonical technical indicator computations

Files to know:
- `indicators/engine.py`
  - shared indicator calculations used across deterministic, ML, and runtime

### `src/fortuna/strategy` and `strategies/`

Purpose:
- strategy DSL
- builtin strategy implementations
- deterministic signal generation

Important idea:
- deterministic strategy logic is the baseline truth source
- model layers advise on top of that baseline rather than replacing it

### `src/fortuna/backtesting`

Purpose:
- compiling signals
- cost-aware backtests
- walk-forward validation and institutional-style evaluation

Use this area when the change affects:
- historical evaluation
- strategy metrics
- realistic cost modeling
- validation and promotion criteria

### `src/fortuna/features`

Purpose:
- feature engineering
- observation construction
- normalization support for ML and RL

### `src/fortuna/ml`

Purpose:
- supervised signal-quality scorer
- artifact metadata
- dataset/training orchestration

Files to know:
- training helpers for scorer artifacts
- metadata/schema-hash handling
- load path used by runtime advisory scoring

### `src/fortuna/rl`

Purpose:
- RL environment
- training
- checkpoint metadata
- inference and promotion behavior

Sub-areas:
- `rl/training/`
- `rl/inference/`
- `rl/evaluation/`

Files to know:
- RL checkpoint resolution and advisory-readiness logic
- per-symbol and global fallback behavior

### `src/fortuna/agentic`

Purpose:
- final decision models
- orchestrator
- learning-row persistence
- notification policy and send path

This is where "what should the operator do now?" becomes explicit.

Key concepts:
- `AgentDecision`
- rationale and confidence
- decision logging
- paper-learning row lifecycle
- notification audit and dedupe

### `src/fortuna/execution`

Purpose:
- paper execution support
- routing/monitoring/account sync style plumbing

Important:
- do not casually treat this as live-order infrastructure
- current project target is advisory plus optional paper-related downstream flow

### `src/fortuna/app`

Purpose:
- runtime composition
- session state
- model status
- operator-facing helpers for dashboard/runtime

Most important file:
- `app/session_engine.py`

That file is the runtime hub. If Cursor only understands one file deeply before
making a cross-cutting change, it should usually be this one.

### `src/fortuna/telegram`

Purpose:
- Telegram polling client
- Telegram command parsing
- Telegram assistant formatting
- conversational normalization into typed tool calls when enabled

Files to know:
- `telegram/parser.py`
- `telegram/assistant.py`
- `telegram/runtime.py`
- `telegram/client.py`
- `agentic/conversational_adapter.py`

### `scripts/`

Purpose:
- operator entrypoints
- training/promotion CLIs
- smoke checks and diagnostics

High-value scripts:
- `run_dashboard.py`
- `run_operator_preflight.py`
- `run_telegram_bot.py`
- `train_ml_signal_scorer.py`
- `run_rl_train.py`
- `promote_model.py`
- `promote_policy.py`

## 5. Runtime flow that Cursor should picture first

The cleanest mental model is this:

```text
symbol request
  -> load bars / cached OHLCV / live state
  -> run deterministic strategies
  -> compute metrics / winner / live signals
  -> optionally load RL vote
  -> optionally load ML scorer vote
  -> agentic orchestration combines evidence
  -> final AgentDecision
  -> decision log
  -> optional paper-learning update
  -> optional Telegram notification or Telegram reply
```

For Telegram specifically:

```text
Telegram command
  -> parser
  -> symbol/contract resolution
  -> FortunaSessionEngine.load_symbol(...)
  -> agent_decisions() + live_signals()
  -> formatted advisory reply
```

## 6. Why `session_engine.py` matters so much

`src/fortuna/app/session_engine.py` is the integration seam across:

- market data loading
- live refresh
- deterministic signal generation
- model loading
- RL/ML participation
- agentic decision creation
- paper-learning wiring
- notification dispatch
- model status exposure to the dashboard

Most features that "feel small" in the UI are actually runtime composition
changes here.

When editing around this file:

- preserve disabled-default behavior
- preserve fail-soft fallback
- keep expensive work lazy when possible
- avoid leaking transient runtime objects into persisted metadata

## 7. Current model stack

### Deterministic layer

Always the baseline.

Outputs:
- strategy results
- winner strategy
- signal actions
- core metrics

### ML scorer

Role:
- estimate signal quality / confidence from labeled examples

Artifacts:
- validated scorer runs
- live pointer under ML live directory

Runtime expectation:
- if missing, stale, invalid, or promotion-gated, deterministic still works

### RL policy

Role:
- advisory stateful action proposal

Artifacts:
- validated/rejected checkpoints
- live pointer resolution by symbol or global path depending on settings

Runtime expectation:
- advisory only
- no model load should be required for deterministic baseline to operate

### Agentic orchestrator

Role:
- combine deterministic, RL, ML, and risk context into one decision

Important:
- this is not an LLM agent framework
- it is a typed domain orchestrator producing structured outputs

## 8. Current Telegram bot behavior

The Telegram bot currently supports:

- help
- search
- analyze equity
- analyze future
- analyze explicit option contract

Examples:

- `/search RELIANCE`
- `/analyze RELIANCE`
- `/analyze RELIANCE FUT`
- `/analyze NIFTY CE 25000 28MAY2026`

The bot currently:

- polls Telegram rather than using webhooks
- filters replies to the configured chat id
- returns final advisory text built from runtime outputs

The bot does not currently try to be an open-ended natural-language planner.
It is command-oriented on purpose.

## 9. Persistence and audit surfaces

Important persistent outputs:

- `logs/agentic/decisions.jsonl`
- `logs/agentic/notifications.jsonl`
- `logs/agentic/learning_rows.jsonl`
- `logs/agentic/learning.jsonl`
- `models/registry/promotions.jsonl`

Cursor should preserve:

- append-auditable behavior
- reproducible metadata
- schema-hash checks where already in use
- explicit live-pointer semantics in registry mode

## 10. Settings and flags that matter

Look in `src/fortuna/config/settings.py`.

Key runtime toggles:

- `FORTUNA_AGENTIC_ENABLED`
- `FORTUNA_AGENTIC_PAPER_LEARNING_ENABLED`
- `FORTUNA_AGENTIC_ML_SCORER_ENABLED`
- `FORTUNA_MODEL_REGISTRY_ENABLED`
- `FORTUNA_MODEL_PROMOTION_REQUIRED`
- `FORTUNA_TELEGRAM_ENABLED`
- `FORTUNA_TELEGRAM_BOT_TOKEN`
- `FORTUNA_TELEGRAM_CHAT_ID`

Operational rule:
- new runtime-affecting behavior should usually be gated through `Settings`

## 11. Where Cursor should add code depending on the task

If the task is about symbol search or derivatives resolution:
- start in `data/instruments.py`

If the task is about final advice, confidence, reasons, or voting:
- start in `agentic/`

If the task is about dashboard/runtime composition:
- start in `app/session_engine.py`

If the task is about model health, load state, or promotion visibility:
- start in `app/model_status.py` and `session_engine.py`

If the task is about Telegram behavior:
- start in `telegram/parser.py`, `telegram/assistant.py`, `telegram/runtime.py`

If the task is about paper-learning labels or outcome refresh:
- start in `agentic/store.py` and `agentic/learning.py`

If the task is about ML artifact generation:
- start in `ml/` and `scripts/train_ml_signal_scorer.py`

If the task is about RL checkpoint loading or promotion:
- start in `rl/inference/`, `rl/training/`, and promotion scripts

## 12. Common extension patterns that fit this repo

Good patterns:

- add a small typed helper instead of a global abstraction
- add focused tests near the changed module
- wire new runtime behavior through `Settings`
- keep pure summary logic outside Streamlit rendering code
- keep notification transport downstream of final decisions

Patterns to avoid:

- putting advisory business logic in the dashboard UI
- introducing broad framework machinery for a narrow typed workflow
- persisting transient DataFrames or large runtime objects into JSONL logs
- making Telegram the source of trading logic instead of a client of it

## 13. Practical test map

Useful slices:

```powershell
uv run pytest tests/agentic -q
uv run pytest tests/execution/test_session_wiring.py -q
uv run pytest tests/app/test_model_status.py -q
uv run pytest tests/app/test_telegram_parser.py tests/app/test_telegram_assistant.py tests/app/test_telegram_runtime.py -q
uv run pytest tests/data/test_smartapi_instruments.py -q
uv run pytest tests/rl -q
uv run pytest tests/ml -q
```

Lint:

```powershell
uv run ruff check src tests scripts
```

Prefer the smallest focused slice that proves the change.

## 14. Cursor workflow recommendation

When Cursor is asked to change Fortuna:

1. read `AGENTS.md`
2. read this file
3. inspect the exact runtime seam involved
4. identify which layer owns the behavior
5. keep deterministic fallback intact
6. add or update focused tests
7. avoid broad refactors unless the current seam is truly broken

## 15. Short version

If Cursor remembers only a few things, make it these:

- Fortuna is a deterministic trading-advisory system first
- ML/RL are additive advisory layers, not replacements
- `session_engine.py` is the runtime hub
- `agentic/` owns final advisory decisions
- Telegram is an interface into the same runtime, not a second system
- persistence, promotions, and learning rows are part of the product, not extras
- fail-soft behavior matters as much as cleverness
