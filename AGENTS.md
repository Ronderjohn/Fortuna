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
- Works from the current task brief plus repo docs.
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

## Current Product Posture

Fortuna has shifted to a **Telegram-first signal product** with a slim
operator console behind it.

Current priority order:

1. keep the deterministic and typed advisory core reliable
2. keep Telegram request handling fast, isolated, and auditable
3. use ML and RL only when promoted, cheap, and clearly additive
4. treat Streamlit as an operator/admin surface, not the main user product
5. keep multimodal and OpenAI-backed routing above typed tools only

Canonical operational docs:

- [docs/how_to_use_fortuna.md](docs/how_to_use_fortuna.md)
- [docs/operator_runbook.md](docs/operator_runbook.md)
- [docs/telegram_bot_quickstart.md](docs/telegram_bot_quickstart.md)
- [docs/current_system_flow.md](docs/current_system_flow.md)

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
uv run python scripts/build_acceptance_bundle.py --help
```

## Settings Principles

New runtime-impacting behavior must be controlled by `Settings`.

Existing agentic flags:

- `FORTUNA_AGENTIC_ENABLED`
- `FORTUNA_AGENTIC_PAPER_LEARNING_ENABLED`
- `FORTUNA_AGENTIC_ML_SCORER_ENABLED`
- `FORTUNA_AGENTIC_LOG_DIR`
- `FORTUNA_MARKET_UNIVERSE_SCREENER_CSV`
- `FORTUNA_MARKET_UNIVERSE_TRADABLE_UNIVERSE_CSV`
- `FORTUNA_MARKET_UNIVERSE_SMARTAPI_SEED_LIMIT`
- `FORTUNA_MARKET_UNIVERSE_AUTO_PREFER_SMARTAPI`
- `FORTUNA_MARKET_UNIVERSE_DEFAULT_LIMIT`
- `FORTUNA_MARKET_UNIVERSE_TURNOVER_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_VOLUME_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_TREND_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_VOLUME_RATIO_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_ACTIVITY_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_ACTION_BIAS_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_REGIME_BIAS_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_SIGNAL_BIAS_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_GLOBAL_BIAS_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_ADAPTIVE_RECENCY_HALFLIFE`
- `FORTUNA_MARKET_UNIVERSE_NIGHTLY_FEEDBACK_ENABLED`
- `FORTUNA_MARKET_UNIVERSE_NIGHTLY_FEEDBACK_MAX_BOOST`
- `FORTUNA_MARKET_UNIVERSE_NIGHTLY_FEEDBACK_LOOKBACK_REPORTS`
- `FORTUNA_MARKET_UNIVERSE_NIGHTLY_REFRESH_BONUS_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_NIGHTLY_REPORT_DIR`
- `FORTUNA_MARKET_UNIVERSE_LONG_HORIZON_FEEDBACK_ENABLED`
- `FORTUNA_MARKET_UNIVERSE_LONG_HORIZON_LOOKBACK_REPORTS`
- `FORTUNA_MARKET_UNIVERSE_EXECUTED_LANE_BOOST_WEIGHT`
- `FORTUNA_MARKET_UNIVERSE_LONG_HORIZON_HALFLIFE`
- `FORTUNA_MARKET_UNIVERSE_FUNDAMENTALS_OVERLAY_ENABLED`
- `FORTUNA_MARKET_UNIVERSE_FUNDAMENTALS_OVERLAY_CSV`
- `FORTUNA_MARKET_UNIVERSE_FUNDAMENTALS_OVERLAY_MAX_BOOST`
- `FORTUNA_MARKET_UNIVERSE_FUNDAMENTALS_QUALITY_WEIGHT`
- `FORTUNA_SCALING_BASKET_SMALL_MAX`
- `FORTUNA_SCALING_BASKET_MEDIUM_MAX`
- `FORTUNA_SCALING_ENFORCE_BASKET_CAP`
- `FORTUNA_NIGHTLY_REPORT_RETENTION_ENABLED`
- `FORTUNA_NIGHTLY_REPORT_RETENTION_COUNT`
- `FORTUNA_OBSERVABILITY_ENABLED`
- `FORTUNA_OBSERVABILITY_LOG_DIR`
- `FORTUNA_OBSERVABILITY_EXPORTER`
- `FORTUNA_OBSERVABILITY_OTLP_ENDPOINT`
- `FORTUNA_TRAINING_CANDIDATE_NIGHTLY_FEEDBACK_ENABLED`
- `FORTUNA_TRAINING_CANDIDATE_VOLUME_RATIO_BONUS_WEIGHT`
- `FORTUNA_TRAINING_CANDIDATE_TRENDING_BONUS_WEIGHT`
- `FORTUNA_TRAINING_CANDIDATE_SCOUT_OVERLAP_BOOST`
- `FORTUNA_TRAINING_CANDIDATE_SCOUT_VOLUME_DENSE_BOOST`
- `FORTUNA_TRAINING_CANDIDATE_NIGHTLY_FEEDBACK_MAX_BOOST`
- `FORTUNA_TRAINING_CANDIDATE_NIGHTLY_FEEDBACK_LOOKBACK_REPORTS`
- `FORTUNA_TRAINING_CANDIDATE_NIGHTLY_REFRESH_BONUS_WEIGHT`
- `FORTUNA_TRAINING_CANDIDATE_NIGHTLY_REPORT_DIR`
- `FORTUNA_TRAINING_CANDIDATE_REMEDIATION_PRESSURE_ENABLED`
- `FORTUNA_TRAINING_CANDIDATE_REMEDIATION_MAX_BOOST`
- `FORTUNA_SHORTLIST_DISCOVERY_ALIGNMENT_ENABLED`
- `FORTUNA_SHORTLIST_DISCOVERY_ALIGNMENT_BOOST`
- `FORTUNA_SHORTLIST_DISCOVERY_ALIGNMENT_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CONTEXT_WEIGHTING_ENABLED`
- `FORTUNA_PORTFOLIO_ALLOCATOR_REGIME_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_ACTIVITY_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_MAX_SINGLE_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_MAX_HIGH_RISK_POSITIONS`
- `FORTUNA_PORTFOLIO_ALLOCATOR_MAX_PER_REGIME`
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
- `FORTUNA_PORTFOLIO_ALLOCATOR_CONTEXT_WEIGHTING_ENABLED`
- `FORTUNA_PORTFOLIO_ALLOCATOR_REGIME_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_ACTIVITY_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_MAX_SINGLE_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_MAX_HIGH_RISK_POSITIONS`
- `FORTUNA_PORTFOLIO_ALLOCATOR_MAX_PER_REGIME`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_ENABLED`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_SAME_SIDE_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_SAME_REGIME_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_SAME_STRATEGY_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_OPEN_SAME_EXPOSURE_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_WEAKER_SAME_SIDE_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_WATCH_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_HIGH_RISK_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_DISCOVERY_ALIGNMENT_ENABLED`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_WEAKER_DISCOVERY_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_RESEARCH_ALIGNMENT_ENABLED`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_MISSING_RESEARCH_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_WEAKER_RESEARCH_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_UNREFRESHED_RESEARCH_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_WEAKER_NIGHTLY_RESEARCH_PENALTY`
- `FORTUNA_PORTFOLIO_ALLOCATOR_CRITIC_BLOCK_THRESHOLD`
- `FORTUNA_PORTFOLIO_ALLOCATOR_NIGHTLY_FEEDBACK_ENABLED`
- `FORTUNA_PORTFOLIO_ALLOCATOR_NIGHTLY_FEEDBACK_LOOKBACK_REPORTS`
- `FORTUNA_PORTFOLIO_ALLOCATOR_NIGHTLY_REFRESH_BONUS_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_NIGHTLY_PROMOTE_BONUS_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_NIGHTLY_REPORT_DIR`
- `FORTUNA_PORTFOLIO_ALLOCATOR_FRESHNESS_PREFERENCE_ENABLED`
- `FORTUNA_PORTFOLIO_ALLOCATOR_FRESHNESS_PREFERENCE_WEIGHT`
- `FORTUNA_PORTFOLIO_ALLOCATOR_SAME_TARGET_BALANCE_ENABLED`
- `FORTUNA_PORTFOLIO_ALLOCATOR_SAME_TARGET_BALANCE_PENALTY`
- `FORTUNA_MODEL_STATUS_NIGHTLY_ALIGNMENT_ENABLED`
- `FORTUNA_MODEL_STATUS_NIGHTLY_ALIGNMENT_LOOKBACK_REPORTS`
- `FORTUNA_MODEL_STATUS_NIGHTLY_REPORT_DIR`
- `FORTUNA_MODEL_STATUS_NIGHTLY_ALIGNMENT_RECOMMENDATION_RATIO`
- `FORTUNA_ACCEPTANCE_BUNDLE_NIGHTLY_ALIGNMENT_MIN_ENABLED_REPORTS`
- `FORTUNA_ACCEPTANCE_BUNDLE_NIGHTLY_ALIGNMENT_WARN_RATIO`
- `FORTUNA_ACCEPTANCE_BUNDLE_NIGHTLY_REFRESHED_ALIGNMENT_WARN_RATIO`
- `FORTUNA_ACCEPTANCE_BUNDLE_TEAM_RESEARCH_ALIGNMENT_WARN_RATIO`
- `FORTUNA_NIGHTLY_AUTOMATION_ENABLED`
- `FORTUNA_NIGHTLY_FAIL_FAST_GLOBAL_BLOCKERS`
- `FORTUNA_NIGHTLY_LANE_GATING_ENABLED`

Add new flags only when they are needed to control runtime behavior, model
artifact locations, or production-like risk.
