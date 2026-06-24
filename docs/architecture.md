# Fortuna Architecture

## Overview

Fortuna is now an **advisory-first multi-agent trading research and decision
system** built on top of a deterministic market-signal core.

It is no longer accurate to describe the repo as only a Phase 1 deterministic
pipeline. The current architecture is:

1. deterministic strategies generate transparent signal context
2. optional ML and RL layers add advisory evidence
3. typed multi-agent workflow services compose discovery, analysis, briefing,
   allocation, and training research
4. nightly artifacts, promotion review, and model-health surfaces audit the
   recurring loop
5. Telegram and dashboard assistants sit above typed tools and do not own the
   advisory core

For the most detailed current-state runtime map, see
`docs/current_system_flow.md`.

## High-level system map

```text
market data / cache / live bars
  -> deterministic strategy signals
  -> optional ML signal-quality vote
  -> optional RL policy vote
  -> agentic orchestrator
  -> final advisory decision
  -> optional paper routing + learning rows
  -> optional Telegram/dashboard assistant surfaces

market universe discovery
  -> Screener CSV seed or registry fallback
  -> OHLCV enrichment for liquidity, trend, activity, and regime
  -> shortlist analysis + briefing
  -> portfolio allocation / critic
  -> training candidates
  -> training research plan
  -> nightly workflow + promotion + acceptance artifacts
```

## Architectural boundaries

### Deterministic core

The deterministic signal path remains the fallback and source of transparent
trading context. If ML, RL, OpenAI routing, Telegram, or workflow helpers fail,
Fortuna should still be able to produce the deterministic baseline output.

Key areas:

- `src/fortuna/strategy/`
- `src/fortuna/strategies/`
- `src/fortuna/indicators/`
- `src/fortuna/backtesting/`
- `src/fortuna/reporting/`

### Advisory extensions

ML and RL layers are additive advisory evidence, not hard replacements for the
deterministic core.

Key areas:

- `src/fortuna/ml/`
- `src/fortuna/rl/`
- `src/fortuna/features/`
- `src/fortuna/paper/`

### Typed multi-agent workflow

Fortuna’s current multi-agent behavior is expressed through typed services and
contracts rather than through a free-form conversational runtime.

Key areas:

- `src/fortuna/agentic/contracts.py`
- `src/fortuna/agentic/tools.py`
- `src/fortuna/app/multi_agent_team.py`
- `src/fortuna/app/agent_roles/` — thin wrappers (`liquidity_scout`,
  `activity_scout`, `universe_scout`, `instrument_analyst`, `briefing_agent`,
  `portfolio_critic`, `research_planner`, `operations_monitor`)
- `src/fortuna/app/market_universe.py`
- `src/fortuna/app/shortlist_analysis.py`
- `src/fortuna/app/shortlist_briefing.py`
- `src/fortuna/app/portfolio_allocator.py`
- `src/fortuna/app/training_candidates.py`
- `src/fortuna/app/training_research.py`
- `src/fortuna/app/operator_workflow.py`

### Runtime and operator surfaces

The dashboard and Telegram assistant are interfaces into the same typed
advisory/runtime system. They should not become separate business-logic owners.

Key areas:

- `src/fortuna/app/session_engine.py`
- `src/fortuna/app/streamlit_app.py`
- `src/fortuna/telegram/`
- `src/fortuna/agentic/conversational_adapter.py`
- `src/fortuna/agentic/openai_router.py`

### Audit, promotion, and acceptance

Recurring artifacts are part of the product. Workflow snapshots, promotion
reviews, nightly reports, acceptance bundles, and model-health summaries should
remain linked and inspectable.

Key areas:

- `src/fortuna/app/model_status.py`
- `src/fortuna/app/promotion_review.py`
- `src/fortuna/app/acceptance_bundle.py`
- `src/fortuna/app/acceptance_alignment.py` — shared refresh context and
  `CrossArtifactAlignmentSummary` for acceptance bundles and promotion review
- `src/fortuna/app/nightly_acceptance.py`
- `src/fortuna/app/workflow_snapshot.py`
- `src/fortuna/app/workflow_artifacts.py`
- `scripts/nightly_train.py`
- `scripts/build_acceptance_bundle.py`
- `scripts/run_nightly_acceptance_dry_run.py`

## Non-negotiable rules

- advisory-first, no live broker execution by default
- deterministic fallback must survive subsystem failure
- no future leakage in ML/RL or derived labels
- typed, auditable interfaces over opaque orchestration
- no network calls in tests
- runtime-impacting behavior behind `Settings` where appropriate

## Guidance for contributors

If you are changing Fortuna:

1. read `AGENTS.md`
2. read `docs/cursor_codebase_guide.md`
3. inspect the exact runtime seam involved
4. preserve deterministic fallback and advisory-only boundaries
5. add focused tests for the touched surface

For the current operating flow and contributor context, see:

- `docs/how_to_use_fortuna.md`
- `docs/operator_runbook.md`
- `docs/current_system_flow.md`
