# Fortuna Agent Architecture

This document describes the current agentic architecture in Fortuna as it
exists now.

Fortuna is not using a generic LLM-agent runtime as its core orchestration
layer today. The implemented system is a typed, auditable advisory pipeline
that combines deterministic strategies, optional ML scoring, optional RL
policies, a native conversational adapter above typed tools, and a final
agentic decision layer.

## 1. Current advisory stack

```text
market data / OHLCV
  -> deterministic strategies
  -> optional ML signal-quality vote
  -> optional RL policy vote
  -> agentic orchestrator
  -> final AgentDecision
  -> decision log
  -> optional paper-learning update
  -> optional Telegram delivery
```

The important thing is that the final advisory decision is structured and
typed. It is not a free-form LLM completion.

## 2. Core agent roles in Fortuna

### Deterministic strategy agents

These are not "agents" in the chat-framework sense. They are the transparent
strategy engines that produce candidate trade actions and metrics.

Examples:
- builtin ORB/MMTS style strategies
- generated/validated strategy definitions

Role:
- propose candidate market actions from price/indicator structure

### ML scorer agent

Role:
- estimate whether a deterministic signal is worth trusting

Implementation shape:
- supervised scorer artifact
- feature-schema-aware load path
- advisory vote only

### RL policy agent

Role:
- provide a stateful advisory action from market/position context

Implementation shape:
- PPO checkpoints
- advisory-readiness metadata
- promotion-gated load path

### Agentic orchestrator

Role:
- combine deterministic, ML, RL, and risk evidence
- produce one final `AgentDecision`

Outputs include:
- action
- confidence
- rationale summary
- reasons
- risk notes

This is the true "main agent" layer in current Fortuna.

## 3. Key implementation modules

| Area | Purpose |
|---|---|
| `src/fortuna/agentic/models.py` | Typed decision, rationale, and vote models |
| `src/fortuna/agentic/orchestrator.py` | Final advisory decision composition |
| `src/fortuna/agentic/agents.py` | Advisory helpers and composition support |
| `src/fortuna/agentic/store.py` | Decision logs, learning rows, notification audit |
| `src/fortuna/app/session_engine.py` | Runtime integration hub |
| `src/fortuna/ml/` | ML scorer artifacts and training |
| `src/fortuna/rl/` | RL training, inference, and checkpoint policy |
| `src/fortuna/telegram/` | Telegram request/response interface |

## 4. Main design rule

**Models and agents may advise. They do not directly own live execution.**

That means:
- deterministic strategy output remains the baseline
- the final decision remains auditable
- Telegram remains downstream of the final `AgentDecision`
- runtime must fail soft when optional agent layers are unavailable

## 5. Why Fortuna is not built on LangGraph/LangChain today

The current agent system is domain-specific and typed enough that plain Python
composition is a better fit right now than a general graph/chat-agent
framework.

Reasons:

- the core problem is structured trading advisory, not open-ended dialogue
- the most important properties are auditability, determinism, and validation
- model loading, promotion gating, and paper-learning persistence are clearer
  as explicit code paths
- the system already has natural integration seams in `session_engine.py` and
  `agentic/`
- a graph framework would add state machinery and indirection before it adds
  meaningful product value

## 6. Where a future LLM-agent layer could fit

A future framework-based layer would make sense only if Fortuna grows into:

- conversational research planning
- tool-calling over multiple market-analysis steps
- memory across multi-turn analyst conversations
- prompt/version tracing as a production concern

If that happens, the likely shape would be:

```text
user conversation layer
  -> tool-calling / search / research planner
  -> Fortuna typed analysis tools
  -> final advisory summarization
```

Even then, the typed advisory core should stay separate from the conversational
layer.

## 6a. What is implemented now

Fortuna now includes a native conversational adapter layer for Telegram and the
dashboard assistant surface:

- free-form operator prompts can be normalized into typed `search` / `analyze`
  tool calls
- the adapter is flag-gated
- the adapter remains above `FortunaAdvisoryTools`
- the adapter does not own execution, promotion, or final `AgentDecision`

## 7. Practical takeaway for contributors

If you are changing agent behavior today:

1. start in `src/fortuna/agentic/`
2. inspect `src/fortuna/app/session_engine.py`
3. preserve deterministic fallback
4. keep outputs typed and auditable
5. keep Telegram downstream of the final decision

## 8. Framework readiness and adoption gate

Technical readiness notes live under:

- `docs/agentic_framework_readiness/`
- [`docs/framework_adoption_gate.md`](framework_adoption_gate.md)

The explicit yes/no adoption gate (Plan 07) is documented in
[`docs/framework_adoption_gate.md`](framework_adoption_gate.md). Completing
readiness work does not approve framework integration; default outcome is
**`not_yet`** until product sign-off.

Current state:

- readiness artifacts are implemented in code and docs
- Fortuna is technically prepared for a narrow framework PoC above typed tools
- Fortuna now also includes a native internal framework layer for safe
  conversational normalization in Telegram and dashboard assistant flows
- Fortuna also now supports an optional OpenAI planner mode above the same
  typed tool surface for routing freer-form operator requests
- The typed operator surface now includes market-universe scans, shortlist
  analysis, shortlist briefings, simple portfolio allocation, and
  training-candidate preparation
- The allocator is now slightly more state-aware: it considers existing open
  positions and recent paper-closed learning outcomes when ranking the
  constrained basket
- The dashboard Assistant tab now also exposes a small workflow console for
  universe, top-setups, allocation, and training-candidate review
- Portfolio/exposure critique now includes directional crowding, overlapping
  underlyings, and open-position overlap when account state is present
- Training-candidate selection now carries a light exposure-aware penalty into
  the ranked manifest so ML/RL prep is less likely to over-focus one underlying
- Shortlist briefings now carry compatible selection rank / exposure-penalty
  signals so operator-facing prioritization stays consistent upstream
- Shortlist analysis now also carries the same ranking metadata, so downstream
  briefing and training layers are inheriting a stable order rather than inventing one
- The dashboard workflow console now shows universe -> shortlist -> briefing ->
  allocation -> training candidates as explicit stages for operator inspection
- Fortuna now also has a higher-level acceptance bundle path that can tie one
  nightly report, workflow snapshot, promotion review evidence, and
  model-health visibility into a single review artifact for operator sign-off
- Fortuna is not automatically approved to adopt a framework in production

Verify automated technical checks:

```powershell
uv run python scripts/check_framework_readiness.py
```

## Related docs

- `docs/cursor_codebase_guide.md`
- `docs/current_system_flow.md`
- `docs/telegram_trading_assistant.md`
- `docs/agentic_framework_readiness/README.md`
- `docs/framework_adoption_gate.md`
- `docs/how_to_use_fortuna.md`
- `AGENTS.md`
