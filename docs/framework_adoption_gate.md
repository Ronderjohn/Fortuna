# Framework adoption gate

Fortuna must satisfy an explicit adoption gate before adding a broader agentic
framework such as LangGraph or LangChain. This document is a **decision gate**,
not a commitment to adopt a framework.

## Decision statement

- Framework adoption is **not approved by default**.
- Fortuna now has a **native internal conversational framework layer** above
  typed advisory tools; this document is about third-party or broader framework
  adoption beyond that native layer.
- Completing Plans 01–06 (technical readiness) does **not** grant permission to
  migrate core trading or advisory logic into a framework runtime.
- The typed advisory core remains **out of scope for migration** even if a
  future proof-of-concept proceeds.

Allowed outcomes:

| Outcome | Meaning |
|---------|---------|
| `not_yet` | Technical readiness may pass, but product need or safeguards are unmet — **default today** |
| `poc_only` | Team approves a narrow, reversible experiment above existing tools |
| `rejected` | Framework integration is not appropriate for Fortuna's advisory model |

Run the automated technical checker:

```powershell
uv run python scripts/check_framework_readiness.py
```

## Pass/fail checklist

| # | Criterion | How to verify | Status after Plans 01–06 |
|---|-----------|---------------|--------------------------|
| 1 | Typed advisory/tool contracts are stable | [`contracts.py`](../src/fortuna/agentic/contracts.py), [`advisory_contracts.md`](advisory_contracts.md), `tests/agentic/test_contracts.py`; breaking changes require explicit contract versioning | **Pass** |
| 2 | Telegram and dashboard share a common analysis tool surface | [`FortunaAdvisoryTools`](../src/fortuna/agentic/tools.py) via [`TelegramAnalysisAssistant`](../src/fortuna/telegram/assistant.py) and [`FortunaSessionEngine.advisory_tools()`](../src/fortuna/app/session_engine.py) | **Pass** |
| 3 | Request evaluation suite exists | [`eval_suite.py`](../src/fortuna/telegram/eval_suite.py), [`v1.yaml`](../tests/fixtures/conversation_eval/v1.yaml); `uv run pytest tests/telegram/test_conversation_eval_suite.py` | **Pass** |
| 4 | Request tracing exists | [`request_audit.py`](../src/fortuna/telegram/request_audit.py), [`telegram_request_audit.md`](telegram_request_audit.md); `uv run pytest tests/telegram/test_request_audit.py` | **Pass** |
| 5 | Team decided command-bot behavior is insufficient | Documented operator sign-off (name, date, decision) — see [Appendix: product sign-off](#appendix-product-sign-off) | **Not met (manual)** |
| 6 | Deterministic fallback preserved in any future adapter | Future adapter maps to `TelegramRequest` → route → tools; orchestrator and execution paths unchanged | **Policy (required for PoC)** |

**Current gate verdict:** criteria 1–4 pass; the native conversational adapter
exists; criterion 5 is still not met for broader framework adoption. Outcome:
**`not_yet`**.

Do not proceed to framework integration until criterion 5 is explicitly satisfied
and criteria 6 remains enforced.

## Valid framework use cases

A framework may help **only** as a layer above existing typed tools:

- Conversational research assistant over `FortunaAdvisoryTools` (search, analyze, health)
- Multi-step clarification that resolves to `TelegramRequest`, not free-form trade plans
- Structured tool-calling planner that selects among typed tool methods
- Prompt/version observability for an **actual** LLM layer (when one exists)

These use cases must preserve:

- Parse → route → tools → format boundary ([`conversation_boundary.md`](conversation_boundary.md))
- Scalar request audit (no full LLM payloads in production logs without review)
- Eval suite parity on mapped inputs

## Invalid use cases — reject integration

Do **not** adopt a framework to:

- Replace deterministic strategy logic or backtest signal compilation
- Replace typed [`AgenticOrchestrator`](../src/fortuna/agentic/orchestrator.py) decision assembly
- Hide model-promotion or registry rules behind prompt logic
- Route core `AgentDecision` through opaque chat completions
- Move execution, risk gates, or paper/live broker paths into conversational runtime
- Adopt LangGraph/LangChain because Plans 01–06 exist (readiness is necessary, not sufficient)

If the primary motivation is "the framework looks powerful," the correct outcome is **`rejected`** or **`not_yet`**.

## Narrow proof-of-concept boundary (if gate passes)

Plan 07 does **not** implement a PoC. If the team later approves `poc_only`, scope
must stay narrow and reversible:

```text
Future module: src/fortuna/agentic/conversational_adapter.py
  Input:  user text (natural language)
  Output: TelegramRequest OR route_telegram_request + execute_route result
  Must NOT:
    - import orchestrator internals directly
    - bypass FortunaAdvisoryTools
    - place or simulate live orders
    - persist full LLM transcripts in production audit logs without review
  Future flag: FORTUNA_CONVERSATIONAL_ADAPTER_ENABLED=false (default off)
  Rollback:     disable flag; TelegramBotRuntime and dashboard unchanged
```

PoC success metrics:

- Eval suite parity on all mapped conversational inputs
- Request audit rows remain compact scalars
- Deterministic command path still works when adapter is disabled
- No regression in `tests/agentic/`, `tests/telegram/`, or session wiring tests

## Explicit non-goals (even after adoption)

- Migrating RL inference, ML promotion, or execution stack into a framework runtime
- Webhook or multi-user Telegram (see Plan 06 boundaries)
- Replacing the dashboard's primary analysis workflow with chat-only UX
- Removing command-oriented `/search` and `/analyze` interfaces

## Related readiness work

Plans 01–06 under [`agentic_framework_readiness/`](agentic_framework_readiness/README.md)
built the prerequisites this gate evaluates. Plan 07 closes that sequence with
policy only — no framework dependency is added to the repo.

## Appendix: product sign-off

Criterion 5 requires a human decision. Record sign-off here or link an ADR when
the team approves moving beyond command-bot behavior.

| Field | Value |
|-------|-------|
| Decision | `not_yet` |
| Date | — |
| Approver | — |
| Notes | Command-oriented Telegram + dashboard tools remain sufficient for current advisory mode. |

When updating, set `Decision` to `poc_only` or `rejected` and fill date/approver.
