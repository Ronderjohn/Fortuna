# Plan 05: Traceability And Request Audit

## Objective

Add request-level tracing and audit surfaces for the interactive advisory path
so future conversational layers are observable before heavyweight LLM tracing
frameworks are introduced.

## Why this matters

People often reach for LangSmith/LangFuse because they need observability.
Fortuna should first capture the domain-specific signals it actually cares
about:

- user request
- parsed intent
- resolved symbol
- invoked tool
- final advisory result
- response delivery outcome

## In scope

- add a lightweight request audit log for Telegram interactions
- log parsed requests and tool outcomes
- correlate request/response flow where practical
- keep logs compact and auditable

## Out of scope

- third-party tracing SaaS
- prompt traces for an LLM that does not exist yet
- verbose debug logging of internal model objects

## Likely files

- `src/fortuna/telegram/runtime.py`
- `src/fortuna/telegram/assistant.py`
- `src/fortuna/agentic/store.py`
- possibly a new request-audit helper module

## Deliverables

1. A request audit sink for Telegram interactions
2. Typed audit entries for request parse and response outcome
3. Operator/developer docs describing how to inspect the traces

## Acceptance criteria

- each Telegram request can be reconstructed at a high level from logs
- logs do not leak oversized runtime objects
- request tracing does not break deterministic fallback
- audit entries are append-friendly and easy to inspect

## Tests

- runtime tests for audit write behavior
- tests for malformed request handling still producing trace entries
- tests for chat-id filtering plus audit behavior

Suggested checks:

```powershell
uv run pytest tests/app/test_telegram_runtime.py tests/agentic -q
uv run ruff check src/fortuna/telegram src/fortuna/agentic
```

## Risks

- logging too much noisy detail
- adding trace code inside user-facing formatting paths
- coupling audit shape to Telegram-specific wording

## Notes for Cursor

- think in terms of domain audit, not generic tracing glamour
- the ideal output is compact and useful during real debugging
