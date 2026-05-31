# Plan 01: Typed Advisory Contracts

## Objective

Stabilize the typed interfaces that define Fortuna's current advisory runtime
before any framework-style conversational layer is introduced.

The goal is to make all downstream tools call stable, explicit contracts rather
than reaching into shifting session/runtime internals.

## Why this comes first

Every later step depends on predictable request/response shapes:

- Telegram request parsing
- instrument analysis responses
- model-health summaries
- future tool-calling interfaces
- request tracing and evaluation

If these shapes are unstable, any framework layer will be built on sand.

## In scope

- formalize request/response models for interactive analysis
- formalize the analysis result shape exposed to user-facing interfaces
- formalize model health summary payloads
- formalize error states for user-facing analysis tools
- identify and remove transient/runtime-only fields from persisted/public payloads

## Out of scope

- introducing LangGraph/LangChain/LangFuse/LangSmith
- redesigning the Telegram UI
- changing core trading strategy logic
- adding natural-language planning

## Likely files

- `src/fortuna/agentic/models.py`
- `src/fortuna/app/model_status.py`
- `src/fortuna/app/session_engine.py`
- `src/fortuna/telegram/parser.py`
- `src/fortuna/telegram/assistant.py`
- possibly a new shared typed contract module under `src/fortuna/app/` or `src/fortuna/agentic/`

## Deliverables

1. A small shared contract layer for:
   - analysis request
   - analysis response
   - model health response
   - standardized advisory/tool error response
2. Telegram assistant updated to use the shared contract surface
3. Session/runtime helpers returning the same normalized output shape
4. Documentation update describing the contracts

## Acceptance criteria

- there is one canonical typed response shape for instrument analysis
- Telegram does not assemble analysis responses from ad hoc dict lookups alone
- model health is exposed through a stable typed structure
- public/runtime payloads do not leak transient DataFrames or other heavy internals
- disabled subsystems still return deterministic-safe outputs

## Tests

- unit tests for the new typed models
- Telegram assistant tests using the canonical response shape
- model-status tests for missing/disabled ML and RL
- session wiring tests for disabled-default behavior

Suggested checks:

```powershell
uv run pytest tests/app/test_model_status.py tests/app/test_telegram_assistant.py -q
uv run pytest tests/execution/test_session_wiring.py -q
uv run ruff check src/fortuna/app src/fortuna/telegram
```

## Risks

- over-abstracting too early instead of extracting only the stable contract
- accidentally persisting user-facing response wrappers into audit logs
- letting the dashboard and Telegram diverge again

## Notes for Cursor

- preserve the current operator-visible behavior where possible
- keep the contract small and typed
- prefer a new shared helper/model module over spreading contract logic across UI code
