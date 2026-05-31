# Plan 03: Conversation Evaluation Suite

## Objective

Create a deterministic evaluation suite for conversational trading-analysis
requests so future changes to the Telegram assistant or a framework layer can be
measured against stable expectations.

## Why this matters

Without an evaluation set, a conversational layer will drift in behavior and
nobody will know whether it became better, worse, or merely different.

## In scope

- define representative user requests for:
  - stock analysis
  - future analysis
  - option analysis
  - symbol search
  - malformed/ambiguous requests
- define expected parsed intent
- define expected tool call and response-shape expectations
- add fixtures for deterministic evaluation

## Out of scope

- LLM grading
- production telemetry
- free-form language scoring

## Likely files

- new tests under `tests/app/` or `tests/telegram/`
- possibly fixture files under `tests/fixtures/`
- docs describing the evaluation cases

## Deliverables

1. A versioned evaluation set of user request cases
2. Expected parse/resolve/analyze outcomes
3. A test harness that can be reused by:
   - Telegram assistant tests
   - future framework adapters

## Acceptance criteria

- there is a stable set of query cases covering stocks, futures, and options
- the suite validates parse intent, symbol normalization, and response class
- malformed or unsupported requests have explicit expected behavior
- the suite is easy to extend when new interfaces are added

## Tests

- parser tests
- assistant/tool tests against the eval fixtures
- regression tests for ambiguous or unsupported queries

Suggested checks:

```powershell
uv run pytest tests/app/test_telegram_parser.py tests/app/test_telegram_assistant.py -q
uv run ruff check tests/app
```

## Risks

- writing a suite that is too coupled to exact wording instead of stable behavior
- covering only happy paths
- creating evaluation cases that depend on live network/data state

## Notes for Cursor

- focus on structured behavior first
- avoid asserting on every line of prose unless that text is intentionally contract-like
