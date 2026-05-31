# Plan 04: Conversation Boundary And Router

## Objective

Define and implement the boundary between conversational intent handling and
Fortuna's typed advisory core.

This plan decides where a future framework layer would sit without letting it
bleed into model loading, strategy logic, or audit persistence.

## Why this matters

Fortuna already has a strong command bot. Before introducing a richer
conversational analyst, the system needs a clear split between:

- user intent parsing
- tool routing
- advisory computation
- response formatting

## In scope

- formalize a request-routing layer for:
  - help
  - search
  - analyze
  - unsupported requests
- define what remains command-oriented and what may become conversational later
- define a response strategy for clarification prompts
- ensure the advisory engine remains tool-driven rather than conversation-driven

## Out of scope

- long-term memory
- open-ended research planning
- external framework integration

## Likely files

- `src/fortuna/telegram/parser.py`
- `src/fortuna/telegram/assistant.py`
- new router/helper module if needed
- bot usage docs

## Deliverables

1. A routing design that clearly separates:
   - intent recognition
   - tool invocation
   - human-readable response formatting
2. Support for graceful clarification-style failures
3. Documentation describing the conversation boundary

## Acceptance criteria

- unsupported or ambiguous messages get a structured fallback response
- analysis remains driven by typed tool calls
- Telegram interaction is smoother without becoming a vague prompt sink
- the conversational boundary is documented for future framework work

## Tests

- parser tests for unsupported or partial requests
- assistant tests for graceful error/help routing
- optional tests for clarification response behavior

Suggested checks:

```powershell
uv run pytest tests/app/test_telegram_parser.py tests/app/test_telegram_assistant.py tests/app/test_telegram_runtime.py -q
uv run ruff check src/fortuna/telegram tests/app
```

## Risks

- overbuilding a chatbot before the tool layer is ready
- letting conversational phrasing become the main API
- losing predictability for operators

## Notes for Cursor

- keep the routing explicit
- optimize for operator clarity over linguistic cleverness
