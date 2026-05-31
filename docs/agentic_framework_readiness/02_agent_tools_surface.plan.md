# Plan 02: Agent Tools Surface

## Objective

Build explicit typed tool surfaces around Fortuna's existing runtime so a
future agentic framework can call safe, well-scoped tools instead of poking at
session internals.

## Why this matters

A framework layer should consume tools like:

- `search_instruments(query)`
- `analyze_instrument(symbol, timeframe, days)`
- `get_model_health()`
- `get_recent_learning_summary(symbol)`

not direct access to large runtime objects.

## In scope

- define a small tool service layer
- implement typed methods around search, analysis, model health, and learning summary
- keep those tools pure or near-pure where possible
- make Telegram use those tools instead of custom internal wiring where sensible

## Out of scope

- conversational memory
- multi-step planning
- any external agent framework
- replacing `FortunaSessionEngine`

## Likely files

- new module such as `src/fortuna/agentic/tools.py` or `src/fortuna/app/analysis_tools.py`
- `src/fortuna/app/session_engine.py`
- `src/fortuna/app/model_status.py`
- `src/fortuna/telegram/assistant.py`
- `src/fortuna/data/instruments.py`

## Deliverables

1. A typed tools module with at least:
   - `search_instruments`
   - `analyze_instrument`
   - `get_model_health`
   - `get_recent_learning_summary`
2. Telegram assistant routed through those tools
3. Clear boundary between:
   - runtime composition
   - user-interface formatting
   - tool payloads

## Acceptance criteria

- Telegram uses the same typed analysis tool any future framework layer would use
- search and analyze are callable without UI-specific assumptions
- tool outputs are small, typed, and stable
- tool errors are explicit and non-throwing where user-facing behavior benefits

## Tests

- tool-level unit tests
- Telegram integration tests using fake tools or fake engine/tool backends
- search tests for equities, futures, and explicit options

Suggested checks:

```powershell
uv run pytest tests/app/test_telegram_assistant.py tests/data/test_smartapi_instruments.py -q
uv run pytest tests/execution/test_session_wiring.py -q
uv run ruff check src/fortuna/app src/fortuna/telegram src/fortuna/agentic
```

## Risks

- building a giant “god tools” module
- duplicating logic already owned by `session_engine.py`
- UI formatting leaking back into tool return shapes

## Notes for Cursor

- keep tools close to current runtime behavior
- do not invent a framework-specific abstraction
- the tools should be good on their own even if no framework is added later
