# Plan 06: Telegram Interface Hardening

## Objective

Strengthen the Telegram interface so it is robust enough to serve as the first
real conversational surface before any framework layer is added.

## In scope

- expand unsupported/invalid input handling
- improve contract-specific option guidance
- improve invalid symbol and missing symbol responses
- verify runtime behavior around offset state, polling, and configured chat filtering
- refine user help text and usage suggestions

## Out of scope

- webhook migration
- multi-user bot support
- natural-language free-form research assistant behavior

## Likely files

- `src/fortuna/telegram/parser.py`
- `src/fortuna/telegram/assistant.py`
- `src/fortuna/telegram/runtime.py`
- `docs/telegram_trading_assistant.md`
- `docs/telegram_bot_quickstart.md`

## Deliverables

1. Better invalid-input and missing-input behavior
2. Stronger option/future guidance messages
3. More complete test coverage around common operator mistakes
4. Updated usage docs with examples and troubleshooting

## Acceptance criteria

- invalid or partial requests produce actionable help
- option requests that cannot resolve explain the expected syntax clearly
- runtime state handling remains stable across restarts
- docs accurately reflect real supported commands and limits

## Tests

- parser tests for missing symbol / malformed option syntax
- assistant tests for invalid resolution and fallback messages
- runtime tests for offset persistence and chat filtering

Suggested checks:

```powershell
uv run pytest tests/app/test_telegram_parser.py tests/app/test_telegram_assistant.py tests/app/test_telegram_runtime.py -q
uv run ruff check src/fortuna/telegram tests/app
```

## Risks

- turning the bot into a fuzzy prompt interpreter
- adding too many special cases without a stable contract layer
- drift between docs and behavior

## Notes for Cursor

- optimize for clarity and stability
- the bot should feel smoother, but still remain deterministic in shape
