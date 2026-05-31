# Conversation evaluation suite

Deterministic regression cases for conversational trading-analysis requests
(Telegram today; future framework adapters later).

## Purpose

Plans 01–02 stabilized typed contracts and tool surfaces. Plan 03 adds a
**versioned evaluation corpus** so parser, tool, and assistant behavior can be
measured without live SmartAPI data or LLM grading.

Harness code lives in
[`src/fortuna/telegram/eval_suite.py`](../src/fortuna/telegram/eval_suite.py).
Fixture data lives in
[`tests/fixtures/conversation_eval/v1.yaml`](../tests/fixtures/conversation_eval/v1.yaml).

## Layers

Each case declares a `layer`:

| Layer | Validates |
|-------|-----------|
| `parse` | `parse_telegram_request()` intent, symbol normalization, timeframe/days |
| `route` | `route_telegram_request()` action and clarification codes |
| `tool` | `FortunaAdvisoryTools` typed responses (`ok`, error codes, resolved symbol, decision action) |
| `assistant` | End-to-end `TelegramAnalysisAssistant.handle_text()` with stable text fragments only |

Tool and assistant layers use shared fakes in
[`tests/support/conversation_eval_fakes.py`](../tests/support/conversation_eval_fakes.py)
so results stay deterministic.

## Fixture schema

```yaml
version: 1
cases:
  - id: unique_case_id
    layer: parse | route | tool | assistant
    input: "/analyze RELIANCE"
    expect:
      kind: analyze
      symbol: RELIANCE
      route_action: analyze
      clarification: empty_analyze_symbol
      tool: analyze_instrument
      ok: true
      error_code: invalid_request
      resolved_symbol: RELIANCE.NS
      segment_contains: EQUITY
      decision_action: BUY
      hit_symbols: [RELIANCE.NS]
      text_contains: ["Advisory only"]
```

Only set fields you need for that layer. The loader ignores unset keys.

## Running the suite

```powershell
uv run pytest tests/telegram/test_conversation_eval_suite.py tests/app/test_telegram_parser.py -q
```

Parser tests reuse parse-layer cases from the same fixture to avoid drift.

## Adding a case

1. Add a row to `tests/fixtures/conversation_eval/v1.yaml` with a new unique `id`.
2. Pick the smallest layer that covers the behavior (`parse` for intent-only changes).
3. Prefer structured `expect` fields over long prose assertions.
4. Run the pytest command above.

## Framework adapter usage

Future conversational layers can import the harness directly:

```python
from fortuna.telegram.eval_suite import (
    default_eval_fixture_path,
    evaluate_tool,
    load_eval_cases,
)
```

Inject your own `FortunaAdvisoryTools` backend when evaluating tool-layer cases in
integration tests; keep production adapters on the same contracts.

## Non-goals

- LLM grading or semantic similarity scoring
- Production telemetry
- Live instrument master or market data dependencies
