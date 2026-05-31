# Conversation boundary

Fortuna's Telegram interface uses an explicit four-layer boundary between
user text and advisory computation. This keeps the system command-oriented and
tool-driven while leaving room for a future conversational adapter.

## Layers

```text
User text
  -> parser (intent recognition)
  -> router (route to tool or static response)
  -> FortunaAdvisoryTools (typed advisory computation)
  -> formatters (human-readable reply)
```

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Parse | [`src/fortuna/telegram/parser.py`](../src/fortuna/telegram/parser.py) | Map prefixes/tags to `TelegramRequest` |
| Route | [`src/fortuna/telegram/router.py`](../src/fortuna/telegram/router.py) | Decide tool vs help/clarify/unsupported |
| Tools | [`src/fortuna/agentic/tools.py`](../src/fortuna/agentic/tools.py) | Typed search/analyze/health calls |
| Format | [`src/fortuna/telegram/formatters.py`](../src/fortuna/telegram/formatters.py) | Operator-facing text only |

The assistant ([`src/fortuna/telegram/assistant.py`](../src/fortuna/telegram/assistant.py))
orchestrates parse → route → execute → format via `execute_route()` and exposes
structured `InteractionResult` metadata for request audit.

See [telegram_request_audit.md](telegram_request_audit.md) for JSONL trace fields.

## Route actions

`route_telegram_request()` returns a `ConversationRoute` with one of:

| Action | When | Tool called? |
|--------|------|--------------|
| `show_help` | `/help`, empty input | No |
| `search` | `/search` with non-empty query | Yes — `search_instruments` |
| `analyze` | `/analyze` with symbol tokens | Yes — `analyze_instrument` |
| `clarify` | `/search` or `/analyze` missing required args, or incomplete option syntax | No |
| `unsupported` | Free text without a known prefix | No |

## Response semantics

Three failure classes are intentionally separate:

1. **Clarification** — user used a known command but omitted required args
   (e.g. bare `/search`). Router returns `clarify` before any tool call.
2. **Unsupported** — text does not match a supported command prefix. Router
   returns `unsupported` with a `/help` hint.
3. **Tool errors** — valid route, but instrument resolution or data load fails.
   Tools return typed `AdvisoryError`; formatters surface the error message.

Clarification copy is defined in [`formatters.py`](../src/fortuna/telegram/formatters.py)
and triggered by router codes in [`router.py`](../src/fortuna/telegram/router.py)
(`empty_search_query`, `empty_analyze_symbol`, `incomplete_option_syntax`).

Clarification and tool-error messages are intentionally separate:

## Command-oriented vs conversational future

Today Fortuna remains **command-oriented**:

- Supported intents are explicit prefixes (`/search`, `/analyze`, `/help`)
- No open-ended natural-language planning or memory

Current implementation note:

- when `FORTUNA_CONVERSATIONAL_ADAPTER_ENABLED=true`, a native conversational
  adapter may normalize certain freer-form prompts into the same typed request
  boundary before routing continues
- the adapter still resolves into `TelegramRequest` -> route -> tools -> format

A future framework adapter should:

1. Produce or map user input to `TelegramRequest` (or a compatible request DTO)
2. Call `route_telegram_request()` then `TelegramAnalysisAssistant.execute_route()`
3. Avoid reaching into `FortunaSessionEngine`, orchestrator internals, or audit stores

See also:

- [advisory_contracts.md](advisory_contracts.md) — typed tool payloads
- [conversation_eval_suite.md](conversation_eval_suite.md) — regression cases including `route` layer

## Framework adapter hook

```python
from fortuna.telegram.assistant import TelegramAnalysisAssistant
from fortuna.telegram.parser import parse_telegram_request
from fortuna.telegram.router import route_telegram_request

assistant = TelegramAnalysisAssistant(settings)
req = parse_telegram_request(user_text)
route = route_telegram_request(req)
reply = assistant.execute_route(route)
```

Inject fakes or alternate `FortunaAdvisoryTools` backends in tests without
changing the boundary.

Before adding LangGraph/LangChain or similar, read
[framework_adoption_gate.md](framework_adoption_gate.md). Default outcome: **`not_yet`**.
