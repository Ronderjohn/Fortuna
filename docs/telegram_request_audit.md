# Telegram request audit

Fortuna records a compact JSONL audit trail for interactive Telegram assistant
requests. Each line captures parse, route, tool, and delivery metadata without
storing full replies, OHLCV, or agent vote payloads.

## Log location

Default path (relative to project root):

```text
logs/telegram/requests.jsonl
```

Override indirectly via `FORTUNA_PROJECT_ROOT` / settings `project_root`, or
inject a custom `TelegramRequestAuditStore` when constructing
`TelegramBotRuntime`.

Enable or disable with:

```text
FORTUNA_TELEGRAM_REQUEST_AUDIT_ENABLED=true
```

When disabled, the runtime does not create the store (zero append overhead).

## Flow

```text
Telegram update
  -> parse + route + tools (InteractionResult)
  -> send_message (delivery_ok / delivery_error)
  -> build_audit_entry -> TelegramRequestAuditStore.append
```

Audit append failures are swallowed so operator replies are never blocked.

Messages rejected by chat-id filtering are not audited and are not sent.

## Field glossary

| Field | Meaning |
|-------|---------|
| `request_id` | 16-char hex correlation id generated per handled update |
| `ts` | UTC ISO8601 timestamp when the audit row was built |
| `channel` | Always `telegram` for this store |
| `chat_id` | Telegram chat id from the update |
| `update_id` | Telegram `update_id` when present |
| `raw_text` | User message text, capped at 500 characters |
| `parsed_kind` | `TelegramRequestKind` value (`help`, `search`, `analyze`, `unknown`) |
| `query` | Search query or analyze token remainder |
| `symbol` | Parsed analyze symbol tokens (before tool resolution) |
| `timeframe` | Parsed analyze timeframe (default `5m`) |
| `days` | Parsed analyze lookback days |
| `route_action` | Router action (`show_help`, `search`, `analyze`, `clarify`, `unsupported`) |
| `clarification` | Clarification code when route is `clarify` |
| `tool` | Advisory tool name when invoked (`search_instruments`, `analyze_instrument`) |
| `tool_ok` | Typed tool response `ok` flag |
| `error_code` | Advisory error code when tool failed |
| `resolved_symbol` | Canonical symbol from analysis snapshot |
| `decision_action` | Final advisory action string when analysis succeeded |
| `hit_count` | Search hit count |
| `response_chars` | Length of formatted reply only |
| `delivery_ok` | Whether `send_message` succeeded |
| `delivery_error` | Exception text when delivery failed |

Null-valued fields are omitted from stored JSON for compact lines.

## Sample line

```json
{"channel":"telegram","chat_id":"12345","days":30,"decision_action":"BUY","delivery_ok":true,"parsed_kind":"analyze","request_id":"a1b2c3d4e5f67890","resolved_symbol":"RELIANCE.NS","response_chars":412,"route_action":"analyze","symbol":"RELIANCE","timeframe":"5m","tool":"analyze_instrument","tool_ok":true,"ts":"2026-05-30T10:15:00+00:00","update_id":10,"raw_text":"/analyze RELIANCE"}
```

## Inspecting logs

PowerShell tail:

```powershell
Get-Content logs/telegram/requests.jsonl -Wait
```

Filter with `jq` (if installed):

```powershell
Get-Content logs/telegram/requests.jsonl | jq 'select(.route_action=="analyze")'
```

Programmatic read:

```python
from fortuna.telegram.request_audit import TelegramRequestAuditStore
from fortuna.config.settings import load_settings

store = TelegramRequestAuditStore.from_settings(load_settings())
rows = store.read_recent(limit=20)
```

## Intentionally excluded

- Full formatted Telegram reply bodies
- OHLCV bars, signal tables, or vote lists
- Full `InstrumentAnalysisResponse.to_dict()` payloads
- Agentic notification audit rows (`logs/agentic/notifications.jsonl`)

Interactive request audit lives in `src/fortuna/telegram/request_audit.py`, separate
from [`NotificationAuditStore`](../src/fortuna/agentic/store.py).

## Related docs

- [conversation_boundary.md](conversation_boundary.md) — parse/route/tools/format layers
- [telegram_trading_assistant.md](telegram_trading_assistant.md) — operator bot overview
