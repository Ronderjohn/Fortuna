# Telegram Trading Assistant

Fortuna can now run as a Telegram polling assistant for symbol search and
advisory analysis requests.

## What it does

The Telegram assistant can:

- search NSE equities and indexed NFO futures
- resolve explicit NFO option contracts
- rank a liquid market universe for watchlist or training prep
- turn shortlisted setups into a simple constrained portfolio allocation
- show model/runtime health
- run Fortuna's current analysis flow for the requested instrument
- reply with the final advisory decision, confidence, key reasons, and current action plan

It uses the same runtime stack as the dashboard:

```text
Telegram message
  -> request parser
  -> conversation router
  -> FortunaAdvisoryTools (valid search/analyze routes only)
  -> telegram/formatters
  -> formatted Telegram reply
```

See also:

- [advisory_contracts.md](advisory_contracts.md) — typed tool payloads
- [conversation_boundary.md](conversation_boundary.md) — parse/route/format split
- [telegram_request_audit.md](telegram_request_audit.md) — JSONL request trace

## Codebase implementation map

The Telegram assistant is intentionally thin and reuses the main advisory
runtime rather than inventing a parallel analysis stack.

- `scripts/run_telegram_bot.py`
  - operator entrypoint for the polling bot
- `src/fortuna/telegram/parser.py`
  - turns `/search ...`, `/analyze ...`, `/universe ...`, `/brief ...`, `/allocate ...`, `/candidates ...`, `/workflow ...`, `/health`, and tag-style prompts into structured requests
- `src/fortuna/telegram/router.py`
  - maps parsed requests to tool execution, help, clarification, or unsupported replies
- `src/fortuna/telegram/assistant.py`
  - orchestrates parse → route → tools → format via `execute_route()`
- `src/fortuna/agentic/tools.py`
  - typed tool surface (`search_instruments`, `analyze_instrument`, `get_market_universe`, `get_shortlist_briefing`, `get_portfolio_allocation`, `get_training_candidates`, `get_model_health`, `get_recent_learning_summary`) plus a composed workflow summary over the same stages
- `src/fortuna/telegram/formatters.py`
  - formats typed search and analysis responses into Telegram text
- `src/fortuna/app/advisory_service.py`
  - builds canonical analysis contracts from `FortunaSessionEngine`
- `src/fortuna/agentic/contracts.py`
  - typed search, analysis, and model-health payloads
- `src/fortuna/telegram/runtime.py`
  - polls Telegram updates, filters by configured chat id, and sends replies
- `src/fortuna/data/instruments.py`
  - resolves equities, front-month or explicit futures, and explicit options
- `src/fortuna/app/session_engine.py`
  - loads bars, runs deterministic strategies, optional ML/RL, and the main agentic decision
- `src/fortuna/agentic/`
  - produces the final advisory action, confidence, rationale, and risk notes

That means the Telegram bot is just another interface to the same decision
engine the dashboard uses. If the main agent says `BUY`, `SELL`, `EXIT_*`,
`DO_NOT_ENTER`, or `HOLD`, the Telegram response is built from that result.

## 1. Telegram bot setup

1. In Telegram, open **BotFather**.
2. Create a bot with `/newbot`.
3. Copy the bot token.
4. Start a chat with your bot and send any message once.
5. Get the chat id for that conversation.

Set these in `.env`:

```env
FORTUNA_AGENTIC_ENABLED=1
FORTUNA_TELEGRAM_ENABLED=1
FORTUNA_TELEGRAM_PROVIDER=telegram
FORTUNA_TELEGRAM_BOT_TOKEN=...
FORTUNA_TELEGRAM_CHAT_ID=...
```

Recommended for the first live check:

- `FORTUNA_AGENTIC_ENABLED=1`
- `FORTUNA_TELEGRAM_ENABLED=1`
- leave real-money execution disabled
- ML/RL can stay disabled; the bot will still work with deterministic analysis

Optional but useful:

```env
FORTUNA_TELEGRAM_DEDUPE_MEMORY=500
FORTUNA_TELEGRAM_MAX_PER_SYMBOL_PER_SESSION=10
FORTUNA_TELEGRAM_MIN_INTERVAL_SECONDS=300
FORTUNA_TELEGRAM_QUIET_HOURS_ENABLED=true
FORTUNA_CONVERSATIONAL_ADAPTER_ENABLED=true
FORTUNA_CONVERSATIONAL_ADAPTER_MODE=heuristic
```

Optional OpenAI planner mode:

```env
FORTUNA_CONVERSATIONAL_ADAPTER_MODE=openai
FORTUNA_OPENAI_API_KEY=...
FORTUNA_OPENAI_MODEL=gpt-5.4-mini
```

## 2. Validate before running

Run:

```powershell
uv run python scripts/run_operator_preflight.py
```

For a deeper Telegram/API sanity check:

```powershell
uv run python scripts/test_smartapi_env.py
```

## 3. Start the Telegram assistant

```powershell
uv run python scripts/run_telegram_bot.py
```

The polling runtime stores its offset under:

```text
logs/telegram_bot/state.json
```

If the conversational adapter is enabled, the bot also accepts selected
free-form prompts and normalizes them into typed advisory tool calls.

If `FORTUNA_CONVERSATIONAL_ADAPTER_MODE=openai`, that planning step is handled
by an OpenAI Responses API tool-calling layer above the same typed Fortuna
tools. It remains advisory-only and does not bypass the main decision core.

## 4. Supported commands and tags

The bot understands either slash commands or tag-style prompts.

### Help

```text
/help
#help
```

### Search

```text
/search RELIANCE
#search NIFTY
/universe 1d 30d limit=10
/brief 5m 20d limit=5
/allocate 5m 20d limit=5 maxpos=3 perexp=1 sameside=2
/candidates ml 5m 20d limit=8
/workflow 5m 20d limit=8
/health
```

This returns matching symbols such as:

- equity: `RELIANCE.NS`
- futures: `RELIANCE.FUT`
- option contracts for an underlying when available from the loaded instrument master

### Analyze equity

```text
/analyze RELIANCE
/analyze ICICIBANK 15m 20d
#analyze TCS
```

### Conversational examples

```text
How is Reliance looking on 15m for 20d?
Should I enter NIFTY CE 25000 28MAY2026?
Find Reliance
Show liquid high-volume stocks
How would you allocate the top liquid stock setups into a small portfolio?
Show model health
```

### Analyze future

```text
/analyze RELIANCE FUT
/analyze NIFTY FUT 15m
/analyze RELIANCE.FUT.27NOV2025
```

### Analyze option

Use explicit contract syntax:

```text
/analyze NIFTY CE 25000 28MAY2026
/analyze BANKNIFTY PE 52000 28MAY2026 5m 10d
/analyze NIFTY.OPT.CE.25000.28MAY2026
```

Canonical option symbol format inside Fortuna is:

```text
BASE.OPT.CE.<STRIKE>.<DDMMMYYYY>
BASE.OPT.PE.<STRIKE>.<DDMMMYYYY>
```

Examples:

- `NIFTY.OPT.CE.25000.28MAY2026`
- `CROMPTON.OPT.PE.400.26MAY2026`

## 5. How the replies should be read

Telegram analysis replies include:

- resolved symbol
- segment (equity / future / option)
- timeframe and lookback
- last bar close
- final advisory decision
- confidence
- top reasons from the agentic rationale
- risk notes if any
- current action plan
- winning deterministic strategy when available

Allocation replies include:

- selected setups
- skipped setups with reasons
- simple allocation weights
- exposure keys and limit notes

Important:

- The reply is **advisory only**.
- A `BUY` / `SELL` means the current closed-bar analysis favors entry.
- An `EXIT_*` means the current advisory favors closing exposure.
- `HOLD` / `DO_NOT_ENTER` means wait for a cleaner setup.

## 6. Suggested smoother conversation flow

Good operator flow:

1. Search first if you're unsure about the symbol:
   - `/search RELIANCE`
2. Analyze the resolved instrument:
   - `/analyze RELIANCE`
   - `/analyze RELIANCE FUT`
3. Re-check on the next bar close before acting.
4. For options, keep the request explicit:
   - `/analyze NIFTY CE 25000 28MAY2026`
5. If several setups look good, ask for a constrained basket:
   - `/allocate 5m 20d limit=5 maxpos=3 perexp=1 sameside=2`

For a smoother conversation style, think of the bot as a short loop:

1. identify the instrument
   - `/search RELIANCE`
   - `/search NIFTY`
2. ask for the advisory read
   - `/analyze RELIANCE`
   - `/analyze RELIANCE FUT`
3. refine the horizon if needed
   - `/analyze RELIANCE 15m 20d`
4. for options, stay contract-specific
   - `/analyze NIFTY CE 25000 28MAY2026`

## 7. Current limitations

- The bot is polling-based, not webhook-based.
- It currently responds only to the configured `FORTUNA_TELEGRAM_CHAT_ID`.
- Futures are supported through the existing instrument registry.
- Options work best when the contract is explicit.
- Generic option-chain exploration is still lighter than the equity/futures search path.
- The heuristic conversational adapter is safety-first; explicit commands remain the most reliable path.
- The OpenAI planner is optional and should be treated as a router above typed tools, not as a trading-decision engine.

## 8. Common mistakes and troubleshooting

| Mistake | What happens | Fix |
|---------|--------------|-----|
| `/analyze NIFTY CE 25000` (no expiry) | Incomplete-option guidance | `/analyze NIFTY CE 25000 28MAY2026` |
| `/analyze` or `/analyze 15m` only | Clarification with examples | `/analyze RELIANCE 15m` |
| Free text without a command prefix | Unsupported reply listing commands | Use `/search`, `/analyze`, or `/help` |
| Unknown symbol | Resolve error with `/search` suggestion | `/search RELIANCE` then `/analyze RELIANCE` |
| Empty search results | No hits plus next-step hints | Broaden query or use explicit option syntax |
| Bot silent / no reply | Chat id filter | Confirm message chat matches `FORTUNA_TELEGRAM_CHAT_ID` |
| Duplicate replies after restart | Offset state | Check `logs/telegram_bot/state.json`; delete only if you accept replay |

Request traces: [telegram_request_audit.md](telegram_request_audit.md) (`logs/telegram/requests.jsonl` when audit is enabled).
