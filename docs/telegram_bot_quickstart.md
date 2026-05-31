# Telegram Bot Quickstart

This is the short operator guide for using the Fortuna Telegram bot.

If you want the deeper implementation and setup notes, read
`docs/telegram_trading_assistant.md`.

## 1. What the bot is for

The bot lets you:

- search for a stock/future/options contract
- ask Fortuna for its current advisory read
- get back the main action, confidence, reasons, and current action plan

It is an advisory interface, not a broker execution bot.

## 2. Minimum setup

In `.env`, keep these enabled:

```env
FORTUNA_AGENTIC_ENABLED=1
FORTUNA_TELEGRAM_ENABLED=1
FORTUNA_TELEGRAM_BOT_TOKEN=...
FORTUNA_TELEGRAM_CHAT_ID=...
FORTUNA_EXECUTION_ENABLED=0
```

Recommended for the first run:

- keep execution disabled
- ML/RL can remain disabled; the bot still works with deterministic analysis

## 3. Before starting

Run:

```powershell
uv run python scripts/run_operator_preflight.py
```

If that passes, start the bot:

```powershell
uv run python scripts/run_telegram_bot.py
```

## 4. Commands you will actually use

### Get help

```text
/help
```

### Search for a symbol

```text
/search RELIANCE
/search NIFTY
```

Use this when you are not sure what symbol or contract format Fortuna expects.

### Analyze an equity

```text
/analyze RELIANCE
/analyze ICICIBANK 15m 20d
```

Meaning:
- `15m` = timeframe
- `20d` = lookback days

### Analyze a future

```text
/analyze RELIANCE FUT
/analyze NIFTY FUT
/analyze RELIANCE.FUT.27NOV2025
```

### Analyze an option

```text
/analyze NIFTY CE 25000 28MAY2026
/analyze BANKNIFTY PE 52000 28MAY2026 5m 10d
```

Best practice:
- keep options explicit by type, strike, and expiry

## 5. How to use the bot properly

The clean operator loop is:

1. search first
   - `/search RELIANCE`
2. pick the instrument you want
3. analyze it
   - `/analyze RELIANCE`
   - `/analyze RELIANCE FUT`
4. if needed, refine timeframe/lookback
   - `/analyze RELIANCE 15m 20d`
5. re-check after the next bar close before acting

For options:

1. decide the exact contract
2. send the explicit option request
3. interpret the reply as contract-specific advice, not just underlying bias

## 6. How to read the reply

A reply usually includes:

- resolved symbol
- segment
- timeframe and lookback
- last bar close
- final decision
- confidence
- key reasons
- risk notes
- action plan

Interpretation:

- `BUY` = current advisory favors long-side entry
- `SELL` = current advisory favors short-side entry if the instrument/setup allows it
- `EXIT_LONG` / `EXIT_SHORT` = current advisory favors closing that side
- `DO_NOT_ENTER` = stay flat
- `HOLD` = wait for a clearer setup

## 7. Suggested conversation examples

### Example A: stock

```text
/search RELIANCE
/analyze RELIANCE
```

### Example B: future

```text
/search NIFTY
/analyze NIFTY FUT
```

### Example C: option

```text
/analyze NIFTY CE 25000 28MAY2026
```

### Example D: refine the view

```text
/analyze ICICIBANK 15m 20d
```

## 8. What the bot does not do

- it does not place live trades by default
- it does not replace the dashboard for deeper inspection
- it does not work best with vague natural-language prompts
- it does not infer a full option chain strategy from a loose sentence

The more specific the instrument request, the better the output.

## 9. If something looks off

Check:

1. `uv run python scripts/run_operator_preflight.py`
2. whether the bot process is running
3. whether your chat id matches the configured Telegram chat id
4. whether the symbol/contract is valid in the loaded instrument master

Common bad → good fixes:

| Bad input | Better input |
|-----------|----------------|
| `/analyze NIFTY CE 25000` | `/analyze NIFTY CE 25000 28MAY2026` |
| `/analyze 15m` | `/analyze RELIANCE 15m` |
| `what should I buy?` | `/search RELIANCE` then `/analyze RELIANCE` |
| `/analyze UNKNOWN` | `/search RELIANCE` first, then analyze the resolved symbol |

Polling offset is stored in `logs/telegram_bot/state.json`. Request audit rows
(when enabled) are in `logs/telegram/requests.jsonl` — see
[telegram_request_audit.md](telegram_request_audit.md).

For deeper details, use:
- `docs/telegram_trading_assistant.md`
- `docs/operator_runbook.md`
