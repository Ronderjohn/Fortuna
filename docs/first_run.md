# First Run Guide

This is the quickest safe path to get Fortuna running for the first time as an
**advisory-first operator workflow**.

For the fuller day-to-day guide, use
[how_to_use_fortuna.md](how_to_use_fortuna.md).

## 1. Install

```powershell
cd C:\dev\Fortuna
uv lock
uv sync
uv sync --group dev --group dashboard --group charts --group smartapi
copy .env.example .env
```

## 2. Configure `.env`

Fill the SmartAPI values you plan to use:

```env
SMARTAPI_API_KEY=...
SMARTAPI_CLIENT_CODE=...
SMARTAPI_PASSWORD=...
SMARTAPI_TOTP_SECRET=...
SMARTAPI_USE_LIVE_FEED=true
```

Safe first-run defaults:

```env
FORTUNA_EXECUTION_ENABLED=0
FORTUNA_AGENTIC_ENABLED=1
FORTUNA_MODEL_REGISTRY_ENABLED=1
FORTUNA_MODEL_PROMOTION_REQUIRED=1
FORTUNA_TELEGRAM_ENABLED=0
FORTUNA_NIGHTLY_AUTOMATION_ENABLED=0
```

Optional conservative config profile already prepared in repo:

```powershell
configs/intraday_first_run.yaml
```

## 3. Run preflight

```powershell
$env:FORTUNA_CONFIG = "configs/intraday_first_run.yaml"
uv run python scripts/run_operator_preflight.py --config configs/intraday_first_run.yaml
```

What you want to see:

- settings loaded
- SmartAPI credentials present if you expect live/cache-backed usage
- agentic enabled only if you want ensemble advisory
- warnings are acceptable for disabled ML or missing RL live pointers on a fresh setup

If you want a deeper SmartAPI validation:

```powershell
uv run python scripts/test_smartapi_env.py
```

## 4. Start the dashboard

```powershell
$env:FORTUNA_CONFIG = "configs/intraday_first_run.yaml"
uv run python scripts/run_dashboard.py
```

Use the dashboard as the main application surface for:

- signals and charts
- agentic advisory
- model-health checks
- workflow and acceptance artifacts

## 5. Try the operator workflow

Build a market universe:

```powershell
uv run python scripts/build_market_universe.py --source auto --limit 15
```

Analyze the top names:

```powershell
uv run python scripts/analyze_market_shortlist.py --analysis-limit 5
```

Create a briefing:

```powershell
uv run python scripts/brief_market_shortlist.py --analysis-limit 5
```

## 6. Rehearse nightly review locally

Run the safe local rehearsal:

```powershell
uv run python scripts/run_nightly_acceptance_dry_run.py
uv run python scripts/run_nightly_acceptance_dry_run.py --mode emit-replay
uv run python scripts/run_nightly_acceptance_dry_run.py --mode verify --report-dir reports/nightly
```

Recommended order:

1. `candidate`
2. `emit-replay`
3. `verify`

If `verify` fails on an older nightly folder, that can simply mean the nightly
artifacts were produced before the newer training-candidate / training-research
/ workflow-snapshot chain existed.

## 7. Next docs

- [how_to_use_fortuna.md](how_to_use_fortuna.md)
- [operator_runbook.md](operator_runbook.md)
- [telegram_trading_assistant.md](telegram_trading_assistant.md)
