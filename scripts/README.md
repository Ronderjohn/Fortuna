# CLI scripts

**Primary workflow:** [Streamlit dashboard](../README.md#quick-start--the-dashboard)

Fortuna's scripts are grouped around a few practical workflows. Most operators
will use the dashboard, preflight checks, training, and promotion commands.

## Dashboard / runtime

| Script | Purpose |
|---|---|
| `run_dashboard.py` | Start the Streamlit dashboard |
| `run_operator_preflight.py` | Read-only operator readiness check before market open |
| `run_telegram_bot.py` | Start the Telegram polling assistant for search and analysis |
| `diagnose_live_feed.py` | Inspect live-feed wiring and runtime state |

## Training

| Script | Purpose |
|---|---|
| `run_rl_train.py` | Train one RL policy checkpoint |
| `nightly_train.py` | Run the nightly RL/ML training and promotion pipeline |
| `train_ml_signal_scorer.py` | Train ML scorer artifacts from agentic learning rows |
| `train_regime_detector.py` | Train the regime detector |

## Promotion

| Script | Purpose |
|---|---|
| `promote_model.py` | Promote RL or ML artifacts through the unified registry |
| `promote_policy.py` | Promote an RL run, optionally overriding the symbol pointer |

## Preflight / smoke / diagnostics

| Script | Purpose |
|---|---|
| `test_smartapi_env.py` | Credential and SmartAPI import smoke test |
| `verify_env.ps1` | Quick local Python/package/test sanity check |
| `check_gpu.py` | Report GPU/CuPy availability |
| `smoke_account_sync.py` | Exercise paper-account sync path |

## Backtest / research

| Script | Purpose |
|---|---|
| `run_backtest.py` | Single-strategy backtest |
| `run_strategy_tester_report.py` | Export strategy tester bundle under `logs/strategy_tester` |
| `run_paper_competition.py` | Multi-strategy paper competition |
| `run_institutional_backtest.py` | Walk-forward institutional backtest suite |
| `run_arena.py` / `run_tournament.py` | Strategy search and tournament workflows |
| `smartapi_backfill.py` | Refresh local OHLCV cache |

## Common commands

```powershell
# Operator preflight
uv run python scripts/run_operator_preflight.py

# Start dashboard
uv run python scripts/run_dashboard.py

# Start Telegram assistant
uv run python scripts/run_telegram_bot.py

# Train RL
uv run --group rl python scripts/run_rl_train.py --help

# Train ML scorer
uv run python scripts/train_ml_signal_scorer.py --help

# Promote validated artifact
uv run python scripts/promote_model.py --help
```
