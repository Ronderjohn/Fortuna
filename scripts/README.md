# CLI scripts (optional)

**Primary workflow:** [Streamlit dashboard](../README.md#streamlit-dashboard-primary-ui) — `uv run python scripts/run_dashboard.py`

These scripts are for batch jobs, CI, and debugging. They may write under `logs/` (gitignored).

| Script | Purpose |
|--------|---------|
| `run_dashboard.py` | Start Streamlit app |
| `run_backtest.py` | Single-strategy vectorbt backtest |
| `run_strategy_tester_report.py` | Export CSV/JSON/charts to `logs/strategy_tester` (`--no-charts` default-friendly) |
| `run_paper_competition.py` | Multi-strategy paper PnL (`export_charts` off by default) |
| `run_institutional_backtest.py` | Walk-forward OOS suite |
| `run_arena.py` / `run_tournament.py` | Param search tournaments |
| `smartapi_backfill.py` | Refresh Parquet cache |
| `test_smartapi_env.py` | Credential / API smoke test |
