# Institutional 5m Backtesting Standard

All Fortuna strategies are evaluated through the same framework so results are comparable.

## Quick start

```powershell
$env:PYTHONPATH = "src"
$env:FORTUNA_CONFIG = "configs/intraday.yaml"

# Backfill 1–2 years first (recommended)
.\.venv\Scripts\python.exe scripts\smartapi_backfill.py --symbol CROMPTON.NS --days 365 --force

# Run institutional pipeline
.\.venv\Scripts\python.exe scripts\run_institutional_backtest.py --symbol CROMPTON.NS --days 365 --min-days 180
```

## What it enforces

| Rule | Implementation |
|------|----------------|
| **Data** | 5m OHLCV, minimum 6 months (`--min-days 180`) |
| **Splits** | 70% train / 15% validation / 15% OOS test (chronological) |
| **Session** | 09:15–15:30 IST, square-off 15:15 |
| **Costs** | Brokerage + exchange + slippage + spread |
| **Optimization** | Param search on **train only** |
| **Final score** | **OOS test** period only |
| **Filters** | PF, Sharpe, DD, min trades, expectancy |
| **Ranking** | Composite (Sharpe, PF, DD, expectancy) — not profit alone |
| **Robustness** | Monte Carlo trade shuffle per strategy |
| **Reports** | Strategy Tester per split (train/val/test) |

## Output layout

```
logs/institutional/CROMPTON.NS_5m_<timestamp>/
  run_meta.json
  oos_test_comparison.csv
  validation_comparison.csv
  ema_crossover/
    train/    summary.json, trades.csv, equity_curve.csv, charts/
    validation/
    test/
    filter_verdict.json
    monte_carlo.json
  mmts/
  ...
```

## Python API

```python
from fortuna.backtesting.standard import InstitutionalBacktestConfig, InstitutionalPipeline

config = InstitutionalBacktestConfig(symbol="CROMPTON.NS", days=365)
run_dir = InstitutionalPipeline(config).run()
```

## Related modules

- `fortuna/reporting/strategy_tester/` — TradingView-style metrics & charts
- `fortuna/paper/` — Walk-forward paper league (optional second validation layer)
- `scripts/run_strategy_tester_report.py` — Full-history TV reports (no split)
