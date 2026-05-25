# First Run Guide

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

## 2. Run EMA crossover backtest

```powershell
python scripts/run_backtest.py `
  --strategy strategies/generated/ema_crossover.json `
  --symbol AAPL `
  --timeframe 1d
```

Expected output: JSON with `metrics`, `score`, and `critic` sections.

## 3. Check results

- Cached data: `data/market_cache/AAPL/1d.parquet`
- Strategy routing: `strategies/validated/` or `strategies/rejected/`
- Score sidecar: `ema_crossover.score.json`

## 4. Try other strategies

```powershell
python scripts/run_backtest.py --strategy strategies/generated/rsi_mean_reversion.json
```

For VWAP reclaim (1h bars):

```powershell
python scripts/run_backtest.py --strategy strategies/generated/vwap_reclaim.json --timeframe 1h
```

## 5. Run tests

```powershell
pytest
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| yfinance empty data | Check symbol and internet; use `--refresh` |
| vectorbt slow first run | Normal; subsequent runs use cache |
| Low score / rejected | Expected on some periods; tune strategy or threshold in `configs/default.yaml` |
