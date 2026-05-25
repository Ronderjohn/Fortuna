# Strategy Walkthrough — EMA Crossover

## File

`strategies/generated/ema_crossover.json`

## Structure

1. **Metadata** — name, symbol, timeframe, tags
2. **Indicators** — `ema_fast` (12), `ema_slow` (26)
3. **Entry** — fast EMA crosses above slow EMA
4. **Exit** — fast EMA crosses below slow EMA
5. **Risk** — 3% stop loss, 6% take profit, 95% position fraction

## Execution flow

```
load_strategy()
  → MarketDataManager.get_ohlcv()
  → IndicatorEngine.compute()
  → StrategyCompiler.compile()  → entries, exits
  → BacktestEngine.run()      → vectorbt Portfolio
  → StrategyScorer.score()
  → StrategyRanker.persist()
```

## Customize

Edit windows in JSON:

```json
"params": { "window": 9 }
```

Change symbol/timeframe via CLI:

```powershell
python scripts/run_backtest.py --strategy strategies/generated/ema_crossover.json --symbol MSFT
```

## Condition types

| Type | Example |
|------|---------|
| `crossover` | `ema_fast` crosses above `ema_slow` |
| `crossunder` | Opposite |
| `compare` | `rsi` < 30 |
| `and` / `or` / `not` | Combine conditions |

## RSI mean reversion

See `strategies/generated/rsi_mean_reversion.json` for `compare` + `and` conditions with an SMA trend filter.
