# Intraday strategy library

Curated **5m NSE** strategies based on widely used intraday approaches (ORB, VWAP, EMA, MACD, Bollinger, Donchian, RSI scalp). These are **not** guaranteed profitable — they are standard baselines for comparable backtesting in Fortuna.

| Strategy | Type | Typical use |
|----------|------|-------------|
| `orb_15m` | Builtin ORB | Opening range breakout |
| `../builtin/mmts.json` | Builtin MMTS | Measured-move trend |
| `macd_trend_5m` | JSON | MACD + 50 EMA filter |
| `bollinger_breakout_5m` | JSON | BB squeeze / expansion |
| `donchian_breakout_20` | JSON | 20-bar high breakout |
| `rsi_scalp_5m` | JSON | RSI(7) oversold scalp |
| `ema_pullback_5m` | JSON | 9/21 EMA pullback |
| `../generated/vwap_reclaim.json` | JSON | VWAP reclaim |
| `../generated/ema_crossover.json` | JSON | EMA cross |

Run full suite:

```powershell
.\scripts\run_intraday_suite.ps1
```
