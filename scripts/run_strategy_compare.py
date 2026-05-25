#!/usr/bin/env python
"""Compare strategies TradingView-style: trades, P&L, profit factor, equity curve CSV."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("FORTUNA_CONFIG", "configs/intraday.yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

import argparse

import pandas as pd

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.config.settings import load_settings
from fortuna.data.manager import MarketDataManager
from fortuna.arena.param_search import _rank_reports
from fortuna.reporting.strategy_report import build_report, print_leaderboard
from fortuna.search.indicator_cache import IndicatorCache
from fortuna.strategy.loader import load_strategy
from fortuna.strategies.builtin.mmts import is_mmts_strategy, run_mmts_backtest
from fortuna.evaluation.scorer import StrategyScorer


def _run_one(
    strategy_path: Path, ohlcv: pd.DataFrame, symbol: str, settings
) -> tuple:
    strategy = load_strategy(strategy_path)
    window = ohlcv.copy()
    if is_mmts_strategy(strategy):
        result = run_mmts_backtest(strategy, window, symbol=symbol, init_cash=settings.init_cash)
    else:
        result = NumPyBacktestRunner(settings).run(strategy, window, symbol=symbol)
    scorer = StrategyScorer(settings)
    report = build_report(
        strategy_name=strategy.name,
        candidate_id=strategy_path.stem,
        symbol=symbol,
        timeframe=strategy.timeframe,
        metrics=result.metrics,
        score=scorer.score(result.metrics, strategy.metadata),
    )
    trades = result.enriched_data.attrs.get("trades")
    return report, trades, result.metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="TradingView-style strategy comparison")
    from fortuna.config.settings import load_settings

    cfg = load_settings()
    parser.add_argument("--symbol", default=cfg.default_symbol)
    parser.add_argument("--timeframe", default=cfg.default_timeframe)
    parser.add_argument("--days", type=int, default=cfg.default_days)
    parser.add_argument("--output", default="logs/compare")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Strategy JSON paths (default: builtin mmts + generated/)",
    )
    args = parser.parse_args()

    settings = load_settings()
    mdm = MarketDataManager(settings, data_source=settings.data_source)
    print(f"Loading {args.symbol} {args.timeframe} ({args.days} days)...", flush=True)
    ohlcv = mdm.get_ohlcv(args.symbol, args.timeframe, days=args.days)
    print(f"  Bars: {len(ohlcv)}", flush=True)

    if args.strategies:
        paths = [Path(p) for p in args.strategies]
    else:
        paths = []
        for rel in [Path("strategies/intraday"), Path("strategies/generated"), Path("strategies/builtin")]:
            if rel.is_dir():
                paths.extend(sorted(rel.glob("*.json")))

    out_dir = Path(args.output) / f"{args.symbol}_{args.timeframe}"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for path in paths:
        if not path.exists():
            print(f"  Skip missing: {path}", flush=True)
            continue
        print(f"\n--- {path.name} ---", flush=True)
        report, trades, metrics = _run_one(path, ohlcv, args.symbol, settings)
        reports.append(report)
        print(
            f"  Total P&L: {report.profit_pct:+.2f}%  |  "
            f"Trades: {report.total_trades}  |  "
            f"Win rate: {report.win_ratio_pct:.1f}%  |  "
            f"Max DD: {report.max_drawdown_pct:.2f}%  |  "
            f"Profit factor: {report.profit_factor:.2f}",
            flush=True,
        )
        if trades is not None and len(trades) > 0:
            trades_path = out_dir / f"{path.stem}_trades.csv"
            trades.to_csv(trades_path, index=False)
            print(f"  Trades saved: {trades_path}", flush=True)
        summary_path = out_dir / f"{path.stem}_summary.json"
        summary_path.write_text(
            pd.Series(metrics.to_dict()).to_json(indent=2),
            encoding="utf-8",
        )
        if is_mmts_strategy(load_strategy(path)):
            from fortuna.strategies.builtin.mmts import mmts_params_from_strategy, simulate_mmts

            strat = load_strategy(path)
            eq, _, _ = simulate_mmts(
                ohlcv,
                mmts_params_from_strategy(strat),
                init_cash=settings.init_cash,
                size_frac=float(strat.risk.position_sizing.value),
            )
            pd.DataFrame({"equity": eq}, index=ohlcv.index[: len(eq)]).to_csv(
                out_dir / f"{path.stem}_equity.csv"
            )

    print("\n" + "=" * 88, flush=True)
    print("STRATEGY COMPARISON (TradingView-style metrics)", flush=True)
    ranked = _rank_reports(reports, "profit_pct")
    print_leaderboard(ranked, top_n=len(ranked))
    winner = ranked[0]
    print(
        f"\nCHAMPION (by P&L among strategies with trades): {winner.strategy_name} "
        f"— {winner.profit_pct:+.2f}% over {winner.total_trades} trades",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
