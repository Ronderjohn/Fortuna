#!/usr/bin/env python
"""Benchmark pipeline stages with timeouts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults

apply_low_spec_gpu_defaults()

from fortuna.backtesting.compiler import StrategyCompiler
from fortuna.backtesting.engine import BacktestEngine
from fortuna.backtesting.numpy_runner import NumPyBacktestRunner
from fortuna.config.settings import load_settings
from fortuna.data.manager import MarketDataManager
from fortuna.indicators.engine import IndicatorEngine
from fortuna.strategy.loader import load_strategy
from fortuna.strategy.schema import IndicatorSpec, IndicatorType
from fortuna.utils.timing import timed_step


def _timed(name: str, fn, timeout_sec: float) -> dict:
    t0 = time.perf_counter()
    details: dict[str, str] = {}
    try:
        with timed_step(name) as d:
            result = fn()
            if isinstance(result, int):
                d["rows"] = str(result)
            elif result is True:
                pass
            details.update(d)
        ms = (time.perf_counter() - t0) * 1000
        ok = ms <= timeout_sec * 1000 * 2
        return {"stage": name, "ms": round(ms, 1), "ok": ok, "error": None, "detail": result}
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return {"stage": name, "ms": round(ms, 1), "ok": False, "error": str(e), "detail": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Fortuna pipeline stages")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--strategy", default="strategies/generated/ema_crossover.json")
    args = parser.parse_args()

    settings = load_settings()
    if args.data_source:
        settings = settings.model_copy(update={"data_source": args.data_source})

    mdm = MarketDataManager(settings, data_source=args.data_source)
    strategy_path = settings.resolve_path(Path(args.strategy))
    strategy = load_strategy(strategy_path)
    strategy.symbol = args.symbol
    strategy.timeframe = args.timeframe

    report: dict = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_source": args.data_source or settings.data_source,
        "timestamp": datetime.now().isoformat(),
        "stages": [],
    }

    ohlcv_box: dict = {"df": None}

    def fetch():
        ohlcv_box["df"] = mdm.get_ohlcv(
            args.symbol, args.timeframe, force_refresh=False, days=args.days
        )
        return len(ohlcv_box["df"])

    report["stages"].append(_timed("fetch", fetch, 30))
    if not report["stages"][-1]["ok"]:
        _write_report(settings, report)
        return 1

    df = ohlcv_box["df"]
    engine = IndicatorEngine()

    def indicators():
        specs = [
            IndicatorSpec(id="ema_fast", type=IndicatorType.EMA, params={"window": 12}),
            IndicatorSpec(id="ema_slow", type=IndicatorType.EMA, params={"window": 26}),
        ]
        engine.compute(df, specs)
        return True

    report["stages"].append(_timed("indicators", indicators, 2))

    enriched_box: dict = {"df": None}

    def compile_signals():
        specs = strategy.indicators
        enriched_box["df"] = engine.compute(df, specs)
        compiler = StrategyCompiler()
        compiler.compile(strategy, enriched_box["df"])
        return True

    report["stages"].append(_timed("compile", compile_signals, 1))

    def numpy_bt():
        NumPyBacktestRunner(settings).run(strategy, df, symbol=args.symbol)
        return True

    report["stages"].append(_timed("numpy_bt", numpy_bt, 5))

    def vectorbt_bt():
        BacktestEngine(settings).run(strategy, df, symbol=args.symbol, fast_metrics=True)
        return True

    try:
        import vectorbt  # noqa: F401
    except ImportError:
        report["stages"].append(
            {
                "stage": "vectorbt",
                "ms": 0.0,
                "ok": True,
                "error": None,
                "detail": "skipped (uv sync --group vbt)",
            }
        )
    else:
        report["stages"].append(_timed("vectorbt", vectorbt_bt, 60))

    _write_report(settings, report)
    all_ok = all(s["ok"] for s in report["stages"])
    for s in report["stages"]:
        status = "OK" if s["ok"] else "FAIL"
        print(f"{s['stage']}: {s['ms']}ms [{status}]" + (f" — {s['error']}" if s["error"] else ""))
    return 0 if all_ok else 1


def _write_report(settings, report: dict) -> None:
    out_dir = settings.resolve_path(Path("logs/benchmarks"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bench_{report['symbol']}_{report['timeframe']}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
