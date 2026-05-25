"""End-to-end institutional backtesting pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from fortuna.backtesting.standard.calendar import filter_session_bars, trading_days_count
from fortuna.backtesting.standard.comparison import (
    build_comparison_table,
    export_comparison,
    print_comparison_table,
)
from fortuna.backtesting.standard.config import InstitutionalBacktestConfig
from fortuna.backtesting.standard.filters import evaluate_strategy
from fortuna.backtesting.standard.robustness import monte_carlo_trade_shuffle
from fortuna.backtesting.standard.runner import InstitutionalRunner
from fortuna.backtesting.standard.splits import (
    chronological_split,
    estimate_calendar_days,
    validate_history_length,
)
from fortuna.config.settings import Settings, get_settings
from fortuna.data.manager import MarketDataManager
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


class InstitutionalPipeline:
    """
    Run all strategies under identical institutional conditions.

    - 6+ months 5m data
    - Train 70% / Val 15% / Test 15%
    - NSE session + square-off
    - Realistic costs
    - OOS test ranking (Sharpe, PF, DD, expectancy — not profit alone)
    """

    def __init__(
        self,
        config: InstitutionalBacktestConfig,
        settings: Optional[Settings] = None,
    ) -> None:
        self.config = config
        self.settings = settings or get_settings()
        self._mdm = MarketDataManager(self.settings, data_source=config.data_source)
        self._runner = InstitutionalRunner(config, self.settings)

    def _strategy_paths(self) -> list[Path]:
        paths: list[Path] = []
        seen: set[str] = set()
        for rel in self.config.strategy_dirs:
            d = self.settings.resolve_path(rel)
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                if p.name not in seen:
                    seen.add(p.name)
                    paths.append(p)
        if not paths:
            raise FileNotFoundError(f"No strategies in {self.config.strategy_dirs}")
        return paths

    def _load_ohlcv(self) -> "pd.DataFrame":
        import pandas as pd

        cfg = self.config
        days = cfg.days or cfg.preferred_history_days
        days = max(days, cfg.min_history_days)
        ohlcv = self._mdm.get_ohlcv(cfg.symbol, cfg.timeframe, days=days)
        ohlcv = filter_session_bars(ohlcv, cfg.session)
        validate_history_length(ohlcv, cfg.min_history_days)
        return ohlcv

    def run(self) -> Path:
        cfg = self.config
        ohlcv = self._load_ohlcv()
        split = chronological_split(ohlcv, cfg.splits)
        cal_days = estimate_calendar_days(ohlcv)
        tdays = trading_days_count(ohlcv)

        run_id = f"{cfg.symbol}_{cfg.timeframe}_{int(time.time())}"
        run_dir = self.settings.resolve_path(cfg.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        grid = None
        if cfg.param_grid_path and cfg.param_grid_path.exists():
            grid = json.loads(cfg.param_grid_path.read_text(encoding="utf-8"))

        meta = {
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "calendar_days": cal_days,
            "trading_days": tdays,
            "bars": len(ohlcv),
            "splits": {
                "train": split.train_range,
                "validation": split.val_range,
                "test": split.test_range,
            },
            "costs": {
                "brokerage": cfg.costs.brokerage_rate,
                "exchange": cfg.costs.exchange_fee_rate,
                "slippage": cfg.costs.slippage_rate,
                "spread": cfg.costs.spread_rate,
            },
            "execution": cfg.execution.value,
            "initial_capital": cfg.initial_capital,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        results = []
        for path in self._strategy_paths():
            logger.info("Institutional backtest: %s", path.name)
            from fortuna.strategies.builtin.dispatch import is_builtin_strategy
            from fortuna.strategy.loader import load_strategy

            if is_builtin_strategy(load_strategy(path)) or not grid:
                res = self._runner.run_strategy_fixed(path, split, symbol=cfg.symbol)
            else:
                res = self._runner.run_strategy(path, split, symbol=cfg.symbol, grid_override=grid)

            strat_dir = run_dir / path.stem
            strat_dir.mkdir(exist_ok=True)
            for phase_name in ("train", "validation", "test"):
                sr = getattr(res, phase_name)
                sr.report.export(strat_dir / phase_name, generate_charts=True)

            test_rep = res.test.report
            verdict = evaluate_strategy(
                test_rep.performance,
                test_rep.risk,
                thresholds=cfg.filters,
                benchmarks=cfg.benchmarks,
                period_days=max(1, cal_days // 7),
            )
            mc = monte_carlo_trade_shuffle(
                res.test.report.trades,
                cfg.initial_capital,
                iterations=cfg.monte_carlo_iterations,
            )
            (strat_dir / "filter_verdict.json").write_text(
                json.dumps(verdict.to_dict(), indent=2),
                encoding="utf-8",
            )
            (strat_dir / "monte_carlo.json").write_text(
                json.dumps(
                    {
                        "iterations": mc.iterations,
                        "median_final_equity": mc.median_final_equity,
                        "prob_profit": mc.prob_profit,
                        "p5_final_equity": mc.p5_final_equity,
                        "p95_final_equity": mc.p95_final_equity,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            results.append(res)

        test_days = max(30, int(cal_days * cfg.splits.test))
        comp = build_comparison_table(
            results,
            phase="test",
            period_days=test_days,
            thresholds=cfg.filters,
            benchmarks=cfg.benchmarks,
        )
        export_comparison(comp, run_dir, "oos_test_comparison")
        print_comparison_table(comp, "INSTITUTIONAL OOS TEST — ranked by composite (Sharpe/PF/DD/expectancy)")

        val_comp = build_comparison_table(
            results,
            phase="validation",
            period_days=max(30, int(cal_days * cfg.splits.validation)),
            thresholds=cfg.filters,
            benchmarks=cfg.benchmarks,
        )
        export_comparison(val_comp, run_dir, "validation_comparison")

        return run_dir
