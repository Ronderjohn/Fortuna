"""Paper competition: every strategy gets the same starting capital."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.config.settings import Settings, get_settings
from fortuna.data.manager import MarketDataManager
from fortuna.paper.account import PaperAccount
from fortuna.paper.engine import PaperTradeEngine
from fortuna.strategy.loader import load_strategy
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PaperCompetitionConfig:
    symbol: str = "ICICIBANK.NS"
    timeframe: str = "5m"
    days: int = 30
    data_source: Optional[str] = None
    strategy_dirs: list[Path] = field(
        default_factory=lambda: [
            Path("strategies/intraday"),
            Path("strategies/generated"),
            Path("strategies/builtin"),
        ]
    )
    init_cash: float = 100_000.0
    output_dir: Path = Path("logs/paper_competition")
    export_charts: bool = False


@dataclass
class PaperCompetitionResult:
    accounts: list[PaperAccount]
    champion: Optional[PaperAccount]
    run_dir: Path
    symbol: str
    timeframe: str
    bars: int


class PaperCompetition:
    """
    Run all strategies on the same OHLCV history.

    Each strategy receives ``init_cash`` independently; results show wins, losses,
    and final equity side by side (TradingView-style paper book).
    """

    def __init__(
        self,
        config: PaperCompetitionConfig,
        settings: Optional[Settings] = None,
    ) -> None:
        self.config = config
        self.settings = settings or get_settings()
        if not self.config.symbol:
            self.config.symbol = self.settings.default_symbol
        if self.config.days <= 0:
            self.config.days = self.settings.default_days
        if config.data_source:
            self.settings = self.settings.model_copy(update={"data_source": config.data_source})
        self.settings = self.settings.model_copy(update={"init_cash": config.init_cash})
        self._mdm = MarketDataManager(self.settings, data_source=config.data_source)
        self._engine = PaperTradeEngine(self.settings)

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

    def run(self) -> PaperCompetitionResult:
        cfg = self.config
        ohlcv = self._mdm.get_ohlcv(cfg.symbol, cfg.timeframe, days=cfg.days)
        logger.info(
            "Paper competition %s %s: %d bars, init_cash=%.0f per strategy",
            cfg.symbol,
            cfg.timeframe,
            len(ohlcv),
            cfg.init_cash,
        )

        accounts: list[PaperAccount] = []
        trade_logs: dict[str, list] = {}
        chart_sessions: list[tuple[Path, object, object]] = []

        for path in self._strategy_paths():
            strategy = load_strategy(path)
            session = self._engine.run(
                strategy,
                ohlcv,
                symbol=cfg.symbol,
                init_cash=cfg.init_cash,
            )
            accounts.append(session.account)
            if cfg.export_charts:
                chart_sessions.append((path, strategy, session))
            trades = session.account.trade_returns
            trade_logs[path.stem] = [
                {"trade": i + 1, "return_pct": round(r * 100, 4)}
                for i, r in enumerate(trades)
            ]

        accounts.sort(key=lambda a: (a.profit_pct, a.wins), reverse=True)
        champion = accounts[0] if accounts else None

        run_dir = self._write_results(accounts, trade_logs, ohlcv)
        if cfg.export_charts and chart_sessions:
            self._export_charts(run_dir, chart_sessions, ohlcv)
        self._print_summary(accounts, champion)

        return PaperCompetitionResult(
            accounts=accounts,
            champion=champion,
            run_dir=run_dir,
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            bars=len(ohlcv),
        )

    def _write_results(
        self,
        accounts: list[PaperAccount],
        trade_logs: dict,
        ohlcv: pd.DataFrame,
    ) -> Path:
        cfg = self.config
        run_id = f"{cfg.symbol}_{cfg.timeframe}_{int(time.time())}"
        run_dir = self.settings.resolve_path(cfg.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "bars": len(ohlcv),
            "period_start": str(ohlcv.index.min()),
            "period_end": str(ohlcv.index.max()),
            "init_cash_per_strategy": cfg.init_cash,
            "competitors": [a.to_dict() for a in accounts],
        }
        (run_dir / "competition_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        import csv

        csv_path = run_dir / "leaderboard.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(accounts[0].to_dict().keys()) if accounts else [],
            )
            writer.writeheader()
            for a in accounts:
                writer.writerow(a.to_dict())

        (run_dir / "trade_returns.json").write_text(
            json.dumps(trade_logs, indent=2),
            encoding="utf-8",
        )
        return run_dir

    def _export_charts(
        self,
        run_dir: Path,
        sessions: list,
        ohlcv: pd.DataFrame,
    ) -> None:
        from fortuna.reporting.strategy_tester.chart_context import export_strategy_chart_bundle
        from fortuna.strategies.builtin.dispatch import is_builtin_strategy, run_builtin_backtest

        cfg = self.config
        charts_root = run_dir / "charts"
        charts_root.mkdir(parents=True, exist_ok=True)

        for path, strategy, session in sessions:
            try:
                if is_builtin_strategy(strategy):
                    bt = run_builtin_backtest(
                        strategy, ohlcv, symbol=cfg.symbol, init_cash=cfg.init_cash
                    )
                else:
                    bt = self._engine._runner.run(strategy, ohlcv, symbol=cfg.symbol)
                export_strategy_chart_bundle(
                    strategy,
                    ohlcv,
                    bt,
                    charts_root / path.stem,
                    strategy_name=path.stem,
                    symbol=cfg.symbol,
                    timeframe=cfg.timeframe,
                    init_cash=cfg.init_cash,
                )[0]
            except Exception as exc:
                logger.warning("Chart export failed for %s: %s", path.stem, exc)

    @staticmethod
    def _print_summary(accounts: list[PaperAccount], champion: Optional[PaperAccount]) -> None:
        print("\n" + "=" * 100)
        print("PAPER COMPETITION — equal starting capital per strategy")
        print("=" * 100)
        for a in accounts:
            print(a.format_row())
        print("=" * 100)
        if champion:
            print(
                f"CHAMPION: {champion.strategy_name} | "
                f"P&L {champion.profit_pct:+.2f}% | "
                f"Wins {champion.wins} / Losses {champion.losses}"
            )
