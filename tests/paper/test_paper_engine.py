"""Paper trade engine smoke test."""

from fortuna.paper.engine import PaperTradeEngine
from fortuna.strategy.loader import load_strategy
from pathlib import Path


def test_paper_session_runs(sample_ohlcv, ema_crossover_path: Path) -> None:
    strategy = load_strategy(ema_crossover_path)
    result = PaperTradeEngine().run(
        strategy, sample_ohlcv.iloc[-45:], symbol="TEST", init_cash=100_000.0
    )
    assert result.account.init_cash == 100_000.0
    assert result.account.total_trades >= 0
    assert result.account.final_equity > 0
