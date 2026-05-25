"""vectorbt backtest integration (slow; requires uv sync --group vbt)."""

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def _require_vectorbt():
    pytest.importorskip("vectorbt")


def test_backtest_runs_vectorbt(
    _require_vectorbt,
    backtest_engine,
    ema_crossover_strategy,
    sample_ohlcv_backtest,
) -> None:
    result = backtest_engine.run(
        ema_crossover_strategy,
        sample_ohlcv_backtest,
        symbol="TEST",
        fast_metrics=True,
    )
    assert result.metrics.total_trades >= 0
    assert result.metrics.final_value > 0
