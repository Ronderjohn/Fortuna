"""Fast NumPy backtest smoke test (no vectorbt)."""

from fortuna.backtesting.numpy_runner import NumPyBacktestRunner


def test_backtest_runs_numpy(
    numpy_backtest_runner: NumPyBacktestRunner,
    ema_crossover_strategy,
    sample_ohlcv_backtest,
) -> None:
    result = numpy_backtest_runner.run(
        ema_crossover_strategy,
        sample_ohlcv_backtest,
        symbol="TEST",
    )
    assert result.metrics.total_trades >= 0
    assert result.metrics.final_value > 0
