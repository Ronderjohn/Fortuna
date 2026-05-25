"""Arena parameter search tests."""

from pathlib import Path

from fortuna.arena.param_search import ParamSearchEngine


def test_param_search_finds_best_variant(
    sample_ohlcv,
    ema_crossover_path: Path,
) -> None:
    window = sample_ohlcv.iloc[-50:].copy()
    engine = ParamSearchEngine(max_workers=1)
    result = engine.search(
        ema_crossover_path,
        window,
        max_candidates=9,
        parallel=False,
        rank_by="profit_pct",
    )
    assert result.variants_tested >= 1
    assert isinstance(result.best_report.profit_pct, float)
    assert result.best_report.total_trades >= 0
    assert len(result.all_reports) >= 1
