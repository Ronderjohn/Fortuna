"""Leaderboard ranking tests."""

from fortuna.arena.leaderboard import StrategyLeaderboard
from fortuna.arena.param_search import ParamSearchResult
from fortuna.reporting.strategy_report import StrategyReport


def _result(name: str, profit: float, win: float) -> ParamSearchResult:
    report = StrategyReport(
        strategy_name=name,
        candidate_id=f"{name}_0",
        symbol="TEST",
        timeframe="5m",
        profit_pct=profit,
        win_ratio_pct=win,
        total_trades=10,
        winning_trades=5,
        losing_trades=5,
        max_drawdown_pct=5.0,
        profit_factor=1.5,
        sharpe_ratio=1.0,
        avg_trade_pct=0.5,
        gross_profit_pct=10.0,
        gross_loss_pct=5.0,
        composite_score=50.0,
        passed_validation=True,
    )
    from pathlib import Path

    from fortuna.strategy.schema import StrategyDefinition

    strat = StrategyDefinition.model_validate({"name": name, "indicators": [], "rules": {}})
    return ParamSearchResult(
        base_name=name,
        source_path=Path("x.json"),
        best_candidate_id=report.candidate_id,
        best_strategy=strat,
        best_report=report,
        all_reports=[report],
        variants_tested=1,
    )


def test_leaderboard_picks_champion_by_profit() -> None:
    board = StrategyLeaderboard(
        [_result("slow", 5.0, 40.0), _result("fast", 15.0, 55.0)],
        rank_by="profit_pct",
    )
    champ = board.overall_winner()
    assert champ is not None
    assert champ.strategy_name == "fast"
