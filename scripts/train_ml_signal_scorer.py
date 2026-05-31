#!/usr/bin/env python
"""Train a Fortuna ML signal scorer from agentic learning rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: E402,F401
from fortuna.config.settings import Settings  # noqa: E402
from fortuna.data.manager import MarketDataManager  # noqa: E402
from fortuna.ml.signal_scorer import SignalScorer  # noqa: E402
from fortuna.ml.training_data import (  # noqa: E402
    build_dataset_from_learning_rows,
    default_run_id,
    load_learning_rows,
)
from fortuna.ml.types import LabelConfig, ScorerMetadata  # noqa: E402
from fortuna.models.promotion import compute_ml_advisory_ready  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Fortuna ML signal scorer")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--strategy-name", default="agentic")
    parser.add_argument("--learning-dir", default="logs/agentic")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--model-kind", choices=("logistic", "hgb"), default="logistic")
    parser.add_argument("--output-dir", default="models/ml_signal_scorer/validated")
    args = parser.parse_args()

    settings = Settings.from_yaml(Path(args.config) if args.config else None)
    learning_rows_path = settings.resolve_path(Path(args.learning_dir)) / "learning_rows.jsonl"
    rows = load_learning_rows(
        learning_rows_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    mdm = MarketDataManager(settings=settings)
    ohlcv = mdm.get_ohlcv(args.symbol, args.timeframe, days=args.days, force_refresh=False)
    dataset = build_dataset_from_learning_rows(
        rows,
        ohlcv,
        eval_fraction=float(args.eval_fraction),
    )

    run_id = args.run_id or default_run_id(args.symbol, args.timeframe)
    label_config = LabelConfig(horizon_bars=int(args.horizon_bars))
    scorer = SignalScorer(random_state=int(args.seed))
    meta = ScorerMetadata(
        run_id=run_id,
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy_name=args.strategy_name,
        label_config=label_config,
        split_ranges=dataset.split_ranges,
        verdict_passed=True,
    )
    trained = scorer.fit(
        dataset.X_train,
        dataset.y_train,
        model_kind=args.model_kind,
        metadata=meta,
        eval_X=dataset.X_eval,
        eval_y=dataset.y_eval,
    )
    ready, reasons = compute_ml_advisory_ready(trained)
    scorer.metadata.verdict_passed = ready
    scorer.metadata.verdict_reasons = list(reasons)
    scorer.metadata.advisory_ready = ready

    output_base = settings.resolve_path(Path(args.output_dir))
    artifact_dir = output_base / run_id
    scorer.save(artifact_dir)

    print(f"[ML] Artifact saved -> {artifact_dir}")
    print(f"     Examples:      {dataset.example_count}")
    print(f"     OOS precision: {scorer.metadata.oos_metrics.precision:.3f}")
    print(f"     OOS ROC-AUC:   {scorer.metadata.oos_metrics.roc_auc:.3f}")
    print(f"     Advisory:      {scorer.metadata.advisory_ready}")
    if scorer.metadata.verdict_reasons:
        print(f"     Reasons:       {', '.join(scorer.metadata.verdict_reasons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
