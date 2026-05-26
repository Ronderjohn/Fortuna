#!/usr/bin/env python
"""Evaluate an existing RL checkpoint against OOS folds and print metrics.

Example:
    uv run python scripts/run_rl_eval.py --run-id 20260601_143022
    uv run python scripts/run_rl_eval.py --checkpoint-dir models/validated/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Warm pre-existing circular: importing calendar first breaks the
# data.manager -> backtesting.standard -> paper -> data.manager cycle.
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: E402,F401

from fortuna.rl.training.checkpoint import PolicyCheckpoint  # noqa: E402
from fortuna.rl.training.trainer import FortunaRLTrainer, TrainerConfig  # noqa: E402


def _find_checkpoint_dir(run_id: str, models_root: Path) -> Path:
    for sub in ("validated", "rejected", "running"):
        cand = models_root / sub / run_id
        if (cand / "policy.zip").exists():
            return cand
    raise SystemExit(f"checkpoint with run_id={run_id} not found under {models_root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--symbol", default=None, help="Override metadata symbol")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    if args.checkpoint_dir:
        cp = Path(args.checkpoint_dir)
    elif args.run_id:
        cp = _find_checkpoint_dir(args.run_id, Path(args.models_root))
    else:
        raise SystemExit("provide --run-id or --checkpoint-dir")

    meta_path = cp / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"metadata.json missing at {meta_path}")
    meta = PolicyCheckpoint.read(meta_path)

    print(f"\n=== RL Checkpoint {meta.run_id} ===")
    print(f"Symbol:          {meta.symbol}  {meta.timeframe}")
    print(f"Policy:          {meta.policy_type}")
    print(f"Total timesteps: {meta.total_timesteps}")
    print(f"Folds:           {meta.n_folds}")
    print(f"OOS Sharpe:      {meta.oos_metrics.sharpe_ratio:.3f}")
    print(f"OOS PF:          {meta.oos_metrics.profit_factor:.3f}")
    print(f"OOS Trades:      {meta.oos_metrics.total_trades}")
    print(f"OOS MaxDD%:      {meta.oos_metrics.max_drawdown_pct * 100:.2f}")
    print(f"Win rate%:       {meta.oos_metrics.win_rate_pct:.2f}")
    print(f"FilterVerdict:   {'PASS' if meta.verdict_passed else 'FAIL'}  score={meta.verdict_score:.1f}")
    if meta.verdict_reasons:
        print(f"  Reasons:       {meta.verdict_reasons}")
    print()

    # Re-evaluate live if symbol+timeframe+days provided.
    if args.symbol or args.timeframe or args.days:
        cfg = TrainerConfig(
            symbol=args.symbol or meta.symbol,
            timeframe=args.timeframe or meta.timeframe,
            days=args.days or 180,
            train_bars=meta.train_bars or 156,
            test_bars=meta.test_bars or 78,
            step_bars=meta.step_bars or 78,
        )
        trainer = FortunaRLTrainer(cfg)
        folds = trainer.load_folds()
        eval_folds = folds[int(len(folds) * 0.7):] or folds[-1:]

        from stable_baselines3 import PPO

        model = PPO.load(str(cp / "policy"), device="cpu")
        metrics, n = trainer.evaluate(model, eval_folds)
        print("=== Re-evaluated against fresh data ===")
        print(f"Fresh trades:    {n}")
        print(f"Fresh Sharpe:    {metrics.risk.sharpe_ratio:.3f}")
        print(f"Fresh PF:        {metrics.performance.profit_factor:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
