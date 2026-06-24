#!/usr/bin/env python
"""Train a Fortuna RL policy via PPO and persist the checkpoint.

Example:
    uv run python scripts/run_rl_train.py `
        --symbol ICICIBANK.NS --timeframe 5m --days 180 `
        --train-bars 156 --test-bars 78 --step-bars 78 `
        --total-timesteps 500000 --n-envs 4 --policy MlpPolicy `
        --checkpoint-dir models/

Output goes to:
    models/validated/{run_id}/   (FilterVerdict.passed = True)
    models/rejected/{run_id}/    (FilterVerdict.passed = False)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults  # noqa: E402

apply_low_spec_gpu_defaults()

# Warm pre-existing circular: importing calendar first breaks the
# data.manager -> backtesting.standard -> paper -> data.manager cycle.
from fortuna.app.training_candidates import (  # noqa: E402
    load_training_candidates_manifest,
    select_training_symbols,
)
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: E402,F401
from fortuna.rl.env.reward import RewardConfig  # noqa: E402
from fortuna.rl.training.trainer import FortunaRLTrainer, TrainerConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Fortuna RL policy")
    parser.add_argument("--symbol", default="")
    parser.add_argument(
        "--candidate-manifest",
        default="",
        help="Optional shortlist-driven training candidate manifest JSON",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="How many RL candidate symbols to train from the manifest",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Train all selected RL candidate symbols sequentially",
    )
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--train-bars", type=int, default=156)
    parser.add_argument("--test-bars", type=int, default=78)
    parser.add_argument("--step-bars", type=int, default=78)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-bars", type=int, default=20)
    parser.add_argument("--ppo-n-steps", type=int, default=2048,
                        help="PPO rollout length per env before each update")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--policy",
        dest="policy_type",
        choices=("MlpPolicy", "CnnPolicy"),
        default="MlpPolicy",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="auto=cuda only for CnnPolicy; MLP defaults to cpu",
    )
    parser.add_argument("--use-subproc", action="store_true",
                        help="Use SubprocVecEnv (parallel env workers)")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=None,
                        help="Override eval episodes; default = len(eval_folds)")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--baseline-sharpe", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-source",
        default=None,
        help="Override settings.data_source (e.g. 'smartapi' for >60 days of 5m)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass the OHLCV cache and refetch from source",
    )
    parser.add_argument(
        "--holding-penalty",
        type=float,
        default=None,
        help="Override RewardConfig.holding_penalty",
    )
    args = parser.parse_args()
    symbols = _resolve_rl_symbols(args)
    if not symbols:
        raise SystemExit("Provide --symbol or --candidate-manifest")

    status = 0
    for symbol in symbols:
        try:
            _train_one(args, symbol)
        except Exception as exc:  # noqa: BLE001
            status = 1
            print(f"[RL] {symbol} failed: {exc}")
    return status


def _resolve_rl_symbols(args) -> list[str]:
    if str(args.symbol or "").strip():
        return [str(args.symbol).strip()]
    if str(args.candidate_manifest or "").strip():
        manifest = load_training_candidates_manifest(Path(args.candidate_manifest))
        top_n = None if args.batch else args.top_n
        return select_training_symbols(manifest, target="rl", top_n=top_n)
    return []


def _train_one(args, symbol: str) -> None:

    cfg_kwargs: dict[str, object] = {
        "symbol": symbol,
        "timeframe": args.timeframe,
        "days": args.days,
        "train_bars": args.train_bars,
        "test_bars": args.test_bars,
        "step_bars": args.step_bars,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "n_bars": args.n_bars,
        "policy_type": args.policy_type,
        "device": args.device,
        "use_subproc": args.use_subproc,
        "ppo_n_steps": args.ppo_n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "checkpoint_dir": Path(args.checkpoint_dir),
        "eval_freq": args.eval_freq,
        "early_stop_patience": args.patience,
        "baseline_sharpe": args.baseline_sharpe,
        "seed": args.seed,
        "data_source": args.data_source,
        "force_refresh": args.force_refresh,
    }

    if args.policy_type == "CnnPolicy":
        from fortuna.rl.models.extractor import TemporalCNNExtractor

        cfg_kwargs["policy_kwargs"] = {
            "features_extractor_class": TemporalCNNExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
        }

    reward = RewardConfig()
    if args.holding_penalty is not None:
        reward = RewardConfig(holding_penalty=float(args.holding_penalty))
    cfg_kwargs["reward_config"] = reward

    trainer = FortunaRLTrainer(TrainerConfig(**cfg_kwargs))
    meta = trainer.train()

    dest = "validated" if meta.verdict_passed else "rejected"
    print(f"\n[RL] Checkpoint saved -> models/{dest}/{meta.run_id}")
    print(f"     Symbol:     {symbol}")
    print(f"     OOS Sharpe: {meta.oos_metrics.sharpe_ratio:.3f}")
    print(f"     PF:         {meta.oos_metrics.profit_factor:.3f}")
    print(f"     Trades:     {meta.oos_metrics.total_trades}")
    print(f"     Passed:     {meta.verdict_passed}")
    return None


if __name__ == "__main__":
    sys.exit(main())
