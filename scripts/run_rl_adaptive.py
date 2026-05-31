#!/usr/bin/env python
"""Iterate one RL training generation via ``AdaptiveOptimizerAgent``.

Steps:
    1. Look up the previous run's metadata.json (most recent under models/rejected
       and models/validated for the given symbol/timeframe).
    2. Seed an ``AdaptiveLearner`` fold history from that metadata so the
       optimizer agent has empirical feedback to perturb against.
    3. Ask ``AdaptiveOptimizerAgent.suggest()`` for the next ``RewardConfig``.
    4. Launch the training run with the suggested config.

Example:
    uv run python scripts/run_rl_adaptive.py `
        --symbol ICICIBANK.NS --timeframe 5m `
        --total-timesteps 1000000 --n-envs 8 --use-subproc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults  # noqa: E402

apply_low_spec_gpu_defaults()

from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: E402,F401

from fortuna.agents.optimizer_agent import AdaptiveOptimizerAgent  # noqa: E402
from fortuna.paper.learner import (  # noqa: E402
    AdaptiveLearner,
    FoldRecord,
    StrategyLearningState,
)
from fortuna.rl.env.reward import RewardConfig  # noqa: E402
from fortuna.rl.training.checkpoint import PolicyCheckpoint  # noqa: E402
from fortuna.rl.training.trainer import FortunaRLTrainer, TrainerConfig  # noqa: E402


def _strategy_stem(symbol: str, timeframe: str) -> str:
    return f"rl_{symbol.replace('.', '_')}_{timeframe}"


def _find_latest_checkpoint(symbol: str, timeframe: str, models_root: Path) -> PolicyCheckpoint | None:
    candidates: list[Path] = []
    for sub in ("validated", "rejected"):
        d = models_root / sub
        if not d.exists():
            continue
        for run in d.iterdir():
            meta = run / "metadata.json"
            if meta.exists():
                try:
                    cp = PolicyCheckpoint.read(meta)
                    if cp.symbol == symbol and cp.timeframe == timeframe:
                        candidates.append(meta)
                except Exception:
                    continue
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return PolicyCheckpoint.read(latest)


def _seed_learner_from_checkpoint(
    learner: AdaptiveLearner, stem: str, cp: PolicyCheckpoint
) -> None:
    """Populate ``AdaptiveLearner`` state from a checkpoint's OOS metrics.

    The optimizer's suggestion logic looks at ``avg_profit`` and ``avg_trades``
    of the last few fold records, so we synthesize one record per checkpoint.
    """
    state = learner.load(stem) or StrategyLearningState(strategy_stem=stem)
    already = {f.candidate_id for f in state.fold_history}
    if cp.run_id in already:
        return
    state.fold_history.append(
        FoldRecord(
            fold_id=len(state.fold_history),
            profit_pct=float(cp.oos_metrics.total_net_profit / 1000.0),
            win_ratio_pct=float(cp.oos_metrics.win_rate_pct),
            total_trades=int(cp.oos_metrics.total_trades),
            candidate_id=cp.run_id,
            params_summary=json.dumps(cp.reward_config),
        )
    )
    state.cumulative_oos_profit_pct = sum(f.profit_pct for f in state.fold_history)
    state.generation += 1
    learner.save(state)


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive RL retrain loop")
    parser.add_argument("--symbol", default="ICICIBANK.NS")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--data-source", default="smartapi")
    parser.add_argument("--train-bars", type=int, default=750)
    parser.add_argument("--test-bars", type=int, default=375)
    parser.add_argument("--step-bars", type=int, default=375)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--use-subproc", action="store_true", default=True)
    parser.add_argument("--no-subproc", dest="use_subproc", action="store_false")
    parser.add_argument("--ppo-n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument(
        "--policy",
        dest="policy_type",
        choices=("MlpPolicy", "CnnPolicy"),
        default="MlpPolicy",
    )
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dry-run", action="store_true", help="Only print suggestion")
    args = parser.parse_args()

    stem = _strategy_stem(args.symbol, args.timeframe)
    learner = AdaptiveLearner(Path("logs/learner"))

    # Seed history from the latest checkpoint if learner state is empty.
    cp = _find_latest_checkpoint(args.symbol, args.timeframe, Path(args.checkpoint_dir))
    if cp is not None:
        _seed_learner_from_checkpoint(learner, stem, cp)
        print(f"=== Seeded AdaptiveLearner from {cp.run_id}")
        print(f"    OOS Sharpe: {cp.oos_metrics.sharpe_ratio:.3f}")
        print(f"    PF:         {cp.oos_metrics.profit_factor:.3f}")
        print(f"    Trades:     {cp.oos_metrics.total_trades}")
    else:
        print("=== No prior checkpoint found — starting cold")

    # Ask the optimizer for a new RewardConfig.
    base = RewardConfig(**cp.reward_config) if cp else RewardConfig()
    agent = AdaptiveOptimizerAgent(Path("logs/learner"))
    suggested = agent.suggest(base, strategy_stem=stem)
    suggested_cfg = RewardConfig(**suggested)

    print("\n=== AdaptiveOptimizerAgent suggestion")
    print(f"  current : {base.to_dict()}")
    print(f"  proposed: {suggested_cfg.to_dict()}")
    for k in suggested:
        if k in base.to_dict() and suggested[k] != getattr(base, k, None):
            print(f"    {k}: {getattr(base, k):.6g} -> {suggested[k]:.6g}")
    print()

    if args.dry_run:
        print("--dry-run: exiting without training.")
        return 0

    cfg = TrainerConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        data_source=args.data_source,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_bars=20,
        warmup_bars=30,
        use_subproc=args.use_subproc,
        ppo_n_steps=args.ppo_n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        policy_type=args.policy_type,
        device=args.device,
        reward_config=suggested_cfg,
        checkpoint_dir=Path(args.checkpoint_dir),
        eval_freq=args.eval_freq,
        early_stop_patience=args.patience,
        seed=args.seed,
    )
    trainer = FortunaRLTrainer(cfg)
    meta = trainer.train()

    dest = "validated" if meta.verdict_passed else "rejected"
    print(f"\n[RL] Generation {meta.run_id} -> models/{dest}")
    print(f"     OOS Sharpe: {meta.oos_metrics.sharpe_ratio:.3f}")
    print(f"     OOS PF:     {meta.oos_metrics.profit_factor:.3f}")
    print(f"     Trades:     {meta.oos_metrics.total_trades}")
    print(f"     Passed:     {meta.verdict_passed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
