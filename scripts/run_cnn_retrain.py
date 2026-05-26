"""Launch a 500k CNN+CUDA retrain with the same reward config v3 landed on.

This isolates the policy-architecture variable: identical reward landscape,
identical fold setup, only ``policy_type`` (MlpPolicy -> CnnPolicy) changes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401,E402
from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults  # noqa: E402

apply_low_spec_gpu_defaults()

import torch  # noqa: E402

from fortuna.rl.env.reward import RewardConfig  # noqa: E402
from fortuna.rl.training.checkpoint import PolicyCheckpoint  # noqa: E402
from fortuna.rl.training.trainer import FortunaRLTrainer, TrainerConfig  # noqa: E402


def _latest_reward_config(
    symbol: str,
    timeframe: str,
    models_root: Path,
    *,
    prefer_run_id: str | None = None,
) -> tuple[RewardConfig, str]:
    """Pick the reward config from the chronologically latest checkpoint.

    Ordering uses ``PolicyCheckpoint.created_at`` rather than file mtime, since
    a smoke-test run mid-pipeline can shift mtime ordering without being the
    "newest" by training generation. If ``prefer_run_id`` is provided and a
    matching checkpoint exists, that one wins.
    """
    candidates: list[tuple[str, PolicyCheckpoint]] = []
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
                        candidates.append((cp.created_at, cp))
                except Exception:
                    continue
    if not candidates:
        return RewardConfig(), "<defaults>"

    if prefer_run_id:
        for _, cp in candidates:
            if cp.run_id == prefer_run_id:
                return RewardConfig(**cp.reward_config), cp.run_id

    candidates.sort(key=lambda kv: kv[0], reverse=True)
    cp = candidates[0][1]
    return RewardConfig(**cp.reward_config), cp.run_id


def main() -> int:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}")

    # Pin to v3 (20260526_001520) explicitly — that run had the
    # AdaptiveOptimizerAgent-tuned reward config; everything before/after may
    # be defaults from smoke tests.
    reward, source_run = _latest_reward_config(
        "ICICIBANK.NS",
        "5m",
        Path("models"),
        prefer_run_id="20260526_001520",
    )
    print(f"Reward config (inherited from {source_run}): {reward.to_dict()}")

    cfg = TrainerConfig(
        symbol="ICICIBANK.NS",
        timeframe="5m",
        days=180,
        data_source="smartapi",
        train_bars=750,
        test_bars=375,
        step_bars=375,
        total_timesteps=500_000,
        # 8 SubprocVecEnv workers feed CUDA fast enough; GTX 1650 is small,
        # so over-subscribing envs gains us very little.
        n_envs=8,
        n_bars=20,
        warmup_bars=30,
        use_subproc=True,
        # Larger PPO rollouts + batches saturate the GPU better; fewer but
        # larger gradient updates per rollout amortize CUDA launch overhead.
        ppo_n_steps=4096,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        policy_type="CnnPolicy",  # routed to TemporalCNNExtractor + MlpPolicy head
        device="cuda",
        reward_config=reward,
        checkpoint_dir=Path("models"),
        eval_freq=25_000,
        early_stop_patience=10,
        seed=44,
    )
    trainer = FortunaRLTrainer(cfg)

    t0 = time.perf_counter()
    meta = trainer.train()
    elapsed = time.perf_counter() - t0
    dest = "validated" if meta.verdict_passed else "rejected"

    print(f"\n[CNN] Generation {meta.run_id} -> models/{dest}")
    print(f"     Wall-clock:        {elapsed:.1f}s ({500_000 / elapsed:.0f} FPS avg)")
    print(f"     OOS Sharpe:        {meta.oos_metrics.sharpe_ratio:.3f}")
    print(f"     OOS PF:            {meta.oos_metrics.profit_factor:.3f}")
    print(f"     OOS Trades:        {meta.oos_metrics.total_trades}")
    print(f"     OOS Win-rate:      {meta.oos_metrics.win_rate_pct:.1f}%")
    print(f"     OOS Max Drawdown:  {meta.oos_metrics.max_drawdown_pct:.2f}%")
    print(f"     Passed:            {meta.verdict_passed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
