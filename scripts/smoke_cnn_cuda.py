"""Smoke test: CnnPolicy on CUDA with SubprocVecEnv (10k steps)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401,E402
from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults  # noqa: E402

apply_low_spec_gpu_defaults()

import torch  # noqa: E402

from fortuna.rl.training.trainer import FortunaRLTrainer, TrainerConfig  # noqa: E402


def main() -> int:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
        print(f"Capability:     {torch.cuda.get_device_capability(0)}")

    cfg = TrainerConfig(
        symbol="ICICIBANK.NS",
        timeframe="5m",
        days=180,
        data_source="smartapi",
        train_bars=750,
        test_bars=375,
        step_bars=375,
        total_timesteps=10_000,
        n_envs=8,
        n_bars=20,
        warmup_bars=30,
        use_subproc=True,
        ppo_n_steps=2048,
        batch_size=512,
        n_epochs=4,
        policy_type="CnnPolicy",
        device="cuda",
        checkpoint_dir=Path("models"),
        eval_freq=50_000,
        early_stop_patience=99,
        seed=99,
    )
    t = FortunaRLTrainer(cfg)
    t0 = time.perf_counter()
    meta = t.train()
    elapsed = time.perf_counter() - t0
    dest = "validated" if meta.verdict_passed else "rejected"
    print(f"CNN+CUDA SMOKE: {elapsed:.1f}s for 10k steps = {10000 / elapsed:.0f} FPS")
    print(f"Run dest: {dest}/{meta.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
