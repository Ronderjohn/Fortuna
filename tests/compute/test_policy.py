"""Compute policy tests (CPU path)."""

from fortuna.compute.policy import ComputePolicy


def test_effective_cpu_workers_without_gpu() -> None:
    policy = ComputePolicy(use_gpu=False, cpu_max_workers_when_gpu=1)
    assert policy.effective_cpu_workers(4) == 4


def test_gpu_indicators_requires_min_bars() -> None:
    policy = ComputePolicy(use_gpu=True, gpu_min_bars=1000, gpu_min_unique_series=2)
    assert policy.should_gpu_indicators(100, 10) is False
