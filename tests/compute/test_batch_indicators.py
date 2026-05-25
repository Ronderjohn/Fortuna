"""Batch indicator computer CPU fallback."""

import pandas as pd

from fortuna.compute.batch_indicators import BatchIndicatorComputer, IndicatorJob, attach_specs
from fortuna.compute.policy import ComputePolicy
from fortuna.strategy.schema import IndicatorSpec, IndicatorType


def test_batch_indicators_cpu(sample_ohlcv: pd.DataFrame) -> None:
    policy = ComputePolicy(use_gpu=False)
    computer = BatchIndicatorComputer(policy)
    jobs = [IndicatorJob("ema", "close", 12), IndicatorJob("ema", "close", 26)]
    arrays = computer.run(sample_ohlcv, jobs)
    specs = [
        IndicatorSpec(id="fast", type=IndicatorType.EMA, params={"window": 12}),
        IndicatorSpec(id="slow", type=IndicatorType.EMA, params={"window": 26}),
    ]
    out = attach_specs(sample_ohlcv, specs, arrays)
    assert "fast" in out.columns
    assert "slow" in out.columns
