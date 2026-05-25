"""Walk-forward black-box fold tests."""

import pandas as pd

from fortuna.paper.blackbox import walk_forward_folds


def test_walk_forward_no_overlap_leakage(sample_ohlcv: pd.DataFrame) -> None:
    folds = walk_forward_folds(
        sample_ohlcv,
        train_bars=20,
        test_bars=10,
        step_bars=10,
        min_folds=1,
    )
    assert len(folds) >= 1
    f0 = folds[0]
    assert len(f0.train) == 20
    assert len(f0.test) == 10
    assert f0.train.index[-1] < f0.test.index[0]
