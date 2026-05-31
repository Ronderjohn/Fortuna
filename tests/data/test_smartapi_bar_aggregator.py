"""Bar aggregator unit tests."""

from __future__ import annotations

import pandas as pd
import pytest

from fortuna.data.sources.smartapi_bar_aggregator import BarAggregator


def test_aggregator_closes_bucket() -> None:
    closed = []

    def on_bar(bar):
        closed.append(bar)

    agg = BarAggregator("5m", on_bar=on_bar)
    ts1 = int(pd.Timestamp("2024-01-02 09:15:00", tz="Asia/Kolkata").timestamp() * 1000)
    ts2 = int(pd.Timestamp("2024-01-02 09:20:00", tz="Asia/Kolkata").timestamp() * 1000)

    assert agg.update({"exchange_timestamp": ts1, "last_traded_price": 25000}) is None
    bar = agg.update({"exchange_timestamp": ts2, "last_traded_price": 25100})
    assert bar is not None
    assert bar.close == 250.0
    assert len(closed) == 1
