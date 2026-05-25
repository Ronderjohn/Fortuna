"""Data source factory smartapi key."""

from __future__ import annotations

from unittest.mock import patch

from fortuna.data.sources import get_data_source


def test_get_data_source_smartapi() -> None:
    with patch("fortuna.data.sources.smartapi_historical.SmartAPIHistoricalSource") as cls:
        cls.return_value = object()
        src = get_data_source("smartapi")
        assert src is cls.return_value
