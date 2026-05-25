"""Data source symbol normalization."""

from fortuna.data.sources.symbols import normalize_for_openchart, normalize_for_yfinance


def test_normalize_yfinance_ns() -> None:
    assert normalize_for_yfinance("reliance") == "RELIANCE.NS"


def test_normalize_openchart_strip() -> None:
    assert normalize_for_openchart("RELIANCE.NS") == "RELIANCE"
