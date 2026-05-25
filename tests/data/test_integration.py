"""Network integration tests (optional)."""

import pytest


@pytest.mark.integration
def test_yfinance_fetch_5m() -> None:
    pytest.importorskip("yfinance")
    from fortuna.data.sources.yfinance_source import YFinanceSource

    df = YFinanceSource().fetch("RELIANCE.NS", "5m", days=5)
    assert len(df) > 10
    assert "close" in df.columns


@pytest.mark.integration
def test_openchart_fetch_5m() -> None:
    pytest.importorskip("openchart")
    from fortuna.data.sources.openchart_source import OpenChartSource

    df = OpenChartSource().fetch("RELIANCE", "5m", days=5)
    assert len(df) > 10
    assert "close" in df.columns


@pytest.mark.integration
def test_smartapi_fetch_5m() -> None:
    pytest.importorskip("SmartApi")
    from fortuna.config.smartapi_settings import get_smartapi_settings

    settings = get_smartapi_settings()
    if not settings.configured:
        pytest.skip("SmartAPI credentials not configured in .env")

    from fortuna.data.sources.smartapi_historical import SmartAPIHistoricalSource

    df = SmartAPIHistoricalSource().fetch("RELIANCE.NS", "5m", days=2)
    assert len(df) > 5
    assert "close" in df.columns
