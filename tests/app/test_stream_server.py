"""Smoke tests for the live bar-stream HTTP server."""

from __future__ import annotations

import json
import urllib.request

import pandas as pd
import pytest

from fortuna.app import stream_server


@pytest.fixture()
def running_server():
    """Spin up a server on a free port with a fake provider, tear it down."""
    captured: dict = {}

    def fake_provider(*, symbol: str, timeframe: str, strategy: str, since: float) -> dict:
        captured["last_call"] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "since": since,
        }
        return {
            "bars": [
                {"time": 1_700_000_000, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
                {"time": 1_700_000_300, "open": 100.5, "high": 102.0, "low": 100.4, "close": 101.8},
            ],
            "markers": [
                {"time": 1_700_000_000, "position": "belowBar", "text": "BUY @ 100.0"},
            ],
            "last": 1_700_000_300,
        }

    port = stream_server.spawn(fake_provider, port=0)
    yield port, captured
    stream_server.stop(port)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def test_health_endpoint(running_server):
    port, _ = running_server
    payload = _get(f"http://127.0.0.1:{port}/fortuna/health")
    assert payload["ok"] is True


def test_bars_endpoint_invokes_provider_with_query(running_server):
    port, captured = running_server
    url = (
        f"http://127.0.0.1:{port}/fortuna/bars?"
        f"symbol=ICICIBANK.NS&tf=5m&strat=mmts&since=1700000000"
    )
    payload = _get(url)
    assert "bars" in payload and "markers" in payload and "last" in payload
    assert len(payload["bars"]) == 2
    assert payload["bars"][0]["close"] == 100.5
    assert payload["last"] == 1_700_000_300
    assert captured["last_call"] == {
        "symbol": "ICICIBANK.NS",
        "timeframe": "5m",
        "strategy": "mmts",
        "since": 1_700_000_000.0,
    }


def test_cors_header_present(running_server):
    port, _ = running_server
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/fortuna/health", timeout=5
    ) as r:
        assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_404_for_unknown_path(running_server):
    port, _ = running_server
    try:
        _get(f"http://127.0.0.1:{port}/nope")
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        assert e.code == 404
        body = json.loads(e.read().decode("utf-8"))
        assert "error" in body
    else:
        pytest.fail("expected 404 HTTPError")


def test_ist_unix_seconds_is_idempotent_for_naive_timestamps():
    """Regression: the chart and the bar-stream must encode timestamps the
    same way so deltas line up with the existing series."""
    from fortuna.app.lightweight_chart import ist_unix_seconds

    ts = pd.Timestamp("2026-05-25 09:15:00")  # tz-naive IST wall clock
    t1 = ist_unix_seconds(ts)
    t2 = ist_unix_seconds(ts.tz_localize("Asia/Kolkata"))
    assert t1 == t2 > 0
