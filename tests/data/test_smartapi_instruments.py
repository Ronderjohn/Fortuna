"""Instrument registry unit tests (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fortuna.data.instruments import InstrumentRegistry


@pytest.fixture
def scrip_fixture(tmp_path: Path) -> Path:
    data = [
        {
            "token": "2885",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "exch_seg": "NSE",
            "instrumenttype": "EQ",
        },
        {
            "token": "999",
            "symbol": "FOO-EQ",
            "name": "FOO",
            "exch_seg": "BSE",
            "instrumenttype": "EQ",
        },
    ]
    path = tmp_path / "OpenAPIScripMaster.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_resolve_reliance_ns(scrip_fixture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fortuna.config import smartapi_settings

    settings = smartapi_settings.SmartAPISettings(scrip_cache_path=scrip_fixture)
    reg = InstrumentRegistry(settings)
    reg.ensure_loaded()
    ref = reg.resolve("RELIANCE.NS")
    assert ref.symboltoken == "2885"
    assert ref.exchange == "NSE"
    assert ref.tradingsymbol == "RELIANCE-EQ"
