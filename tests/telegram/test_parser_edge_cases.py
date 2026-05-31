from __future__ import annotations

from fortuna.telegram.parser import parse_telegram_request


def test_parse_incomplete_option_keeps_underlying_symbol_only():
    req = parse_telegram_request("/analyze NIFTY CE 25000")
    assert req.symbol == "NIFTY"
    assert "CE" in req.query


def test_parse_malformed_option_expiry_passthrough():
    req = parse_telegram_request("/analyze NIFTY CE 25000 BADEXPIRY")
    assert req.symbol == "NIFTY.OPT.CE.25000.BADEXPIRY"
