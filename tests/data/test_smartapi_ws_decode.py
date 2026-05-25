"""WebSocket message decode tests."""

from fortuna.data.sources.smartapi_ws_decode import decode_ws_message


def test_decode_dict_passthrough() -> None:
    msg = {"last_traded_price": 250000, "exchange_timestamp": 1_700_000_000}
    assert decode_ws_message(msg) == msg


def test_decode_json_bytes() -> None:
    raw = b'{"last_traded_price": 10000}'
    out = decode_ws_message(raw)
    assert out is not None
    assert out["last_traded_price"] == 10000
