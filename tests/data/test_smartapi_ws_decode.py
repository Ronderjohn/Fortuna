"""WebSocket message decode tests."""

import struct

from fortuna.data.sources.smartapi_ws_decode import decode_ws_message, parse_snap_quote_binary


def test_decode_dict_passthrough() -> None:
    msg = {"last_traded_price": 250000, "exchange_timestamp": 1_700_000_000}
    assert decode_ws_message(msg) == msg


def test_decode_json_bytes() -> None:
    raw = b'{"last_traded_price": 10000}'
    out = decode_ws_message(raw)
    assert out is not None
    assert out["last_traded_price"] == 10000


def test_parse_snap_quote_binary_ltp() -> None:
    buf = bytearray(51)
    buf[0] = 1  # LTP mode
    buf[1] = 1  # NSE CM
    buf[2:27] = b"2885".ljust(25, b"\x00")
    struct.pack_into("<q", buf, 35, 1_700_000_000)
    struct.pack_into("<q", buf, 43, 132_950)  # paise
    out = parse_snap_quote_binary(bytes(buf))
    assert out is not None
    assert out["last_traded_price"] == 132_950
    assert out["token"] == "2885"
