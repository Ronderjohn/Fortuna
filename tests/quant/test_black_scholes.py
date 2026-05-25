"""Black-Scholes reference benchmark tests."""

from fortuna.quant.black_scholes import black_scholes_call, implied_vol_sanity_check


def test_black_scholes_call_positive() -> None:
    price = black_scholes_call(
        spot=100.0,
        strike=100.0,
        time_years=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
    )
    assert price > 0


def test_implied_vol_recovery() -> None:
    vol = 0.25
    price = black_scholes_call(100, 100, 1.0, 0.05, vol)
    recovered = implied_vol_sanity_check(price, 100, 100, 1.0, risk_free_rate=0.05)
    assert recovered is not None
    assert abs(recovered - vol) < 0.01
