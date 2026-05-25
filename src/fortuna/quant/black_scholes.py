"""
Black–Scholes reference utilities (benchmark only — not used for equity signals).

Useful as a theoretical baseline when comparing strategy Sharpe / drawdown
to a delta-hedged option book, similar to black-box validation against a
closed-form model.
"""

from __future__ import annotations

import math
from typing import Optional


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    """European call price (no dividends)."""
    if time_years <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0)
    sqrt_t = math.sqrt(time_years)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_years) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    return spot * _norm_cdf(d1) - strike * math.exp(-risk_free_rate * time_years) * _norm_cdf(d2)


def implied_vol_sanity_check(
    market_price: float,
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float = 0.05,
    vol_guess: float = 0.2,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> Optional[float]:
    """Newton-style IV solve; returns None if no convergence."""
    vol = vol_guess
    for _ in range(max_iter):
        price = black_scholes_call(spot, strike, time_years, risk_free_rate, vol)
        diff = price - market_price
        if abs(diff) < tol:
            return vol
        # Vega approximation
        sqrt_t = math.sqrt(time_years)
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * vol**2) * time_years) / (
            vol * sqrt_t
        )
        vega = spot * math.exp(-0.5 * d1 * d1) * sqrt_t / math.sqrt(2 * math.pi)
        if vega < 1e-12:
            return None
        vol = max(1e-6, vol - diff / vega)
    return None
