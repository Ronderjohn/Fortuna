"""Symbol normalization across data providers."""

from __future__ import annotations


def normalize_for_yfinance(symbol: str) -> str:
    """Ensure NSE symbols use .NS suffix for yfinance."""
    s = symbol.upper().strip()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    if "." not in s:
        return f"{s}.NS"
    return s


def normalize_for_openchart(symbol: str) -> str:
    """Strip exchange suffix for OpenChart NSE EQ segment."""
    s = symbol.upper().strip()
    for suffix in (".NS", ".BO", ".NSE"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def normalize_for_smartapi(symbol: str) -> str:
    """Canonical NSE symbol for SmartAPI (RELIANCE.NS)."""
    s = normalize_for_yfinance(symbol)
    return s
