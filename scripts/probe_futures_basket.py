"""Probe SmartAPI futures coverage + history depth for the nightly basket."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: F401, E402
from fortuna.data.instruments import InstrumentRegistry  # noqa: E402


BASKET = [
    "ICICIBANK", "HDFCBANK", "RELIANCE", "INFY", "TCS",
    "SBIN", "AXISBANK", "BHARTIARTL", "LT", "TATASTEEL",
]


def main() -> int:
    reg = InstrumentRegistry()
    reg.ensure_loaded()

    print(f"{'Base':<13} {'Front-month tradingsymbol':<28} {'expiry':<12} {'lot':<6}")
    print("-" * 65)
    missing: list[str] = []
    for base in BASKET:
        try:
            ref = reg.resolve(f"{base}.FUT")
            print(f"{base:<13} {ref.tradingsymbol:<28} {str(ref.expiry):<12} {ref.lot_size}")
        except KeyError:
            missing.append(base)
            print(f"{base:<13} <missing>")
    print()
    print(f"Missing futures: {missing or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
