"""Quick standalone diagnostic — connect to SmartAPI WS and dump ticks for 25 s."""

from __future__ import annotations

import time

import fortuna  # noqa: F401  -- ensure package init runs once first
from fortuna.app.session_engine import FortunaSessionEngine  # noqa: F401
from fortuna.config.settings import load_settings
from fortuna.data.manager import MarketDataManager
from fortuna.data.sources.smartapi_bar_aggregator import OhlcvBar
from fortuna.data.sources.smartapi_live import SmartAPILiveBarFeed
from fortuna.utils.insecure_ssl import maybe_disable_ssl_verification
from fortuna.utils.logging import setup_logging

maybe_disable_ssl_verification()


def main() -> None:
    setup_logging("INFO")
    settings = load_settings()
    mdm = MarketDataManager(settings, data_source="smartapi")

    tick_count = {"n": 0}
    bar_count = {"n": 0}

    def on_tick(b: OhlcvBar) -> None:
        tick_count["n"] += 1
        if tick_count["n"] <= 5 or tick_count["n"] % 50 == 0:
            print(
                f"[TICK #{tick_count['n']}] {b.datetime} O={b.open} H={b.high} "
                f"L={b.low} C={b.close} V={b.volume}"
            )

    def on_bar(b: OhlcvBar) -> None:
        bar_count["n"] += 1
        print(f"[BAR CLOSED #{bar_count['n']}] {b.datetime} close={b.close}")

    feed = SmartAPILiveBarFeed(mdm, "ICICIBANK.NS", "5m", on_bar=on_bar, on_tick=on_tick)
    feed.start(background=True)
    print("WS thread started; waiting 25 s for data…")
    for i in range(25):
        time.sleep(1)
        if i % 5 == 4:
            print(f"  t={i+1}s ticks={tick_count['n']} bars={bar_count['n']}")
    feed.stop()
    print(
        f"DONE — total ticks={tick_count['n']} bars closed={bar_count['n']}"
    )


if __name__ == "__main__":
    main()
