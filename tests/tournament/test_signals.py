"""Tournament signal mapping tests."""

from fortuna.search.signals import emit_signal
from fortuna.strategy.schema import StrategyDefinition, TradeSide


def test_long_entry_is_buy() -> None:
    s = StrategyDefinition.model_validate(
        {"name": "long", "side": TradeSide.LONG, "indicators": [], "rules": {}}
    )
    assert emit_signal(s, entry=True, exit_sig=False) == "BUY"


def test_long_exit_is_sell() -> None:
    s = StrategyDefinition.model_validate(
        {"name": "long", "side": TradeSide.LONG, "indicators": [], "rules": {}}
    )
    assert emit_signal(s, entry=False, exit_sig=True) == "SELL"
