"""SmartAPIBroker: every call must raise (stub guards real-money flow)."""

from __future__ import annotations

import pytest

from fortuna.execution.smartapi_broker import SmartAPIBroker


def test_constructor_raises():
    with pytest.raises(NotImplementedError, match="live SmartAPI trading"):
        SmartAPIBroker()
