"""Live execution layer for Fortuna (Tier 1).

This package converts the dashboard's per-bar ``LiveSignal`` dictionaries
into actual orders, simulated fills, and an auditable P&L journal — the
bridge from "I have signals" to "I have measurable live P&L".

Public surface:

- :class:`Broker` — abstract broker interface.
- :class:`PaperBroker` — virtual portfolio that simulates fills on the
  *next* bar's open with slippage from ``MarketCostModel``.
- :class:`SmartAPIBroker` — stub for real-money trading (raises until
  Tier 1 follow-up enables it).
- :class:`LiveAccount` — cash + positions + realized/unrealized P&L.
- :class:`OrderJournal` — JSONL-on-disk event log.
- :class:`ExecutionRouter` — per-bar orchestrator.
- :class:`RiskGate` / :class:`PositionSizer` — pre-trade guards.
- :class:`ExecutionMonitor` — thread-safe in-memory event feed for the
  dashboard.

Toggle the whole stack via ``settings.execution_enabled`` (default
``False``).
"""

from fortuna.execution.account import EquityPoint, LiveAccount
from fortuna.execution.broker import Broker
from fortuna.execution.journal import OrderJournal
from fortuna.execution.monitor import EventSeverity, ExecutionEvent, ExecutionMonitor
from fortuna.execution.paper_broker import PaperBroker
from fortuna.execution.reconciliation import (
    MatchedPair,
    PaperOrder,
    ReconciliationReport,
    append_reconciliation_to_eod,
    paper_orders_from_journal,
    reconcile_paper_vs_real,
    render_reconciliation_markdown,
)
from fortuna.execution.risk import (
    PositionSizer,
    RiskConfig,
    RiskGate,
    RiskVerdict,
    load_risk_config,
)
from fortuna.execution.router import ExecutionRouter
from fortuna.execution.smartapi_account import (
    AccountSnapshot,
    BrokerHolding,
    BrokerOrder,
    BrokerPosition,
    BrokerTrade,
    FundsSnapshot,
    SmartAPIAccountReader,
)
from fortuna.execution.smartapi_broker import SmartAPIBroker
from fortuna.execution.types import (
    Holding,
    OrderAck,
    OrderIntent,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Trade,
)

__all__ = [
    "AccountSnapshot",
    "Broker",
    "BrokerHolding",
    "BrokerOrder",
    "BrokerPosition",
    "BrokerTrade",
    "EquityPoint",
    "EventSeverity",
    "ExecutionEvent",
    "ExecutionMonitor",
    "ExecutionRouter",
    "FundsSnapshot",
    "Holding",
    "LiveAccount",
    "MatchedPair",
    "OrderAck",
    "OrderIntent",
    "OrderJournal",
    "OrderStatus",
    "OrderType",
    "PaperBroker",
    "PaperOrder",
    "Position",
    "PositionSizer",
    "ReconciliationReport",
    "RiskConfig",
    "RiskGate",
    "RiskVerdict",
    "Side",
    "SmartAPIAccountReader",
    "SmartAPIBroker",
    "Trade",
    "append_reconciliation_to_eod",
    "load_risk_config",
    "paper_orders_from_journal",
    "reconcile_paper_vs_real",
    "render_reconciliation_markdown",
]
