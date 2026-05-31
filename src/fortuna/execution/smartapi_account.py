"""Read-only adapter for SmartAPI account endpoints.

This module is **read-only by design** — it lives separately from
:mod:`fortuna.execution.smartapi_broker` (which is the write surface and
still stubbed) so accidental imports can never place an order.

Wraps the five SmartConnect endpoints we need for portfolio-aware paper
trading and EOD reconciliation:

- ``rmsLimit``    → :class:`FundsSnapshot`      (cash / margin / collateral)
- ``position``    → :class:`BrokerPosition`     (intraday + carryforward F&O / equity)
- ``holding``     → :class:`BrokerHolding`      (long-term DEMAT shares)
- ``orderBook``   → :class:`BrokerOrder`        (today's orders, all states)
- ``tradeBook``   → :class:`BrokerTrade`        (today's executed fills)

The reader is intentionally duck-typed against the SmartConnect client —
any object with the five methods above works, which makes it trivial to
test with a fake client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from fortuna.data.instruments import InstrumentRef, InstrumentRegistry
from fortuna.data.sources.smartapi_session import SmartAPISession
from fortuna.execution.types import Side
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================== dataclasses


@dataclass(frozen=True)
class FundsSnapshot:
    """Account funds from SmartAPI ``rmsLimit``.

    Field semantics follow Angel's RMS payload (Hindi names paraphrased to
    English):

    - ``net``                 — total account value (cash + collateral + MTM)
    - ``available_cash``      — withdrawable cash
    - ``available_margin``    — total margin available for new positions
    - ``used_margin``         — margin currently blocked
    - ``m2m``                 — mark-to-market on open positions
    - ``collateral``          — pledged-stock collateral
    """

    net: float
    available_cash: float
    available_margin: float
    used_margin: float
    m2m: float = 0.0
    collateral: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerPosition:
    """One row from ``position`` — an intraday or carry-forward position."""

    tradingsymbol: str
    symboltoken: str
    exchange: str
    producttype: str  # INTRADAY / CARRYFORWARD / DELIVERY / MARGIN
    side: Side
    quantity: int  # absolute (positive)
    avg_price: float
    last_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fortuna_symbol: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def signed_qty(self) -> int:
        return self.quantity if self.side is Side.LONG else -self.quantity

    @property
    def market_value(self) -> float:
        return (self.last_price or self.avg_price) * self.quantity


@dataclass(frozen=True)
class BrokerHolding:
    """One row from ``holding`` — a long-term DEMAT share holding."""

    tradingsymbol: str
    symboltoken: str
    exchange: str
    quantity: int
    avg_price: float
    last_price: Optional[float] = None
    pnl: Optional[float] = None
    fortuna_symbol: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        return (self.last_price or self.avg_price) * self.quantity


@dataclass(frozen=True)
class BrokerOrder:
    """One row from ``orderBook`` — every order placed today (any status)."""

    order_id: str
    unique_order_id: Optional[str]
    tradingsymbol: str
    symboltoken: str
    exchange: str
    transaction_type: str  # BUY / SELL
    order_type: str        # MARKET / LIMIT / SL / SL-M
    product_type: str
    quantity: int
    filled_qty: int
    price: Optional[float]
    average_price: Optional[float]
    status: str            # complete / cancelled / rejected / open
    update_time: Optional[str]
    text: Optional[str]    # rejection reason, if any
    fortuna_symbol: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerTrade:
    """One row from ``tradeBook`` — an executed fill."""

    order_id: str
    tradingsymbol: str
    symboltoken: str
    exchange: str
    transaction_type: str  # BUY / SELL
    quantity: int
    fill_price: float
    fill_time: Optional[datetime]
    producttype: Optional[str] = None
    fortuna_symbol: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class AccountSnapshot:
    """Composed read of all five endpoints at one moment in time."""

    funds: FundsSnapshot
    positions: list[BrokerPosition]
    holdings: list[BrokerHolding]
    orders: list[BrokerOrder]
    trades: list[BrokerTrade]
    taken_at: datetime
    errors: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        return bool(self.errors)


# =============================================================== symbol mapping


def fortuna_symbol_for(
    ref: InstrumentRef,
    registry: Optional[InstrumentRegistry] = None,
) -> str:
    """Map an ``InstrumentRef`` back to Fortuna's canonical symbol string.

    - Equity (``RELIANCE-EQ``)          → ``RELIANCE.NS``
    - Front-month future                → ``RELIANCE.FUT``
    - Back-month future (27NOV2025)     → ``RELIANCE.FUT.27NOV2025``
    """
    if not ref.is_future:
        return f"{ref.symbol}.NS"

    # For futures, ``InstrumentRef.symbol`` is already ``BASE.FUT`` (the
    # registry pre-suffixes it). The underlying base lives in ``ref.name``.
    base = (ref.name or ref.symbol.replace(".FUT", "")).upper()
    if registry is None:
        if ref.expiry is not None:
            return f"{base}.FUT.{_format_expiry(ref.expiry)}"
        return f"{base}.FUT"

    try:
        front = registry._front_future(base)  # noqa: SLF001 — public read accessor
    except Exception:  # noqa: BLE001
        front = None
    if front is not None and front.expiry == ref.expiry:
        return f"{base}.FUT"
    if ref.expiry is not None:
        return f"{base}.FUT.{_format_expiry(ref.expiry)}"
    return f"{base}.FUT"


def _format_expiry(d: date) -> str:
    return d.strftime("%d%b%Y").upper()


# =============================================================== reader


class SmartAPIAccountReadError(RuntimeError):
    """Raised when an account read fails after retries."""


class SmartAPIAccountReader:
    """Read-only adapter over the SmartConnect account endpoints.

    Constructed with either a live :class:`SmartAPISession` (production) or
    a duck-typed ``client`` (tests). The session form lazily fetches the
    underlying SmartConnect instance via ``session.get_client()`` so token
    refresh stays the session's responsibility.

    Parameters
    ----------
    session
        A logged-in :class:`SmartAPISession`. Mutually exclusive with
        ``client``.
    client
        A pre-built SmartConnect-shaped object (any object with
        ``rmsLimit``, ``position``, ``holding``, ``orderBook``,
        ``tradeBook`` methods). Useful for tests.
    registry
        Optional :class:`InstrumentRegistry` for back-mapping tradingsymbols
        to Fortuna symbols. When omitted the reader still returns rows but
        ``fortuna_symbol`` will be ``None``.
    """

    def __init__(
        self,
        *,
        session: Optional[SmartAPISession] = None,
        client: Optional[Any] = None,
        registry: Optional[InstrumentRegistry] = None,
    ) -> None:
        if session is None and client is None:
            raise ValueError("SmartAPIAccountReader needs either session= or client=")
        if session is not None and client is not None:
            raise ValueError("pass session= OR client=, not both")
        self._session = session
        self._client = client
        self._registry = registry

    # ------------------------------------------------------------------ client
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        assert self._session is not None
        return self._session.get_client()

    # =================================================================== reads
    def snapshot_funds(self) -> FundsSnapshot:
        client = self._get_client()
        raw = self._safe_call(client.rmsLimit, label="rmsLimit")
        data = _payload_data(raw) or {}
        return FundsSnapshot(
            net=_as_float(data.get("net")),
            available_cash=_as_float(data.get("availablecash"), data.get("availableCash")),
            available_margin=_as_float(
                data.get("net"), data.get("availablecash"), data.get("availableCash")
            ),
            used_margin=_as_float(data.get("utilisedDebits"), data.get("utiliseddebits")),
            m2m=_as_float(data.get("m2munrealized"), data.get("m2mrealized")),
            collateral=_as_float(data.get("collateral")),
            raw=raw if isinstance(raw, dict) else {},
        )

    def snapshot_positions(self) -> list[BrokerPosition]:
        client = self._get_client()
        raw = self._safe_call(client.position, label="position")
        rows = _payload_rows(raw)
        out: list[BrokerPosition] = []
        for r in rows:
            netqty = _as_int(r.get("netqty"), r.get("netQty"), r.get("netquantity"))
            if netqty == 0:
                # Flat row — skip; Angel echoes back closed positions for
                # the day with netqty=0, which we don't want to seed.
                continue
            side = Side.LONG if netqty > 0 else Side.SHORT
            qty = abs(netqty)
            avg = _parse_position_avg_price(r, qty=qty)
            last_price = _as_float_or_none(r.get("ltp"), r.get("lastTradedPrice"))
            unreal = _as_float(r.get("unrealised"), r.get("unrealisedpnl"))
            if avg <= 0 and last_price is not None and qty > 0 and unreal != 0:
                # Broker-reported MTM: long P&L = (ltp - avg) * qty
                if netqty > 0:
                    avg = last_price - unreal / qty
                else:
                    avg = last_price + unreal / qty
            sym = str(r.get("tradingsymbol", "")).upper()
            ts = str(r.get("symboltoken", ""))
            exch = str(r.get("exchange", "")).upper()
            out.append(
                BrokerPosition(
                    tradingsymbol=sym,
                    symboltoken=ts,
                    exchange=exch,
                    producttype=str(r.get("producttype", "")).upper(),
                    side=side,
                    quantity=qty,
                    avg_price=max(0.0, avg),
                    last_price=last_price,
                    unrealized_pnl=_as_float(r.get("unrealised"), r.get("unrealisedpnl")),
                    realized_pnl=_as_float(r.get("realised"), r.get("realisedpnl")),
                    fortuna_symbol=self._map_tradingsymbol(sym, exch),
                    raw=r,
                )
            )
        return out

    def snapshot_holdings(self) -> list[BrokerHolding]:
        client = self._get_client()
        raw = self._safe_call(client.holding, label="holding")
        rows = _payload_rows(raw)
        out: list[BrokerHolding] = []
        for r in rows:
            qty = _as_int(r.get("quantity"), r.get("realisedquantity"))
            if qty == 0:
                continue
            sym = str(r.get("tradingsymbol", "")).upper()
            exch = str(r.get("exchange", "")).upper()
            out.append(
                BrokerHolding(
                    tradingsymbol=sym,
                    symboltoken=str(r.get("symboltoken", "")),
                    exchange=exch,
                    quantity=qty,
                    avg_price=_as_float(r.get("averageprice"), r.get("avgPrice")),
                    last_price=_as_float_or_none(r.get("ltp"), r.get("lastTradedPrice")),
                    pnl=_as_float_or_none(r.get("profitandloss"), r.get("pnl")),
                    fortuna_symbol=self._map_tradingsymbol(sym, exch),
                    raw=r,
                )
            )
        return out

    def snapshot_orders(self) -> list[BrokerOrder]:
        client = self._get_client()
        raw = self._safe_call(client.orderBook, label="orderBook")
        rows = _payload_rows(raw)
        out: list[BrokerOrder] = []
        for r in rows:
            sym = str(r.get("tradingsymbol", "")).upper()
            exch = str(r.get("exchange", "")).upper()
            out.append(
                BrokerOrder(
                    order_id=str(r.get("orderid", r.get("orderId", ""))),
                    unique_order_id=str(r.get("uniqueorderid", "")) or None,
                    tradingsymbol=sym,
                    symboltoken=str(r.get("symboltoken", "")),
                    exchange=exch,
                    transaction_type=str(r.get("transactiontype", "")).upper(),
                    order_type=str(r.get("ordertype", "")).upper(),
                    product_type=str(r.get("producttype", "")).upper(),
                    quantity=_as_int(r.get("quantity"), r.get("orderQuantity")),
                    filled_qty=_as_int(
                        r.get("filledshares"), r.get("filledQty"), r.get("filledqty"),
                    ),
                    price=_as_float_or_none(r.get("price")),
                    average_price=_as_float_or_none(r.get("averageprice"), r.get("avgPrice")),
                    status=str(r.get("status", r.get("orderstatus", ""))).lower(),
                    update_time=str(r.get("updatetime", "")) or None,
                    text=str(r.get("text", "")) or None,
                    fortuna_symbol=self._map_tradingsymbol(sym, exch),
                    raw=r,
                )
            )
        return out

    def snapshot_trades(self) -> list[BrokerTrade]:
        client = self._get_client()
        raw = self._safe_call(client.tradeBook, label="tradeBook")
        rows = _payload_rows(raw)
        out: list[BrokerTrade] = []
        for r in rows:
            sym = str(r.get("tradingsymbol", "")).upper()
            exch = str(r.get("exchange", "")).upper()
            out.append(
                BrokerTrade(
                    order_id=str(r.get("orderid", r.get("orderId", ""))),
                    tradingsymbol=sym,
                    symboltoken=str(r.get("symboltoken", "")),
                    exchange=exch,
                    transaction_type=str(r.get("transactiontype", "")).upper(),
                    quantity=_as_int(r.get("fillsize"), r.get("quantity")),
                    fill_price=_as_float(r.get("fillprice"), r.get("tradeprice"), r.get("price")),
                    fill_time=_parse_ts(r.get("filltime"), r.get("exchangeorderupdatetime")),
                    producttype=str(r.get("producttype", "")).upper() or None,
                    fortuna_symbol=self._map_tradingsymbol(sym, exch),
                    raw=r,
                )
            )
        return out

    def full_snapshot(self) -> AccountSnapshot:
        """Read all five endpoints; per-endpoint failures degrade gracefully."""
        errors: list[str] = []
        funds = FundsSnapshot(net=0.0, available_cash=0.0, available_margin=0.0, used_margin=0.0)
        positions: list[BrokerPosition] = []
        holdings: list[BrokerHolding] = []
        orders: list[BrokerOrder] = []
        trades: list[BrokerTrade] = []

        try:
            funds = self.snapshot_funds()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"funds: {exc}")
            logger.exception("[account_reader] funds snapshot failed")
        try:
            positions = self.snapshot_positions()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"positions: {exc}")
            logger.exception("[account_reader] positions snapshot failed")
        try:
            holdings = self.snapshot_holdings()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"holdings: {exc}")
            logger.exception("[account_reader] holdings snapshot failed")
        try:
            orders = self.snapshot_orders()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"orders: {exc}")
            logger.exception("[account_reader] orders snapshot failed")
        try:
            trades = self.snapshot_trades()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"trades: {exc}")
            logger.exception("[account_reader] trades snapshot failed")

        return AccountSnapshot(
            funds=funds,
            positions=positions,
            holdings=holdings,
            orders=orders,
            trades=trades,
            taken_at=datetime.now(),
            errors=errors,
        )

    # ============================================================== mapping
    def _map_tradingsymbol(self, tradingsymbol: str, exchange: str) -> Optional[str]:
        """Best-effort map Angel tradingsymbol → Fortuna canonical symbol.

        Returns ``None`` when the registry is absent or the symbol can't be
        resolved (e.g. options / unsupported segment); the dashboard still
        renders the row but uses the raw tradingsymbol as the join key.
        """
        if self._registry is None or not tradingsymbol:
            return None
        try:
            ref = self._registry.resolve(tradingsymbol)
        except Exception:  # noqa: BLE001
            return None
        return fortuna_symbol_for(ref, registry=self._registry)

    # ============================================================== helpers
    def _safe_call(self, fn, *, label: str) -> Any:
        try:
            return fn()
        except TypeError:
            # Some SmartConnect methods take an empty arg list — others want
            # ``orderBook(None)``. Try once more with no args explicit.
            return fn()
        except Exception as exc:  # noqa: BLE001
            raise SmartAPIAccountReadError(f"{label} failed: {exc}") from exc


# =============================================================== payload helpers


def _payload_data(raw: Any) -> Any:
    """Return ``raw["data"]`` if the SmartAPI envelope is present."""
    if isinstance(raw, dict):
        if "data" in raw:
            return raw["data"]
        return raw
    return raw


def _payload_rows(raw: Any) -> list[dict]:
    """Normalize ``rmsLimit``-style envelope OR raw list into a list of dicts."""
    data = _payload_data(raw)
    if data is None:
        return []
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        # Some endpoints return ``{"data": {"some_key": [...]}}`` — be tolerant.
        for v in data.values():
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        return [data]
    return []


def _as_float(*candidates: Any) -> float:
    for c in candidates:
        if c is None or c == "":
            continue
        try:
            return float(c)
        except (TypeError, ValueError):
            continue
    return 0.0


def _first_positive_float(*candidates: Any) -> float:
    """First parseable float > 0 (Angel often sends ``0`` placeholders)."""
    for c in candidates:
        if c is None or c == "":
            continue
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return 0.0


def _as_float_or_none(*candidates: Any) -> Optional[float]:
    for c in candidates:
        if c is None or c == "":
            continue
        try:
            return float(c)
        except (TypeError, ValueError):
            continue
    return None


def _as_int(*candidates: Any) -> int:
    for c in candidates:
        if c is None or c == "":
            continue
        try:
            return int(float(c))
        except (TypeError, ValueError):
            continue
    return 0


def _parse_position_avg_price(row: dict, *, qty: int) -> float:
    """Resolve average entry price from Angel ``position`` row fields.

    Intraday MIS rows usually populate ``avgnetprice`` / ``netavgprice``.
    Overnight NRML / carry-forward rows often leave those at zero and only
    populate ``cfbuyavgprice`` / ``cfsellavgprice`` or ``netvalue`` instead.
    """
    if qty <= 0:
        return 0.0

    avg = _first_positive_float(
        row.get("avgnetprice"),
        row.get("netavgprice"),
        row.get("netprice"),
        row.get("avgPrice"),
        row.get("buyavgprice"),
        row.get("sellavgprice"),
        row.get("cfbuyavgprice"),
        row.get("cfsellavgprice"),
        row.get("weightedavgprice"),
        row.get("totalbuyavgprice"),
        row.get("totalsellavgprice"),
    )
    if avg > 0:
        return avg

    # netvalue is total signed position value; divide by qty for per-unit avg.
    netvalue = _as_float(row.get("netvalue"), row.get("netValue"))
    if netvalue != 0:
        return abs(netvalue) / qty

    buyqty = _as_int(row.get("buyqty"), row.get("cfbuyqty"))
    buyavg = _first_positive_float(row.get("buyavgprice"), row.get("cfbuyavgprice"))
    sellqty = _as_int(row.get("sellqty"), row.get("cfsellqty"))
    sellavg = _first_positive_float(row.get("sellavgprice"), row.get("cfsellavgprice"))
    if buyqty > 0 and buyavg > 0:
        return buyavg
    if sellqty > 0 and sellavg > 0:
        return sellavg

    totalbuy = _as_float(row.get("totalbuyvalue"), row.get("totalbuyamount"))
    if buyqty > 0 and totalbuy > 0:
        return totalbuy / buyqty

    return 0.0


def _parse_ts(*candidates: Any) -> Optional[datetime]:
    """SmartAPI returns timestamps in a few formats; try the common ones."""
    for c in candidates:
        if not c:
            continue
        s = str(c).strip()
        for fmt in (
            "%d-%b-%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
    return None


__all__ = [
    "AccountSnapshot",
    "BrokerHolding",
    "BrokerOrder",
    "BrokerPosition",
    "BrokerTrade",
    "FundsSnapshot",
    "SmartAPIAccountReadError",
    "SmartAPIAccountReader",
    "fortuna_symbol_for",
]
