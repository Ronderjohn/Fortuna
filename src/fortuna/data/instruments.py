"""Angel One instrument master (OpenAPIScripMaster.json) lookup.

Supports two segments today:

- **NSE cash equity** (``exch_seg=NSE``, ``instrumenttype=EQ``) — referenced
  as ``RELIANCE.NS`` / ``RELIANCE-EQ`` / ``RELIANCE``.
- **NSE futures** (``exch_seg=NFO``, ``instrumenttype=FUTSTK``/``FUTIDX``) —
  referenced as ``RELIANCE.FUT`` (current-month contract) or
  ``RELIANCE.FUT.27NOV2025`` (explicit expiry, ``DDMMMYYYY`` format).
  ``RELIANCE.FUT`` resolves to the **nearest non-expired** contract.

The registry surfaces equities and futures through the same ``catalog()`` /
``search()`` API so the Streamlit picker can show both in one list.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from fortuna.config.smartapi_settings import SmartAPISettings, get_smartapi_settings
from fortuna.data.sources.symbols import normalize_for_smartapi
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

# SmartWebSocketV2 exchange-type codes
NSE_CM_EXCHANGE_TYPE = 1  # NSE cash market
NSE_FO_EXCHANGE_TYPE = 2  # NSE futures & options

_DEFAULT_MAX_AGE_HOURS = 24
_FUT_SUFFIX = ".FUT"
_OPT_SUFFIX = ".OPT"
_EXPIRY_FMT = "%d%b%Y"  # Angel scrip-master format, e.g. "27NOV2025"
_OPTION_TYPE_MAP = {"C": "CE", "CE": "CE", "P": "PE", "PE": "PE"}
_OPTION_SYMBOL_RE = re.compile(r"^(?P<head>[A-Z0-9]+?)(?P<cp>CE|PE|C|P)(?P<strike>\d+(?:\.\d+)?)$")
# NSE always lists three monthly stock-future contracts at any given time
# (near, mid, far month). The picker surfaces all of them by default so a
# user can search/select beyond just the front-month expiry.
_DEFAULT_CONTRACT_DEPTH = 3


@dataclass(frozen=True)
class InstrumentRef:
    """Resolved SmartAPI instrument for a Fortuna symbol.

    Fields populated for cash equity stay as-is. For futures, ``expiry`` and
    ``lot_size`` carry the contract-specific metadata and ``exchange`` is
    ``"NFO"`` with ``exchange_type=NSE_FO_EXCHANGE_TYPE``.
    """

    symbol: str
    tradingsymbol: str
    symboltoken: str
    exchange: str
    exchange_type: int = NSE_CM_EXCHANGE_TYPE
    instrumenttype: str = "EQ"
    expiry: Optional[date] = None
    lot_size: Optional[int] = None
    name: Optional[str] = None
    option_type: Optional[str] = None
    strike: Optional[float] = None

    @property
    def is_future(self) -> bool:
        return self.instrumenttype in {"FUTSTK", "FUTIDX"}

    @property
    def is_option(self) -> bool:
        return self.instrumenttype in {"OPTSTK", "OPTIDX"}


@dataclass(frozen=True)
class SymbolSearchHit:
    """Autocomplete result for dashboard symbol search."""

    display: str
    symbol: str
    tradingsymbol: str
    segment: str = "EQUITY"


class InstrumentRegistry:
    """Download, cache, and resolve Angel OpenAPIScripMaster entries."""

    def __init__(
        self,
        settings: Optional[SmartAPISettings] = None,
        *,
        project_root: Optional[Path] = None,
        contract_depth: int = _DEFAULT_CONTRACT_DEPTH,
    ) -> None:
        self._settings = settings or get_smartapi_settings()
        self._project_root = project_root
        # How many active contracts per futures base to expose in
        # ``catalog()`` / ``search()``. NSE lists 3 monthly contracts;
        # 1 = front-month only, 3 = full near/mid/far chain.
        self._contract_depth = max(1, int(contract_depth))
        # Equity index: keyed by both base ("RELIANCE") and tradingsymbol
        # ("RELIANCE-EQ").
        self._by_key: dict[str, InstrumentRef] = {}
        self._equity_bases: list[str] = []
        # Futures index: base ("RELIANCE") -> list of contracts sorted by
        # expiry ascending. Also indexed by Angel's full tradingsymbol
        # ("RELIANCE25NOVFUT") for direct token lookup.
        self._futures_by_base: dict[str, list[InstrumentRef]] = {}
        self._futures_by_symbol: dict[str, InstrumentRef] = {}
        self._futures_bases: list[str] = []
        self._options_by_base: dict[str, list[InstrumentRef]] = {}
        self._options_by_symbol: dict[str, InstrumentRef] = {}
        self._options_bases: list[str] = []
        self._loaded = False

    @property
    def cache_path(self) -> Path:
        return self._settings.resolve_path(self._project_root)

    def _base_symbol(self, symbol: str) -> str:
        """Strip cosmetic suffixes to recover the underlying base symbol.

        ``RELIANCE.NS`` / ``RELIANCE-EQ`` / ``RELIANCE.FUT`` /
        ``RELIANCE.FUT.27NOV2025`` all collapse to ``"RELIANCE"``.
        """
        s = symbol.upper().strip()
        # Strip a trailing ``.FUT[.<expiry>]`` segment first so equity-style
        # suffixes underneath are still picked up.
        if _FUT_SUFFIX in s:
            s = s.split(_FUT_SUFFIX, 1)[0]
        if _OPT_SUFFIX in s:
            s = s.split(_OPT_SUFFIX, 1)[0]
        for suffix in (".NS", ".BO", ".NSE"):
            if s.endswith(suffix):
                return s[: -len(suffix)]
        if s.endswith("-EQ"):
            return s[:-3]
        return s

    def ensure_loaded(self, *, force_refresh: bool = False) -> None:
        if self._loaded and not force_refresh:
            return
        path = self.cache_path
        if force_refresh or not path.exists() or self._is_cache_stale(path):
            self.download_master(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Unexpected scrip master format at {path}")
        self._index_records(raw)
        self._loaded = True

    def _is_cache_stale(self, path: Path) -> bool:
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        return age_hours > _DEFAULT_MAX_AGE_HOURS

    def download_master(self, path: Optional[Path] = None) -> Path:
        """Fetch OpenAPIScripMaster.json from Angel and write to cache."""
        dest = path or self.cache_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = self._settings.scrip_master_url
        logger.info("Downloading instrument master from %s", url)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            raise ValueError("Empty OpenAPIScripMaster response")
        dest.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Cached %s instruments -> %s", len(data), dest)
        return dest

    @staticmethod
    def _parse_expiry(raw: str) -> Optional[date]:
        s = (raw or "").strip().upper()
        if not s:
            return None
        try:
            return datetime.strptime(s, _EXPIRY_FMT).date()
        except ValueError:
            return None

    @staticmethod
    def _parse_lot_size(raw: Any) -> Optional[int]:
        try:
            value = int(float(str(raw).strip()))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _parse_strike(raw: Any) -> Optional[float]:
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        if value >= 100.0:
            value = value / 100.0
        if abs(value - int(value)) < 1e-9:
            return float(int(value))
        return value

    @staticmethod
    def _parse_query_strike(raw: str) -> Optional[float]:
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            return None
        if abs(value - int(value)) < 1e-9:
            return float(int(value))
        return value

    @staticmethod
    def _parse_option_type(symbol: str) -> Optional[str]:
        m = _OPTION_SYMBOL_RE.match(symbol.upper().strip())
        if not m:
            return None
        return _OPTION_TYPE_MAP.get(m.group("cp"))

    def _index_records(self, records: list[dict[str, Any]]) -> None:
        self._by_key.clear()
        self._futures_by_base.clear()
        self._futures_by_symbol.clear()
        self._options_by_base.clear()
        self._options_by_symbol.clear()
        equity_bases: set[str] = set()
        # Skip currency derivatives + indices. Stock/index futures and
        # options are indexed for search and explicit resolution.
        skip_types = frozenset({"AMXIDX", "INDEX", "OPTCUR", "FUTCUR"})
        futures_tmp: dict[str, list[InstrumentRef]] = {}
        options_tmp: dict[str, list[InstrumentRef]] = {}

        for row in records:
            exch = str(row.get("exch_seg", "")).upper()
            inst_type = str(row.get("instrumenttype", "")).upper()
            sym = str(row.get("symbol", "")).upper()
            token = str(row.get("token", ""))
            if not sym or not token:
                continue
            if inst_type in skip_types:
                continue

            if exch == "NSE":
                # Angel cash equities: instrumenttype "" or "EQ" and symbol like RELIANCE-EQ
                is_cash_eq = inst_type == "EQ" or sym.endswith("-EQ") or (
                    not inst_type and "-" not in sym and len(sym) < 20
                )
                if not is_cash_eq:
                    continue
                base = sym.split("-")[0] if "-" in sym else sym
                ref = InstrumentRef(
                    symbol=base,
                    tradingsymbol=sym,
                    symboltoken=token,
                    exchange="NSE",
                    exchange_type=NSE_CM_EXCHANGE_TYPE,
                    instrumenttype=inst_type or "EQ",
                    name=str(row.get("name", "")).upper() or None,
                )
                self._by_key[base] = ref
                self._by_key[sym] = ref
                equity_bases.add(base)
                continue

            if exch == "NFO" and inst_type in {"FUTSTK", "FUTIDX"}:
                base = str(row.get("name", "")).upper().strip()
                if not base:
                    continue
                expiry = self._parse_expiry(str(row.get("expiry", "")))
                ref = InstrumentRef(
                    symbol=f"{base}{_FUT_SUFFIX}",
                    tradingsymbol=sym,
                    symboltoken=token,
                    exchange="NFO",
                    exchange_type=NSE_FO_EXCHANGE_TYPE,
                    instrumenttype=inst_type,
                    expiry=expiry,
                    lot_size=self._parse_lot_size(row.get("lotsize")),
                    name=base,
                )
                futures_tmp.setdefault(base, []).append(ref)
                self._futures_by_symbol[sym] = ref
                continue

            if exch == "NFO" and inst_type in {"OPTSTK", "OPTIDX"}:
                base = str(row.get("name", "")).upper().strip()
                if not base:
                    continue
                expiry = self._parse_expiry(str(row.get("expiry", "")))
                option_type = self._parse_option_type(sym)
                strike = self._parse_strike(row.get("strike"))
                ref = InstrumentRef(
                    symbol=f"{base}{_OPT_SUFFIX}",
                    tradingsymbol=sym,
                    symboltoken=token,
                    exchange="NFO",
                    exchange_type=NSE_FO_EXCHANGE_TYPE,
                    instrumenttype=inst_type,
                    expiry=expiry,
                    lot_size=self._parse_lot_size(row.get("lotsize")),
                    name=base,
                    option_type=option_type,
                    strike=strike,
                )
                options_tmp.setdefault(base, []).append(ref)
                self._options_by_symbol[sym] = ref
                continue

        # Sort each future's contract chain by expiry (None expiries sink to
        # the back) so resolve_future() picks the nearest contract first.
        far_future = date.max
        for base, refs in futures_tmp.items():
            refs.sort(key=lambda r: r.expiry or far_future)
            self._futures_by_base[base] = refs
        for base, refs in options_tmp.items():
            refs.sort(
                key=lambda r: (
                    r.expiry or far_future,
                    float(r.strike or 0.0),
                    str(r.option_type or ""),
                )
            )
            self._options_by_base[base] = refs

        self._equity_bases = sorted(equity_bases)
        self._futures_bases = sorted(self._futures_by_base.keys())
        self._options_bases = sorted(self._options_by_base.keys())
        logger.info(
            "Indexed %s NSE equities, %s NFO futures, and %s NFO options across %s bases",
            len(self._equity_bases),
            sum(len(v) for v in self._futures_by_base.values()),
            sum(len(v) for v in self._options_by_base.values()),
            len(set(self._futures_bases) | set(self._options_bases)),
        )

    @property
    def equity_count(self) -> int:
        self.ensure_loaded()
        return len(self._equity_bases)

    @property
    def futures_base_count(self) -> int:
        self.ensure_loaded()
        return len(self._futures_bases)

    @property
    def options_base_count(self) -> int:
        self.ensure_loaded()
        return len(self._options_bases)

    def catalog(self) -> list[SymbolSearchHit]:
        """Full picker list — every NSE equity plus every active futures contract.

        For futures, all non-expired contracts up to ``contract_depth`` are
        surfaced (default 3 = near/mid/far month).
        """
        self.ensure_loaded()
        hits: list[SymbolSearchHit] = []
        for base in self._equity_bases:
            ref = self._by_key.get(base)
            if ref is not None:
                hits.append(self._hit_from_ref(ref))
        for base in self._futures_bases:
            for idx, ref in enumerate(self._active_contracts(base)):
                hits.append(self._hit_from_future(ref, is_front=(idx == 0)))
        return hits

    def search(self, query: str, *, limit: int = 20) -> list[SymbolSearchHit]:
        """Prefix/substring match across both equities and futures.

        Each matching futures base contributes up to ``contract_depth``
        active contracts (one row per expiry).
        """
        self.ensure_loaded()
        q = query.upper().strip()
        if not q:
            popular = ("RELIANCE", "ICICIBANK", "TCS", "HDFCBANK", "INFY", "SBIN", "GAIL")
            hits: list[SymbolSearchHit] = []
            for base in popular:
                ref = self._by_key.get(base)
                if ref:
                    hits.append(self._hit_from_ref(ref))
                for idx, fut in enumerate(self._active_contracts(base)):
                    hits.append(self._hit_from_future(fut, is_front=(idx == 0)))
            return hits[:limit]

        # Score both segments together — equities slightly outrank futures
        # on identical scores so the cash market shows first.
        scored: list[tuple[int, int, str, str]] = []
        for base in self._equity_bases:
            if base.startswith(q):
                scored.append((0, 0, base, "EQ"))
            elif q in base:
                scored.append((1, 0, base, "EQ"))
        for base in self._futures_bases:
            if base.startswith(q):
                scored.append((0, 1, base, "FUT"))
            elif q in base:
                scored.append((1, 1, base, "FUT"))
        scored.sort(key=lambda t: (t[0], t[1], len(t[2]), t[2]))

        out: list[SymbolSearchHit] = []
        seen_eq: set[str] = set()
        seen_fut: set[tuple[str, str]] = set()  # (base, tradingsymbol)
        for _rank, _seg_rank, base, segment in scored:
            if segment == "EQ":
                if base in seen_eq:
                    continue
                ref = self._by_key.get(base)
                if ref is None:
                    continue
                out.append(self._hit_from_ref(ref))
                seen_eq.add(base)
            else:
                contracts = self._active_contracts(base)
                if not contracts:
                    continue
                for idx, fut in enumerate(contracts):
                    key = (base, fut.tradingsymbol)
                    if key in seen_fut:
                        continue
                    out.append(self._hit_from_future(fut, is_front=(idx == 0)))
                    seen_fut.add(key)
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
        return out

    def search_options(
        self,
        base: str,
        *,
        option_type: Optional[str] = None,
        expiry: Optional[date] = None,
        strike: Optional[float] = None,
        limit: int = 20,
    ) -> list[SymbolSearchHit]:
        self.ensure_loaded()
        chain = self._options_by_base.get(base.upper().strip()) or []
        if not chain:
            return []
        opt_kind = str(option_type or "").upper().strip() or None
        out: list[SymbolSearchHit] = []
        for ref in chain:
            if opt_kind and (ref.option_type or "") != opt_kind:
                continue
            if expiry is not None and ref.expiry != expiry:
                continue
            if strike is not None and ref.strike is not None and abs(ref.strike - strike) > 1e-6:
                continue
            out.append(self._hit_from_option(ref))
            if len(out) >= limit:
                break
        return out

    def _front_future(self, base: str, *, on: Optional[date] = None) -> Optional[InstrumentRef]:
        """Return the nearest non-expired contract for ``base`` (or None).

        Kept for callers that only care about the front month (e.g.
        ``RELIANCE.FUT`` resolution).
        """
        active = self._active_contracts(base, depth=1, on=on)
        if active:
            return active[0]
        chain = self._futures_by_base.get(base.upper())
        return chain[-1] if chain else None

    def _active_contracts(
        self,
        base: str,
        *,
        depth: Optional[int] = None,
        on: Optional[date] = None,
    ) -> list[InstrumentRef]:
        """Return up to ``depth`` non-expired contracts for ``base``.

        Contracts are returned ordered by expiry ascending (front-month
        first). When every contract has expired the chain's tail is
        returned so the picker still surfaces something.
        """
        chain = self._futures_by_base.get(base.upper())
        if not chain:
            return []
        today = on or date.today()
        n = self._contract_depth if depth is None else max(1, int(depth))
        active = [ref for ref in chain if ref.expiry is None or ref.expiry >= today]
        if not active:
            return [chain[-1]]
        return active[:n]

    @staticmethod
    def _hit_from_ref(ref: InstrumentRef) -> SymbolSearchHit:
        fortuna_sym = normalize_for_smartapi(ref.symbol)
        return SymbolSearchHit(
            display=f"{ref.symbol} — {ref.tradingsymbol} (NSE)",
            symbol=fortuna_sym,
            tradingsymbol=ref.tradingsymbol,
            segment="EQUITY",
        )

    @staticmethod
    def _hit_from_future(ref: InstrumentRef, *, is_front: bool = True) -> SymbolSearchHit:
        """Build a picker hit for one futures contract.

        Front-month contracts keep the bare ``BASE.FUT`` symbol so existing
        watchlists/strategies continue to track the rolling front-month.
        Back-month contracts use the explicit ``BASE.FUT.DDMMMYYYY`` form
        so the picker can show one row per expiry without collapsing.
        """
        base = ref.name or ref.symbol.replace(_FUT_SUFFIX, "")
        if ref.expiry is not None:
            exp_label = ref.expiry.strftime("%d-%b-%Y")
            tag = "front" if is_front else "back"
            display = f"{base} FUT {exp_label} — {ref.tradingsymbol} (NFO, {tag})"
            if is_front:
                fortuna_symbol = ref.symbol  # e.g. "RELIANCE.FUT"
            else:
                fortuna_symbol = (
                    f"{ref.symbol}.{ref.expiry.strftime(_EXPIRY_FMT).upper()}"
                )
        else:
            display = f"{base} FUT — {ref.tradingsymbol} (NFO)"
            fortuna_symbol = ref.symbol
        return SymbolSearchHit(
            display=display,
            symbol=fortuna_symbol,
            tradingsymbol=ref.tradingsymbol,
            segment="FUTURES",
        )

    @staticmethod
    def _hit_from_option(ref: InstrumentRef) -> SymbolSearchHit:
        base = ref.name or ref.symbol.replace(_OPT_SUFFIX, "")
        strike = f"{ref.strike:g}" if ref.strike is not None else "NA"
        expiry = ref.expiry.strftime("%d-%b-%Y") if ref.expiry is not None else "NA"
        fortuna_symbol = ref.symbol
        if ref.option_type and ref.strike is not None and ref.expiry is not None:
            fortuna_symbol = (
                f"{ref.symbol}.{ref.option_type}.{strike}."
                f"{ref.expiry.strftime(_EXPIRY_FMT).upper()}"
            )
        display = (
            f"{base} {ref.option_type or 'OPT'} {strike} {expiry} — "
            f"{ref.tradingsymbol} (NFO)"
        )
        return SymbolSearchHit(
            display=display,
            symbol=fortuna_symbol,
            tradingsymbol=ref.tradingsymbol,
            segment="OPTIONS",
        )

    def resolve(self, symbol: str) -> InstrumentRef:
        """Resolve a Fortuna symbol to an SmartAPI ``InstrumentRef``.

        Accepts:

        - Equity: ``RELIANCE.NS``, ``RELIANCE-EQ``, ``RELIANCE``.
        - Futures: ``RELIANCE.FUT`` (current-month), ``RELIANCE.FUT.27NOV2025``
          (explicit expiry), or Angel's raw tradingsymbol like
          ``RELIANCE25NOVFUT``.
        - Options: ``RELIANCE.OPT.CE.2500.27NOV2025`` or Angel's raw option
          tradingsymbol when known.
        """
        self.ensure_loaded()
        raw = symbol.upper().strip()

        # Direct Angel tradingsymbol for a future (e.g. "CROMPTON25MAYFUT").
        if raw in self._futures_by_symbol:
            return self._futures_by_symbol[raw]
        if raw in self._options_by_symbol:
            return self._options_by_symbol[raw]

        if _FUT_SUFFIX in raw:
            base = self._base_symbol(raw)
            tail = raw.split(_FUT_SUFFIX, 1)[1].lstrip(".")
            if tail:
                target = self._parse_expiry(tail)
                chain = self._futures_by_base.get(base) or []
                for ref in chain:
                    if ref.expiry == target:
                        return ref
                raise KeyError(
                    f"No NFO future for {symbol!r} with expiry {tail!r}. "
                    "Check the date or refresh the scrip master."
                )
            front = self._front_future(base)
            if front is None:
                raise KeyError(
                    f"No NFO futures chain for {symbol!r} (base={base!r}). "
                    "Refresh the scrip master or pick a different symbol."
                )
            return front

        if _OPT_SUFFIX in raw:
            base = self._base_symbol(raw)
            tail = raw.split(_OPT_SUFFIX, 1)[1].lstrip(".")
            parts = [p for p in tail.split(".") if p]
            if len(parts) < 3:
                raise KeyError(
                    f"Option symbol {symbol!r} must look like BASE.OPT.CE.2500.27NOV2025"
                )
            opt_type = _OPTION_TYPE_MAP.get(parts[0].upper())
            strike = self._parse_query_strike(parts[1])
            expiry = self._parse_expiry(parts[2])
            if opt_type is None or strike is None or expiry is None:
                raise KeyError(
                    "Could not parse option contract from "
                    f"{symbol!r}; use BASE.OPT.CE.2500.27NOV2025"
                )
            chain = self._options_by_base.get(base) or []
            for ref in chain:
                if ref.option_type != opt_type or ref.expiry != expiry or ref.strike is None:
                    continue
                if abs(ref.strike - strike) <= 1e-6:
                    return ref
            raise KeyError(
                f"No NFO option for {symbol!r}. Refresh the scrip master or check expiry/strike."
            )

        base = self._base_symbol(raw)
        ref = self._by_key.get(base)
        if ref is None:
            raise KeyError(
                f"No NSE EQ instrument for {symbol!r} (base={base!r}). "
                "Refresh scrip master or check symbol."
            )
        return ref
