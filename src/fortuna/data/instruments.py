"""Angel One instrument master (OpenAPIScripMaster.json) lookup."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fortuna.config.smartapi_settings import SmartAPISettings, get_smartapi_settings
from fortuna.data.sources.symbols import normalize_for_smartapi
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

# WebSocket exchange_type for NSE cash market (SmartWebSocketV2.NSE_CM)
NSE_CM_EXCHANGE_TYPE = 1

_DEFAULT_MAX_AGE_HOURS = 24


@dataclass(frozen=True)
class InstrumentRef:
    """Resolved SmartAPI instrument for a Fortuna symbol."""

    symbol: str
    tradingsymbol: str
    symboltoken: str
    exchange: str
    exchange_type: int = NSE_CM_EXCHANGE_TYPE
    instrumenttype: str = "EQ"


@dataclass(frozen=True)
class SymbolSearchHit:
    """Autocomplete result for dashboard symbol search."""

    display: str
    symbol: str
    tradingsymbol: str


class InstrumentRegistry:
    """Download, cache, and resolve Angel OpenAPIScripMaster entries."""

    def __init__(
        self,
        settings: Optional[SmartAPISettings] = None,
        *,
        project_root: Optional[Path] = None,
    ) -> None:
        self._settings = settings or get_smartapi_settings()
        self._project_root = project_root
        self._by_key: dict[str, InstrumentRef] = {}
        self._equity_bases: list[str] = []
        self._loaded = False

    @property
    def cache_path(self) -> Path:
        return self._settings.resolve_path(self._project_root)

    def _base_symbol(self, symbol: str) -> str:
        s = symbol.upper().strip()
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

    def _index_records(self, records: list[dict[str, Any]]) -> None:
        self._by_key.clear()
        bases: set[str] = set()
        skip_types = frozenset(
            {"OPTSTK", "FUTSTK", "FUTIDX", "OPTIDX", "AMXIDX", "INDEX", "OPTCUR", "FUTCUR"}
        )
        for row in records:
            exch = str(row.get("exch_seg", "")).upper()
            if exch != "NSE":
                continue
            inst_type = str(row.get("instrumenttype", "")).upper()
            sym = str(row.get("symbol", "")).upper()
            token = str(row.get("token", ""))
            if not sym or not token:
                continue
            if inst_type in skip_types:
                continue
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
                instrumenttype=inst_type,
            )
            self._by_key[base] = ref
            self._by_key[sym] = ref
            bases.add(base)
        self._equity_bases = sorted(bases)

    @property
    def equity_count(self) -> int:
        self.ensure_loaded()
        return len(self._equity_bases)

    def catalog(self) -> list[SymbolSearchHit]:
        """Full NSE equity list for dashboard symbol picker (sorted by symbol)."""
        self.ensure_loaded()
        return [self._hit_from_ref(self._by_key[b]) for b in self._equity_bases if b in self._by_key]

    def search(self, query: str, *, limit: int = 20) -> list[SymbolSearchHit]:
        """Prefix/substring match on NSE equity symbols for UI autocomplete."""
        self.ensure_loaded()
        q = query.upper().strip()
        if not q:
            popular = ("RELIANCE", "ICICIBANK", "TCS", "HDFCBANK", "INFY", "SBIN", "GAIL")
            hits: list[SymbolSearchHit] = []
            for base in popular:
                ref = self._by_key.get(base)
                if ref:
                    hits.append(self._hit_from_ref(ref))
            return hits[:limit]

        matches: list[tuple[int, str]] = []
        for base in self._equity_bases:
            if base.startswith(q):
                matches.append((0, base))
            elif q in base:
                matches.append((1, base))
        matches.sort(key=lambda x: (x[0], len(x[1]), x[1]))
        out: list[SymbolSearchHit] = []
        seen: set[str] = set()
        for _, base in matches:
            if base in seen:
                continue
            ref = self._by_key.get(base)
            if ref is None:
                continue
            seen.add(base)
            out.append(self._hit_from_ref(ref))
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _hit_from_ref(ref: InstrumentRef) -> SymbolSearchHit:
        fortuna_sym = normalize_for_smartapi(ref.symbol)
        return SymbolSearchHit(
            display=f"{ref.symbol} — {ref.tradingsymbol} (NSE)",
            symbol=fortuna_sym,
            tradingsymbol=ref.tradingsymbol,
        )

    def resolve(self, symbol: str) -> InstrumentRef:
        """Map RELIANCE.NS / RELIANCE-EQ / RELIANCE to InstrumentRef."""
        self.ensure_loaded()
        base = self._base_symbol(symbol)
        ref = self._by_key.get(base)
        if ref is None:
            raise KeyError(
                f"No NSE EQ instrument for {symbol!r} (base={base!r}). "
                "Refresh scrip master or check symbol."
            )
        return ref
