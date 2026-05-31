"""Native agentic conversational layer above typed Fortuna advisory tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRegistry
from fortuna.telegram.parser import TelegramRequest, TelegramRequestKind, parse_telegram_request
from fortuna.telegram.router import (
    ClarificationCode,
    ConversationRoute,
    RouteAction,
    route_telegram_request,
)

_ANALYZE_HINTS = frozenset(
    {
        "analyze",
        "analysis",
        "advise",
        "advice",
        "buy",
        "sell",
        "enter",
        "entry",
        "exit",
        "looking",
        "look",
        "check",
        "should",
        "view",
    }
)
_SEARCH_HINTS = frozenset({"search", "find", "lookup", "show", "list"})
_HELP_HINTS = frozenset({"help", "commands"})
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "from",
        "give",
        "i",
        "is",
        "it",
        "looking",
        "market",
        "me",
        "of",
        "on",
        "please",
        "should",
        "stock",
        "tell",
        "the",
        "to",
        "view",
        "what",
        "with",
    }
)
_TIMEFRAME_RE = re.compile(r"\b(1m|5m|15m|30m|1h|1d)\b", re.IGNORECASE)
_DAYS_RE = re.compile(r"\b(?:days=)?(\d{1,3})d\b", re.IGNORECASE)
_OPTION_RE = re.compile(
    r"\b(?P<underlying>[A-Z][A-Z0-9]+)\s+(?P<cp>CE|PE|C|P)\s+(?P<strike>\d+(?:\.\d+)?)\s+"
    r"(?P<expiry>\d{1,2}[A-Z]{3}\d{4}|\d{1,2}-[A-Z]{3}-\d{4}|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_FUTURE_RE = re.compile(
    r"\b(?P<underlying>[A-Z][A-Z0-9]+)\s+(?:FUT|FUTURE)"
    r"(?:\s+(?P<expiry>\d{1,2}[A-Z]{3}\d{4}|\d{1,2}-[A-Z]{3}-\d{4}|\d{4}-\d{2}-\d{2}))?\b",
    re.IGNORECASE,
)
_RAW_EXPIRY_RE = re.compile(
    r"\b(\d{1,2}[A-Z]{3}\d{4}|\d{1,2}-[A-Z]{3}-\d{4}|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


class AdapterSource(str, Enum):
    COMMAND = "command"
    CONVERSATIONAL = "conversational"


@dataclass(frozen=True)
class AdapterPlan:
    request: TelegramRequest
    route: ConversationRoute
    source: AdapterSource
    confidence: float
    rationale: str


class FortunaConversationalAdapter:
    """Map freer-form operator text onto Fortuna's typed search/analyze tools."""

    def __init__(
        self,
        settings: Settings,
        *,
        registry: Optional[InstrumentRegistry] = None,
        catalog: Optional[SymbolCatalog] = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.catalog = catalog

    def plan(self, text: str) -> AdapterPlan:
        req = parse_telegram_request(text)
        route = route_telegram_request(req)
        if route.action != RouteAction.UNSUPPORTED:
            return AdapterPlan(
                request=req,
                route=route,
                source=AdapterSource.COMMAND,
                confidence=1.0,
                rationale="explicit command matched the primary parser",
            )
        return self._plan_conversational(req.raw_text)

    def _plan_conversational(self, text: str) -> AdapterPlan:
        raw = (text or "").strip()
        if not raw:
            req = TelegramRequest(TelegramRequestKind.HELP, raw_text=raw)
            return AdapterPlan(
                request=req,
                route=route_telegram_request(req),
                source=AdapterSource.CONVERSATIONAL,
                confidence=1.0,
                rationale="empty freeform request maps to help",
            )

        words = {tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", raw)}
        timeframe = self._extract_timeframe(raw)
        days = self._extract_days(raw)

        option_req = self._option_request(raw, timeframe=timeframe, days=days)
        if option_req is not None:
            return AdapterPlan(
                request=option_req,
                route=route_telegram_request(option_req),
                source=AdapterSource.CONVERSATIONAL,
                confidence=0.95,
                rationale="freeform request matched explicit option contract syntax",
            )

        future_req = self._future_request(raw, timeframe=timeframe, days=days)
        if future_req is not None:
            return AdapterPlan(
                request=future_req,
                route=route_telegram_request(future_req),
                source=AdapterSource.CONVERSATIONAL,
                confidence=0.9,
                rationale="freeform request matched a futures contract pattern",
            )

        if words & _HELP_HINTS:
            req = TelegramRequest(TelegramRequestKind.HELP, raw_text=raw)
            return AdapterPlan(
                request=req,
                route=route_telegram_request(req),
                source=AdapterSource.CONVERSATIONAL,
                confidence=0.84,
                rationale="freeform request asked for usage help",
            )

        candidate = self._best_symbol_candidate(raw)
        if candidate is None:
            req = TelegramRequest(TelegramRequestKind.UNKNOWN, raw_text=raw, query=raw)
            return AdapterPlan(
                request=req,
                route=ConversationRoute(
                    RouteAction.CLARIFY,
                    req,
                    clarification=ClarificationCode.FREEFORM_INSTRUMENT_NEEDED,
                ),
                source=AdapterSource.CONVERSATIONAL,
                confidence=0.2,
                rationale="could not infer a stable instrument from freeform text",
            )

        wants_search = bool(words & _SEARCH_HINTS) and not bool(words & _ANALYZE_HINTS)
        if wants_search:
            req = TelegramRequest(TelegramRequestKind.SEARCH, raw_text=raw, query=candidate)
            return AdapterPlan(
                request=req,
                route=route_telegram_request(req),
                source=AdapterSource.CONVERSATIONAL,
                confidence=0.72,
                rationale="freeform request looked like symbol discovery/search intent",
            )

        req = TelegramRequest(
            TelegramRequestKind.ANALYZE,
            raw_text=raw,
            query=candidate,
            symbol=candidate,
            timeframe=timeframe,
            days=days,
        )
        return AdapterPlan(
            request=req,
            route=route_telegram_request(req),
            source=AdapterSource.CONVERSATIONAL,
            confidence=0.78,
            rationale="freeform request was normalized into a typed analysis request",
        )

    def _best_symbol_candidate(self, text: str) -> Optional[str]:
        catalog = self._symbol_catalog()
        registry = self._registry()
        cleaned = re.sub(_TIMEFRAME_RE, " ", text)
        cleaned = re.sub(_RAW_EXPIRY_RE, " ", cleaned)
        cleaned = re.sub(r"[/#.,!?]", " ", cleaned)
        tokens = [tok.upper() for tok in cleaned.split() if tok]
        candidates = [tok for tok in tokens if tok.lower() not in _STOPWORDS]
        if not candidates:
            return None

        for token in candidates:
            if ".NS" in token or ".FUT" in token or ".OPT." in token:
                return token
            try:
                registry.resolve(token)
                return token
            except KeyError:
                pass

        hits = []
        for token in candidates:
            found = catalog.search(token, limit=3)
            if found:
                hits.append((token, found[0].symbol))
        if not hits:
            return None
        first_symbol = hits[0][1]
        if all(symbol == first_symbol for _, symbol in hits):
            return first_symbol
        return hits[0][0]

    @staticmethod
    def _extract_timeframe(text: str) -> str:
        match = _TIMEFRAME_RE.search(text)
        return match.group(1) if match is not None else "5m"

    @staticmethod
    def _extract_days(text: str) -> int:
        match = _DAYS_RE.search(text)
        return max(1, int(match.group(1))) if match is not None else 30

    def _option_request(self, text: str, *, timeframe: str, days: int) -> Optional[TelegramRequest]:
        match = _OPTION_RE.search(text)
        if match is None:
            return None
        from fortuna.telegram.parser import _normalize_expiry

        option_type = match.group("cp").upper()
        option_type = "CE" if option_type == "C" else "PE" if option_type == "P" else option_type
        expiry = _normalize_expiry(match.group("expiry"))
        strike = match.group("strike")
        underlying = match.group("underlying").upper()
        symbol = f"{underlying}.OPT.{option_type}.{strike}.{expiry}"
        return TelegramRequest(
            TelegramRequestKind.ANALYZE,
            raw_text=text,
            query=f"{underlying} {option_type} {strike} {expiry}",
            symbol=symbol,
            timeframe=timeframe,
            days=days,
        )

    def _future_request(self, text: str, *, timeframe: str, days: int) -> Optional[TelegramRequest]:
        match = _FUTURE_RE.search(text)
        if match is None:
            return None
        from fortuna.telegram.parser import _normalize_expiry

        underlying = match.group("underlying").upper()
        expiry = match.group("expiry")
        symbol = f"{underlying}.FUT"
        query = f"{underlying} FUT"
        if expiry:
            normalized = _normalize_expiry(expiry)
            symbol = f"{symbol}.{normalized}"
            query = f"{query} {normalized}"
        return TelegramRequest(
            TelegramRequestKind.ANALYZE,
            raw_text=text,
            query=query,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
        )

    def _registry(self) -> InstrumentRegistry:
        if self.registry is None:
            self.registry = InstrumentRegistry()
        self.registry.ensure_loaded()
        return self.registry

    def _symbol_catalog(self) -> SymbolCatalog:
        if self.catalog is None:
            self.catalog = SymbolCatalog(self._registry())
        self.catalog.ensure_loaded()
        return self.catalog
