"""Telegram-facing search and analysis assistant built on Fortuna runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from fortuna.agentic.conversational_adapter import (
    AdapterPlan,
    AdapterSource,
    FortunaConversationalAdapter,
)
from fortuna.agentic.tools import FortunaAdvisoryTools
from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRegistry
from fortuna.telegram.formatters import (
    format_clarification,
    format_help,
    format_instrument_analysis,
    format_instrument_search,
    format_unsupported,
)
from fortuna.telegram.parser import TelegramRequest, parse_telegram_request
from fortuna.telegram.router import ConversationRoute, RouteAction, route_telegram_request


@dataclass(frozen=True)
class InteractionResult:
    reply: str
    request: TelegramRequest
    route: ConversationRoute
    source: str = AdapterSource.COMMAND.value
    source_confidence: float = 1.0
    source_rationale: str = ""
    tool: str | None = None
    tool_ok: bool | None = None
    error_code: str | None = None
    resolved_symbol: str | None = None
    decision_action: str | None = None
    hit_count: int | None = None


@dataclass
class TelegramAnalysisAssistant:
    settings: Settings
    engine_factory: Optional[Callable[[], Any]] = None
    registry: Optional[InstrumentRegistry] = None
    catalog: Optional[SymbolCatalog] = None
    tools: Optional[FortunaAdvisoryTools] = None
    adapter: Optional[FortunaConversationalAdapter] = None

    def handle_text(self, text: str) -> str:
        return self.handle_interaction(text).reply

    def handle_interaction(self, text: str) -> InteractionResult:
        plan = self._plan(text)
        return self.execute_route(
            plan.route,
            source=plan.source.value,
            source_confidence=plan.confidence,
            source_rationale=plan.rationale,
        )

    def execute_route(
        self,
        route: ConversationRoute,
        *,
        source: str = AdapterSource.COMMAND.value,
        source_confidence: float = 1.0,
        source_rationale: str = "",
    ) -> InteractionResult:
        if route.action == RouteAction.SHOW_HELP:
            return InteractionResult(
                reply=format_help(),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
            )
        if route.action == RouteAction.CLARIFY:
            return InteractionResult(
                reply=format_clarification(route.clarification),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
            )
        if route.action == RouteAction.UNSUPPORTED:
            return InteractionResult(
                reply=format_unsupported(),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
            )
        if route.action == RouteAction.SEARCH:
            response = self._tools().search_instruments(route.request.query)
            error_code = response.error.code.value if response.error is not None else None
            return InteractionResult(
                reply=format_instrument_search(response),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="search_instruments",
                tool_ok=response.ok,
                error_code=error_code,
                hit_count=len(response.hits),
            )
        if route.action == RouteAction.ANALYZE:
            response = self._tools().analyze_instrument(
                route.request.symbol,
                timeframe=route.request.timeframe,
                days=route.request.days,
            )
            error_code = response.error.code.value if response.error is not None else None
            resolved_symbol = (
                response.instrument.resolved_symbol if response.instrument is not None else None
            )
            decision_action = response.decision.action if response.decision is not None else None
            return InteractionResult(
                reply=format_instrument_analysis(response),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="analyze_instrument",
                tool_ok=response.ok,
                error_code=error_code,
                resolved_symbol=resolved_symbol,
                decision_action=decision_action,
            )
        return InteractionResult(
            reply=format_unsupported(),
            request=route.request,
            route=route,
            source=source,
            source_confidence=source_confidence,
            source_rationale=source_rationale,
        )

    def _tools(self) -> FortunaAdvisoryTools:
        if self.tools is not None:
            return self.tools
        return FortunaAdvisoryTools(
            settings=self.settings,
            engine_factory=self.engine_factory,
            registry=self.registry,
            catalog=self.catalog,
        )

    def _plan(self, text: str) -> AdapterPlan:
        if getattr(self.settings, "conversational_adapter_enabled", False):
            return self._adapter().plan(text)
        req = parse_telegram_request(text)
        route = route_telegram_request(req)
        return AdapterPlan(
            request=req,
            route=route,
            source=AdapterSource.COMMAND,
            confidence=1.0,
            rationale="conversational adapter disabled; using direct command parsing",
        )

    def _adapter(self) -> FortunaConversationalAdapter:
        if self.adapter is not None:
            return self.adapter
        return FortunaConversationalAdapter(
            settings=self.settings,
            registry=self.registry,
            catalog=self.catalog,
        )
