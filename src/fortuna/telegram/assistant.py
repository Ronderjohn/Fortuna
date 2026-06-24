"""Telegram-facing search and analysis assistant built on Fortuna runtime."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from fortuna.agentic.chart_analyzer import ChartImageAnalyzer
from fortuna.agentic.contracts import MediaAttachment, SignalLayerUsage, SignalResponse
from fortuna.agentic.conversational_adapter import (
    AdapterPlan,
    AdapterSource,
    FortunaConversationalAdapter,
)
from fortuna.agentic.redaction import looks_like_prompt_injection, redact_secrets
from fortuna.agentic.tools import FortunaAdvisoryTools
from fortuna.app.forecasting import merge_forecasts_into_signal, run_forecast_lane
from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRegistry
from fortuna.telegram.formatters import (
    format_clarification,
    format_help,
    format_instrument_analysis,
    format_instrument_search,
    format_market_universe,
    format_model_health,
    format_multi_agent_workflow,
    format_portfolio_allocation,
    format_shortlist_briefing,
    format_training_candidates,
    format_training_research_plan,
    format_unsupported,
)
from fortuna.telegram.parser import TelegramRequest, TelegramRequestKind, parse_telegram_request
from fortuna.telegram.router import ConversationRoute, RouteAction, route_telegram_request
from fortuna.telegram.session_store import TelegramConversationState

_TIMEFRAME_RE = re.compile(r"\b(1m|5m|15m|30m|1h|1d)\b", re.IGNORECASE)
_DAYS_RE = re.compile(r"\b(?:days=)?(\d{1,3})d\b", re.IGNORECASE)
_RISK_INTENT_RE = re.compile(
    r"\b(risk|risky|lot|lots|stop|sl|invalidation|capital|budget|max\s+loss)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _CacheEntry:
    reply: str
    signal_response: SignalResponse | None
    created_at: float


@dataclass(frozen=True)
class _ValueCacheEntry:
    value: Any
    created_at: float


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
    request_id: str = ""
    signal_response: SignalResponse | None = None
    modality: str = "text"
    attachment_kind: str | None = None


@dataclass
class TelegramAnalysisAssistant:
    settings: Settings
    engine_factory: Optional[Callable[[], Any]] = None
    registry: Optional[InstrumentRegistry] = None
    catalog: Optional[SymbolCatalog] = None
    tools: Optional[FortunaAdvisoryTools] = None
    adapter: Optional[FortunaConversationalAdapter] = None
    chart_analyzer: Optional[ChartImageAnalyzer] = None
    _response_cache: dict[tuple[str, ...], _CacheEntry] = field(default_factory=dict, init=False)
    _image_cache: dict[tuple[str, ...], _ValueCacheEntry] = field(default_factory=dict, init=False)
    _forecast_cache: dict[tuple[str, ...], _ValueCacheEntry] = field(
        default_factory=dict,
        init=False,
    )
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def handle_text(self, text: str) -> str:
        return self.handle_interaction(text).reply

    def handle_interaction(
        self,
        text: str,
        *,
        chat_id: str = "",
        request_id: str = "",
        conversation_state: TelegramConversationState | None = None,
        is_admin: bool = False,
        reply_style: str | None = None,
        attachment: MediaAttachment | None = None,
    ) -> InteractionResult:
        effective_text = redact_secrets(
            self._rewrite_follow_up(text, conversation_state),
            enabled=bool(getattr(self.settings, "signal_secret_redaction_enabled", True)),
        )
        if attachment is not None:
            return self._handle_image_interaction(
                effective_text,
                attachment=attachment,
                request_id=request_id,
                conversation_state=conversation_state,
                is_admin=is_admin,
                reply_style=reply_style,
            )
        plan = self._plan(effective_text)
        return self.execute_route(
            plan.route,
            source=plan.source.value,
            source_confidence=plan.confidence,
            source_rationale=plan.rationale,
            request_id=request_id,
            is_admin=is_admin,
            reply_style=reply_style,
        )

    def execute_route(
        self,
        route: ConversationRoute,
        *,
        source: str = AdapterSource.COMMAND.value,
        source_confidence: float = 1.0,
        source_rationale: str = "",
        request_id: str = "",
        is_admin: bool = False,
        reply_style: str | None = None,
    ) -> InteractionResult:
        effective_reply_style = self._effective_reply_style(reply_style)
        if route.action == RouteAction.SHOW_HELP:
            help_text = format_help(
                advanced=is_admin or self._help_explicitly_advanced(route.request.raw_text),
                expert_commands_visible=bool(
                    getattr(self.settings, "signal_expert_commands_visible", False)
                ),
            )
            return InteractionResult(
                reply=help_text,
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                request_id=request_id,
            )
        if route.action == RouteAction.CLARIFY:
            return InteractionResult(
                reply=format_clarification(route.clarification),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                request_id=request_id,
            )
        if route.action == RouteAction.UNSUPPORTED:
            return InteractionResult(
                reply=format_unsupported(
                    expert_commands_visible=bool(
                        getattr(self.settings, "signal_expert_commands_visible", False)
                    )
                ),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                request_id=request_id,
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
                request_id=request_id,
            )
        if route.action == RouteAction.UNIVERSE:
            response = self._tools().get_market_universe(
                limit=route.request.limit,
                timeframe=route.request.timeframe,
                days=route.request.days,
                source=route.request.source,
            )
            error_code = response.error.code.value if response.error is not None else None
            return InteractionResult(
                reply=format_market_universe(response),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="get_market_universe",
                tool_ok=response.ok,
                error_code=error_code,
                hit_count=len(response.candidates),
                request_id=request_id,
            )
        if route.action == RouteAction.RESEARCH:
            analysis_limit = max(1, route.request.limit)
            response = self._tools().get_training_research_plan(
                universe_limit=max(analysis_limit * 2, 12),
                analysis_limit=analysis_limit,
                timeframe=route.request.timeframe,
                days=route.request.days,
                source=route.request.source,
                refresh_target=route.request.target,
            )
            error_code = response.error.code.value if response.error is not None else None
            return InteractionResult(
                reply=format_training_research_plan(response, target=route.request.target),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="get_training_research_plan",
                tool_ok=response.ok,
                error_code=error_code,
                hit_count=len(response.rows),
                request_id=request_id,
            )
        if route.action == RouteAction.BRIEF:
            analysis_limit = max(1, route.request.limit)
            response = self._tools().get_shortlist_briefing(
                universe_limit=max(analysis_limit * 2, 10),
                analysis_limit=analysis_limit,
                timeframe=route.request.timeframe,
                days=route.request.days,
                source=route.request.source,
            )
            error_code = response.error.code.value if response.error is not None else None
            return InteractionResult(
                reply=format_shortlist_briefing(response),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="get_shortlist_briefing",
                tool_ok=response.ok,
                error_code=error_code,
                hit_count=len(response.items),
                request_id=request_id,
            )
        if route.action == RouteAction.ALLOCATE:
            analysis_limit = max(1, route.request.limit)
            response = self._tools().get_portfolio_allocation(
                universe_limit=max(analysis_limit * 2, 10),
                analysis_limit=analysis_limit,
                max_positions=route.request.max_positions,
                max_per_exposure=route.request.max_per_exposure,
                max_same_side=route.request.max_same_side,
                timeframe=route.request.timeframe,
                days=route.request.days,
                source=route.request.source,
            )
            error_code = response.error.code.value if response.error is not None else None
            return InteractionResult(
                reply=format_portfolio_allocation(response),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="get_portfolio_allocation",
                tool_ok=response.ok,
                error_code=error_code,
                hit_count=len(response.items),
                request_id=request_id,
            )
        if route.action == RouteAction.CANDIDATES:
            analysis_limit = max(1, route.request.limit)
            response = self._tools().get_training_candidates(
                universe_limit=max(analysis_limit * 2, 12),
                analysis_limit=analysis_limit,
                timeframe=route.request.timeframe,
                days=route.request.days,
                source=route.request.source,
            )
            error_code = response.error.code.value if response.error is not None else None
            return InteractionResult(
                reply=format_training_candidates(response, target=route.request.target),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="get_training_candidates",
                tool_ok=response.ok,
                error_code=error_code,
                hit_count=len(response.candidates),
                request_id=request_id,
            )
        if route.action == RouteAction.WORKFLOW:
            analysis_limit = max(1, route.request.limit)
            response = self._tools().get_multi_agent_workflow(
                universe_limit=max(analysis_limit * 2, 10),
                analysis_limit=analysis_limit,
                timeframe=route.request.timeframe,
                days=route.request.days,
                source=route.request.source,
            )
            return InteractionResult(
                reply=format_multi_agent_workflow(response),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="get_multi_agent_workflow",
                tool_ok=response.ok,
                hit_count=len(getattr(getattr(response, "research", None), "rows", ()) or ()),
                request_id=request_id,
            )
        if route.action == RouteAction.HEALTH:
            response = self._tools().get_model_health()
            return InteractionResult(
                reply=format_model_health(response),
                request=route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="get_model_health",
                tool_ok=True,
                request_id=request_id,
            )
        if route.action == RouteAction.ANALYZE:
            return self._execute_signal_analysis(
                route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                request_id=request_id,
                is_admin=is_admin,
                reply_style=effective_reply_style,
                modality="text",
            )
        if route.action == RouteAction.RISK:
            return self._execute_signal_analysis(
                route.request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                request_id=request_id,
                is_admin=is_admin,
                reply_style=effective_reply_style,
                modality="text",
                risk_intent=True,
            )
        return InteractionResult(
            reply=format_unsupported(
                expert_commands_visible=bool(
                    getattr(self.settings, "signal_expert_commands_visible", False)
                )
            ),
            request=route.request,
            route=route,
            source=source,
            source_confidence=source_confidence,
            source_rationale=source_rationale,
            request_id=request_id,
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
        if looks_like_prompt_injection(text):
            req = parse_telegram_request(text)
            route = route_telegram_request(req)
            return AdapterPlan(
                request=req,
                route=route,
                source=AdapterSource.COMMAND,
                confidence=0.4,
                rationale="prompt-injection-like content bypassed model routing",
            )
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

    def _help_explicitly_advanced(self, raw_text: str) -> bool:
        lowered = str(raw_text or "").strip().lower()
        return "admin" in lowered or "advanced" in lowered

    def _effective_reply_style(self, reply_style: str | None) -> str:
        style = str(reply_style or getattr(self.settings, "signal_reply_style", "compact")).strip()
        return style.lower() if style else "compact"

    def _rewrite_follow_up(
        self,
        text: str,
        conversation_state: TelegramConversationState | None,
    ) -> str:
        raw = str(text or "").strip()
        if not raw or conversation_state is None:
            return raw
        if raw.startswith("/") or raw.startswith("#"):
            return raw
        symbol = str(
            conversation_state.last_symbol or conversation_state.last_resolved_symbol or ""
        ).strip()
        if not symbol:
            return raw
        lowered = raw.lower()
        timeframe_match = _TIMEFRAME_RE.search(raw)
        days_match = _DAYS_RE.search(raw)
        looks_like_follow_up = (
            raw.lower() in {"1m", "5m", "15m", "30m", "1h", "1d"}
            or lowered.startswith("what about")
            or lowered.startswith("and ")
            or lowered.startswith("about ")
            or lowered.startswith("for ")
            or lowered.endswith("?")
        )
        if not looks_like_follow_up and timeframe_match is None and days_match is None:
            return raw
        timeframe = (
            timeframe_match.group(1)
            if timeframe_match is not None
            else conversation_state.last_timeframe
        )
        days = (
            max(1, int(days_match.group(1)))
            if days_match is not None
            else max(1, int(conversation_state.last_days or 30))
        )
        if "search" in lowered:
            return f"/search {symbol}"
        if _RISK_INTENT_RE.search(raw) is not None:
            return f"/risk {symbol} {timeframe} {days}d"
        return f"/analyze {symbol} {timeframe} {days}d"

    def _build_signal_response(
        self,
        response: Any,
        *,
        request_id: str,
        modality: str = "text",
    ) -> SignalResponse | None:
        if not getattr(response, "ok", False):
            return None
        instrument = getattr(response, "instrument", None)
        decision = getattr(response, "decision", None)
        if instrument is None:
            return None
        if decision is None:
            verdict = "HOLD"
            confidence = 0.0
            summary = "No advisory decision was produced."
            reasons: tuple[str, ...] = ()
            risk_notes: tuple[str, ...] = ()
            action_plan = "Wait for a clearer setup."
        else:
            verdict = str(decision.action or "HOLD")
            confidence = float(decision.confidence or 0.0)
            summary = str(decision.summary or "").strip()
            reasons = tuple(decision.reasons or ())
            risk_notes = tuple(decision.risk_notes or ())
            action_plan = str(decision.action_plan or "").strip()
        winning_strategy = str(getattr(response, "winning_strategy", "") or "").strip()
        market_context = f"Winning strategy: {winning_strategy}" if winning_strategy else None
        structured_signal = getattr(response, "structured_signal", None)
        replay_risk = getattr(response, "replay_risk", None)
        lot_risk = getattr(response, "futures_lot_risk", None)
        oi_enrichment = getattr(response, "oi_enrichment", None)
        data_freshness = ""
        last_bar_time = getattr(instrument, "last_bar_time", None)
        if last_bar_time is not None:
            if hasattr(last_bar_time, "strftime"):
                data_freshness = f"Last bar: {last_bar_time.strftime('%Y-%m-%d %H:%M')}"
            else:
                data_freshness = f"Last bar: {last_bar_time}"
        return SignalResponse(
            resolved_symbol=str(instrument.resolved_symbol),
            segment=str(instrument.segment_label),
            timeframe=str(instrument.timeframe),
            lookback_days=max(1, int(instrument.lookback_days)),
            signal_verdict=verdict,
            confidence=confidence,
            summary=summary,
            reasons=reasons,
            risk_notes=risk_notes,
            action_plan=action_plan,
            data_freshness=data_freshness,
            market_context=market_context,
            used_layers=self._resolve_used_layers(response),
            request_modality=modality,
            stop_loss=getattr(structured_signal, "stop_loss", None),
            target_price=getattr(structured_signal, "target_price", None),
            invalidation=str(getattr(structured_signal, "invalidation", "") or ""),
            setup_type=str(getattr(structured_signal, "setup_type", "") or ""),
            risk_reward=getattr(structured_signal, "risk_reward", None),
            replay_summary=str(getattr(replay_risk, "summary", "") or ""),
            lot_risk_summary=str(getattr(lot_risk, "summary", "") or ""),
            lot_stop_loss_amount=getattr(lot_risk, "cost_adjusted_loss_per_lot", None),
            volatility_bucket=str(
                getattr(structured_signal, "volatility_bucket", "")
                or getattr(lot_risk, "volatility_bucket", "")
                or ""
            ),
            oi_summary=str(
                getattr(structured_signal, "oi_summary", "")
                or getattr(oi_enrichment, "summary", "")
                or ""
            ),
            expert_notes=self._expert_notes(
                structured_signal=structured_signal,
                replay_risk=replay_risk,
                lot_risk=lot_risk,
            ),
            request_id=str(request_id or ""),
        )

    def _resolve_used_layers(self, response: Any) -> SignalLayerUsage:
        ml_enabled = bool(getattr(self.settings, "signal_ml_enabled_in_request_path", True))
        agentic_ml_enabled = bool(getattr(self.settings, "agentic_ml_scorer_enabled", False))
        rl_mode = str(getattr(self.settings, "signal_rl_mode", "warm") or "warm").strip().lower()
        rl_signals = tuple(
            str(getattr(sig, "name", "") or "").lower()
            for sig in getattr(response, "signals", ()) or ()
        )
        ml_status = "skipped"
        if ml_enabled and agentic_ml_enabled and getattr(response, "decision", None) is not None:
            ml_status = "used"
        elif ml_enabled:
            ml_status = "not_ready"
        rl_status = "skipped"
        if rl_mode == "off":
            rl_status = "skipped"
        elif any("rl" in name for name in rl_signals):
            rl_status = "used"
        elif bool(getattr(self.settings, "agentic_enabled", False)):
            rl_status = "not_ready" if rl_mode == "required" else "skipped"
        else:
            rl_status = "not_ready"
        return SignalLayerUsage(
            deterministic="used",
            ml=ml_status,
            rl=rl_status,
        )

    def _cache_key(
        self,
        request: TelegramRequest,
        *,
        reply_style: str,
        expert: bool,
    ) -> tuple[str, ...]:
        return (
            request.kind.value,
            str(request.symbol or "").upper(),
            str(request.timeframe or "5m").lower(),
            str(max(1, int(request.days or 30))),
            str(request.raw_text or "").strip().lower(),
            reply_style,
            "expert" if expert else "default",
        )

    def _cache_get(self, key: tuple[str, ...]) -> _CacheEntry | None:
        if not bool(getattr(self.settings, "signal_cache_enabled", True)):
            return None
        ttl = max(1, int(getattr(self.settings, "signal_result_ttl_seconds", 20) or 20))
        now = time.time()
        with self._cache_lock:
            entry = self._response_cache.get(key)
            if entry is None:
                return None
            if now - entry.created_at > ttl:
                self._response_cache.pop(key, None)
                return None
            return entry

    def _cache_put(self, key: tuple[str, ...], entry: _CacheEntry) -> None:
        if not bool(getattr(self.settings, "signal_cache_enabled", True)):
            return
        with self._cache_lock:
            self._response_cache[key] = entry

    def _handle_image_interaction(
        self,
        text: str,
        *,
        attachment: MediaAttachment,
        request_id: str,
        conversation_state: TelegramConversationState | None,
        is_admin: bool,
        reply_style: str | None,
    ) -> InteractionResult:
        if not bool(getattr(self.settings, "signal_image_input_enabled", False)):
            req = parse_telegram_request("/help")
            route = route_telegram_request(req)
            return InteractionResult(
                reply=(
                    "Image analysis is disabled right now. "
                    "Send a text request or enable chart intake."
                ),
                request=req,
                route=route,
                request_id=request_id,
                error_code="image_disabled",
                modality="image",
                attachment_kind=attachment.kind,
            )
        image_key = (
            attachment.content_hash,
            str(text or "").strip().lower(),
        )
        cached = self._value_cache_get(
            self._image_cache,
            image_key,
            ttl_seconds=max(
                1,
                int(getattr(self.settings, "signal_image_cache_ttl_seconds", 300) or 300),
            ),
        )
        analysis = cached
        if analysis is None:
            image_bytes = Path(attachment.local_path).read_bytes()
            analysis = self._chart_analyzer().analyze(image_bytes=image_bytes, caption=text)
            self._value_cache_put(self._image_cache, image_key, analysis)
        analysis_symbol = str(getattr(analysis, "symbol", "") or "").strip()
        if not getattr(analysis, "ok", False) or not analysis_symbol:
            req = parse_telegram_request(text or "/help")
            route = route_telegram_request(req)
            reply = (
                "I could not safely identify a chart instrument from that image. "
                "Add a caption like 'Crompton Futures' or send /search CROMPTON first."
            )
            return InteractionResult(
                reply=reply,
                request=req,
                route=route,
                request_id=request_id,
                tool="chart_image_analyzer",
                tool_ok=False,
                error_code=(
                    analysis.error.code.value
                    if getattr(analysis, "error", None) is not None
                    else "image_clarify"
                ),
                modality="image_with_caption" if text.strip() else "image",
                attachment_kind=attachment.kind,
            )
        timeframe = str(getattr(analysis, "timeframe", "") or "").strip() or (
            conversation_state.last_timeframe if conversation_state is not None else "5m"
        )
        days = max(1, int(conversation_state.last_days or 30)) if conversation_state else 30
        req = TelegramRequest(
            kind=TelegramRequestKind.ANALYZE,
            raw_text=text or f"/analyze {analysis.symbol}",
            symbol=str(analysis.symbol),
            query=str(analysis.symbol),
            timeframe=timeframe,
            days=days,
        )
        route = route_telegram_request(req)
        image_source = (
            AdapterSource.OPENAI.value
            if self._chart_analyzer().available()
            else AdapterSource.COMMAND.value
        )
        return self._execute_signal_analysis(
            req,
            route=route,
            source=image_source,
            source_confidence=float(getattr(analysis, "confidence", 0.0) or 0.0),
            source_rationale=str(getattr(analysis, "summary", "") or "chart image analysis"),
            request_id=request_id,
            is_admin=is_admin,
            reply_style=self._effective_reply_style(reply_style),
            modality="image_with_caption" if text.strip() else "image",
            attachment=attachment,
        )

    def _execute_signal_analysis(
        self,
        request: TelegramRequest,
        *,
        route: ConversationRoute,
        source: str,
        source_confidence: float,
        source_rationale: str,
        request_id: str,
        is_admin: bool,
        reply_style: str,
        modality: str,
        attachment: MediaAttachment | None = None,
        risk_intent: bool = False,
    ) -> InteractionResult:
        cache_key = self._cache_key(
            request,
            reply_style=reply_style,
            expert=is_admin,
        ) + (modality,)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return InteractionResult(
                reply=cached.reply,
                request=request,
                route=route,
                source=source,
                source_confidence=source_confidence,
                source_rationale=source_rationale,
                tool="analyze_instrument",
                tool_ok=True,
                resolved_symbol=(
                    cached.signal_response.resolved_symbol
                    if cached.signal_response is not None
                    else None
                ),
                decision_action=(
                    cached.signal_response.signal_verdict
                    if cached.signal_response is not None
                    else None
                ),
                request_id=request_id,
                signal_response=cached.signal_response,
                modality=modality,
                attachment_kind=attachment.kind if attachment is not None else None,
            )
        response = self._tools().analyze_instrument(
            request.symbol,
            timeframe=request.timeframe,
            days=request.days,
        )
        error_code = response.error.code.value if response.error is not None else None
        resolved_symbol = (
            response.instrument.resolved_symbol if response.instrument is not None else None
        )
        decision_action = response.decision.action if response.decision is not None else None
        signal_response = self._build_signal_response(
            response,
            request_id=request_id,
            modality=modality,
        )
        if signal_response is not None and response.ok:
            signal_response = self._apply_user_budget_context(
                signal_response,
                raw_text=request.raw_text,
            )
            if risk_intent:
                signal_response = self._apply_risk_intent_enrichment(signal_response)
            forecast_key = (
                signal_response.resolved_symbol,
                signal_response.timeframe,
                str(signal_response.lookback_days),
                signal_response.data_freshness,
                str(request.raw_text or "").lower(),
            )
            forecasts = self._value_cache_get(
                self._forecast_cache,
                forecast_key,
                ttl_seconds=max(
                    1,
                    int(getattr(self.settings, "signal_forecast_cache_ttl_seconds", 60) or 60),
                ),
            )
            if forecasts is None:
                forecasts = run_forecast_lane(
                    raw_text=request.raw_text,
                    response=response,
                    signal_response=signal_response,
                    settings=self.settings,
                    engine_factory=self.engine_factory,
                )
                self._value_cache_put(self._forecast_cache, forecast_key, forecasts)
            signal_response = merge_forecasts_into_signal(
                signal_response,
                tuple(forecasts or ()),
            )
        reply = format_instrument_analysis(
            response,
            signal_response=signal_response,
            reply_style=reply_style,
            expert=is_admin,
        )
        if response.ok:
            self._cache_put(
                cache_key,
                _CacheEntry(
                    reply=reply,
                    signal_response=signal_response,
                    created_at=time.time(),
                ),
            )
        return InteractionResult(
            reply=reply,
            request=request,
            route=route,
            source=source,
            source_confidence=source_confidence,
            source_rationale=source_rationale,
            tool="analyze_instrument",
            tool_ok=response.ok,
            error_code=error_code,
            resolved_symbol=resolved_symbol,
            decision_action=decision_action,
            request_id=request_id,
            signal_response=signal_response,
            modality=modality,
            attachment_kind=attachment.kind if attachment is not None else None,
        )

    def _expert_notes(
        self,
        *,
        structured_signal: Any,
        replay_risk: Any,
        lot_risk: Any,
    ) -> tuple[str, ...]:
        notes: list[str] = []
        if structured_signal is not None:
            rr = getattr(structured_signal, "risk_reward", None)
            if rr is not None:
                notes.append(f"reward_to_risk={float(rr):.2f}")
        if replay_risk is not None and getattr(replay_risk, "ok", False):
            notes.append(str(getattr(replay_risk, "summary", "") or ""))
        if lot_risk is not None and getattr(lot_risk, "available", False):
            notes.append(str(getattr(lot_risk, "summary", "") or ""))
        return tuple(note for note in notes if note)

    def _apply_user_budget_context(
        self,
        signal_response: SignalResponse,
        *,
        raw_text: str,
    ) -> SignalResponse:
        text = str(raw_text or "").lower()
        budget = _extract_budget_in_inr(text)
        if budget is None or signal_response.lot_stop_loss_amount is None:
            return signal_response
        fits = float(signal_response.lot_stop_loss_amount) <= float(budget)
        note = (
            f"user_budget={budget:,.0f} INR fits one-lot stop risk"
            if fits
            else f"user_budget={budget:,.0f} INR is below one-lot stop risk"
        )
        risk_notes = tuple(signal_response.risk_notes or ()) + (note,)
        return replace(signal_response, risk_notes=risk_notes[:5])

    def _apply_risk_intent_enrichment(self, signal_response: SignalResponse) -> SignalResponse:
        reasons = tuple(signal_response.reasons or ())
        risk_notes = tuple(signal_response.risk_notes or ())
        expert_notes = tuple(signal_response.expert_notes or ())
        if signal_response.replay_summary and signal_response.replay_summary not in expert_notes:
            expert_notes = expert_notes + (signal_response.replay_summary,)
        if (
            signal_response.lot_risk_summary
            and signal_response.lot_risk_summary not in risk_notes
        ):
            risk_notes = risk_notes + (signal_response.lot_risk_summary,)
        if signal_response.stop_loss is not None and signal_response.target_price is not None:
            guidance = (
                f"stop={float(signal_response.stop_loss):.2f} "
                f"target={float(signal_response.target_price):.2f}"
            )
            if guidance not in reasons:
                reasons = reasons + (guidance,)
        return replace(
            signal_response,
            reasons=reasons[:5],
            risk_notes=risk_notes[:5],
            expert_notes=expert_notes[:4],
        )

    def _chart_analyzer(self) -> ChartImageAnalyzer:
        if self.chart_analyzer is not None:
            return self.chart_analyzer
        return ChartImageAnalyzer(self.settings)

    def _value_cache_get(
        self,
        cache: dict[tuple[str, ...], _ValueCacheEntry],
        key: tuple[str, ...],
        *,
        ttl_seconds: int,
    ) -> Any:
        now = time.time()
        with self._cache_lock:
            entry = cache.get(key)
            if entry is None:
                return None
            if now - entry.created_at > max(1, ttl_seconds):
                cache.pop(key, None)
                return None
            return entry.value

    def _value_cache_put(
        self,
        cache: dict[tuple[str, ...], _ValueCacheEntry],
        key: tuple[str, ...],
        value: Any,
    ) -> None:
        with self._cache_lock:
            cache[key] = _ValueCacheEntry(value=value, created_at=time.time())


def _extract_budget_in_inr(text: str) -> float | None:
    patterns = (
        r"(?:risk|budget|capital|max loss)\s*(?:of|is|=|:)?\s*(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*)",
        r"(?:rs\.?|inr|₹)\s*([0-9][0-9,]*)\s*(?:risk|budget|capital|max loss)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return None
