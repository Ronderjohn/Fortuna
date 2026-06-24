"""Typed advisory tool surface for interactive and agent callers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from fortuna.agentic.contracts import (
    AdvisoryError,
    AdvisoryErrorCode,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSearchHitSummary,
    InstrumentSearchResponse,
    LearningSummaryStatus,
    MarketUniverseResponse,
    ModelHealthResponse,
    MultiAgentWorkflowResponse,
    PortfolioAllocationResponse,
    ShortlistAnalysisResponse,
    ShortlistBriefingResponse,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
)
from fortuna.app.advisory_service import analyze_instrument as build_instrument_analysis
from fortuna.app.market_universe import build_market_universe
from fortuna.app.model_status import build_learning_summary, build_model_status
from fortuna.app.multi_agent_team import build_multi_agent_workflow
from fortuna.app.portfolio_allocator import build_portfolio_allocation
from fortuna.app.session_engine import FortunaSessionEngine
from fortuna.app.shortlist_analysis import analyze_market_shortlist
from fortuna.app.shortlist_briefing import build_shortlist_briefing
from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.app.training_candidates import build_training_candidates
from fortuna.app.training_research import build_training_research_plan
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRegistry


@dataclass
class FortunaAdvisoryTools:
    settings: Settings
    engine_factory: Optional[Callable[[], Any]] = None
    registry: Optional[InstrumentRegistry] = None
    catalog: Optional[SymbolCatalog] = None
    _registry: Optional[InstrumentRegistry] = field(default=None, init=False, repr=False)
    _catalog: Optional[SymbolCatalog] = field(default=None, init=False, repr=False)

    def search_instruments(self, query: str, *, limit: int = 10) -> InstrumentSearchResponse:
        text = (query or "").strip()
        if not text:
            return InstrumentSearchResponse(
                ok=False,
                query=text,
                error=AdvisoryError(
                    code=AdvisoryErrorCode.INVALID_REQUEST,
                    message="Usage: /search RELIANCE",
                ),
            )

        catalog = self._symbol_catalog()
        hits = catalog.search(text, limit=8)
        option_hits = []
        registry = self._instrument_registry()
        if hasattr(registry, "search_options"):
            option_hits = registry.search_options(text, limit=4)

        merged = [*hits, *option_hits]
        normalized = [self._normalize_search_hit(hit, registry=registry) for hit in merged]
        summaries = tuple(
            InstrumentSearchHitSummary.from_hit(hit)
            for hit in normalized[: max(1, int(limit))]
        )
        return InstrumentSearchResponse(ok=True, query=text, hits=summaries)

    def _normalize_search_hit(self, hit: Any, *, registry: InstrumentRegistry) -> Any:
        symbol = str(getattr(hit, "symbol", "") or "").strip()
        segment = str(getattr(hit, "segment", "") or "").upper()
        if not symbol:
            return hit
        if segment not in {"OPTIONS", "FUTURES"}:
            return hit
        needs_upgrade = (
            symbol.endswith(".OPT")
            or symbol.endswith(".FUT")
            or ".OPT." not in symbol and segment == "OPTIONS"
        )
        if not needs_upgrade:
            return hit
        tradingsymbol = str(getattr(hit, "tradingsymbol", "") or "").strip()
        if not tradingsymbol:
            return hit
        try:
            ref = registry.resolve(tradingsymbol)
        except Exception:  # noqa: BLE001
            return hit
        canonical = symbol
        if ref.is_option and ref.option_type and ref.strike is not None and ref.expiry is not None:
            strike = f"{ref.strike:g}"
            canonical = (
                f"{ref.symbol}.{ref.option_type}.{strike}."
                f"{ref.expiry.strftime('%d%b%Y').upper()}"
            )
        elif ref.is_future and ref.expiry is not None:
            base_symbol = ref.symbol
            canonical = (
                base_symbol
                if symbol.endswith(".FUT")
                else f"{base_symbol}.{ref.expiry.strftime('%d%b%Y').upper()}"
            )
        if canonical == symbol:
            return hit
        return replace(hit, symbol=canonical)

    def analyze_instrument(
        self,
        symbol: str,
        *,
        timeframe: str = "5m",
        days: int = 30,
        force_refresh: bool = False,
    ) -> InstrumentAnalysisResponse:
        request = InstrumentAnalysisRequest(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            force_refresh=force_refresh,
        )
        return build_instrument_analysis(
            request=request,
            settings=self.settings,
            engine_factory=self._engine_factory,
            registry=self._instrument_registry(),
        )

    def get_model_health(self) -> ModelHealthResponse:
        return build_model_status(self._engine())

    def get_recent_learning_summary(self, symbol: str | None = None) -> LearningSummaryStatus:
        engine = self._engine()
        store = getattr(engine, "_agentic_learning_store", None)
        return build_learning_summary(store, symbol=symbol)

    def get_market_universe(
        self,
        *,
        limit: int | None = None,
        timeframe: str = "1d",
        days: int = 30,
        source: str = "auto",
    ) -> MarketUniverseResponse:
        return build_market_universe(
            settings=self.settings,
            registry=self._instrument_registry(),
            limit=limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def analyze_market_shortlist(
        self,
        *,
        universe_limit: int = 10,
        analysis_limit: int = 5,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
    ) -> ShortlistAnalysisResponse:
        return analyze_market_shortlist(
            settings=self.settings,
            engine_factory=self._engine_factory,
            registry=self._instrument_registry(),
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def get_training_candidates(
        self,
        *,
        universe_limit: int = 15,
        analysis_limit: int = 8,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
    ) -> TrainingCandidateResponse:
        return build_training_candidates(
            settings=self.settings,
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def get_training_research_plan(
        self,
        *,
        universe_limit: int = 15,
        analysis_limit: int = 8,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
        ml_top_n: int = 5,
        rl_top_n: int = 3,
        selection_policy: str = "diversified",
        refresh_data: bool = False,
        refresh_target: str = "all",
        refresh_timeframe: str | None = None,
        refresh_days: int | None = None,
        force_refresh: bool = False,
        ) -> TrainingResearchPlanResponse:
        return build_training_research_plan(
            settings=self.settings,
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
            ml_top_n=ml_top_n,
            rl_top_n=rl_top_n,
            selection_policy=selection_policy,
            refresh_data=refresh_data,
            refresh_target=refresh_target,
            refresh_timeframe=refresh_timeframe,
            refresh_days=refresh_days,
            force_refresh=force_refresh,
        )

    def get_multi_agent_workflow(
        self,
        *,
        universe_limit: int = 15,
        analysis_limit: int = 8,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
        max_positions: int = 3,
        max_per_exposure: int = 1,
        max_same_side: int = 2,
        ml_top_n: int = 5,
        rl_top_n: int = 3,
        selection_policy: str = "diversified",
        refresh_research_data: bool = False,
        research_refresh_target: str = "all",
        research_refresh_timeframe: str | None = None,
        research_refresh_days: int | None = None,
        force_refresh: bool = False,
    ) -> MultiAgentWorkflowResponse:
        return build_multi_agent_workflow(
            settings=self.settings,
            engine_factory=self._engine_factory,
            registry=self._instrument_registry(),
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
            max_positions=max_positions,
            max_per_exposure=max_per_exposure,
            max_same_side=max_same_side,
            ml_top_n=ml_top_n,
            rl_top_n=rl_top_n,
            selection_policy=selection_policy,
            refresh_research_data=refresh_research_data,
            research_refresh_target=research_refresh_target,
            research_refresh_timeframe=research_refresh_timeframe,
            research_refresh_days=research_refresh_days,
            force_refresh=force_refresh,
        )

    def get_shortlist_briefing(
        self,
        *,
        universe_limit: int = 10,
        analysis_limit: int = 5,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
    ) -> ShortlistBriefingResponse:
        return build_shortlist_briefing(
            settings=self.settings,
            engine_factory=self._engine_factory,
            registry=self._instrument_registry(),
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def get_portfolio_allocation(
        self,
        *,
        universe_limit: int = 10,
        analysis_limit: int = 5,
        max_positions: int = 3,
        max_per_exposure: int = 1,
        max_same_side: int = 2,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
    ) -> PortfolioAllocationResponse:
        return build_portfolio_allocation(
            settings=self.settings,
            engine_factory=self._engine_factory,
            registry=self._instrument_registry(),
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            max_positions=max_positions,
            max_per_exposure=max_per_exposure,
            max_same_side=max_same_side,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def _engine_factory(self) -> Any:
        if self.engine_factory is not None:
            return self.engine_factory()
        return FortunaSessionEngine(self.settings)

    def _engine(self) -> Any:
        return self._engine_factory()

    def _instrument_registry(self) -> InstrumentRegistry:
        if self.registry is not None:
            self.registry.ensure_loaded()
            return self.registry
        if self._registry is None:
            self._registry = InstrumentRegistry()
        self._registry.ensure_loaded()
        return self._registry

    def _symbol_catalog(self) -> SymbolCatalog:
        if self.catalog is not None:
            self.catalog.ensure_loaded()
            return self.catalog
        if self._catalog is None:
            self._catalog = SymbolCatalog(self._instrument_registry())
        self._catalog.ensure_loaded()
        return self._catalog


def build_advisory_tools(
    settings: Settings,
    *,
    engine: Optional[FortunaSessionEngine] = None,
) -> FortunaAdvisoryTools:
    if engine is not None:
        return FortunaAdvisoryTools(
            settings=settings,
            engine_factory=lambda: engine,
            registry=engine._registry,
        )
    return FortunaAdvisoryTools(settings=settings)
