"""Typed advisory tool surface for interactive and agent callers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fortuna.agentic.contracts import (
    AdvisoryError,
    AdvisoryErrorCode,
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSearchHitSummary,
    InstrumentSearchResponse,
    LearningSummaryStatus,
    ModelHealthResponse,
)
from fortuna.app.advisory_service import analyze_instrument as build_instrument_analysis
from fortuna.app.model_status import build_learning_summary, build_model_status
from fortuna.app.session_engine import FortunaSessionEngine
from fortuna.app.symbol_catalog import SymbolCatalog
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
        summaries = tuple(
            InstrumentSearchHitSummary.from_hit(hit) for hit in merged[: max(1, int(limit))]
        )
        return InstrumentSearchResponse(ok=True, query=text, hits=summaries)

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
