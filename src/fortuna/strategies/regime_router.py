"""Regime-aware router that selects which deterministic strategies to apply.

Pairs ``RegimeDetector`` (Phase 2.2) with the existing deterministic JSON
strategies (Phase 1) to deliver "right strategy, right regime" execution
without RL involvement.

Routing table (defaults, override via constructor):

    TRENDING -> strategies that lean on momentum / breakout follow-through:
                orb (15-min opening range), donchian_breakout, ema_pullback,
                macd_trend, ivorb.

    RANGING  -> mean-reversion / fade strategies:
                rsi_scalp, bollinger_breakout (fade extremes), vwap_reclaim,
                lsvwap.

    VOLATILE -> conservative / wide-stop strategies; many traders sit out:
                mmts (only this — wide stops, single high-conviction signal).
                Empty allow-list means "do not trade this regime".

The router returns the *subset of strategy file paths* that should be
evaluated for the current bar. ``compute_live_signals`` is then called only
on that subset, with the regime label attached to every output ``LiveSignal``.

Routing is best-effort — when ``RegimeDetector`` is unavailable or classifies
nothing, the router falls back to "all strategies allowed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.rl.inference.regime_detector import (
    REGIMES,
    RegimeDetector,
    load_detector_if_available,
)
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


# Default routing table — strategy *file-stem* (no .json suffix) per regime.
# A name only needs to be present; missing strategies are silently skipped.
DEFAULT_ROUTING: dict[str, tuple[str, ...]] = {
    "TRENDING": (
        "orb_15m",
        "donchian_breakout_20",
        "ema_pullback_5m",
        "macd_trend_5m",
        "ivorb_5m",
        "ivorb",
    ),
    "RANGING": (
        "rsi_scalp_5m",
        "bollinger_breakout_5m",
        "vwap_reclaim",
        "lsvwap_5m",
        "lsvwap",
    ),
    "VOLATILE": (
        "mmts",
    ),
}


@dataclass
class RegimeRouting:
    """One bar's worth of routing decision — what to evaluate + why."""

    regime: Optional[str]
    """``"TRENDING"`` / ``"RANGING"`` / ``"VOLATILE"`` / ``None`` (unknown)."""
    allowed: list[Path] = field(default_factory=list)
    """The subset of ``strategy_paths`` cleared to run on this bar."""
    skipped: list[Path] = field(default_factory=list)
    """The complement — paths that the router muted for this regime."""
    confidence: float = 0.0
    """Routing confidence: max probability across regime classes, 0..1."""

    @property
    def has_decision(self) -> bool:
        return self.regime is not None


class RegimeRouter:
    """Selects which deterministic strategies to run based on session regime.

    Designed to be cheap: ``RegimeDetector`` runs in ~1 ms per session, and
    we cache the classification per (symbol, session_date) so we only pay
    that cost once per day per symbol.
    """

    def __init__(
        self,
        detector: Optional[RegimeDetector] = None,
        routing: Optional[dict[str, tuple[str, ...]]] = None,
        *,
        fallback_allow_all: bool = True,
    ) -> None:
        self.detector = detector
        self.routing = routing or DEFAULT_ROUTING
        self.fallback_allow_all = bool(fallback_allow_all)
        # Per-day cache: key=(symbol, session_date_iso) -> regime label.
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}

    @classmethod
    def from_disk(
        cls,
        detector_path: Path | str = "models/regime/classifier.joblib",
        routing: Optional[dict[str, tuple[str, ...]]] = None,
    ) -> "RegimeRouter":
        det = load_detector_if_available(detector_path)
        return cls(detector=det, routing=routing)

    @property
    def is_available(self) -> bool:
        return self.detector is not None

    def classify(
        self, ohlcv_session: pd.DataFrame, *, symbol: str = ""
    ) -> tuple[Optional[str], float]:
        """Return ``(regime, confidence)`` for the most recent session.

        Caches the result per (symbol, date) to skip repeat classification
        inside the same trading day.
        """
        if self.detector is None or ohlcv_session is None or ohlcv_session.empty:
            return None, 0.0
        last_idx = pd.Timestamp(ohlcv_session.index[-1])
        cache_key = (symbol, last_idx.normalize().isoformat())
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            label = self.detector.predict(ohlcv_session)
            if label is None:
                return None, 0.0
            proba = self.detector.predict_proba(ohlcv_session) or {}
            conf = float(proba.get(label, 0.0)) if proba else 0.0
        except Exception as exc:  # noqa: BLE001
            logger.debug("regime classify failed: %s", exc)
            return None, 0.0

        self._cache[cache_key] = (label, conf)
        return label, conf

    def route(
        self,
        strategy_paths: list[Path],
        ohlcv_session: pd.DataFrame,
        *,
        symbol: str = "",
    ) -> RegimeRouting:
        """Pick the subset of ``strategy_paths`` to evaluate for the current bar."""
        if not strategy_paths:
            return RegimeRouting(regime=None, allowed=[], skipped=[])

        regime, conf = self.classify(ohlcv_session, symbol=symbol)
        if regime is None or regime not in self.routing:
            if self.fallback_allow_all:
                return RegimeRouting(regime=regime, allowed=list(strategy_paths),
                                     skipped=[], confidence=conf)
            return RegimeRouting(regime=regime, allowed=[],
                                 skipped=list(strategy_paths), confidence=conf)

        allowed_stems = set(self.routing[regime])
        allowed: list[Path] = []
        skipped: list[Path] = []
        for p in strategy_paths:
            if p.stem in allowed_stems:
                allowed.append(p)
            else:
                skipped.append(p)

        # If routing matched none of our paths, fall back to all-on so the
        # dashboard isn't empty — never let a configuration gap break trading.
        if not allowed and self.fallback_allow_all:
            logger.warning(
                "[regime_router] regime=%s matched none of %d strategy paths; "
                "falling back to all-allow", regime, len(strategy_paths),
            )
            allowed = list(strategy_paths)
            skipped = []

        return RegimeRouting(
            regime=regime,
            allowed=allowed,
            skipped=skipped,
            confidence=conf,
        )


__all__ = [
    "DEFAULT_ROUTING",
    "REGIMES",
    "RegimeRouter",
    "RegimeRouting",
]
