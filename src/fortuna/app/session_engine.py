"""Orchestrates symbol load, parallel backtests, and live feed for the dashboard."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from fortuna.app.config import AppConfig
from fortuna.app.live_session import LiveSessionBridge
from fortuna.app.live_signals import (
    LiveSignal,
    compute_live_signals,
    compute_live_signals_with_rl,
)
from fortuna.app.parallel_runner import BatchRunResult, ParallelStrategyRunner
from fortuna.app.strategy_paths import list_strategy_paths
from fortuna.config.settings import Settings
from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.data.instruments import InstrumentRegistry
from fortuna.data.manager import MarketDataManager
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_index_to_naive_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Force the DatetimeIndex to tz-naive IST, regardless of input dtype.

    Cache writes are tz-naive IST; live bars arrive tz-aware Asia/Kolkata.
    Mixing them in a single index breaks ``sort_index`` with
    "Cannot compare tz-naive and tz-aware timestamps".
    """
    if df is None or df.empty:
        return df
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            df = df.copy()
            df.index = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        return df
    # Mixed/object dtype index — coerce element-wise.
    coerced = pd.DatetimeIndex(
        [
            (
                pd.Timestamp(t).tz_convert("Asia/Kolkata").tz_localize(None)
                if pd.Timestamp(t).tzinfo is not None
                else pd.Timestamp(t)
            )
            for t in idx
        ],
        name=idx.name,
    )
    df = df.copy()
    df.index = coerced
    return df


@dataclass
class SessionState:
    symbol: str = ""
    timeframe: str = "5m"
    days: int = 30
    ohlcv: Optional[pd.DataFrame] = None
    batch: Optional[BatchRunResult] = None
    load_error: Optional[str] = None
    live_signals: dict[str, LiveSignal] = field(default_factory=dict)
    last_bar_time: Optional[pd.Timestamp] = None
    last_signal_refresh: Optional[pd.Timestamp] = None
    rl_run_id: Optional[str] = None
    rl_available: bool = False


class FortunaSessionEngine:
    """Thread-safe session for one active symbol (dashboard backend)."""

    def __init__(self, settings: Settings, app_config: Optional[AppConfig] = None) -> None:
        self.settings = settings.model_copy(update={"data_source": "smartapi"})
        self.app_config = app_config or AppConfig.from_settings(settings)
        self._mdm = MarketDataManager(self.settings, data_source="smartapi")
        self._registry = InstrumentRegistry()
        self._runner = ParallelStrategyRunner(self.settings, self.app_config)
        self._lock = threading.RLock()
        self._state = SessionState()
        self._live: Optional[LiveSessionBridge] = None
        self._rl_generator = self._load_rl_generator()
        self._regime_router = self._load_regime_router()

    def _load_regime_router(self):
        """Load the trained ``RegimeDetector`` for regime-aware strategy routing.

        Falls back to a stub router (``is_available=False``) when no classifier
        is present, which keeps the dashboard on the all-strategies code path.
        """
        try:
            from fortuna.strategies.regime_router import RegimeRouter
        except Exception as exc:  # noqa: BLE001
            logger.debug("regime router imports unavailable: %s", exc)
            return None
        try:
            return RegimeRouter.from_disk("models/regime/classifier.joblib")
        except Exception as exc:  # noqa: BLE001
            logger.debug("RegimeRouter init failed: %s", exc)
            return None

    def reload_regime_router(self) -> bool:
        """Pick up a freshly-trained RegimeDetector without restarting."""
        self._regime_router = self._load_regime_router()
        return bool(self._regime_router and getattr(self._regime_router, "is_available", False))

    @property
    def regime_router(self):
        return self._regime_router

    def _load_rl_generator(self):
        """Load the validated RL policy (if any) at startup.

        Returns ``None`` silently if no checkpoint is available — the
        deterministic signal path continues to drive the dashboard.
        """
        try:
            from fortuna.rl.inference import (
                RLSignalGenerator,
                resolve_live_checkpoint_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("RL inference imports unavailable: %s", exc)
            return None

        try:
            live_dir = resolve_live_checkpoint_dir(Path("models"))
            if live_dir is None:
                return RLSignalGenerator(None)  # explicit unavailable instance
            return RLSignalGenerator(live_dir)
        except Exception as exc:  # noqa: BLE001
            logger.debug("RLSignalGenerator failed to initialize: %s", exc)
            return None

    def reload_rl_generator(self) -> bool:
        """Pick up a freshly promoted ``models/live/`` policy without restart.

        Returns ``True`` if a policy is loaded after the reload, ``False`` otherwise.
        """
        self._rl_generator = self._load_rl_generator()
        return bool(self._rl_generator and getattr(self._rl_generator, "is_available", False))

    @property
    def rl_generator(self):
        return self._rl_generator

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    def symbol_catalog(self) -> SymbolCatalog:
        cat = SymbolCatalog(self._registry)
        cat.ensure_loaded()
        return cat

    def load_symbol(
        self,
        symbol: str,
        *,
        timeframe: Optional[str] = None,
        days: Optional[int] = None,
        force_refresh: bool = False,
    ) -> SessionState:
        tf = timeframe or self.settings.default_timeframe
        d = days if days is not None else self.settings.default_days
        sym = symbol.upper().strip()
        # Only append the cash-equity ``.NS`` suffix when the input is a bare
        # base symbol. Futures (``CROMPTON.FUT``) and explicit-exchange
        # tradingsymbols already carry a dot and must be left untouched.
        if "." not in sym and not sym.endswith("-EQ"):
            sym = f"{sym}.NS"

        with self._lock:
            self.stop_live()
            self._state = SessionState(symbol=sym, timeframe=tf, days=d)

        try:
            self._registry.resolve(sym)
            if force_refresh:
                ohlcv = self._mdm.get_ohlcv(sym, tf, days=d, force_refresh=True)
            else:
                ohlcv = self._mdm.get_ohlcv(sym, tf, days=d)
            paths = list_strategy_paths(self.settings, self.app_config)
            batch = self._runner.run_parallel(ohlcv, paths, symbol=sym, timeframe=tf)
            signals = compute_live_signals_with_rl(
                paths, ohlcv,
                rl_generator=self._rl_generator,
                regime_router=self._regime_router,
                symbol=sym,
            )
            with self._lock:
                self._state.ohlcv = ohlcv
                self._state.batch = batch
                self._state.live_signals = signals
                self._state.last_bar_time = pd.Timestamp(ohlcv.index[-1]) if len(ohlcv) else None
                self._state.last_signal_refresh = pd.Timestamp.utcnow()
                self._state.rl_available = bool(
                    self._rl_generator and getattr(self._rl_generator, "is_available", False)
                )
                meta = getattr(self._rl_generator, "metadata", None) if self._rl_generator else None
                self._state.rl_run_id = getattr(meta, "run_id", None) if meta else None
            logger.info("Loaded %s: %d bars, %d strategies", sym, len(ohlcv), len(batch.results))
        except Exception as e:
            logger.exception("Load failed for %s", sym)
            with self._lock:
                self._state.load_error = str(e)
        return self.state

    def get_ohlcv_for_chart(self) -> pd.DataFrame:
        with self._lock:
            if self._state.ohlcv is None:
                return pd.DataFrame()
            ohlcv = self._state.ohlcv
            forming = self._live.forming_bar_df() if self._live else None
        if forming is not None and not forming.empty:
            # Defensive: align both indices to tz-naive IST so concat/sort
            # never sees mixed tz-aware/tz-naive timestamps.
            ohlcv = _normalize_index_to_naive_ist(ohlcv)
            forming = _normalize_index_to_naive_ist(forming)
            merged = pd.concat([ohlcv, forming]).sort_index()
            return merged[~merged.index.duplicated(keep="last")]
        return ohlcv

    def start_live(self) -> None:
        with self._lock:
            if self._state.ohlcv is None or not self._state.symbol:
                raise ValueError("Load a symbol before starting live feed")
            sym, tf = self._state.symbol, self._state.timeframe
            ohlcv = self._state.ohlcv.copy()

        def on_bar_closed() -> None:
            self._refresh_strategies_after_bar()

        self._live = LiveSessionBridge(
            self._mdm,
            sym,
            tf,
            ohlcv,
            on_bar_closed=on_bar_closed,
        )
        self._live.start()

    def stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def is_live(self) -> bool:
        return self._live is not None and self._live.is_running

    def _refresh_strategies_after_bar(self) -> None:
        """Triggered when a new closed bar arrives — full backtest + signal refresh."""
        with self._lock:
            if self._live is None:
                return
            ohlcv = self._live.ohlcv
            sym = self._state.symbol
            tf = self._state.timeframe
        if ohlcv is None or ohlcv.empty:
            return
        paths = list_strategy_paths(self.settings, self.app_config)
        batch = self._runner.run_parallel(ohlcv, paths, symbol=sym, timeframe=tf)
        signals = compute_live_signals_with_rl(
            paths, ohlcv,
            rl_generator=self._rl_generator,
            regime_router=self._regime_router,
            symbol=sym,
        )
        with self._lock:
            self._state.ohlcv = ohlcv
            self._state.batch = batch
            self._state.live_signals = signals
            self._state.last_bar_time = pd.Timestamp(ohlcv.index[-1]) if len(ohlcv) else None
            self._state.last_signal_refresh = pd.Timestamp.utcnow()

    def refresh_live_signals(self) -> dict[str, LiveSignal]:
        """Recompute live signals on the latest OHLCV (incl. forming bar) without re-backtesting.

        Cheap enough to call per Streamlit refresh / per WebSocket tick.
        """
        ohlcv = self.get_ohlcv_for_chart()
        if ohlcv is None or ohlcv.empty:
            return {}
        paths = list_strategy_paths(self.settings, self.app_config)
        with self._lock:
            sym = self._state.symbol
        signals = compute_live_signals_with_rl(
            paths, ohlcv,
            rl_generator=self._rl_generator,
            regime_router=self._regime_router,
            symbol=sym,
        )
        with self._lock:
            self._state.live_signals = signals
            self._state.last_bar_time = pd.Timestamp(ohlcv.index[-1])
            self._state.last_signal_refresh = pd.Timestamp.utcnow()
        return signals

    def live_signals(self) -> dict[str, LiveSignal]:
        with self._lock:
            return dict(self._state.live_signals)

    def poll_live(self) -> int:
        """Apply queued live events; return count processed."""
        if self._live is None:
            return 0
        n = self._live.drain_events()
        with self._lock:
            if self._live is not None:
                self._state.ohlcv = self._live.ohlcv
        if n > 0:
            self.refresh_live_signals()
        return n
