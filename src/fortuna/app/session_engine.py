"""Orchestrates symbol load, parallel backtests, and live feed for the dashboard."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from fortuna.app.config import AppConfig
from fortuna.app.live_session import LiveSessionBridge
from fortuna.app.live_signals import (
    LiveSignal,
    compute_live_signals_with_rl,
)
from fortuna.app.parallel_runner import BatchRunResult, ParallelStrategyRunner
from fortuna.app.strategy_paths import list_strategy_paths
from fortuna.app.symbol_catalog import SymbolCatalog
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRegistry
from fortuna.data.manager import MarketDataManager
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_latest_ml_artifact(base: Path) -> Optional[Path]:
    if not base.is_dir():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _enrich_ohlcv_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    from fortuna.indicators.registry import (
        compute_atr,
        compute_bollinger,
        compute_ema,
        compute_macd,
        compute_rsi,
        compute_volume_sma,
        compute_vwap,
    )

    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["atr"] = compute_atr(out)
    out["vwap"] = compute_vwap(out)
    out["volume_sma"] = compute_volume_sma(out)
    out["ema_fast"] = compute_ema(out, "close", 9)
    out["ema_slow"] = compute_ema(out, "close", 21)
    out["rsi"] = compute_rsi(out, "close", 14)
    macd = compute_macd(out, "close")
    out["macd_hist"] = macd["macd_hist"]
    bb = compute_bollinger(out, "close")
    spread = (bb["bb_upper"] - bb["bb_lower"]).replace(0, pd.NA)
    out["bb_pct_b"] = (out["close"] - bb["bb_lower"]) / spread
    return out


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
    agent_decisions: dict[str, Any] = field(default_factory=dict)


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
        self._regime_router = self._load_regime_router()
        self._agentic_orchestrator = self._build_agentic_orchestrator()
        self._agentic_store = self._build_agentic_store()
        self._agentic_learning_store = self._build_agentic_learning_store()
        self._notification_audit_store = self._build_notification_audit_store()
        self._notifier = self._build_notifier()
        self._notification_dispatcher = self._build_notification_dispatcher()
        # Execution stack (Tier 1) — lazy: built only when settings says so.
        self._execution_router = self._build_execution_router()
        # Portfolio-aware paper simulation (Phase 1) — read-only adapter
        # over SmartAPI account endpoints. Active only when both
        # execution_enabled and execution_account_sync are set.
        # Initialize attributes BEFORE _build_account_reader so it can set
        # _account_sync_error on the same instance during construction.
        self._broker_snapshot = None  # AccountSnapshot | None
        self._account_sync_error: Optional[str] = None
        self._account_reader = self._build_account_reader()
        # Cache live signals per symbol for the "What to do now" panel (held book).
        self._signals_by_symbol: dict[str, dict] = {}
        self._rl_by_symbol: dict[str, Any] = {}
        self._agent_decisions_by_symbol: dict[str, Any] = {}
        if self._account_reader is not None:
            self.sync_account_from_broker()

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

    # --------------------------------------------------------------- Tier 1
    def _build_execution_router(self):
        """Construct the paper-broker / risk / monitor stack if enabled.

        Returns ``None`` when ``settings.execution_enabled`` is False — the
        dashboard then renders read-only signals as before. Failures during
        instantiation log + degrade to None so the dashboard never goes
        down because of an execution problem.
        """
        if not (
            getattr(self.settings, "execution_enabled", False)
            or (
                getattr(self.settings, "agentic_enabled", False)
                and getattr(self.settings, "agentic_paper_learning_enabled", False)
            )
        ):
            return None
        try:
            from fortuna.execution import (
                ExecutionMonitor,
                ExecutionRouter,
                LiveAccount,
                OrderJournal,
                PaperBroker,
                PositionSizer,
                RiskGate,
                load_risk_config,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[exec] failed to import execution package: %s", exc)
            return None

        try:
            cfg = load_risk_config(self.settings.execution_config_path)
            account = LiveAccount(init_cash=cfg.init_cash)
            monitor = ExecutionMonitor()
            journal = OrderJournal()
            broker_kind = getattr(self.settings, "execution_broker", "paper").lower()
            if broker_kind == "smartapi":
                logger.warning(
                    "[exec] settings.execution_broker=smartapi is not yet "
                    "implemented; falling back to paper broker",
                )
            broker = PaperBroker(account=account, journal=journal)
            router = ExecutionRouter(
                broker=broker,
                account=account,
                risk_gate=RiskGate(cfg),
                sizer=PositionSizer(cfg),
                monitor=monitor,
                journal=journal,
                config=cfg,
            )
            logger.info("[exec] router enabled (init_cash=%s)", cfg.init_cash)
            return router
        except Exception:  # noqa: BLE001
            logger.exception("[exec] router init failed; execution disabled")
            return None

    @property
    def execution_router(self):
        return self._execution_router

    @property
    def execution_enabled(self) -> bool:
        return self._execution_router is not None

    @property
    def live_account(self):
        return self._execution_router.account if self._execution_router else None

    @property
    def execution_monitor(self):
        return self._execution_router.monitor if self._execution_router else None

    # -------------------------------------------------------------- Agentic
    def _resolve_ml_artifact_dir(self) -> Optional[Path]:
        art_base = self.settings.resolve_path(self.settings.ml_scorer_artifact_dir)
        registry_on = bool(getattr(self.settings, "model_registry_enabled", False))
        if registry_on:
            from fortuna.models import ModelKind, resolve_live

            ml_base = art_base.parent if art_base.name == "validated" else art_base
            allow_fallback = not bool(getattr(self.settings, "model_promotion_required", True))
            return resolve_live(
                ModelKind.ML_SCORER,
                models_root=self.settings.resolve_path(Path("models")),
                ml_base=ml_base,
                allow_implicit_fallback=allow_fallback,
            )
        return _resolve_latest_ml_artifact(art_base)

    def _build_agentic_orchestrator(self):
        if not getattr(self.settings, "agentic_enabled", False):
            return None
        try:
            from fortuna.agentic import AgenticOrchestrator

            ml_scorer = None
            if getattr(self.settings, "agentic_ml_scorer_enabled", False):
                try:
                    from fortuna.ml.signal_scorer import SignalScorer

                    latest = self._resolve_ml_artifact_dir()
                    if latest is not None:
                        registry_on = bool(getattr(self.settings, "model_registry_enabled", False))
                        require_promotion = registry_on and bool(
                            getattr(self.settings, "model_promotion_required", True)
                        )
                        ml_scorer = SignalScorer.load(
                            latest,
                            require_promotion=require_promotion,
                        )
                        if ml_scorer is None:
                            logger.warning(
                                "[agentic] ml scorer load rejected at %s (promotion gate)",
                                latest,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[agentic] ml scorer load failed: %s", exc)

            return AgenticOrchestrator(
                paper_tracking=bool(
                    getattr(self.settings, "agentic_paper_learning_enabled", False)
                ),
                ml_scorer=ml_scorer,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[agentic] orchestrator init failed: %s", exc)
            return None

    def _build_agentic_store(self):
        if not getattr(self.settings, "agentic_enabled", False):
            return None
        try:
            from fortuna.agentic import AgenticDecisionStore

            return AgenticDecisionStore(self.settings.resolve_path(self.settings.agentic_log_dir))
        except Exception as exc:  # noqa: BLE001
            logger.exception("[agentic] store init failed: %s", exc)
            return None

    def _build_agentic_learning_store(self):
        if not getattr(self.settings, "agentic_enabled", False):
            return None
        try:
            from fortuna.agentic import AgenticLearningStore

            return AgenticLearningStore(
                self.settings.resolve_path(self.settings.agentic_log_dir)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[agentic] learning store init failed: %s", exc)
            return None

    def _build_notification_audit_store(self):
        if not getattr(self.settings, "agentic_enabled", False):
            return None
        try:
            from fortuna.agentic.store import NotificationAuditStore

            return NotificationAuditStore(
                self.settings.resolve_path(self.settings.agentic_log_dir)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[agentic] notification audit store init failed: %s", exc)
            return None

    def _build_notification_dispatcher(self):
        if self._notifier is None:
            return None
        try:
            from fortuna.agentic.notification_dispatch import NotificationDispatcher
            from fortuna.agentic.notification_policy import (
                NotificationPolicy,
                NotificationPolicyEngine,
            )

            policy = NotificationPolicy(
                max_per_symbol_per_session=int(
                    getattr(self.settings, "telegram_max_per_symbol_per_session", 10)
                ),
                min_interval_seconds=int(
                    getattr(self.settings, "telegram_min_interval_seconds", 300)
                ),
                quiet_hours_enabled=bool(
                    getattr(self.settings, "telegram_quiet_hours_enabled", True)
                ),
            )
            engine = NotificationPolicyEngine(policy=policy)
            dedupe_keys: set[str] = set()
            if self._notification_audit_store is not None:
                dedupe_keys = self._notification_audit_store.recent_dedupe_keys(
                    limit=max(1, int(getattr(self.settings, "telegram_dedupe_memory", 500)))
                )
            dispatcher = NotificationDispatcher(
                self._notifier,
                policy_engine=engine,
                audit_store=self._notification_audit_store,
                dedupe_keys=dedupe_keys,
                dedupe_limit=int(getattr(self.settings, "telegram_dedupe_memory", 500)),
            )
            return dispatcher
        except Exception as exc:  # noqa: BLE001
            logger.exception("[agentic] notification dispatcher init failed: %s", exc)
            return None

    def _build_notifier(self):
        if not (
            getattr(self.settings, "agentic_enabled", False)
            and getattr(self.settings, "telegram_enabled", False)
        ):
            return None
        provider = str(getattr(self.settings, "telegram_provider", "telegram")).lower()
        if provider != "telegram":
            logger.warning("[agentic] unsupported telegram_provider=%s", provider)
            return None
        try:
            from fortuna.agentic import TelegramNotifier

            return TelegramNotifier(
                bot_token=getattr(self.settings, "telegram_bot_token", ""),
                chat_id=getattr(self.settings, "telegram_chat_id", ""),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[agentic] notifier init failed: %s", exc)
            return None

    # --------------------------------------------------------- Phase 1 / 2
    def _build_account_reader(self):
        """Construct the SmartAPI account reader if portfolio sync is on.

        Returns ``None`` when:
        - ``execution_enabled`` is False (no point syncing into a non-existent account), or
        - ``execution_account_sync`` is False, or
        - SmartAPI credentials are missing, or
        - the SmartAPISession cannot be built.

        All failures degrade silently (dashboard still works) and the
        reason is captured in ``self._account_sync_error`` so the UI can
        explain it.
        """
        if not getattr(self.settings, "execution_enabled", False):
            return None
        if not getattr(self.settings, "execution_account_sync", False):
            return None
        try:
            from fortuna.config.smartapi_settings import get_smartapi_settings
            from fortuna.data.sources.smartapi_session import SmartAPISession
            from fortuna.execution.smartapi_account import SmartAPIAccountReader
        except Exception as exc:  # noqa: BLE001
            self._account_sync_error = f"imports failed: {exc}"
            logger.exception("[account] failed to import SmartAPI account reader")
            return None
        if not get_smartapi_settings().configured:
            self._account_sync_error = (
                "SmartAPI credentials missing; set SMARTAPI_API_KEY / CLIENT_CODE / "
                "PASSWORD / TOTP_SECRET in .env"
            )
            logger.warning("[account] %s", self._account_sync_error)
            return None
        try:
            session = SmartAPISession.get()
            return SmartAPIAccountReader(session=session, registry=self._registry)
        except Exception as exc:  # noqa: BLE001
            self._account_sync_error = f"session init failed: {exc}"
            logger.exception("[account] SmartAPISession init failed")
            return None

    def sync_account_from_broker(self) -> Optional[dict]:
        """Pull a fresh ``AccountSnapshot`` and seed ``LiveAccount``.

        Returns the ``applied`` dict from ``LiveAccount.seed_from_broker``
        (counts of positions / holdings added, cash before/after). Returns
        ``None`` when account sync is unavailable (no reader, no router, or
        an exception). Errors are captured in ``self._account_sync_error``
        rather than raised so the dashboard can keep rendering.
        """
        if self._account_reader is None or self._execution_router is None:
            return None
        try:
            snapshot = self._account_reader.full_snapshot()
        except Exception as exc:  # noqa: BLE001
            self._account_sync_error = str(exc)
            logger.exception("[account] full_snapshot failed")
            return None
        applied = self.live_account.seed_from_broker(snapshot)
        self._signals_by_symbol.clear()
        self._rl_by_symbol.clear()
        self._agent_decisions_by_symbol.clear()
        with self._lock:
            self._broker_snapshot = snapshot
            self._account_sync_error = (
                "; ".join(snapshot.errors) if snapshot.errors else None
            )
        logger.info(
            "[account] synced from broker: cash=%s positions=%s holdings=%s errors=%s",
            applied["init_cash_after"],
            applied["positions_added"],
            applied["holdings_added"],
            applied["errors"],
        )
        return applied

    @property
    def account_reader(self):
        return self._account_reader

    @property
    def broker_snapshot(self):
        """Last successful AccountSnapshot or ``None``."""
        return self._broker_snapshot

    @property
    def account_sync_error(self) -> Optional[str]:
        return self._account_sync_error

    @property
    def account_sync_enabled(self) -> bool:
        return self._account_reader is not None

    def run_reconciliation(self):
        """Phase 2: diff today's paper journal against the broker tradeBook.

        Returns ``None`` when:
        - execution isn't enabled,
        - account sync is off (no broker tradebook to compare against), or
        - the paper journal is missing.

        Otherwise returns a :class:`ReconciliationReport`. The caller is
        expected to either render it (dashboard) or append to the EOD file.
        """
        if self._execution_router is None or self._account_reader is None:
            return None
        from fortuna.execution.reconciliation import reconcile_paper_vs_real

        journal = self._execution_router.journal
        if journal is None:
            return None
        try:
            rows = journal.read_today()
        except Exception:  # noqa: BLE001
            logger.exception("[reconciliation] failed to read paper journal")
            return None
        try:
            trades = self._account_reader.snapshot_trades()
        except Exception as exc:  # noqa: BLE001
            self._account_sync_error = f"tradeBook: {exc}"
            logger.exception("[reconciliation] failed to read broker tradeBook")
            trades = []
        return reconcile_paper_vs_real(rows, trades)

    def _rl_generator_for_symbol(self, symbol: str):
        """RL policy for ``symbol`` via per-symbol checkpoint (optional global fallback)."""
        sym = self._normalize_symbol_key(symbol)
        if sym in self._rl_by_symbol:
            return self._rl_by_symbol[sym]
        try:
            import fortuna.models as models_pkg
            import fortuna.rl.inference as rl_inference
        except Exception:  # noqa: BLE001
            self._rl_by_symbol[sym] = None
            return None
        allow_global = bool(getattr(self.settings, "rl_allow_global_policy", False))
        registry_on = bool(getattr(self.settings, "model_registry_enabled", False))
        models_root = self.settings.resolve_path(Path("models"))
        if registry_on:
            allow_fallback = not bool(getattr(self.settings, "model_promotion_required", True))
            live_dir = models_pkg.resolve_live(
                models_pkg.ModelKind.RL_POLICY,
                models_root=models_root,
                symbol=sym,
                allow_implicit_fallback=allow_fallback,
            )
            if live_dir is None and allow_global:
                live_dir = rl_inference.resolve_live_checkpoint_dir(
                    models_root,
                    allow_latest_validated=allow_fallback,
                )
        else:
            live_dir = rl_inference.resolve_checkpoint_for_symbol(
                sym,
                models_root,
                allow_global_fallback=allow_global,
            )
        if live_dir is None:
            self._rl_by_symbol[sym] = None
            return None
        gen = rl_inference.RLSignalGenerator(live_dir)
        self._rl_by_symbol[sym] = gen if gen.is_available else None
        return self._rl_by_symbol[sym]

    @property
    def rl_generator(self):
        with self._lock:
            sym = self._state.symbol
        if not sym:
            return None
        return self._rl_generator_for_symbol(sym)

    def reload_rl_generator(self) -> bool:
        """Pick up freshly promoted per-symbol policies without restart."""
        self._rl_by_symbol.clear()
        with self._lock:
            sym = self._state.symbol
        gen = self._rl_generator_for_symbol(sym) if sym else None
        return bool(gen and getattr(gen, "is_available", False))

    def reload_ml_scorer(self) -> bool:
        """Rebuild agentic orchestrator to pick up a promoted ML scorer."""
        if not getattr(self.settings, "agentic_enabled", False):
            return False
        self._agentic_orchestrator = self._build_agentic_orchestrator()
        orch = self._agentic_orchestrator
        if orch is None:
            return False
        ml_agent = getattr(orch, "_ml", None)
        scorer = getattr(ml_agent, "_scorer", None) if ml_agent else None
        return scorer is not None

    def model_status(self):
        """Read-only model health snapshot for the dashboard."""
        return self.advisory_tools().get_model_health()

    def market_universe(
        self,
        *,
        limit: int | None = None,
        timeframe: str = "1d",
        days: int = 30,
        source: str = "auto",
    ):
        """Read-only liquid universe snapshot for training and agent callers."""
        return self.advisory_tools().get_market_universe(
            limit=limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def market_shortlist(
        self,
        *,
        universe_limit: int = 10,
        analysis_limit: int = 5,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
    ):
        """Analyze the top ranked market-universe candidates through the advisory stack."""
        return self.advisory_tools().analyze_market_shortlist(
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def training_candidates(
        self,
        *,
        universe_limit: int = 15,
        analysis_limit: int = 8,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
    ):
        """Return shortlist-derived ML/RL training candidates."""
        return self.advisory_tools().get_training_candidates(
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def training_research_plan(
        self,
        *,
        universe_limit: int = 15,
        analysis_limit: int = 8,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
        selection_policy: str = "diversified",
        refresh_data: bool = False,
        refresh_target: str = "all",
        refresh_timeframe: str | None = None,
        refresh_days: int | None = None,
        force_refresh: bool = False,
    ):
        """Return a typed training-research remediation plan for ML/RL prep."""
        return self.advisory_tools().get_training_research_plan(
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
            selection_policy=selection_policy,
            refresh_data=refresh_data,
            refresh_target=refresh_target,
            refresh_timeframe=refresh_timeframe,
            refresh_days=refresh_days,
            force_refresh=force_refresh,
        )

    def multi_agent_workflow(
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
    ):
        """Return the explicit typed multi-agent workflow composition."""
        return self.advisory_tools().get_multi_agent_workflow(
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

    def shortlist_briefing(
        self,
        *,
        universe_limit: int = 10,
        analysis_limit: int = 5,
        timeframe: str = "5m",
        days: int = 30,
        source: str = "auto",
    ):
        """Return a cross-symbol briefing for the best shortlisted setups."""
        return self.advisory_tools().get_shortlist_briefing(
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def portfolio_allocation(
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
    ):
        return self.advisory_tools().get_portfolio_allocation(
            universe_limit=universe_limit,
            analysis_limit=analysis_limit,
            max_positions=max_positions,
            max_per_exposure=max_per_exposure,
            max_same_side=max_same_side,
            timeframe=timeframe,
            days=days,
            source=source,
        )

    def advisory_tools(self):
        """Shared typed tool surface for advisory callers."""
        from fortuna.agentic.tools import FortunaAdvisoryTools

        return FortunaAdvisoryTools(
            settings=self.settings,
            engine_factory=lambda: self,
            registry=self._registry,
        )

    def conversational_assistant(self):
        """Shared agentic conversational layer for Telegram and dashboard callers."""
        from fortuna.telegram.assistant import TelegramAnalysisAssistant

        return TelegramAnalysisAssistant(
            settings=self.settings,
            engine_factory=lambda: self,
            registry=self._registry,
            tools=self.advisory_tools(),
        )

    def analyze_instrument(self, request):
        """Run typed instrument analysis through the shared advisory service."""
        return self.advisory_tools().analyze_instrument(
            request.symbol,
            timeframe=request.timeframe,
            days=request.days,
            force_refresh=request.force_refresh,
        )

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
        self._signals_by_symbol.clear()
        self._agent_decisions_by_symbol.clear()
        self._rl_by_symbol.clear()

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
                rl_generator=self._rl_generator_for_symbol(sym),
                regime_router=self._regime_router,
                symbol=sym,
            )
            decisions = self._compute_agent_decisions(
                sym,
                signals,
                ohlcv,
                timeframe=tf,
                notify=False,
            )
            with self._lock:
                self._state.ohlcv = ohlcv
                self._state.batch = batch
                self._state.live_signals = signals
                self._state.agent_decisions = decisions
                self._state.last_bar_time = pd.Timestamp(ohlcv.index[-1]) if len(ohlcv) else None
                self._state.last_signal_refresh = pd.Timestamp.utcnow()
                rl_gen = self._rl_generator_for_symbol(sym)
                self._state.rl_available = bool(
                    rl_gen and getattr(rl_gen, "is_available", False)
                )
                meta = getattr(rl_gen, "metadata", None) if rl_gen else None
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
            last_closed = pd.Timestamp(ohlcv.index.max()) if not ohlcv.empty else None
            forming_ts = pd.Timestamp(forming.index[0])
            # Drop a regressive forming bar (stale exchange clock pinned to the open).
            if last_closed is not None and forming_ts < last_closed:
                return ohlcv
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

    def live_stats(self) -> dict[str, object]:
        if self._live is None:
            return {"tick_count": 0, "last_tick_at": None, "last_error": None}
        return self._live.live_stats()

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
            rl_generator=self._rl_generator_for_symbol(sym),
            regime_router=self._regime_router,
            symbol=sym,
        )
        decisions = self._compute_agent_decisions(
            sym,
            signals,
            ohlcv,
            timeframe=tf,
            notify=True,
        )
        if getattr(self.settings, "agentic_paper_learning_enabled", False) and decisions:
            self._drive_agentic_paper_learning(sym, ohlcv, decisions)
        else:
            self._drive_execution_router(sym, ohlcv, signals)
        if decisions and self._agentic_learning_store is not None:
            self._resolve_agentic_learning_outcomes(sym, ohlcv)
        with self._lock:
            self._state.ohlcv = ohlcv
            self._state.batch = batch
            self._state.live_signals = signals
            self._state.agent_decisions = decisions
            self._state.last_bar_time = pd.Timestamp(ohlcv.index[-1]) if len(ohlcv) else None
            self._state.last_signal_refresh = pd.Timestamp.utcnow()

    def _drive_execution_router(
        self, symbol: str, ohlcv: pd.DataFrame, signals: dict
    ) -> None:
        """Forward the last closed bar + per-strategy LiveSignals to the router.

        Silently no-ops when execution is disabled or any inputs are
        malformed; the dashboard will continue to render the read-only
        signal panel unchanged.
        """
        if self._execution_router is None:
            return
        if ohlcv is None or ohlcv.empty:
            return
        try:
            last_row = ohlcv.iloc[-1]
            last_ts = pd.Timestamp(ohlcv.index[-1])
            bar = {
                "ts": last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts,
                "open": float(last_row["open"]),
                "high": float(last_row["high"]),
                "low": float(last_row["low"]),
                "close": float(last_row["close"]),
                "volume": float(last_row.get("volume", 0.0)),
            }
            self._execution_router.on_bar_closed(symbol, bar, signals)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[exec] router.on_bar_closed failed for %s; execution path skipped",
                symbol,
            )

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
            rl_generator=self._rl_generator_for_symbol(sym),
            regime_router=self._regime_router,
            symbol=sym,
        )
        decisions = self._compute_agent_decisions(
            sym,
            signals,
            ohlcv,
            timeframe=self._state.timeframe,
            notify=False,
        )
        with self._lock:
            closed = self._state.ohlcv
            self._state.live_signals = signals
            self._state.agent_decisions = decisions
            # Dashboard label = last *closed* bar; forming bar can lag on illiquid names.
            if closed is not None and not closed.empty:
                self._state.last_bar_time = pd.Timestamp(closed.index[-1])
            self._state.last_signal_refresh = pd.Timestamp.utcnow()
        return signals

    def live_signals(self) -> dict[str, LiveSignal]:
        with self._lock:
            return dict(self._state.live_signals)

    def agent_decisions(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state.agent_decisions)

    def _normalize_symbol_key(self, symbol: str) -> str:
        sym = symbol.upper().strip()
        if "." not in sym and not sym.endswith("-EQ"):
            sym = f"{sym}.NS"
        return sym

    def live_signals_for_symbols(
        self,
        symbols: list[str],
        *,
        clear_cache: bool = False,
    ) -> dict[str, dict]:
        """Compute (or return cached) live signals for each held symbol.

        Strategies are evaluated on **that symbol's** OHLCV — not only on the
        sidebar focus symbol. JSON files may still say ``ICICIBANK.NS`` in
        metadata; the indicator walk uses the fetched bars for ``symbol``.

        RL is attached only when a per-symbol checkpoint exists under
        ``models/live/by_symbol/`` (no cross-symbol ICICI policy on CROMPTON).
        """
        if clear_cache:
            self._signals_by_symbol.clear()
        paths = list_strategy_paths(self.settings, self.app_config)
        tf = self.settings.default_timeframe
        days = self.settings.default_days
        out: dict[str, dict] = {}
        for raw in symbols:
            if not raw:
                continue
            sym = self._normalize_symbol_key(raw)
            if sym in self._signals_by_symbol:
                out[sym] = dict(self._signals_by_symbol[sym])
                continue
            try:
                self._registry.resolve(sym)
                ohlcv = self._mdm.get_ohlcv(sym, tf, days=days, force_refresh=False)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[signals] skip %s: %s", sym, exc)
                out[sym] = {}
                self._signals_by_symbol[sym] = {}
                continue
            if ohlcv is None or ohlcv.empty:
                out[sym] = {}
                self._signals_by_symbol[sym] = {}
                continue
            rl_gen = self._rl_generator_for_symbol(sym)
            signals = compute_live_signals_with_rl(
                paths,
                ohlcv,
                rl_generator=rl_gen,
                regime_router=self._regime_router,
                symbol=sym,
            )
            self._signals_by_symbol[sym] = signals
            out[sym] = dict(signals)
        return out

    def _compute_agent_decisions(
        self,
        symbol: str,
        signals: dict,
        ohlcv: pd.DataFrame,
        *,
        timeframe: str,
        notify: bool,
    ) -> dict[str, Any]:
        if self._agentic_orchestrator is None:
            return {}
        if ohlcv is None or ohlcv.empty:
            return {}
        try:
            ctx = self._agent_context(symbol, timeframe, signals, ohlcv)
            decision = self._agentic_orchestrator.decide(ctx)
            if self._agentic_store is not None:
                self._agentic_store.append_decision(decision)
            if notify:
                decision = self._send_agentic_notification(decision)
            if self._agentic_learning_store is not None:
                try:
                    self._agentic_learning_store.append_pending(
                        decision,
                        timeframe=timeframe,
                        ohlcv=ohlcv,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("[agentic] failed to append learning row")
            return {ctx.symbol: decision}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[agentic] decision failed for %s: %s", symbol, exc)
            return {}

    def _agent_context(
        self,
        symbol: str,
        timeframe: str,
        signals: dict,
        ohlcv: pd.DataFrame,
    ):
        from fortuna.agentic import AgentRunContext

        sym = self._normalize_symbol_key(symbol)
        account = self.live_account
        current_side: Optional[str] = None
        position_qty = 0
        is_holding = False
        account_summary: dict[str, Any] = {}
        if account is not None:
            account_summary = account.to_summary()
            pos = account.positions.get(sym)
            if pos is not None:
                current_side = pos.side.value
                position_qty = int(pos.qty)
            elif sym in account.holdings:
                current_side = "LONG"
                position_qty = int(account.holdings[sym].qty)
                is_holding = True

        last = ohlcv.iloc[-1]
        ts = pd.Timestamp(ohlcv.index[-1]).to_pydatetime()
        metadata: dict[str, Any] = {"timeframe": timeframe}
        try:
            from fortuna.agentic import bar_idx_for_decision

            bar_idx = bar_idx_for_decision(ohlcv, ts, timeframe=timeframe)
            if bar_idx is not None:
                metadata["bar_idx"] = bar_idx
        except Exception:  # noqa: BLE001
            logger.debug("[agentic] bar_idx lookup failed for %s", sym, exc_info=True)
        metadata.update(self._market_context_metadata(signals or {}))
        if getattr(self.settings, "agentic_ml_scorer_enabled", False):
            try:
                metadata["enriched"] = _enrich_ohlcv_for_ml(ohlcv)
            except Exception:  # noqa: BLE001
                logger.debug("[agentic] ohlcv enrich failed for %s", sym, exc_info=True)
        return AgentRunContext(
            symbol=sym,
            timeframe=timeframe,
            signals=signals or {},
            bar_time=ts,
            bar_close=float(last["close"]),
            current_side=current_side,
            position_qty=position_qty,
            is_holding=is_holding,
            account_summary=account_summary,
            metadata=metadata,
        )

    def _market_context_metadata(self, signals: dict[str, Any]) -> dict[str, Any]:
        actionable: list[tuple[str, Any]] = []
        buy_count = 0
        sell_count = 0
        regime = ""
        regime_confidence = 0.0
        for name, sig in (signals or {}).items():
            action = str(getattr(sig, "action", "") or "").upper()
            if action in {"BUY", "SELL"}:
                actionable.append((str(name), sig))
                if action == "BUY":
                    buy_count += 1
                elif action == "SELL":
                    sell_count += 1
            raw_regime = str(getattr(sig, "regime", "") or "").upper()
            if raw_regime and not regime:
                regime = raw_regime
            raw_conf = float(getattr(sig, "regime_confidence", 0.0) or 0.0)
            regime_confidence = max(regime_confidence, raw_conf)
        metadata: dict[str, Any] = {
            "actionable_signal_count": len(actionable),
            "buy_signal_count": buy_count,
            "sell_signal_count": sell_count,
        }
        if actionable:
            primary_name, primary_signal = actionable[0]
            metadata["primary_signal"] = primary_name
            metadata["primary_signal_action"] = str(
                getattr(primary_signal, "action", "") or ""
            ).upper()
        if regime:
            metadata["regime"] = regime
            metadata["regime_confidence"] = round(regime_confidence, 4)
        return metadata

    def _send_agentic_notification(self, decision):
        if self._notification_dispatcher is None:
            return decision
        return self._notification_dispatcher.dispatch(decision)

    def _drive_agentic_paper_learning(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        decisions: dict[str, Any],
    ) -> None:
        if self._execution_router is None:
            return
        decision = decisions.get(self._normalize_symbol_key(symbol)) or decisions.get(symbol)
        if decision is None or not getattr(decision.action, "is_actionable", False):
            return
        sig = self._decision_to_live_signal(decision)
        self._drive_execution_router(symbol, ohlcv, {f"agentic:{decision.decision_hash}": sig})
        if self._agentic_learning_store is not None:
            try:
                self._agentic_learning_store.record_event(
                    decision.decision_hash,
                    "paper_submitted",
                    {
                        "bar_time": decision.bar_time.isoformat()
                        if decision.bar_time
                        else None,
                        "bar_close": decision.bar_close,
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception("[agentic] failed to record paper_submitted learning event")
            self._record_agentic_risk_blocks(decision.decision_hash)
        if self._agentic_store is not None:
            try:
                from fortuna.agentic import PaperLearningEvent

                self._agentic_store.append_learning_event(
                    PaperLearningEvent(
                        ts=datetime.now(),
                        symbol=decision.symbol,
                        action=decision.action,
                        decision_hash=decision.decision_hash,
                        event_type="submitted_to_paper_router",
                        confidence=float(decision.confidence),
                        payload={
                            "bar_time": decision.bar_time.isoformat()
                            if decision.bar_time
                            else None,
                            "bar_close": decision.bar_close,
                        },
                    )
                )
            except Exception:  # noqa: BLE001
                logger.exception("[agentic] failed to write paper learning event")

    def _record_agentic_risk_blocks(self, decision_hash: str) -> None:
        if self._agentic_learning_store is None or self.execution_monitor is None:
            return
        tag = f"agentic:{decision_hash}"
        try:
            for evt in self.execution_monitor.tail(50):
                if evt.event_type != "risk_breach":
                    continue
                if evt.strategy != tag:
                    continue
                self._agentic_learning_store.record_event(
                    decision_hash,
                    "risk_blocked",
                    {
                        "rule": evt.payload.get("rule"),
                        "reason": evt.message or evt.payload.get("reason"),
                        "strategy": tag,
                    },
                )
        except Exception:  # noqa: BLE001
            logger.exception("[agentic] failed to record risk_blocked learning event")

    def _resolve_agentic_learning_outcomes(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
    ) -> None:
        if self._agentic_learning_store is None or ohlcv is None or ohlcv.empty:
            return
        try:
            from fortuna.agentic import resolve_pending_for_symbol

            trades = self.live_account.trades if self.live_account is not None else []
            sym = self._normalize_symbol_key(symbol)
            resolve_pending_for_symbol(
                self._agentic_learning_store,
                sym,
                ohlcv,
                trades,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[agentic] failed to resolve learning outcomes for %s", symbol)

    def _decision_to_live_signal(self, decision) -> LiveSignal:
        action = decision.action.value
        label = "EXIT" if action.startswith("EXIT") else action
        color = {
            "BUY": "#26A69A",
            "SELL": "#EF5350",
            "EXIT_LONG": "#FF9800",
            "EXIT_SHORT": "#FF9800",
        }.get(action, "#9E9E9E")
        return LiveSignal(
            strategy_name=f"agentic:{decision.decision_hash}",
            action=action,
            label=label,
            bar_time=pd.Timestamp(decision.bar_time or datetime.now()),
            bar_close=float(decision.bar_close or 0.0),
            enter=action in {"BUY", "SELL"},
            exit=action in {"EXIT_LONG", "EXIT_SHORT"},
            in_position=bool(decision.current_side),
            side=decision.current_side,
            color=color,
            exit_reason="agentic_advisory" if action.startswith("EXIT") else None,
            rl_action=decision.rl_action,
            regime=decision.regime,
        )

    def agent_decisions_for_symbols(
        self,
        symbols: list[str],
        *,
        clear_cache: bool = False,
    ) -> dict[str, Any]:
        """Return agentic advisory decisions for held/watchlist symbols."""
        if clear_cache:
            self._agent_decisions_by_symbol.clear()
        if self._agentic_orchestrator is None:
            return {}
        signals_by_symbol = self.live_signals_for_symbols(
            symbols,
            clear_cache=clear_cache,
        )
        out: dict[str, Any] = {}
        tf = self.settings.default_timeframe
        days = self.settings.default_days
        for raw in symbols:
            sym = self._normalize_symbol_key(raw)
            if sym in self._agent_decisions_by_symbol and not clear_cache:
                out[sym] = self._agent_decisions_by_symbol[sym]
                continue
            signals = signals_by_symbol.get(sym) or {}
            try:
                ohlcv = self._mdm.get_ohlcv(sym, tf, days=days, force_refresh=False)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[agentic] skip %s: %s", sym, exc)
                continue
            decisions = self._compute_agent_decisions(
                sym,
                signals,
                ohlcv,
                timeframe=tf,
                notify=False,
            )
            if sym in decisions:
                out[sym] = decisions[sym]
                self._agent_decisions_by_symbol[sym] = decisions[sym]
        return out

    def poll_live(self) -> int:
        """Apply queued live events; return count processed."""
        if self._live is None:
            return 0
        n = self._live.drain_events()
        with self._lock:
            if self._live is not None:
                self._state.ohlcv = self._live.ohlcv
                forming = self._live.forming_bar_df()
                if forming is not None and not forming.empty:
                    self._state.last_bar_time = pd.Timestamp(forming.index[-1])
                elif self._state.ohlcv is not None and not self._state.ohlcv.empty:
                    self._state.last_bar_time = pd.Timestamp(self._state.ohlcv.index[-1])
        if n > 0:
            self.refresh_live_signals()
        return n
