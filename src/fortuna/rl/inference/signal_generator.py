"""``RLSignalGenerator`` — drop-in RL signal source for the live dashboard.

If the policy checkpoint is absent or fails to load, the generator returns
``None`` from every ``predict()`` call and ``is_available`` stays ``False`` so
``compute_live_signals_with_rl`` can fall back to the deterministic path
silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

from fortuna.app.live_signals import SignalType
from fortuna.features.builder import FeatureBuilder
from fortuna.features.position import PositionState
from fortuna.features.registry import feature_registry_hash
from fortuna.features.rl_indicators import enrich_for_rl_last_bar
from fortuna.rl.env.actions import ACTION_TO_SIGNAL
from fortuna.rl.env.session import is_squareoff_time
from fortuna.rl.training.checkpoint import PolicyCheckpoint
from fortuna.utils.logging import get_logger
from fortuna.utils.timing import timed_step

logger = get_logger(__name__)


class RegimeDetectorLike(Protocol):
    """Minimal interface ``RLSignalGenerator`` needs from a regime detector."""

    def predict(self, ohlcv_session: pd.DataFrame) -> Optional[str]: ...


@dataclass
class RLLiveSignal:
    """Output of ``RLSignalGenerator.predict()`` for the dashboard layer."""

    signal: SignalType
    action_id: int
    timestamp: pd.Timestamp
    bar_close: float
    source: str = "rl_policy"
    metadata: dict[str, Any] = None  # type: ignore[assignment]


def resolve_live_checkpoint_dir(
    root: Path | str = "models",
    *,
    allow_latest_validated: bool = True,
) -> Optional[Path]:
    """Resolve the active live policy directory.

    Three options, tried in order:
        1. A ``models/live`` symlink (POSIX-friendly).
        2. A ``models/live.json`` pointer file containing ``{"run_id": "..."}``.
        3. The lexicographically last directory under ``models/validated``.

    Returns ``None`` if nothing is available — caller is expected to silently
    fall back to the deterministic signal path.
    """
    root_p = Path(root)
    live = root_p / "live"
    if live.exists() and (live / "policy.zip").exists():
        return live

    pointer = root_p / "live.json"
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            run_id = payload.get("run_id")
            if run_id:
                cand = root_p / "validated" / run_id
                if (cand / "policy.zip").exists():
                    return cand
        except Exception as exc:  # noqa: BLE001
            logger.debug("live pointer parse failed: %s", exc)

    if allow_latest_validated:
        validated = root_p / "validated"
        if validated.exists():
            dirs = sorted(p for p in validated.iterdir() if p.is_dir())
            for cand in reversed(dirs):
                if (cand / "policy.zip").exists():
                    return cand
    return None


def resolve_live_checkpoint_dir_for_symbol(
    symbol: str,
    root: Path | str = "models",
) -> Optional[Path]:
    """Per-symbol live policy under ``models/live/by_symbol/<SYMBOL_KEY>/``.

    Returns ``None`` when no symbol-specific checkpoint exists — callers
    should **not** fall back to the global ``models/live`` policy trained on
    a different symbol (e.g. ICICIBANK) when advising on CROMPTON.
    """
    sym_key = symbol.upper().strip().replace(".", "_")
    per_sym = Path(root) / "live" / "by_symbol" / sym_key
    ptr = per_sym / "live.json"
    if ptr.is_file():
        from fortuna.models.promotion import resolve_live_pointer

        resolved = resolve_live_pointer(ptr)
        if resolved is not None:
            return resolved
    if per_sym.exists() and (per_sym / "policy.zip").exists():
        return per_sym
    return None


def resolve_checkpoint_for_symbol(
    symbol: str,
    root: Path | str = "models",
    *,
    allow_global_fallback: bool = False,
) -> Optional[Path]:
    """Resolve live checkpoint for a symbol, optionally falling back to global."""
    per_sym = resolve_live_checkpoint_dir_for_symbol(symbol, root)
    if per_sym is not None:
        return per_sym
    if allow_global_fallback:
        return resolve_live_checkpoint_dir(root)
    return None


class RLSignalGenerator:
    """Loads a validated policy + normalizer and emits RL signals per bar.

    If a ``RegimeDetector`` is supplied, the generator will look for regime-specific
    checkpoints under ``models/live/<REGIME>/`` (e.g. ``models/live/TRENDING/``).
    When the current regime cannot be classified or a regime-specific policy is
    missing, the generator silently uses the default policy in ``models/live``.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path | str] = None,
        *,
        regime_detector: Optional["RegimeDetectorLike"] = None,
    ) -> None:
        self._model = None
        self._builder: Optional[FeatureBuilder] = None
        self._metadata: Optional[PolicyCheckpoint] = None
        self._checkpoint_dir: Optional[Path] = None
        self._regime_detector = regime_detector
        self._regime_policies: dict[str, tuple[Any, FeatureBuilder, PolicyCheckpoint]] = {}
        self._current_regime: Optional[str] = None
        self._load_error: Optional[str] = None

        path = Path(checkpoint_dir) if checkpoint_dir else None
        if path is None or not path.exists() or not (path / "policy.zip").exists():
            self._load_error = "no_checkpoint"
            logger.info("[RL] no policy checkpoint found — deterministic fallback active")
            return

        try:
            self._load(path)
            self._checkpoint_dir = path
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.warning("[RL] failed to load policy from %s: %s", path, exc)
            self._model = None
            self._builder = None
            self._metadata = None
            return

        if self._metadata is not None and not self._metadata.advisory_ready:
            logger.warning(
                "[RL] policy %s loaded for inspection but not advisory-ready (%s)",
                self._metadata.run_id,
                ", ".join(self._metadata.failure_modes) or "checks failed",
            )

        # Discover regime-specific policies (optional).
        if self._regime_detector is not None:
            self._discover_regime_policies(path)

    def _resolve_active_policy(
        self,
    ) -> tuple[Any, FeatureBuilder, Optional[PolicyCheckpoint]]:
        if self._current_regime and self._current_regime in self._regime_policies:
            model, builder, meta = self._regime_policies[self._current_regime]
            return model, builder, meta
        return self._model, self._builder, self._metadata

    def _load(self, path: Path) -> None:
        from stable_baselines3 import PPO  # heavy import — defer until needed

        meta_path = path / "metadata.json"
        if meta_path.exists():
            self._metadata = PolicyCheckpoint.read(meta_path)
            saved_hash = self._metadata.feature_registry_hash
            current_hash = feature_registry_hash()
            if saved_hash and saved_hash != current_hash:
                raise ValueError(
                    f"feature_registry_hash mismatch: checkpoint={saved_hash}, "
                    f"current={current_hash}"
                )

        normalizer_path = path / "normalizer.json"
        if not normalizer_path.exists():
            raise ValueError("missing normalizer.json — refusing load")

        self._model = PPO.load(str(path / "policy"), device="cpu")

        n_bars = self._metadata.n_bars if self._metadata else 20
        builder = FeatureBuilder(n_bars=n_bars)
        builder.load(normalizer_path)
        self._builder = builder

        sharpe = self._metadata.oos_metrics.sharpe_ratio if self._metadata else float("nan")
        run_id = self._metadata.run_id if self._metadata else path.name
        logger.info("[RL] loaded policy %s OOS Sharpe=%.3f", run_id, sharpe)

    def _discover_regime_policies(self, root: Path) -> None:
        """Look for ``<root>/<REGIME>/policy.zip`` siblings."""
        from stable_baselines3 import PPO

        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.upper()
            if (sub / "policy.zip").exists():
                try:
                    model = PPO.load(str(sub / "policy"), device="cpu")
                    meta = self._metadata
                    n_bars = 20
                    meta_path = sub / "metadata.json"
                    if meta_path.exists():
                        meta = PolicyCheckpoint.read(meta_path)
                        saved_hash = meta.feature_registry_hash
                        current_hash = feature_registry_hash()
                        if saved_hash and saved_hash != current_hash:
                            raise ValueError(
                                f"feature_registry_hash mismatch: checkpoint={saved_hash}, "
                                f"current={current_hash}"
                            )
                        n_bars = meta.n_bars
                    builder = FeatureBuilder(n_bars=n_bars)
                    normalizer_path = sub / "normalizer.json"
                    if not normalizer_path.exists():
                        raise ValueError("missing normalizer.json — refusing load")
                    builder.load(normalizer_path)
                    self._regime_policies[name] = (model, builder, meta)  # type: ignore[arg-type]
                    logger.info("[RL] discovered regime-specific policy: %s", name)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("regime policy %s load failed: %s", sub, exc)

    # -------------------------------------------------- public surface
    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def is_available(self) -> bool:
        return self._model is not None and self._builder is not None

    @property
    def is_advisory_ready(self) -> bool:
        _model, _builder, meta = self._resolve_active_policy()
        if meta is None:
            return False
        return bool(meta.advisory_ready)

    @property
    def metadata(self) -> Optional[PolicyCheckpoint]:
        return self._metadata

    @property
    def checkpoint_dir(self) -> Optional[Path]:
        return self._checkpoint_dir

    @property
    def active_regime(self) -> Optional[str]:
        return self._current_regime

    def set_active_regime(self, regime: Optional[str]) -> None:
        """Switch the active regime-specific policy.

        Called once per session by the dashboard layer after classifying the
        current day's OHLCV. ``None`` reverts to the default policy.
        """
        if regime is None:
            self._current_regime = None
            return
        key = regime.upper()
        if key in self._regime_policies:
            self._current_regime = key
            logger.info("[RL] active regime -> %s", key)
        else:
            self._current_regime = None

    def classify_and_set_regime(self, ohlcv_session: pd.DataFrame) -> Optional[str]:
        """Run the detector on one session and switch to the matching policy."""
        if self._regime_detector is None:
            return None
        regime = self._regime_detector.predict(ohlcv_session)
        self.set_active_regime(regime)
        return regime

    def predict(
        self,
        indicator_row: pd.Series,
        position_state: PositionState,
        timestamp: pd.Timestamp,
    ) -> Optional[RLLiveSignal]:
        """Run one inference pass. Returns ``None`` if no policy loaded.

        Must complete in < 50 ms on CPU (see Phase 2 latency budget).
        """
        if self._model is None or self._builder is None:
            return None
        model, builder, meta = self._resolve_active_policy()
        if meta is not None and not meta.advisory_ready:
            return None

        with timed_step("rl_inference") as details:
            obs = builder.build(indicator_row, position_state, timestamp)
            action_arr, _ = model.predict(obs[np.newaxis], deterministic=True)
            action_id = int(action_arr[0])

            if is_squareoff_time(timestamp) and position_state.is_open:
                # Mirror the env's square-off override so live trading respects 15:15 IST.
                from fortuna.rl.env.actions import ACTION_EXIT_LONG, ACTION_EXIT_SHORT

                action_id = ACTION_EXIT_LONG if position_state.is_long else ACTION_EXIT_SHORT
                details["forced_exit"] = "1"

            signal = ACTION_TO_SIGNAL.get(action_id, SignalType.HOLD)
            details["action"] = str(action_id)
            details["signal"] = signal.value

        return RLLiveSignal(
            signal=signal,
            action_id=action_id,
            timestamp=pd.Timestamp(timestamp),
            bar_close=float(indicator_row.get("close", 0.0)),
            source="rl_policy",
            metadata={
                "run_id": meta.run_id if meta else "unknown",
                "policy_type": meta.policy_type if meta else "unknown",
                "active_regime": self._current_regime,
            },
        )

    def predict_from_ohlcv(
        self,
        ohlcv: pd.DataFrame,
        position_state: PositionState,
    ) -> Optional[RLLiveSignal]:
        """Convenience: enrich the last bar and predict from raw OHLCV."""
        if self._model is None or ohlcv is None or ohlcv.empty:
            return None
        row = enrich_for_rl_last_bar(ohlcv)
        return self.predict(row, position_state, row.name)
