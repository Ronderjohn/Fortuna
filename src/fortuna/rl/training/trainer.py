"""``FortunaRLTrainer`` — orchestrates PPO training on Phase 1 walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from fortuna.backtesting.standard.config import MarketCostModel
from fortuna.backtesting.standard.filters import FilterVerdict
from fortuna.config.settings import Settings, get_settings
from fortuna.data.manager import MarketDataManager
from fortuna.features.builder import FeatureBuilder
from fortuna.features.position import PositionState
from fortuna.features.rl_indicators import enrich_for_rl
from fortuna.paper.blackbox import WalkForwardFold, walk_forward_folds
from fortuna.reporting.strategy_tester.metrics import compute_all_metrics
from fortuna.reporting.strategy_tester.trade import TradeRecord, TradeSide
from fortuna.rl.env.actions import (
    ACTION_ENTER_LONG,
    ACTION_ENTER_SHORT,
    ACTION_EXIT_LONG,
    ACTION_EXIT_SHORT,
)
from fortuna.rl.env.reward import RewardConfig
from fortuna.rl.env.session import is_squareoff_time
from fortuna.rl.evaluation.baseline import baseline_sharpe_for_symbol
from fortuna.rl.training.callbacks import FortunaEvalCallback, TensorboardCallback
from fortuna.rl.training.checkpoint import (
    OOSMetricsSummary,
    PolicyCheckpoint,
    build_failure_modes,
    compute_advisory_ready,
)
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


def _gpu_available() -> bool:
    import os

    if os.environ.get("FORTUNA_USE_GPU", "0") not in ("1", "true", "True"):
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _resolve_device(cfg: "TrainerConfig") -> str:
    """Pick the actual torch device given config + policy type.

    Policy:
        - explicit "cpu" / "cuda" wins
        - "auto" uses GPU only for CnnPolicy (or any non-MLP policy); MLP
          stays on CPU because per-batch GPU transfer dominates the tiny
          network's compute.
    """
    if cfg.device != "auto":
        return cfg.device
    if cfg.policy_type != "MlpPolicy" and _gpu_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainerConfig:
    symbol: str = "ICICIBANK.NS"
    timeframe: str = "5m"
    days: int = 180

    train_bars: int = 156
    test_bars: int = 78
    step_bars: int = 78
    min_folds: int = 2

    total_timesteps: int = 500_000
    n_envs: int = 4
    n_bars: int = 20
    warmup_bars: int = 30
    # SubprocVecEnv is now opt-in but recommended on multi-core boxes — it
    # routes around the GIL and unlocks ~Nx env-step throughput where N is
    # roughly min(n_envs, physical_cores).
    use_subproc: bool = False

    policy_type: str = "MlpPolicy"
    policy_kwargs: dict[str, Any] = field(default_factory=lambda: {"net_arch": [256, 256, 128]})
    # ``device='auto'`` defers to ``_resolve_device(self)``. PPO with MlpPolicy
    # is almost always faster on CPU (transfer overhead dominates the tiny
    # network). Set ``device='cuda'`` only for CnnPolicy.
    device: str = "auto"
    learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

    reward_config: RewardConfig = field(default_factory=RewardConfig)
    cost_model: MarketCostModel = field(default_factory=MarketCostModel)

    seed: Optional[int] = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("models"))
    train_eval_split: float = 0.7

    eval_freq: int = 10_000
    early_stop_patience: int = 5
    baseline_sharpe: Optional[float] = None

    # Optional data-source override (e.g. "smartapi" supports months of 5m
    # history vs yfinance's 60-day cap). ``None`` falls back to settings.data_source.
    data_source: Optional[str] = None
    force_refresh: bool = False


# Re-exported from the lean ``fortuna.rl.env.factory`` module so that
# SubprocVecEnv child processes do NOT have to re-import this trainer
# module (which pulls in torch + stable_baselines3). On low-end CUDA
# boxes that import + CUDA-context init was observed to serialize 8
# spawning workers behind the NVIDIA driver and stall startup by hours.
# Keeping the factory in a torch-free module fixes that.
from fortuna.rl.env.factory import EnvFactory as _EnvFactory  # noqa: E402


class FortunaRLTrainer:
    """Top-level PPO orchestrator."""

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        *,
        settings: Optional[Settings] = None,
    ) -> None:
        self.cfg = config or TrainerConfig()
        self.settings = settings or get_settings()
        source = self.cfg.data_source or self.settings.data_source
        self._mdm = MarketDataManager(self.settings, data_source=source)

    # ------------------------------------------------------------------ data
    def load_folds(self) -> list[WalkForwardFold]:
        ohlcv = self._mdm.get_ohlcv(
            self.cfg.symbol,
            self.cfg.timeframe,
            days=self.cfg.days,
            force_refresh=self.cfg.force_refresh,
        )
        if ohlcv is None or len(ohlcv) == 0:
            raise RuntimeError(
                f"No OHLCV available for {self.cfg.symbol} {self.cfg.timeframe} "
                f"with days={self.cfg.days}"
            )
        return walk_forward_folds(
            ohlcv,
            train_bars=self.cfg.train_bars,
            test_bars=self.cfg.test_bars,
            step_bars=self.cfg.step_bars,
            min_folds=self.cfg.min_folds,
        )

    # ---------------------------------------------------------------- policy
    def _resolve_policy_kwargs(self) -> tuple[str, dict[str, Any]]:
        """Translate ``cfg.policy_type`` into the SB3 (policy_name, kwargs) tuple.

        - ``"MlpPolicy"``: returned as-is.
        - ``"CnnPolicy"``: rewritten to ``MlpPolicy`` with
          ``features_extractor_class=TemporalCNNExtractor`` injected — this
          is the canonical SB3 path for custom CNN extractors on non-image
          flat observations. A smaller ``net_arch`` is used on top of the
          128-dim extractor output since the extractor already does the
          heavy lifting.
        """
        if self.cfg.policy_type == "CnnPolicy":
            from fortuna.rl.models.extractor import TemporalCNNExtractor

            user_kwargs = dict(self.cfg.policy_kwargs or {})
            # Don't carry MLP-shaped net_arch into the CNN head — override
            # with a compact policy/value head unless the user explicitly
            # set one suitable for the CNN.
            net_arch = user_kwargs.pop("net_arch", [128, 64])
            features_dim = user_kwargs.pop("cnn_features_dim", 128)
            extractor_kwargs = {
                "features_dim": features_dim,
                "n_bars": self.cfg.n_bars,
            }
            policy_kwargs: dict[str, Any] = {
                "features_extractor_class": TemporalCNNExtractor,
                "features_extractor_kwargs": extractor_kwargs,
                "net_arch": net_arch,
                **user_kwargs,
            }
            return "MlpPolicy", policy_kwargs

        return self.cfg.policy_type, dict(self.cfg.policy_kwargs or {})

    # ------------------------------------------------------------------- env
    def _make_env_factory(
        self,
        folds: list[WalkForwardFold],
        train_mode: bool,
    ) -> _EnvFactory:
        return _EnvFactory(
            folds=folds,
            reward_config=self.cfg.reward_config,
            cost_model=self.cfg.cost_model,
            n_bars=self.cfg.n_bars,
            warmup_bars=self.cfg.warmup_bars,
            train_mode=train_mode,
        )

    def _build_vec_env(
        self,
        folds: list[WalkForwardFold],
        n_envs: int,
        train_mode: bool,
    ) -> VecEnv:
        factories = [self._make_env_factory(folds, train_mode) for _ in range(n_envs)]
        if self.cfg.use_subproc and n_envs > 1:
            # ``spawn`` is the only safe start method on Windows.
            return SubprocVecEnv(factories, start_method="spawn")
        return DummyVecEnv(factories)

    # ------------------------------------------------------------------ train
    def train(self) -> PolicyCheckpoint:
        folds = self.load_folds()
        if len(folds) < 2:
            raise RuntimeError(
                f"Need at least 2 folds for train/eval split, got {len(folds)}"
            )

        split = max(1, int(len(folds) * self.cfg.train_eval_split))
        train_folds = folds[:split]
        eval_folds = folds[split:] if split < len(folds) else folds[-1:]

        logger.info(
            "[RL] Training on %d folds, evaluating on %d. Total fold count=%d",
            len(train_folds),
            len(eval_folds),
            len(folds),
        )
        # Note: with SubprocVecEnv each child process pays its own pandas
        # warm-up (~600 ms) on its first call to enrich_for_rl. The env's
        # ``_enriched_cache`` ensures we never recompute the same fold twice
        # per env, so steady-state cost is amortized. We deliberately do NOT
        # pre-enrich in the main process because spawn would still need to
        # re-pickle and ship every enriched DataFrame to each child.

        vec_env = self._build_vec_env(train_folds, self.cfg.n_envs, train_mode=True)
        eval_env = self._build_vec_env(eval_folds, 1, train_mode=False)

        device = _resolve_device(self.cfg)
        logger.info("[RL] device=%s, n_envs=%d, use_subproc=%s",
                    device, self.cfg.n_envs, self.cfg.use_subproc)

        # Enable cuDNN auto-tuner for our fixed conv input shape — picks the
        # fastest kernel implementation once per shape, then memoizes.
        if device == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:  # noqa: BLE001
                pass

        # Resolve the SB3-side policy + extractor wiring. SB3's literal
        # ``CnnPolicy`` only accepts image-shape Boxes (HxWxC), so for our
        # flat 1-D temporal observation we keep ``MlpPolicy`` and inject the
        # custom 1-D CNN via ``features_extractor_class``.
        sb3_policy, resolved_policy_kwargs = self._resolve_policy_kwargs()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.cfg.checkpoint_dir) / "running" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        tb_log = out_dir / "tb_logs"

        model = PPO(
            policy=sb3_policy,
            env=vec_env,
            learning_rate=self.cfg.learning_rate,
            n_steps=self.cfg.ppo_n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_range=self.cfg.clip_range,
            policy_kwargs=resolved_policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log=str(tb_log),
            seed=self.cfg.seed,
        )

        eval_cb = FortunaEvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(out_dir / "best"),
            log_path=str(out_dir / "eval_logs"),
            eval_freq=max(self.cfg.eval_freq // self.cfg.n_envs, 100),
            n_eval_episodes=max(1, len(eval_folds)),
            deterministic=True,
            verbose=1,
            baseline_sharpe=self.cfg.baseline_sharpe,
            patience=self.cfg.early_stop_patience,
        )

        try:
            model.learn(
                total_timesteps=self.cfg.total_timesteps,
                callback=[eval_cb, TensorboardCallback()],
                progress_bar=False,
            )
        finally:
            try:
                vec_env.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                eval_env.close()
            except Exception:  # noqa: BLE001
                pass

        # OOS evaluation on all eval folds with deterministic policy.
        oos_metrics, fold_summaries, n_trades_total = self.evaluate(model, eval_folds)
        period_days = max(1, len(eval_folds) * self.cfg.test_bars // 75)
        verdict = FilterVerdict.evaluate(oos_metrics, period_days=period_days)

        baseline_sharpe = self.cfg.baseline_sharpe
        if baseline_sharpe is None:
            baseline_sharpe = baseline_sharpe_for_symbol(
                self.cfg.symbol,
                self.cfg.timeframe,
                settings=self.settings,
            )
        oos_sharpe = float(oos_metrics.risk.sharpe_ratio)
        beats_baseline: Optional[bool] = None
        if baseline_sharpe is not None:
            beats_baseline = oos_sharpe >= float(baseline_sharpe)

        # Package + persist checkpoint.
        meta = PolicyCheckpoint(
            run_id=run_id,
            symbol=self.cfg.symbol,
            timeframe=self.cfg.timeframe,
            obs_shape=list(self._scratch_builder().obs_shape),
            policy_type=self.cfg.policy_type,
            total_timesteps=self.cfg.total_timesteps,
            n_folds=len(folds),
            oos_metrics=OOSMetricsSummary.from_strategy_metrics(oos_metrics),
            verdict_passed=bool(verdict.passed),
            verdict_reasons=list(verdict.reasons),
            verdict_score=float(verdict.score),
            reward_config=self.cfg.reward_config.to_dict(),
            train_bars=self.cfg.train_bars,
            test_bars=self.cfg.test_bars,
            step_bars=self.cfg.step_bars,
            n_envs=self.cfg.n_envs,
            n_bars=self.cfg.n_bars,
            seed=self.cfg.seed,
            sb3_version=_pkg_version("stable_baselines3"),
            torch_version=torch.__version__,
            notes=f"n_eval_trades={n_trades_total}",
            baseline_sharpe=baseline_sharpe,
            beats_baseline=beats_baseline,
            oos_fold_metrics=[m.to_dict() for m in fold_summaries],
        )
        meta.failure_modes = build_failure_modes(meta)
        meta.advisory_ready, _ = compute_advisory_ready(meta)

        dest_root = "validated" if verdict.passed else "rejected"
        final_dir = Path(self.cfg.checkpoint_dir) / dest_root / run_id
        final_dir.mkdir(parents=True, exist_ok=True)

        model.save(str(final_dir / "policy"))
        # Snapshot a fresh normalizer state from an env stepped over the eval folds
        # so live inference starts with sensible stats.
        normalizer_builder = self._scratch_builder()
        self._warm_normalizer(normalizer_builder, eval_folds)
        normalizer_builder.save(final_dir / "normalizer.json")

        meta.write(final_dir / "metadata.json")

        # Persist a FoldRecord so the next AdaptiveOptimizerAgent.suggest()
        # has empirical RewardConfig feedback for this strategy_stem.
        try:
            self._update_adaptive_learner_state(oos_metrics, run_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("AdaptiveLearner update failed (non-fatal): %s", exc)

        logger.info(
            "[RL] %s -> %s  OOS Sharpe=%.3f  PF=%.3f  trades=%d  passed=%s",
            run_id,
            final_dir,
            oos_metrics.risk.sharpe_ratio,
            oos_metrics.performance.profit_factor,
            oos_metrics.performance.total_trades,
            verdict.passed,
        )
        return meta

    def _update_adaptive_learner_state(self, metrics, run_id: str) -> None:
        """Append a ``FoldRecord`` to ``logs/learner/rl_policy.json``.

        Stemmed by symbol+timeframe so each instrument tracks its own history.
        """
        from fortuna.paper.learner import (
            AdaptiveLearner,
            FoldRecord,
            StrategyLearningState,
        )

        learner = AdaptiveLearner(Path("logs/learner"))
        stem = f"rl_{self.cfg.symbol.replace('.', '_')}_{self.cfg.timeframe}"
        state = learner.load(stem) or StrategyLearningState(strategy_stem=stem)

        record = FoldRecord(
            fold_id=len(state.fold_history),
            profit_pct=float(metrics.performance.total_net_profit / 1000.0),  # currency -> rough %
            win_ratio_pct=float(metrics.performance.win_rate_pct),
            total_trades=int(metrics.performance.total_trades),
            candidate_id=run_id,
            params_summary=str(self.cfg.reward_config.to_dict()),
        )
        state.fold_history.append(record)
        state.cumulative_oos_profit_pct = sum(f.profit_pct for f in state.fold_history)
        state.generation += 1
        learner.save(state)
        logger.info(
            "[RL] AdaptiveLearner state updated for %s (generation=%d, history=%d)",
            stem,
            state.generation,
            len(state.fold_history),
        )

    # ---------------------------------------------------------------- eval
    def evaluate(
        self,
        model: PPO,
        eval_folds: list[WalkForwardFold],
    ) -> tuple[Any, list[OOSMetricsSummary], int]:
        """Run the deterministic policy over OOS folds and return aggregated metrics."""
        trades: list[TradeRecord] = []
        fold_summaries: list[OOSMetricsSummary] = []
        capital = 100_000.0

        builder = FeatureBuilder(n_bars=self.cfg.n_bars, warmup_bars=self.cfg.warmup_bars)

        for fold in eval_folds:
            enriched = enrich_for_rl(fold.test)
            if len(enriched) <= self.cfg.n_bars + 1:
                continue

            fold_trades: list[TradeRecord] = []
            builder.reset()
            position = PositionState()
            empty_pos = PositionState()
            for i in range(self.cfg.n_bars):
                row = enriched.iloc[i]
                builder.build(row, empty_pos, row.name)

            for i in range(self.cfg.n_bars, len(enriched)):
                row = enriched.iloc[i]
                timestamp = row.name
                price = float(row["close"])

                obs = builder.build(row, position, timestamp)
                action, _ = model.predict(obs[np.newaxis], deterministic=True)
                action = int(action[0])

                if is_squareoff_time(timestamp) and position.is_open:
                    action = ACTION_EXIT_LONG if position.is_long else ACTION_EXIT_SHORT

                if action == ACTION_ENTER_LONG and position.is_flat:
                    position.open_long(price, timestamp)
                elif action == ACTION_ENTER_SHORT and position.is_flat:
                    position.open_short(price, timestamp)
                elif action == ACTION_EXIT_LONG and position.is_long:
                    self._close_to_trade(fold_trades, position, price, timestamp)
                elif action == ACTION_EXIT_SHORT and position.is_short:
                    self._close_to_trade(fold_trades, position, price, timestamp)

                position.step()

            # Close any lingering position at fold end.
            if position.is_open:
                last_row = enriched.iloc[-1]
                self._close_to_trade(
                    fold_trades, position, float(last_row["close"]), last_row.name
                )

            fold_metrics, _ = compute_all_metrics(fold_trades, capital)
            fold_summaries.append(OOSMetricsSummary.from_strategy_metrics(fold_metrics))
            trades.extend(fold_trades)

        metrics, _ = compute_all_metrics(trades, capital)
        return metrics, fold_summaries, len(trades)

    def _close_to_trade(
        self,
        trades: list[TradeRecord],
        position: PositionState,
        price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        side = position.side
        entry_price = position.entry_price
        entry_time = position.entry_time
        size = position.size

        realized_pct = position.close(price, timestamp)
        gross = (price - entry_price) if side == "LONG" else (entry_price - price)
        pnl_currency = gross * size
        cost_currency = self.cfg.cost_model.round_trip_rate * abs(entry_price) * size
        holding = pd.Timestamp(timestamp) - pd.Timestamp(entry_time)
        if not isinstance(holding, pd.Timedelta):
            holding = pd.Timedelta(0)

        trades.append(
            TradeRecord(
                entry_time=pd.Timestamp(entry_time),
                exit_time=pd.Timestamp(timestamp),
                entry_price=float(entry_price),
                exit_price=float(price),
                side=TradeSide.LONG if side == "LONG" else TradeSide.SHORT,
                qty=float(size),
                pnl=float(pnl_currency - cost_currency),
                pnl_percent=float(realized_pct),
                holding_time=holding,
                commission=float(cost_currency),
            )
        )

    # ------------------------------------------------------- helpers
    def _scratch_builder(self) -> FeatureBuilder:
        return FeatureBuilder(n_bars=self.cfg.n_bars, warmup_bars=self.cfg.warmup_bars)

    def _warm_normalizer(
        self,
        builder: FeatureBuilder,
        eval_folds: list[WalkForwardFold],
    ) -> None:
        empty_pos = PositionState()
        for fold in eval_folds:
            enriched = enrich_for_rl(fold.test)
            for i in range(len(enriched)):
                row = enriched.iloc[i]
                builder.build(row, empty_pos, row.name)


def _pkg_version(name: str) -> str:
    try:
        import importlib

        mod = importlib.import_module(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"
