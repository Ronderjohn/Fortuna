"""Train, load, and predict signal-quality scores."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from fortuna.ml.artifacts import load_artifact, save_artifact
from fortuna.ml.features import (
    ML_FEATURE_NAMES,
    build_feature_row,
    feature_vector_from_row,
    ml_feature_schema_hash,
)
from fortuna.ml.types import (
    ClassificationMetrics,
    ScorerMetadata,
    SignalScoreResult,
)


def _build_pipeline(model_kind: str, random_state: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_kind == "hgb":
        clf = HistGradientBoostingClassifier(random_state=random_state, max_iter=100)
    else:
        clf = LogisticRegression(max_iter=500, random_state=random_state)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> ClassificationMetrics:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    n = len(y_true)
    if n == 0:
        return ClassificationMetrics()

    pos_rate = float(np.mean(y_true))
    try:
        roc = float(roc_auc_score(y_true, y_proba)) if len(set(y_true.tolist())) > 1 else 0.5
    except ValueError:
        roc = 0.5

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        brier_score=float(brier_score_loss(y_true, y_proba)),
        n_samples=n,
        positive_rate=pos_rate,
    )


class SignalScorer:
    """Supervised signal-quality classifier with fail-soft inference."""

    def __init__(
        self,
        *,
        random_state: int = 42,
        pipeline: Optional[object] = None,
        metadata: Optional[ScorerMetadata] = None,
    ) -> None:
        self.random_state = random_state
        self._pipeline = pipeline
        self.metadata = metadata

    @property
    def is_available(self) -> bool:
        return self._pipeline is not None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        model_kind: Literal["logistic", "hgb"] = "logistic",
        metadata: Optional[ScorerMetadata] = None,
        eval_X: Optional[np.ndarray] = None,
        eval_y: Optional[np.ndarray] = None,
    ) -> ScorerMetadata:
        if len(y) == 0:
            raise ValueError("Cannot fit signal scorer on empty dataset")
        if len(set(y.tolist())) < 2:
            raise ValueError("Cannot fit signal scorer: labels must contain both classes")

        import sklearn

        pipeline = _build_pipeline(model_kind, self.random_state)
        pipeline.fit(X, y)
        self._pipeline = pipeline

        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)[:, 1]
        train_metrics = compute_classification_metrics(y, y_pred, y_proba)

        oos_metrics = ClassificationMetrics()
        if eval_X is not None and eval_y is not None and len(eval_y) > 0:
            eval_pred = pipeline.predict(eval_X)
            eval_proba = pipeline.predict_proba(eval_X)[:, 1]
            oos_metrics = compute_classification_metrics(eval_y, eval_pred, eval_proba)

        base_meta = metadata or ScorerMetadata(
            run_id="unassigned",
            symbol="",
            timeframe="",
        )
        self.metadata = ScorerMetadata(
            run_id=base_meta.run_id,
            symbol=base_meta.symbol,
            timeframe=base_meta.timeframe,
            strategy_name=base_meta.strategy_name,
            model_kind=model_kind,
            label_config=base_meta.label_config,
            feature_names=ML_FEATURE_NAMES,
            feature_schema_hash=ml_feature_schema_hash(),
            split_ranges=list(base_meta.split_ranges),
            train_metrics=train_metrics,
            oos_metrics=oos_metrics,
            random_state=self.random_state,
            sklearn_version=sklearn.__version__,
            created_at=base_meta.created_at,
            verdict_passed=base_meta.verdict_passed,
            verdict_reasons=list(base_meta.verdict_reasons),
        )
        from fortuna.models.promotion import compute_ml_advisory_ready

        ready, _ = compute_ml_advisory_ready(self.metadata)
        self.metadata.advisory_ready = ready
        return self.metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("SignalScorer is not fitted or loaded")
        return self._pipeline.predict_proba(X)[:, 1]

    def score_signal(
        self,
        enriched: pd.DataFrame,
        bar_idx: int,
        *,
        action: str,
        position_side: str = "FLAT",
        bar_time: Optional[pd.Timestamp] = None,
    ) -> SignalScoreResult:
        if self._pipeline is None:
            return SignalScoreResult.unavailable()

        feat = build_feature_row(
            enriched,
            bar_idx,
            action=action,
            position_side=position_side,
            bar_time=bar_time,
        )
        X = feature_vector_from_row(feat).reshape(1, -1)
        proba = float(self.predict_proba(X)[0])
        pred_class = 1 if proba >= 0.5 else 0
        model_id = self.metadata.run_id if self.metadata else None
        quality = "favorable" if pred_class == 1 else "unfavorable"
        reason = f"ML scorer P(favorable)={proba:.2f} → {quality} for {action}"
        return SignalScoreResult(
            available=True,
            probability=proba,
            predicted_class=pred_class,
            model_id=model_id,
            reason=reason,
            feature_names=ML_FEATURE_NAMES,
            metadata={"action": action, "bar_idx": bar_idx},
        )

    def save(self, artifact_dir: Path | str, metadata: Optional[ScorerMetadata] = None) -> Path:
        if self._pipeline is None:
            raise RuntimeError("Cannot save unfitted SignalScorer")
        if self.metadata is None and metadata is None:
            raise ValueError("metadata is required to save artifact")
        if metadata is not None and self.metadata is not None:
            meta = ScorerMetadata(
                run_id=metadata.run_id,
                symbol=metadata.symbol,
                timeframe=metadata.timeframe,
                strategy_name=metadata.strategy_name,
                model_kind=self.metadata.model_kind,
                label_config=metadata.label_config,
                feature_names=self.metadata.feature_names,
                feature_schema_hash=self.metadata.feature_schema_hash,
                split_ranges=list(metadata.split_ranges),
                train_metrics=self.metadata.train_metrics,
                oos_metrics=self.metadata.oos_metrics,
                random_state=self.metadata.random_state,
                sklearn_version=self.metadata.sklearn_version,
                created_at=metadata.created_at,
                verdict_passed=metadata.verdict_passed,
                verdict_reasons=list(metadata.verdict_reasons),
                advisory_ready=metadata.advisory_ready,
            )
        else:
            meta = metadata or self.metadata
        path = save_artifact(artifact_dir, self._pipeline, meta)
        self.metadata = meta
        return path

    @classmethod
    def load(
        cls,
        artifact_dir: Path | str,
        *,
        require_promotion: bool = False,
    ) -> Optional["SignalScorer"]:
        pipeline, meta = load_artifact(artifact_dir)
        if pipeline is None or meta is None:
            return None
        if meta.feature_schema_hash and meta.feature_schema_hash != ml_feature_schema_hash():
            return None
        if require_promotion and not meta.verdict_passed:
            return None
        if require_promotion and not meta.advisory_ready:
            return None
        return cls(
            random_state=meta.random_state,
            pipeline=pipeline,
            metadata=meta,
        )


def score_or_neutral(
    scorer: Optional[SignalScorer],
    enriched: pd.DataFrame,
    bar_idx: int,
    *,
    action: str,
    position_side: str = "FLAT",
    bar_time: Optional[pd.Timestamp] = None,
) -> SignalScoreResult:
    """Score via loaded scorer or return neutral unavailable result."""
    if scorer is None or not scorer.is_available:
        return SignalScoreResult.unavailable()
    try:
        return scorer.score_signal(
            enriched,
            bar_idx,
            action=action,
            position_side=position_side,
            bar_time=bar_time,
        )
    except Exception as exc:  # noqa: BLE001
        return SignalScoreResult.unavailable(reason=f"ML scorer failed: {exc}")
