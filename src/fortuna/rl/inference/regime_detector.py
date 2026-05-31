"""Market regime classifier (Phase 2.2).

Labels each session as TRENDING / RANGING / VOLATILE. Used by
``RLSignalGenerator`` to optionally load a regime-specific policy
checkpoint (``models/live/<regime>/``) when one is available.

Training is offline and supervised:
    1. Compute per-session features (ADX-like, ATR%/close, realized vol,
       BB-width, volume ratio).
    2. Label sessions by an heuristic based on price/vol patterns.
    3. Fit a small ``LogisticRegression``.
    4. Persist via joblib alongside the policy in ``models/regime/``.

The detector is **optional** — when no classifier is present,
``predict()`` returns ``None`` and the inference path stays on the default policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

REGIMES = ("TRENDING", "RANGING", "VOLATILE")


# ----------------------------------------------- feature engineering


def _session_features(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-session features from a single trading day's OHLCV."""
    if df is None or len(df) < 10:
        return {}

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64) if "volume" in df.columns else None

    # ADX proxy: average abs daily return / range ratio.
    tr = np.maximum.reduce(
        [
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ]
    )
    atr = float(np.mean(tr)) if len(tr) else 0.0
    atr_pct = atr / float(np.mean(close)) if np.mean(close) > 0 else 0.0

    # Realized vol: std of returns.
    ret = np.diff(close) / close[:-1]
    real_vol = float(np.std(ret))

    # BB-width: 2*std / mean over the session.
    bb_width = (2 * float(np.std(close)) / float(np.mean(close))) if np.mean(close) > 0 else 0.0

    # Trend strength: net move / total path.
    net = float(abs(close[-1] - close[0])) / float(np.mean(close)) if np.mean(close) > 0 else 0.0
    path = float(np.sum(np.abs(np.diff(close)))) / float(np.mean(close)) if np.mean(close) > 0 else 0.0
    trend_strength = net / path if path > 0 else 0.0

    # Volume ratio: late session vs early session.
    if volume is not None and len(volume) >= 10:
        early = float(np.mean(volume[: len(volume) // 2]))
        late = float(np.mean(volume[len(volume) // 2:]))
        vol_ratio = late / early if early > 0 else 1.0
    else:
        vol_ratio = 1.0

    return {
        "atr_pct": atr_pct,
        "real_vol": real_vol,
        "bb_width": bb_width,
        "trend_strength": trend_strength,
        "vol_ratio": vol_ratio,
    }


FEATURE_ORDER = ("atr_pct", "real_vol", "bb_width", "trend_strength", "vol_ratio")


def _label_session(features: dict[str, float]) -> str:
    """Heuristic labeling for bootstrapping the classifier.

    Phase 2.2 default rule:
        VOLATILE if atr_pct > 0.015 AND trend_strength < 0.3
        TRENDING if trend_strength > 0.5
        RANGING otherwise
    """
    if not features:
        return "RANGING"
    atr = features["atr_pct"]
    trend = features["trend_strength"]
    if atr > 0.015 and trend < 0.3:
        return "VOLATILE"
    if trend > 0.5:
        return "TRENDING"
    return "RANGING"


def _per_session_groups(df: pd.DataFrame) -> list[pd.DataFrame]:
    if df is None or df.empty:
        return []
    idx = pd.DatetimeIndex(df.index)
    by_day = df.groupby(idx.normalize())
    return [grp for _, grp in by_day if len(grp) >= 20]


# ----------------------------------------------- detector


@dataclass
class RegimeDetectorMetadata:
    accuracy: float
    n_train: int
    n_test: int
    confusion_matrix: list[list[int]]


class RegimeDetector:
    """``LogisticRegression`` over per-session features."""

    def __init__(self) -> None:
        self._classifier = None
        self._scaler = None
        self.metadata: Optional[RegimeDetectorMetadata] = None

    @property
    def is_trained(self) -> bool:
        return self._classifier is not None

    # ------------------------------------------- API
    def predict(self, ohlcv_session: pd.DataFrame) -> Optional[str]:
        """Classify the regime for a single session's OHLCV. ``None`` if untrained."""
        if not self.is_trained:
            return None
        feats = _session_features(ohlcv_session)
        if not feats:
            return None
        x = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=np.float64)
        x = self._scaler.transform(x)
        idx = int(self._classifier.predict(x)[0])
        return REGIMES[idx]

    def predict_proba(self, ohlcv_session: pd.DataFrame) -> Optional[dict[str, float]]:
        if not self.is_trained:
            return None
        feats = _session_features(ohlcv_session)
        if not feats:
            return None
        x = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=np.float64)
        x = self._scaler.transform(x)
        proba = self._classifier.predict_proba(x)[0]
        return {REGIMES[i]: float(p) for i, p in enumerate(proba)}

    def train(
        self,
        ohlcv: pd.DataFrame,
        *,
        test_fraction: float = 0.2,
        random_state: int = 42,
    ) -> RegimeDetectorMetadata:
        """Fit the classifier on session-grouped OHLCV with heuristic labels."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        sessions = _per_session_groups(ohlcv)
        if len(sessions) < 10:
            raise ValueError(
                f"Need at least 10 sessions for regime detector training; got {len(sessions)}"
            )

        feat_rows: list[list[float]] = []
        labels: list[int] = []
        for session in sessions:
            feats = _session_features(session)
            if not feats:
                continue
            feat_rows.append([feats[k] for k in FEATURE_ORDER])
            labels.append(REGIMES.index(_label_session(feats)))

        if not feat_rows:
            raise ValueError("No usable sessions for regime training")

        X = np.array(feat_rows, dtype=np.float64)
        y = np.array(labels, dtype=np.int64)

        # If all sessions ended up with the same label, fall back to a no-op
        # detector that always predicts that label.
        if len(set(y.tolist())) == 1:
            single_label = REGIMES[int(y[0])]

            class _ConstantClassifier:
                def __init__(self, label: int) -> None:
                    self._label = int(label)

                def predict(self, X):  # noqa: D401
                    return np.full(len(X), self._label, dtype=np.int64)

                def predict_proba(self, X):  # noqa: D401
                    p = np.zeros((len(X), len(REGIMES)), dtype=np.float64)
                    p[:, self._label] = 1.0
                    return p

            self._scaler = StandardScaler().fit(X)
            self._classifier = _ConstantClassifier(int(y[0]))
            self.metadata = RegimeDetectorMetadata(
                accuracy=1.0,
                n_train=len(X),
                n_test=0,
                confusion_matrix=[[len(X)]],
            )
            logger.info("[Regime] all sessions labeled %s; using constant classifier", single_label)
            return self.metadata

        # Only stratify when every present class has at least 2 samples.
        unique, counts = np.unique(y, return_counts=True)
        can_stratify = bool(np.all(counts >= 2)) and len(unique) > 1
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_fraction,
            random_state=random_state,
            stratify=y if can_stratify else None,
        )

        self._scaler = StandardScaler().fit(X_train)
        Xs_train = self._scaler.transform(X_train)
        Xs_test = self._scaler.transform(X_test)

        # scikit-learn 1.8 removed the deprecated ``multi_class`` argument.
        # Default solver behaviour ("lbfgs") already handles multinomial fitting.
        self._classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            random_state=random_state,
        )
        self._classifier.fit(Xs_train, y_train)

        preds = self._classifier.predict(Xs_test)
        acc = float(np.mean(preds == y_test)) if len(y_test) else 1.0
        cm = confusion_matrix(y_test, preds, labels=list(range(len(REGIMES)))).tolist()

        self.metadata = RegimeDetectorMetadata(
            accuracy=acc,
            n_train=len(X_train),
            n_test=len(X_test),
            confusion_matrix=cm,
        )
        logger.info(
            "[Regime] trained on %d sessions, test acc=%.3f",
            len(X_train),
            acc,
        )
        return self.metadata

    # ------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classifier": self._classifier,
            "scaler": self._scaler,
            "metadata": self.metadata,
            "feature_order": FEATURE_ORDER,
            "regimes": REGIMES,
        }
        joblib.dump(payload, path)

    def load(self, path: str | Path) -> None:
        import joblib

        payload = joblib.load(path)
        self._classifier = payload["classifier"]
        self._scaler = payload["scaler"]
        self.metadata = payload.get("metadata")


def load_detector_if_available(path: Path | str = "models/regime/classifier.joblib") -> Optional[RegimeDetector]:
    """Convenience loader — returns ``None`` silently if not present."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        det = RegimeDetector()
        det.load(p)
        return det
    except Exception as exc:  # noqa: BLE001
        logger.debug("RegimeDetector load failed at %s: %s", p, exc)
        return None
