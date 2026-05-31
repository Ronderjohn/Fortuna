"""Supervised ML signal-quality scoring for Fortuna."""

from fortuna.ml.features import ML_FEATURE_NAMES, build_feature_matrix, build_feature_row
from fortuna.ml.labels import (
    build_signal_examples,
    events_from_entry_exit_columns,
    forward_return_at_bar,
    label_from_forward_return,
)
from fortuna.ml.signal_scorer import SignalScorer, score_or_neutral
from fortuna.ml.types import (
    ClassificationMetrics,
    LabelConfig,
    ScorerMetadata,
    SignalExample,
    SignalScoreResult,
    SplitRange,
)

__all__ = [
    "ClassificationMetrics",
    "LabelConfig",
    "MLSignalScorerAgent",
    "ML_FEATURE_NAMES",
    "ScorerMetadata",
    "SignalExample",
    "SignalScoreResult",
    "SignalScorer",
    "SplitRange",
    "build_feature_matrix",
    "build_feature_row",
    "build_signal_examples",
    "events_from_entry_exit_columns",
    "forward_return_at_bar",
    "label_from_forward_return",
    "score_or_neutral",
]


def __getattr__(name: str):
    if name == "MLSignalScorerAgent":
        from fortuna.ml.agent import MLSignalScorerAgent

        return MLSignalScorerAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
