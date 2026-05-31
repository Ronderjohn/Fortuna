"""Persist and load ML signal scorer artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fortuna.ml.types import ScorerMetadata

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


def save_artifact(
    artifact_dir: Path | str,
    pipeline: Any,
    metadata: ScorerMetadata,
) -> Path:
    """Save sklearn pipeline and metadata.json to artifact_dir."""
    import joblib

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / MODEL_FILENAME)
    (out / METADATA_FILENAME).write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )
    return out


def load_metadata(artifact_dir: Path | str) -> Optional[ScorerMetadata]:
    path = Path(artifact_dir) / METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        return ScorerMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def load_pipeline(artifact_dir: Path | str) -> Optional[Any]:
    model_path = Path(artifact_dir) / MODEL_FILENAME
    if not model_path.is_file():
        return None
    try:
        import joblib

        return joblib.load(model_path)
    except Exception:
        return None


def load_artifact(artifact_dir: Path | str) -> tuple[Optional[Any], Optional[ScorerMetadata]]:
    """Load pipeline and metadata; either may be None on failure."""
    meta = load_metadata(artifact_dir)
    pipeline = load_pipeline(artifact_dir)
    if pipeline is None or meta is None:
        return None, None
    return pipeline, meta


def validated_dir(base: Path | str, run_id: str) -> Path:
    return Path(base) / "validated" / run_id


def rejected_dir(base: Path | str, run_id: str) -> Path:
    return Path(base) / "rejected" / run_id
