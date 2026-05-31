#!/usr/bin/env python
"""Promote a validated artifact to live via the unified promotion registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.models import ModelKind, PromotionRecord, PromotionStatus
from fortuna.models.promotion import promote as registry_promote
from fortuna.models.promotion import compute_ml_advisory_ready
from fortuna.rl.training.checkpoint import PolicyCheckpoint


def _locate_rl_checkpoint(
    run_id: str,
    models_root: Path,
    *,
    allow_rejected: bool,
) -> Path:
    validated_dir = models_root / "validated" / run_id
    if validated_dir.exists() and (validated_dir / "policy.zip").exists():
        return validated_dir
    if allow_rejected:
        rejected_dir = models_root / "rejected" / run_id
        if rejected_dir.exists() and (rejected_dir / "policy.zip").exists():
            print("NOTE: promoting from rejected/ — use only for confidence-filter mode.")
            return rejected_dir
    raise SystemExit(
        f"RL checkpoint not found for run_id={run_id} "
        f"(validated/, allow_rejected={allow_rejected})"
    )


def _locate_ml_checkpoint(
    run_id: str,
    ml_base: Path,
    *,
    allow_rejected: bool,
) -> Path:
    validated_dir = ml_base / "validated" / run_id
    if validated_dir.exists() and (validated_dir / "model.joblib").exists():
        return validated_dir
    if allow_rejected:
        rejected_dir = ml_base / "rejected" / run_id
        if rejected_dir.exists() and (rejected_dir / "model.joblib").exists():
            return rejected_dir
    raise SystemExit(f"ML scorer not found for run_id={run_id}")


def promote_rl(
    run_id: str,
    models_root: Path,
    *,
    allow_rejected: bool = False,
    symbol: str | None = None,
) -> Path:
    source_dir = _locate_rl_checkpoint(run_id, models_root, allow_rejected=allow_rejected)
    cp = PolicyCheckpoint.read(source_dir / "metadata.json")
    if symbol:
        from dataclasses import replace

        cp = replace(cp, symbol=symbol)
    record = PromotionRecord.from_rl_checkpoint(
        cp,
        source_dir,
        status=PromotionStatus.VALIDATED,
    )
    return registry_promote(record, models_root=models_root, promoted_by="cli")


def promote_ml(
    run_id: str,
    models_root: Path,
    ml_base: Path,
    *,
    allow_rejected: bool = False,
) -> Path:
    from fortuna.ml.artifacts import load_metadata

    source_dir = _locate_ml_checkpoint(run_id, ml_base, allow_rejected=allow_rejected)
    meta = load_metadata(source_dir)
    if meta is None:
        raise SystemExit(f"metadata missing for ML run_id={run_id}")
    ready, _ = compute_ml_advisory_ready(meta)
    record = PromotionRecord.from_scorer_metadata(
        meta,
        source_dir,
        status=PromotionStatus.VALIDATED,
        advisory_ready=ready,
    )
    return registry_promote(record, models_root=models_root, promoted_by="cli", ml_base=ml_base)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a model artifact to live")
    parser.add_argument("--kind", choices=["rl_policy", "ml_scorer"], default="rl_policy")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--ml-base", default="models/ml_signal_scorer")
    parser.add_argument("--symbol", default="", help="Optional RL symbol override")
    parser.add_argument("--allow-rejected", action="store_true")
    args = parser.parse_args()

    models_root = Path(args.models_root)
    if args.kind == "rl_policy":
        out = promote_rl(
            args.run_id,
            models_root,
            allow_rejected=args.allow_rejected,
            symbol=args.symbol or None,
        )
    else:
        out = promote_ml(
            args.run_id,
            models_root,
            Path(args.ml_base),
            allow_rejected=args.allow_rejected,
        )
    print(f"promoted run_id={args.run_id} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
