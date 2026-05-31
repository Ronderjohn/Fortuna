#!/usr/bin/env python
"""Promote a validated RL policy to live (global or per-symbol via registry)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.models import PromotionRecord, PromotionStatus
from fortuna.models.promotion import promote as registry_promote
from fortuna.rl.training.checkpoint import PolicyCheckpoint


def _locate_checkpoint(
    run_id: str,
    models_root: Path,
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
        f"checkpoint not found for run_id={run_id} "
        f"(validated/, allow_rejected={allow_rejected})"
    )


def promote(
    run_id: str,
    models_root: Path = Path("models"),
    *,
    allow_rejected: bool = False,
    symbol: str | None = None,
) -> Path:
    source_dir = _locate_checkpoint(run_id, models_root, allow_rejected)
    cp = PolicyCheckpoint.read(source_dir / "metadata.json")
    if symbol:
        from dataclasses import replace

        cp = replace(cp, symbol=symbol)
    record = PromotionRecord.from_rl_checkpoint(
        cp,
        source_dir,
        status=PromotionStatus.VALIDATED,
    )
    return registry_promote(record, models_root=models_root, promoted_by="manual")


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote an RL policy to live")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--symbol", default="", help="Promote as per-symbol live pointer")
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help="Also accept run_ids under models/rejected/",
    )
    args = parser.parse_args()
    out = promote(
        args.run_id,
        Path(args.models_root),
        allow_rejected=args.allow_rejected,
        symbol=args.symbol or None,
    )
    print(f"promoted run_id={args.run_id} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
