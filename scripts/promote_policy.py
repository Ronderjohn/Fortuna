#!/usr/bin/env python
"""Promote a validated RL policy to live (global or per-symbol via registry)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.workflow_snapshot import (
    format_workflow_snapshot_summary,
    load_workflow_snapshot_summary,
)
from fortuna.models import PromotionRecord, PromotionStatus
from fortuna.models.promotion import promote as registry_promote
from fortuna.rl.training.checkpoint import PolicyCheckpoint


def _resolve_workflow_snapshot(path_text: str) -> str | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_file():
        raise SystemExit(f"workflow snapshot not found: {path}")
    return str(path.resolve())


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
    workflow_snapshot_path: str | None = None,
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
    record.workflow_snapshot_path = workflow_snapshot_path
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
    parser.add_argument(
        "--workflow-snapshot",
        default="",
        help="Optional workflow snapshot JSON to reference from the promotion audit/pointer",
    )
    args = parser.parse_args()
    workflow_snapshot_path = _resolve_workflow_snapshot(args.workflow_snapshot)
    out = promote(
        args.run_id,
        Path(args.models_root),
        allow_rejected=args.allow_rejected,
        symbol=args.symbol or None,
        workflow_snapshot_path=workflow_snapshot_path,
    )
    print(f"promoted run_id={args.run_id} -> {out}")
    summary = load_workflow_snapshot_summary(workflow_snapshot_path)
    if summary is not None:
        print(format_workflow_snapshot_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
