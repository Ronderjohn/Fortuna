#!/usr/bin/env python
"""Promote a validated RL policy to ``models/live``.

On POSIX systems this creates a symlink ``models/live -> models/validated/<run_id>``.
On Windows (where symlinks require Developer Mode or admin) it writes a pointer
file ``models/live.json`` with ``{"run_id": "..."}`` that ``resolve_live_checkpoint_dir``
understands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _locate_checkpoint(
    run_id: str, models_root: Path, allow_rejected: bool
) -> Path:
    """Find a checkpoint by run_id under validated/ (or rejected/ if allowed)."""
    validated_dir = models_root / "validated" / run_id
    if validated_dir.exists() and (validated_dir / "policy.zip").exists():
        return validated_dir
    if allow_rejected:
        rejected_dir = models_root / "rejected" / run_id
        if rejected_dir.exists() and (rejected_dir / "policy.zip").exists():
            print(f"NOTE: promoting from rejected/ — use only for confidence-filter mode.")
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
) -> Path:
    source_dir = _locate_checkpoint(run_id, models_root, allow_rejected)

    live_link = models_root / "live"
    live_json = models_root / "live.json"

    # Try a symlink first (best on POSIX).
    if live_link.exists() or live_link.is_symlink():
        try:
            live_link.unlink()
        except Exception:  # noqa: BLE001
            pass

    try:
        target = source_dir.resolve()
        os.symlink(target, live_link, target_is_directory=True)
        if live_json.exists():
            live_json.unlink()
        return live_link
    except (OSError, NotImplementedError) as exc:
        # Fall back to a JSON pointer (Windows without Developer Mode).
        live_json.write_text(
            json.dumps({"run_id": run_id, "path": str(source_dir)}, indent=2),
            encoding="utf-8",
        )
        print(f"symlink unavailable ({exc}); wrote pointer file -> {live_json}")
        return live_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote an RL policy to models/live")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models-root", default="models")
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help="Also accept run_ids under models/rejected/ "
        "(useful when the best available policy is below FilterVerdict but "
        "still usable as a confidence-filter overlay).",
    )
    args = parser.parse_args()
    out = promote(args.run_id, Path(args.models_root), allow_rejected=args.allow_rejected)
    print(f"promoted run_id={args.run_id} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
