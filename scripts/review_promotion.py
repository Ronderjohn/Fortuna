#!/usr/bin/env python
"""Review the currently promoted model artifact and linked workflow context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.model_activation import build_lane_activation_summary, format_lane_activation
from fortuna.app.promotion_review import (
    build_promotion_review,
    export_promotion_review,
    format_promotion_review,
)
from fortuna.config.settings import get_settings
from fortuna.models.metadata import ModelKind


def main() -> int:
    parser = argparse.ArgumentParser(description="Review a promoted model artifact")
    parser.add_argument("--kind", choices=["rl_policy", "ml_scorer"], default="rl_policy")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--ml-base", default="models/ml_signal_scorer")
    parser.add_argument("--symbol", default="", help="Optional symbol filter for RL reviews")
    parser.add_argument("--run-id", default="", help="Optional run id filter")
    parser.add_argument("--json", action="store_true", help="Emit JSON review payload")
    parser.add_argument("--out", default="", help="Optional path to export the review artifact")
    parser.add_argument(
        "--format",
        choices=["json", "md"],
        default="md",
        help="Export format for --out (default: md)",
    )
    parser.add_argument(
        "--activation-only",
        action="store_true",
        help="Print lane activation summary without requiring a promoted audit row",
    )
    args = parser.parse_args()

    kind = ModelKind(args.kind)
    settings = get_settings()
    lane = "ml" if kind == ModelKind.ML_SCORER else "rl"
    activation = build_lane_activation_summary(
        settings,
        lane=lane,
        symbol=args.symbol or None,
        models_root=Path(args.models_root),
        ml_base=Path(args.ml_base),
    )

    if args.activation_only:
        if args.json:
            print(json.dumps(activation.to_dict(), indent=2))
        else:
            print("\n".join(format_lane_activation(activation)))
        return 0 if activation.active else 1

    review = build_promotion_review(
        kind=kind,
        models_root=Path(args.models_root),
        ml_base=Path(args.ml_base),
        symbol=args.symbol or None,
        run_id=args.run_id or None,
        settings=settings,
    )
    if review is None:
        print("\n".join(format_lane_activation(activation)))
        raise SystemExit("no matching promoted artifact found")

    if args.out:
        written = export_promotion_review(review, args.out, fmt=args.format)
        print(f"review artifact -> {written}")

    if args.json:
        print(json.dumps(review.to_dict(), indent=2))
    else:
        print(format_promotion_review(review))
    return 0


if __name__ == "__main__":
    sys.exit(main())
