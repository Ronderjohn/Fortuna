#!/usr/bin/env python
"""Build a higher-level multi-agent acceptance bundle for operators."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.acceptance_bundle import (  # noqa: E402
    export_acceptance_bundle,
    format_acceptance_bundle,
    gather_acceptance_bundle,
)
from fortuna.config.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Fortuna multi-agent acceptance bundle")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--ml-base", default="models/ml_signal_scorer")
    parser.add_argument("--symbol", default="", help="Optional RL symbol filter")
    parser.add_argument(
        "--nightly-report",
        default="",
        help="Optional nightly JSON report path to use as acceptance evidence",
    )
    parser.add_argument(
        "--nightly-report-dir",
        default="",
        help="Optional nightly report directory; the newest JSON report will be used",
    )
    parser.add_argument(
        "--workflow-snapshot",
        default="",
        help="Existing workflow snapshot JSON path to include",
    )
    parser.add_argument(
        "--build-workflow",
        action="store_true",
        help="Build and export a fresh workflow snapshot before bundling",
    )
    parser.add_argument("--workflow-out", default="reports/acceptance/workflow_snapshot.json")
    parser.add_argument("--universe-limit", type=int, default=10)
    parser.add_argument("--analysis-limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--source",
        default="auto",
        choices=("auto", "screener", "registry"),
    )
    parser.add_argument(
        "--skip-model-health",
        action="store_true",
        help="Skip the runtime model-health snapshot",
    )
    parser.add_argument("--out", default="reports/acceptance/multi_agent_acceptance.md")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    bundle = gather_acceptance_bundle(
        settings=settings,
        models_root=settings.resolve_path(Path(args.models_root)),
        ml_base=settings.resolve_path(Path(args.ml_base)),
        symbol=args.symbol or None,
        nightly_report_path=(
            settings.resolve_path(Path(args.nightly_report))
            if args.nightly_report
            else None
        ),
        nightly_report_dir=(
            settings.resolve_path(Path(args.nightly_report_dir))
            if args.nightly_report_dir
            else None
        ),
        workflow_snapshot_path=(
            settings.resolve_path(Path(args.workflow_snapshot))
            if args.workflow_snapshot
            else None
        ),
        build_workflow=bool(args.build_workflow),
        workflow_out=settings.resolve_path(Path(args.workflow_out)),
        universe_limit=args.universe_limit,
        analysis_limit=args.analysis_limit,
        candidate_limit=args.candidate_limit,
        timeframe=args.timeframe,
        days=args.days,
        source=args.source,
        include_model_health=not args.skip_model_health,
    )

    out_path = settings.resolve_path(Path(args.out))
    written = export_acceptance_bundle(bundle, out_path, fmt=args.format)
    print(format_acceptance_bundle(bundle))
    print(f"bundle artifact -> {written}")
    return 1 if bundle.overall_status == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
