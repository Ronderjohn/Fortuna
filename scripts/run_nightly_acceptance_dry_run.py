#!/usr/bin/env python
"""Build a fully local nightly acceptance dry run or verify existing artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.acceptance_bundle import (  # noqa: E402
    export_acceptance_bundle,
    format_acceptance_bundle,
)
from fortuna.app.nightly_acceptance import (  # noqa: E402
    build_candidate_nightly_dry_run,
    build_nightly_emit_replay,
    verify_acceptance_replay_chain,
)
from fortuna.config.settings import Settings  # noqa: E402


def _format_replay_linkage(linkage) -> str:
    parts = [f"status={linkage.overall_status}"]
    if getattr(linkage, "run_identifier", None):
        parts.append(f"run={linkage.run_identifier}")
    if getattr(linkage, "missing_artifacts", ()):
        parts.append(f"missing={','.join(linkage.missing_artifacts)}")
    if getattr(linkage, "broken_links", ()):
        parts.append(f"broken={','.join(linkage.broken_links)}")
    if getattr(linkage, "linkage_warnings", ()):
        parts.append(f"warnings={' | '.join(linkage.linkage_warnings)}")
    parts.append(f"summary={linkage.summary or '—'}")
    return " ".join(parts)


def _legacy_replay_hint(linkage) -> str:
    missing = set(getattr(linkage, "missing_artifacts", ()) or ())
    legacy_markers = {"training_candidates", "training_research", "workflow_snapshot"}
    if getattr(linkage, "overall_status", "") != "broken":
        return ""
    if not legacy_markers.issubset(missing):
        return ""
    return (
        "hint=This nightly directory appears to predate training-candidate, "
        "training-research, or workflow-snapshot artifact emission. "
        "Use a newer nightly run or the local candidate -> emit-replay -> verify sequence."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local nightly acceptance dry run, emit-based replay, or verify "
            "linkage for an existing nightly report directory."
        )
    )
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument(
        "--mode",
        choices=("candidate", "emit-replay", "verify"),
        default="candidate",
        help=(
            "candidate: synthetic fixture dry run (default); "
            "emit-replay: route through nightly_basket emit_* entrypoints; "
            "verify: check linkage for an existing nightly report directory"
        ),
    )
    parser.add_argument("--out-dir", default="reports/acceptance/dry_run")
    parser.add_argument(
        "--report-dir",
        default="",
        help="Nightly report directory for --mode verify (defaults to --out-dir/reports/nightly)",
    )
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument(
        "--source",
        default="screener",
        choices=("screener", "registry", "auto"),
    )
    parser.add_argument("--run-id", default="dry_run_acceptance")
    parser.add_argument(
        "--bundle-out",
        default="reports/acceptance/dry_run/acceptance_bundle.md",
    )
    parser.add_argument("--bundle-format", choices=("md", "json"), default="md")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    root = settings.resolve_path(Path(args.out_dir))
    bundle_out = settings.resolve_path(Path(args.bundle_out))

    if args.mode == "verify":
        report_dir = (
            settings.resolve_path(Path(args.report_dir))
            if args.report_dir
            else root / "reports" / "nightly"
        )
        verify_settings = settings.model_copy(update={"project_root": root})
        linkage, bundle = verify_acceptance_replay_chain(
            settings=verify_settings,
            report_dir=report_dir,
            models_root=root / "models",
            ml_base=root / "models" / "ml_signal_scorer",
            symbol=args.symbol,
        )
        if bundle_out:
            export_acceptance_bundle(bundle, bundle_out, fmt=args.bundle_format)
        artifacts = None
    elif args.mode == "emit-replay":
        bundle, artifacts = build_nightly_emit_replay(
            settings=settings,
            out_dir=root,
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            source=args.source,
            run_id=args.run_id,
            export_bundle_path=bundle_out,
            export_bundle_format=args.bundle_format,
        )
        linkage = artifacts.linkage
    else:
        bundle, artifacts = build_candidate_nightly_dry_run(
            settings=settings,
            out_dir=root,
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            source=args.source,
            run_id=args.run_id,
            export_bundle_path=bundle_out,
            export_bundle_format=args.bundle_format,
        )
        linkage = artifacts.linkage

    print(format_acceptance_bundle(bundle))
    if args.mode == "verify":
        print(f"verify_report_dir={report_dir}")
    else:
        print(f"dry_run_root={artifacts.root}")
        if artifacts.nightly_report_path:
            print(f"nightly_report={artifacts.nightly_report_path}")
        if artifacts.workflow_snapshot_path:
            print(f"workflow_snapshot={artifacts.workflow_snapshot_path}")
        if artifacts.promotion_review_path:
            print(f"promotion_review={artifacts.promotion_review_path}")
        if artifacts.acceptance_bundle_path:
            print(f"acceptance_bundle={artifacts.acceptance_bundle_path}")
    if linkage is not None:
        print("replay_linkage=" + _format_replay_linkage(linkage))
        hint = _legacy_replay_hint(linkage)
        if hint:
            print(hint)
    return 1 if bundle.overall_status == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
