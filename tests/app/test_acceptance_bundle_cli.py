from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "build_acceptance_bundle.py"
    spec = importlib.util.spec_from_file_location("fortuna_acceptance_bundle_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_workflow_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": "auto",
                "timeframe": "5m",
                "lookback_days": 20,
                "universe": {"metrics": {"count": 15}},
                "shortlist": {"metrics": {"count": 6}},
                "briefing": {"metrics": {"candidates": 4}},
                "allocation": {
                    "metrics": {"selected_count": 3, "skipped_count": 1, "max_positions": 3}
                },
                "candidates": {"metrics": {"ml_count": 5, "rl_count": 3}},
            }
        ),
        encoding="utf-8",
    )


def _write_nightly_report(
    path: Path,
    workflow_path: Path,
    training_candidate_path: Path,
    training_research_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "started_at": "2026-06-03T01:00:00",
                "ended_at": "2026-06-03T01:10:00",
                "total_duration_s": 600.0,
                "basket": ["RELIANCE.NS"],
                "overall_status": "ok",
                "steps": [
                    {
                        "name": "basket_resolution",
                        "status": "ok",
                        "detail": {"mode": "derived_training_candidates"},
                    },
                    {
                        "name": "training_candidates",
                        "status": "ok",
                        "detail": {
                            "path": str(training_candidate_path),
                            "count": 5,
                            "ml_count": 4,
                            "rl_count": 2,
                            "selection_policy": "ranked",
                        },
                    },
                    {
                        "name": "training_research",
                        "status": "ok",
                        "detail": {
                            "path": str(training_research_path),
                            "count": 1,
                            "ml_count": 1,
                            "rl_count": 1,
                            "selection_policy": "ranked",
                            "refresh_target": "all",
                            "discovery_preferred_count": 1,
                            "discovery_regime_mix": "trending=1",
                            "discovery_summary": "preferred=1 | setup_posture=rl | posture=rl",
                            "discovery_recommended_refresh_target": "rl",
                        },
                    },
                    {
                        "name": "training_execution_target",
                        "status": "ok",
                        "detail": {
                            "target": "rl",
                            "run_ml": False,
                            "run_rl": True,
                            "selection_source": "training_research",
                        },
                    },
                    {
                        "name": "workflow_snapshot",
                        "status": "ok",
                        "detail": {"path": str(workflow_path)},
                    },
                    {
                        "name": "promote_best",
                        "status": "ok",
                        "detail": {"count": 0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_acceptance_bundle_cli_exports_markdown(tmp_path, monkeypatch, capsys):
    module = _load_script()
    workflow = tmp_path / "reports" / "workflow_snapshot.json"
    training_candidates = tmp_path / "reports" / "nightly" / "training_candidates.json"
    training_research = tmp_path / "reports" / "nightly" / "training_research_plan.json"
    nightly = tmp_path / "reports" / "nightly" / "20260603_0100.json"
    _write_workflow_snapshot(workflow)
    training_candidates.parent.mkdir(parents=True, exist_ok=True)
    training_candidates.write_text("{}", encoding="utf-8")
    training_research.write_text("{}", encoding="utf-8")
    _write_nightly_report(nightly, workflow, training_candidates, training_research)
    out = tmp_path / "reports" / "acceptance.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_acceptance_bundle.py",
            "--nightly-report",
            str(nightly),
            "--models-root",
            str(tmp_path / "models"),
            "--ml-base",
            str(tmp_path / "models" / "ml_signal_scorer"),
            "--skip-model-health",
            "--out",
            str(out),
            "--format",
            "md",
        ],
    )
    rc = module.main()
    text = capsys.readouterr().out
    assert rc == 0
    assert "acceptance bundle: warn" in text
    assert "nightly_report=" in text
    assert "bundle artifact ->" in text
    assert out.is_file()
    assert "# Multi-Agent Acceptance Bundle" in out.read_text(encoding="utf-8")
