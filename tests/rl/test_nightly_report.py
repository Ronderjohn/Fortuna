from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from fortuna.ml.types import ClassificationMetrics, ScorerMetadata
from fortuna.rl.training.checkpoint import OOSMetricsSummary


def _load_nightly_train_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "nightly_train.py"
    spec = importlib.util.spec_from_file_location("fortuna_nightly_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_ml_artifact(
    artifact_dir: Path,
    *,
    run_id: str,
    symbol: str,
    timeframe: str = "5m",
    precision: float = 0.7,
    roc_auc: float = 0.72,
    accuracy: float = 0.68,
    n_samples: int = 24,
    verdict_passed: bool = True,
    advisory_ready: bool = True,
):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model.joblib").write_bytes(b"\x00")
    meta = ScorerMetadata(
        run_id=run_id,
        symbol=symbol,
        timeframe=timeframe,
        verdict_passed=verdict_passed,
        advisory_ready=advisory_ready,
        oos_metrics=ClassificationMetrics(
            precision=precision,
            roc_auc=roc_auc,
            accuracy=accuracy,
            n_samples=n_samples,
        ),
    )
    (artifact_dir / "metadata.json").write_text(
        json.dumps(meta.to_dict(), indent=2),
        encoding="utf-8",
    )


def test_write_report_includes_workflow_snapshot_section(tmp_path):
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=600.0,
        basket=["RELIANCE.NS"],
        overall_status="ok",
    )
    report.add(
        nightly_train.StepResult(
            name="workflow_snapshot",
            status="ok",
            started_at="2026-06-03T01:00:01",
            ended_at="2026-06-03T01:00:01",
            duration_s=0.0,
            detail={
                "path": "reports/nightly/workflow_snapshot.json",
                "team_role_count": 6,
                "team_ok_role_count": 5,
                "team_nightly_report_count": 2,
                "team_nightly_enabled_reports": 2,
                "team_nightly_aligned_reports": 1,
                "team_nightly_latest_target_mix": "both=1, rl=1",
                "team_nightly_latest_refreshed_target_mix": "both=1",
                "team_nightly_recommended_target": "all",
                "team_nightly_recommended_force_refresh": True,
                "team_research_headline": "4 ML/RL research rows prepared",
                "team_research_selection_policy": "diversified",
                "team_research_refresh_target": "rl",
                "team_research_discovery_recommended_target": "ml",
                "team_research_effective_target": "all",
                "team_research_target_mismatch": True,
                "team_research_refresh_requested": True,
                "team_research_refreshed_count": 2,
                "team_research_ml_count": 3,
                "team_research_rl_count": 2,
                "universe_count": 10,
                "shortlist_count": 5,
                "briefing_candidates": 3,
                "research_count": 4,
                "research_ml_count": 3,
                "research_rl_count": 2,
                "research_selection_policy": "diversified",
                "research_refresh_target": "rl",
                "research_discovery_recommended_refresh_target": "ml",
                "research_effective_refresh_target": "all",
                "research_target_mismatch": True,
                "research_refresh_requested": True,
                "research_refreshed_count": 2,
                "allocation_research_alignment_enabled": True,
                "allocation_research_target_mix": "both=1, rl=1",
                "allocation_refreshed_research_target_mix": "both=1",
                "ml_count": 4,
                "rl_count": 2,
            },
        )
    )
    out = nightly_train.write_report(report, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## Workflow Snapshot" in text
    assert "workflow_snapshot.json" in text
    assert "universe=10 shortlist=5 briefing_candidates=3 ml=4 rl=2" in text
    assert "roles=5/6 nightly=1/2 target=all force_refresh=1" in text
    assert "Team nightly target mix: `both=1, rl=1`" in text
    assert "Team refreshed nightly target mix: `both=1`" in text
    assert (
        "Team research posture: 4 ML/RL research rows prepared "
        "(ml=3 rl=2 policy=diversified target=rl discovery_target=ml "
        "effective_target=all refreshed=2 mismatch=1)"
    ) in text
    assert (
        "count=4 ml=3 rl=2 selection_policy=diversified refresh_target=rl "
        "discovery_target=ml effective_target=all refreshed=2 mismatch=1"
    ) in text
    assert "Allocation research alignment: `both=1, rl=1`" in text
    assert "Refreshed allocation research alignment: `both=1`" in text


def test_build_training_execution_plan_prefers_effective_ml_target():
    nightly_train = _load_nightly_train_module()
    plan = nightly_train._build_training_execution_plan(  # noqa: SLF001
        ["RELIANCE.NS", "RELIANCE.FUT", "TCS.NS"],
        {"target": "all", "requested_target": "all"},
        {"effective_refresh_target": "ml"},
    )

    assert plan["target"] == "ml"
    assert plan["run_ml"] is True
    assert plan["run_rl"] is False
    assert plan["allow_rl_promotion"] is False
    assert plan["cash_symbol_count"] == 2
    assert plan["ml_symbol_count"] == 2


def test_build_training_execution_plan_falls_back_to_rl_basket_target():
    nightly_train = _load_nightly_train_module()
    plan = nightly_train._build_training_execution_plan(  # noqa: SLF001
        ["RELIANCE.NS", "RELIANCE.FUT"],
        {"target": "rl", "requested_target": "all"},
        None,
    )

    assert plan["target"] == "rl"
    assert plan["run_ml"] is False
    assert plan["run_rl"] is True
    assert plan["allow_ml_promotion"] is False
    assert plan["allow_rl_promotion"] is True
    assert plan["ml_symbol_count"] == 0


def test_build_training_execution_plan_runs_both_for_mixed_target():
    nightly_train = _load_nightly_train_module()
    plan = nightly_train._build_training_execution_plan(  # noqa: SLF001
        ["RELIANCE.NS", "TCS.NS", "RELIANCE.FUT"],
        {"target": "all", "requested_target": "all"},
        {
            "effective_refresh_target": "all",
            "discovery_recommended_refresh_target": "all",
            "selection_source": "training_research",
        },
    )

    assert plan["target"] == "all"
    assert plan["run_ml"] is True
    assert plan["run_rl"] is True
    assert plan["allow_ml_promotion"] is False
    assert plan["allow_rl_promotion"] is False
    assert plan["cash_symbol_count"] == 2
    assert plan["ml_symbol_count"] == 2
    assert plan["rl_symbol_count"] == 3
    assert plan["selection_source"] == "training_research"


def test_train_ml_scorer_symbol_invokes_ml_cli(monkeypatch, tmp_path):
    nightly_train = _load_nightly_train_module()
    captured = {}

    class DummyProc:
        returncode = 0
        stdout = "[ML] ok"
        stderr = ""

    def _run(cmd, cwd, capture_output, text, check):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        return DummyProc()

    monkeypatch.setattr(nightly_train.subprocess, "run", _run)

    meta = nightly_train.train_ml_scorer_symbol(
        "RELIANCE.NS",
        timeframe="5m",
        days=120,
        repo_root=tmp_path,
        learning_dir="logs/agentic",
        output_dir=tmp_path / "models" / "ml_signal_scorer" / "validated",
        model_kind="hgb",
        horizon_bars=5,
        eval_fraction=0.3,
        seed=99,
    )

    assert meta["symbol"] == "RELIANCE.NS"
    assert meta["model_kind"] == "hgb"
    assert captured["cwd"] == tmp_path
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False
    assert captured["cmd"] == [
        "uv",
        "run",
        "--group",
        "ml",
        "python",
        "scripts/train_ml_signal_scorer.py",
        "--symbol",
        "RELIANCE.NS",
        "--timeframe",
        "5m",
        "--learning-dir",
        "logs/agentic",
        "--output-dir",
        str(tmp_path / "models" / "ml_signal_scorer" / "validated"),
        "--days",
        "120",
        "--horizon-bars",
        "5",
        "--eval-fraction",
        "0.3",
        "--seed",
        "99",
        "--model-kind",
        "hgb",
    ]


def test_write_report_includes_training_candidates_section(tmp_path):
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=600.0,
        basket=["RELIANCE.NS"],
        overall_status="ok",
    )
    report.add(
        nightly_train.StepResult(
            name="training_candidates",
            status="ok",
            started_at="2026-06-03T01:00:01",
            ended_at="2026-06-03T01:00:01",
            duration_s=0.0,
            detail={
                "path": "reports/nightly/training_candidates.json",
                "count": 6,
                "ml_count": 5,
                "rl_count": 3,
                "selection_policy": "diversified",
            },
        )
    )
    out = nightly_train.write_report(report, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## Training Candidates" in text
    assert "training_candidates.json" in text
    assert "count=6 ml=5 rl=3 selection_policy=diversified" in text


def test_write_report_includes_training_research_section(tmp_path):
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=600.0,
        basket=["RELIANCE.NS"],
        overall_status="ok",
    )
    report.add(
        nightly_train.StepResult(
            name="training_research",
            status="ok",
            started_at="2026-06-03T01:00:01",
            ended_at="2026-06-03T01:00:01",
            duration_s=0.0,
            detail={
                "path": "reports/nightly/training_research_plan.json",
                "count": 4,
                "ml_count": 3,
                "rl_count": 2,
                "selection_policy": "diversified",
                "refresh_target": "all",
                "discovery_recommended_refresh_target": "rl",
                "effective_refresh_target": "rl",
                "refresh_requested": True,
                "refreshed_count": 2,
                "discovery_preferred_count": 2,
                "discovery_regime_mix": "trending=2, volatile=1",
                "discovery_summary": (
                    "preferred=2 | liquidity=TCS.NS,SBIN.NS | "
                    "activity=SBIN.NS,TCS.NS | regimes=trending=2, volatile=1"
                ),
            },
        )
    )
    out = nightly_train.write_report(report, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## Training Research Plan" in text
    assert "training_research_plan.json" in text
    assert (
        "count=4 ml=3 rl=2 selection_policy=diversified "
        "refresh_target=all discovery_target=rl effective_target=rl refreshed=2"
    ) in text
    assert (
        "preferred=2 regimes=trending=2, volatile=1 "
        "summary=preferred=2 | liquidity=TCS.NS,SBIN.NS | "
        "activity=SBIN.NS,TCS.NS | regimes=trending=2, volatile=1"
    ) in text


def test_write_report_includes_promotion_review_section(tmp_path):
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=600.0,
        basket=["RELIANCE.NS"],
        overall_status="ok",
    )
    report.add(
        nightly_train.StepResult(
            name="promotion_reviews",
            status="ok",
            started_at="2026-06-03T01:09:00",
            ended_at="2026-06-03T01:09:01",
            duration_s=1.0,
            detail={
                "count": 1,
                "format": "md",
                "model_kinds": ["rl_policy"],
                "paths": ["reports/nightly/promotions/RELIANCE_NS__run_ready.md"],
            },
        )
    )

    out = nightly_train.write_report(report, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## Promotion Reviews" in text
    assert "Model kinds: `rl_policy`" in text
    assert "RELIANCE_NS__run_ready.md" in text


def test_promote_best_per_symbol_threads_workflow_snapshot(tmp_path):
    nightly_train = _load_nightly_train_module()
    run_dir = tmp_path / "validated" / "run_ready"
    run_dir.mkdir(parents=True)
    (run_dir / "policy.zip").write_bytes(b"\x00")
    (run_dir / "normalizer.json").write_text("{}", encoding="utf-8")
    nightly_train.PolicyCheckpoint(
        run_id="run_ready",
        symbol="RELIANCE.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=10, sharpe_ratio=1.2),
        verdict_passed=True,
        verdict_score=0.8,
        advisory_ready=True,
    ).write(run_dir / "metadata.json")
    workflow = tmp_path / "reports" / "workflow_snapshot.json"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("{}", encoding="utf-8")

    result = nightly_train.promote_best_per_symbol(
        tmp_path,
        workflow_snapshot_path=str(workflow),
    )

    assert result["count"] == 1
    assert result["workflow_snapshot_path"] == str(workflow)
    ptr = tmp_path / "live" / "by_symbol" / "RELIANCE_NS" / "live.json"
    payload = json.loads(ptr.read_text(encoding="utf-8"))
    assert payload["workflow_snapshot_path"] == str(workflow)


def test_promote_best_ml_scorer_threads_workflow_snapshot(tmp_path):
    nightly_train = _load_nightly_train_module()
    models_root = tmp_path / "models"
    ml_base = models_root / "ml_signal_scorer"
    _write_ml_artifact(
        ml_base / "validated" / "ml_rel",
        run_id="ml_rel",
        symbol="RELIANCE.NS",
        precision=0.69,
        roc_auc=0.73,
    )
    _write_ml_artifact(
        ml_base / "validated" / "ml_tcs",
        run_id="ml_tcs",
        symbol="TCS.NS",
        precision=0.66,
        roc_auc=0.68,
    )
    workflow = tmp_path / "reports" / "workflow_snapshot.json"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("{}", encoding="utf-8")

    result = nightly_train.promote_best_ml_scorer(
        models_root,
        ml_base,
        eligible_symbols=["RELIANCE.NS"],
        timeframe="5m",
        workflow_snapshot_path=str(workflow),
    )

    assert result["count"] == 1
    assert result["model_kind"] == "ml_scorer"
    assert result["workflow_snapshot_path"] == str(workflow)
    ptr = ml_base / "live" / "live.json"
    payload = json.loads(ptr.read_text(encoding="utf-8"))
    assert payload["workflow_snapshot_path"] == str(workflow)
    assert payload["run_id"] == "ml_rel"


def test_emit_promotion_review_artifacts_writes_markdown(tmp_path):
    nightly_train = _load_nightly_train_module()
    run_dir = tmp_path / "models" / "validated" / "run_ready"
    run_dir.mkdir(parents=True)
    (run_dir / "policy.zip").write_bytes(b"\x00")
    (run_dir / "normalizer.json").write_text("{}", encoding="utf-8")
    nightly_train.PolicyCheckpoint(
        run_id="run_ready",
        symbol="RELIANCE.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=10, sharpe_ratio=1.2),
        verdict_passed=True,
        verdict_score=0.8,
        verdict_reasons=["strong_oos"],
        advisory_ready=True,
    ).write(run_dir / "metadata.json")
    workflow = tmp_path / "reports" / "workflow_snapshot.json"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        json.dumps(
            {
                "source": "auto",
                "timeframe": "5m",
                "lookback_days": 10,
                "universe": {"metrics": {"count": 12}},
                "shortlist": {"metrics": {"count": 6}},
                "briefing": {"metrics": {"candidates": 3}},
                "candidates": {"metrics": {"ml_count": 4, "rl_count": 2}},
            }
        ),
        encoding="utf-8",
    )
    promote_result = nightly_train.promote_best_per_symbol(
        tmp_path / "models",
        workflow_snapshot_path=str(workflow),
    )

    meta = nightly_train.emit_promotion_review_artifacts(
        promote_result,
        models_root=tmp_path / "models",
        report_dir=tmp_path / "reports" / "nightly",
        fmt="md",
    )

    assert meta["count"] == 1
    written = Path(meta["paths"][0])
    assert written.is_file()
    assert "# Promotion Review" in written.read_text(encoding="utf-8")


def test_emit_promotion_review_artifacts_supports_ml(tmp_path):
    nightly_train = _load_nightly_train_module()
    models_root = tmp_path / "models"
    ml_base = models_root / "ml_signal_scorer"
    _write_ml_artifact(
        ml_base / "validated" / "ml_rel",
        run_id="ml_rel",
        symbol="RELIANCE.NS",
        precision=0.7,
        roc_auc=0.75,
    )
    workflow = tmp_path / "reports" / "workflow_snapshot.json"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        json.dumps(
            {
                "source": "auto",
                "timeframe": "5m",
                "lookback_days": 10,
                "universe": {"metrics": {"count": 12}},
                "shortlist": {"metrics": {"count": 6}},
                "briefing": {"metrics": {"candidates": 3}},
                "candidates": {"metrics": {"ml_count": 4, "rl_count": 2}},
            }
        ),
        encoding="utf-8",
    )
    promote_result = nightly_train.promote_best_ml_scorer(
        models_root,
        ml_base,
        eligible_symbols=["RELIANCE.NS"],
        timeframe="5m",
        workflow_snapshot_path=str(workflow),
    )

    meta = nightly_train.emit_promotion_review_artifacts(
        promote_result,
        models_root=models_root,
        report_dir=tmp_path / "reports" / "nightly",
        fmt="md",
    )

    assert meta["count"] == 1
    assert meta["model_kind"] == "ml_scorer"
    written = Path(meta["paths"][0])
    assert written.is_file()
    text = written.read_text(encoding="utf-8")
    assert "# Promotion Review" in text
    assert "Promotion Review — ml_scorer" in text


def test_merge_promotion_results_builds_mixed_bundle():
    nightly_train = _load_nightly_train_module()
    merged = nightly_train._merge_promotion_results(  # noqa: SLF001
        {
            "model_kind": "ml_scorer",
            "workflow_snapshot_path": "reports/workflow_snapshot.json",
            "promoted": [
                {
                    "symbol": "RELIANCE.NS",
                    "run_id": "ml_rel",
                    "pointer": "models/ml_signal_scorer/live/live.json",
                }
            ],
        },
        {
            "model_kind": "rl_policy",
            "workflow_snapshot_path": "reports/workflow_snapshot.json",
            "promoted": [
                {
                    "symbol": "RELIANCE.NS",
                    "run_id": "rl_rel",
                    "pointer": "models/live/by_symbol/RELIANCE_NS/live.json",
                }
            ],
        },
    )

    assert merged["count"] == 2
    assert merged["model_kind"] == "mixed"
    assert merged["model_kinds"] == ["ml_scorer", "rl_policy"]
    assert merged["counts_by_kind"] == {"ml_scorer": 1, "rl_policy": 1}
    promoted = merged["promoted"]
    assert promoted[0]["model_kind"] == "ml_scorer"
    assert promoted[1]["model_kind"] == "rl_policy"


def test_emit_promotion_review_artifacts_supports_mixed_bundle(tmp_path):
    nightly_train = _load_nightly_train_module()
    models_root = tmp_path / "models"
    ml_base = models_root / "ml_signal_scorer"
    _write_ml_artifact(
        ml_base / "validated" / "ml_rel",
        run_id="ml_rel",
        symbol="RELIANCE.NS",
        precision=0.7,
        roc_auc=0.75,
    )
    run_dir = models_root / "validated" / "rl_rel"
    run_dir.mkdir(parents=True)
    (run_dir / "policy.zip").write_bytes(b"\x00")
    (run_dir / "normalizer.json").write_text("{}", encoding="utf-8")
    nightly_train.PolicyCheckpoint(
        run_id="rl_rel",
        symbol="RELIANCE.NS",
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=10, sharpe_ratio=1.2),
        verdict_passed=True,
        verdict_score=0.8,
        advisory_ready=True,
    ).write(run_dir / "metadata.json")
    workflow = tmp_path / "reports" / "workflow_snapshot.json"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        json.dumps(
            {
                "source": "auto",
                "timeframe": "5m",
                "lookback_days": 10,
                "universe": {"metrics": {"count": 12}},
                "shortlist": {"metrics": {"count": 6}},
                "briefing": {"metrics": {"candidates": 3}},
                "candidates": {"metrics": {"ml_count": 4, "rl_count": 2}},
            }
        ),
        encoding="utf-8",
    )
    ml_promote = nightly_train.promote_best_ml_scorer(
        models_root,
        ml_base,
        eligible_symbols=["RELIANCE.NS"],
        timeframe="5m",
        workflow_snapshot_path=str(workflow),
    )
    rl_promote = nightly_train.promote_best_per_symbol(
        models_root,
        workflow_snapshot_path=str(workflow),
    )
    promote_result = nightly_train._merge_promotion_results(  # noqa: SLF001
        ml_promote,
        rl_promote,
    )

    meta = nightly_train.emit_promotion_review_artifacts(
        promote_result,
        models_root=models_root,
        report_dir=tmp_path / "reports" / "nightly",
        fmt="md",
    )

    assert meta["count"] == 2
    assert meta["model_kind"] == "mixed"
    assert meta["model_kinds"] == ["ml_scorer", "rl_policy"]
    texts = [Path(path).read_text(encoding="utf-8") for path in meta["paths"]]
    assert any("Promotion Review — ml_scorer" in text for text in texts)
    assert any("Promotion Review — rl_policy" in text for text in texts)


def test_finalize_overall_status_blocked_when_lanes_skipped_for_prerequisites():
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=60.0,
        basket=["RELIANCE.NS"],
    )
    report.add(
        nightly_train.StepResult(
            name="lane_readiness",
            status="ok",
            started_at="2026-06-03T01:00:01",
            ended_at="2026-06-03T01:00:01",
            duration_s=0.0,
            detail={
                "global_blockers": ["smartapi_credentials_missing"],
                "lane_decisions": [],
            },
        )
    )
    report.add(
        nightly_train.StepResult(
            name="backfill",
            status="skip",
            started_at="2026-06-03T01:00:02",
            ended_at="2026-06-03T01:00:02",
            duration_s=0.0,
            detail={"skip_class": "missing_prerequisites"},
        )
    )
    assert nightly_train.finalize_overall_status(report) == "blocked"


def test_finalize_overall_status_ok_when_training_completed():
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=600.0,
        basket=["RELIANCE.NS"],
    )
    report.add(
        nightly_train.StepResult(
            name="backfill:RELIANCE.NS",
            status="ok",
            started_at="2026-06-03T01:00:01",
            ended_at="2026-06-03T01:00:02",
            duration_s=1.0,
            detail={"rows": 100},
        )
    )
    assert nightly_train.finalize_overall_status(report) == "ok"


def test_finalize_overall_status_skipped_for_policy_only():
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=10.0,
        basket=["RELIANCE.NS"],
    )
    report.add(
        nightly_train.StepResult(
            name="ml_train",
            status="skip",
            started_at="2026-06-03T01:00:01",
            ended_at="2026-06-03T01:00:01",
            duration_s=0.0,
            detail={"skip_class": "policy", "reason": "target_prefers_rl"},
        )
    )
    assert nightly_train.finalize_overall_status(report) == "skipped"


def test_should_abort_after_lane_readiness_reads_typed_flag():
    nightly_train = _load_nightly_train_module()
    assert nightly_train._should_abort_after_lane_readiness({"should_abort": True}) is True
    assert nightly_train._should_abort_after_lane_readiness({"should_abort": False}) is False
    assert nightly_train._should_abort_after_lane_readiness(None) is False


def test_write_report_includes_lane_readiness_section(tmp_path):
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=60.0,
        basket=["RELIANCE.NS"],
        overall_status="blocked",
    )
    report.add(
        nightly_train.StepResult(
            name="lane_readiness",
            status="ok",
            started_at="2026-06-03T01:00:01",
            ended_at="2026-06-03T01:00:01",
            duration_s=0.0,
            detail={
                "global_blockers": ["smartapi_credentials_missing"],
                "lane_decisions": [
                    {
                        "lane": "backfill",
                        "disposition": "block",
                        "skip_class": "missing_prerequisites",
                        "detail": "credentials missing",
                    }
                ],
            },
        )
    )
    md_path = nightly_train.write_report(report, tmp_path / "reports" / "nightly")
    text = md_path.read_text(encoding="utf-8")
    assert "## Lane readiness" in text
    assert "smartapi_credentials_missing" in text


def test_write_report_includes_scaling_posture_section(tmp_path):
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=60.0,
        basket=["RELIANCE.NS", "TCS.NS"],
        overall_status="ok",
    )
    report.add(
        nightly_train.StepResult(
            name="scaling_posture",
            status="ok",
            started_at="2026-06-03T01:00:01",
            ended_at="2026-06-03T01:00:01",
            duration_s=0.0,
            detail={
                "basket_posture": "medium_ok",
                "basket_count": 2,
                "recommended_max_symbols": 15,
                "within_recommended": True,
                "compute": {
                    "machine_class": "mid",
                    "physical_ram_gb": 16.0,
                    "rl_n_envs": 4,
                },
                "warnings": [],
            },
        )
    )
    md_path = nightly_train.write_report(report, tmp_path / "reports" / "nightly")
    text = md_path.read_text(encoding="utf-8")
    assert "## Scaling posture" in text
    assert "medium_ok" in text
    assert "recommended max" in text.lower() or "Recommended max" in text


def test_write_report_includes_artifact_retention_section(tmp_path):
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=60.0,
        basket=["RELIANCE.NS"],
        overall_status="ok",
    )
    report.add(
        nightly_train.StepResult(
            name="artifact_retention",
            status="ok",
            started_at="2026-06-03T01:10:01",
            ended_at="2026-06-03T01:10:01",
            duration_s=0.0,
            detail={
                "enabled": True,
                "policy": "keep_newest_30_json_reports",
                "retained_count": 30,
                "pruned_count": 5,
                "detail": "retained 30 pruned 5 from 35 total",
            },
        )
    )

    md_path = nightly_train.write_report(report, tmp_path / "reports" / "nightly")
    text = md_path.read_text(encoding="utf-8")
    assert "## Artifact retention" in text
    assert "keep_newest_30_json_reports" in text
    assert "Pruned count: `5`" in text
