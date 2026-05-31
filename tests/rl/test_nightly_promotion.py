from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fortuna.rl.training.checkpoint import OOSMetricsSummary, PolicyCheckpoint


def _load_promote_helper():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "nightly_train.py"
    spec = importlib.util.spec_from_file_location("nightly_train_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.promote_best_per_symbol


def _write_checkpoint(
    root: Path,
    bucket: str,
    run_id: str,
    *,
    symbol: str = "RELIANCE.NS",
    verdict_score: float,
    verdict_passed: bool,
    advisory_ready: bool,
) -> Path:
    cp_dir = root / bucket / run_id
    cp_dir.mkdir(parents=True)
    (cp_dir / "policy.zip").write_bytes(b"\x00")
    (cp_dir / "normalizer.json").write_text("{}", encoding="utf-8")
    PolicyCheckpoint(
        run_id=run_id,
        symbol=symbol,
        timeframe="5m",
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=1),
        verdict_passed=verdict_passed,
        verdict_score=verdict_score,
        advisory_ready=advisory_ready,
    ).write(cp_dir / "metadata.json")
    return cp_dir


def test_promote_best_per_symbol_prefers_best_advisory_ready_checkpoint(tmp_path):
    promote_best_per_symbol = _load_promote_helper()
    _write_checkpoint(
        tmp_path,
        "validated",
        "run_ready",
        verdict_score=0.8,
        verdict_passed=True,
        advisory_ready=True,
    )
    _write_checkpoint(
        tmp_path,
        "rejected",
        "run_rejected",
        verdict_score=0.95,
        verdict_passed=False,
        advisory_ready=False,
    )

    result = promote_best_per_symbol(tmp_path)

    assert result["count"] == 1
    assert result["promoted"][0]["run_id"] == "run_ready"
    ptr = tmp_path / "live" / "by_symbol" / "RELIANCE_NS" / "live.json"
    assert ptr.is_file()
    assert not (tmp_path / "live" / "by_symbol" / "RELIANCE_NS" / "metadata.json").exists()
