"""Tests for nightly pre-flight helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fortuna.config.settings import Settings
from fortuna.utils.nightly_preflight import (
    PreflightResult,
    build_lane_readiness,
    classify_data_blocker,
    find_stale_fortuna_streamlit_pids,
    kill_stale_fortuna_streamlit,
    lane_readiness_for_step,
    recommend_rl_env,
    resolve_nightly_abort,
    run_nightly_preflight,
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    payload = {
        "env": "test",
        "log_level": "WARNING",
        "data_cache_dir": tmp_path / "cache",
        "duckdb_path": tmp_path / "cache" / "fortuna.duckdb",
        "project_root": tmp_path,
        "agentic_enabled": True,
        "agentic_ml_scorer_enabled": True,
    }
    payload.update(overrides)
    return Settings(**payload)


def _execution_plan(**overrides) -> dict:
    payload = {
        "target": "all",
        "run_ml": True,
        "run_rl": True,
        "allow_ml_promotion": False,
        "allow_rl_promotion": False,
        "ml_symbol_count": 2,
        "rl_symbol_count": 2,
    }
    payload.update(overrides)
    return payload


def test_recommend_rl_env_low_ram_caps_parallelism() -> None:
    n, subproc, notes = recommend_rl_env(8, True, physical_ram=8.0)
    assert n == 2
    assert subproc is False
    assert notes


def test_recommend_rl_env_mid_ram_caps_at_four() -> None:
    n, subproc, notes = recommend_rl_env(8, True, physical_ram=16.0)
    assert n == 4
    assert subproc is True
    assert notes


def test_recommend_rl_env_high_ram_keeps_request() -> None:
    n, subproc, notes = recommend_rl_env(8, True, physical_ram=32.0)
    assert n == 8
    assert subproc is True
    assert notes == []


def test_find_stale_streamlit_pids_non_windows() -> None:
    with patch("fortuna.utils.nightly_preflight.sys.platform", "linux"):
        assert find_stale_fortuna_streamlit_pids(Path("/tmp/Fortuna")) == []


def test_kill_stale_streamlit_invokes_taskkill(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(r"C:\dev\Fortuna")
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight.find_stale_fortuna_streamlit_pids",
        lambda _root: [23388, 24284],
    )
    calls: list[list[str]] = []

    def _run(cmd, **kwargs):  # noqa: ANN001
        calls.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr("fortuna.utils.nightly_preflight.subprocess.run", _run)
    killed = kill_stale_fortuna_streamlit(repo)
    assert killed == [23388, 24284]
    assert calls == [
        ["taskkill", "/PID", "23388", "/T", "/F"],
        ["taskkill", "/PID", "24284", "/T", "/F"],
    ]


def test_run_preflight_applies_tune_and_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(r"C:\dev\Fortuna")
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight.physical_ram_gb",
        lambda: 16.0,
    )
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight.kill_stale_fortuna_streamlit",
        lambda _root: [23388],
    )
    result = run_nightly_preflight(
        repo,
        requested_n_envs=8,
        requested_use_subproc=True,
    )
    assert isinstance(result, PreflightResult)
    assert result.killed_pids == [23388]
    assert result.n_envs == 4
    assert result.use_subproc is True
    assert result.as_detail()["killed_count"] == 1


def test_run_preflight_can_skip_tune(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(r"C:\dev\Fortuna")
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight.physical_ram_gb",
        lambda: 8.0,
    )
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight.kill_stale_fortuna_streamlit",
        lambda _root: [],
    )
    result = run_nightly_preflight(
        repo,
        requested_n_envs=8,
        auto_tune_envs=False,
        kill_streamlit=False,
    )
    assert result.n_envs == 8
    assert result.killed_pids == []


def test_classify_data_blocker_detects_global_failures() -> None:
    assert (
        classify_data_blocker("NameResolutionError: Failed to resolve host")
        == "smartapi_unreachable"
    )
    assert classify_data_blocker("CERTIFICATE_VERIFY_FAILED") == "smartapi_ssl_blocked"
    assert classify_data_blocker("401 Unauthorized") == "smartapi_auth_failed"


def test_build_lane_readiness_blocks_smartapi_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: False,
    )
    settings = _settings(tmp_path)
    decisions, blockers, should_abort, readiness, scaling = build_lane_readiness(
        settings,
        execution_plan=_execution_plan(),
        data_source="smartapi",
        symbols=["RELIANCE.NS"],
    )
    assert "smartapi_credentials_missing" in blockers
    assert should_abort is True
    assert readiness is not None
    assert scaling is not None
    assert "basket_posture" in scaling
    by_lane = {row.lane: row for row in decisions}
    assert by_lane["backfill"].disposition == "block"
    assert by_lane["ml"].disposition == "block"
    assert by_lane["rl"].disposition == "block"
    assert by_lane["promotion"].disposition == "block"


def test_build_lane_readiness_ml_only_skips_rl_by_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: True,
    )
    settings = _settings(tmp_path)
    decisions, blockers, _, _, _ = build_lane_readiness(
        settings,
        execution_plan=_execution_plan(
            target="ml",
            run_ml=True,
            run_rl=False,
            allow_ml_promotion=True,
        ),
        data_source="cache",
        symbols=["RELIANCE.NS"],
    )
    assert blockers == []
    by_lane = {row.lane: row for row in decisions}
    assert by_lane["ml"].disposition == "run"
    assert by_lane["rl"].disposition == "skip"
    assert by_lane["rl"].skip_class == "policy"
    assert by_lane["promotion"].disposition == "run"


def test_build_lane_readiness_mixed_target_skips_promotion_by_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: True,
    )
    settings = _settings(tmp_path)
    decisions, _, _, _, _ = build_lane_readiness(
        settings,
        execution_plan=_execution_plan(target="all"),
        data_source="cache",
        symbols=["RELIANCE.NS"],
    )
    promo = next(row for row in decisions if row.lane == "promotion")
    assert promo.disposition == "skip"
    assert promo.skip_class == "policy"


def test_lane_readiness_for_step_includes_runtime_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: True,
    )
    settings = _settings(tmp_path)
    detail = lane_readiness_for_step(
        settings,
        execution_plan=_execution_plan(target="ml", run_rl=False, allow_ml_promotion=True),
        data_source="cache",
        symbols=["RELIANCE.NS"],
    )
    assert detail["lane_decisions"]
    assert detail["runtime_readiness"] is not None


def test_resolve_nightly_abort_when_all_requested_lanes_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: False,
    )
    settings = _settings(tmp_path)
    decisions, _, should_abort, _, _ = build_lane_readiness(
        settings,
        execution_plan=_execution_plan(),
        data_source="smartapi",
        symbols=["RELIANCE.NS"],
    )
    assert should_abort is True
    assert resolve_nightly_abort(decisions, _execution_plan()) is True


def test_build_lane_readiness_scaling_low_ram_basket_exceeds_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: True,
    )
    settings = _settings(tmp_path)
    symbols = [f"SYM{i}.NS" for i in range(12)]
    _, _, _, _, scaling = build_lane_readiness(
        settings,
        execution_plan=_execution_plan(),
        data_source="cache",
        symbols=symbols,
        physical_ram_gb=8.0,
    )
    assert scaling is not None
    assert scaling["basket_posture"] in {"small_only", "large_not_recommended"}
    assert scaling["within_recommended"] is False


def test_build_lane_readiness_scaling_mid_ram_within_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: True,
    )
    settings = _settings(tmp_path)
    symbols = [f"SYM{i}.NS" for i in range(10)]
    _, _, _, _, scaling = build_lane_readiness(
        settings,
        execution_plan=_execution_plan(),
        data_source="cache",
        symbols=symbols,
        physical_ram_gb=16.0,
    )
    assert scaling is not None
    assert scaling["basket_posture"] == "medium_ok"
    assert scaling["within_recommended"] is True


def test_build_lane_readiness_includes_scaling_in_lane_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: True,
    )
    settings = _settings(tmp_path)
    symbols = [f"SYM{i}.NS" for i in range(12)]
    detail = lane_readiness_for_step(
        settings,
        execution_plan=_execution_plan(),
        data_source="cache",
        symbols=symbols,
    )
    assert detail.get("scaling_posture") is not None
    assert detail["scaling_posture"]["basket_count"] == 12


def test_build_lane_readiness_enforce_cap_adds_global_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fortuna.utils.nightly_preflight._smartapi_configured",
        lambda: True,
    )
    settings = _settings(tmp_path, scaling_enforce_basket_cap=True)
    symbols = [f"SYM{i}.NS" for i in range(20)]
    _, blockers, _, _, _ = build_lane_readiness(
        settings,
        execution_plan=_execution_plan(),
        data_source="cache",
        symbols=symbols,
        physical_ram_gb=8.0,
    )
    assert "basket_exceeds_compute_budget" in blockers
