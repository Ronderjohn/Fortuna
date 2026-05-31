"""Tests for nightly pre-flight helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fortuna.utils.nightly_preflight import (
    PreflightResult,
    find_stale_fortuna_streamlit_pids,
    kill_stale_fortuna_streamlit,
    recommend_rl_env,
    run_nightly_preflight,
)


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
