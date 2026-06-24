"""Tests for typed compute budget and scaling posture helpers."""

from __future__ import annotations

from pathlib import Path

from fortuna.app.scaling_posture import (
    apply_nightly_report_retention,
    build_compute_budget_summary,
    build_scaling_posture_summary,
)
from fortuna.config.settings import Settings


def _settings(tmp_path: Path, **overrides) -> Settings:
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    payload = {
        "env": "test",
        "log_level": "WARNING",
        "data_cache_dir": tmp_path / "cache",
        "duckdb_path": tmp_path / "cache" / "fortuna.duckdb",
        "project_root": tmp_path,
    }
    payload.update(overrides)
    return Settings(**payload)


def test_build_compute_budget_summary_low_ram_tier(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    summary = build_compute_budget_summary(settings, physical_ram_gb=8.0)
    assert summary.machine_class == "low"
    assert summary.recommended_max_symbols == 8
    assert summary.rl_n_envs == 2
    assert summary.rl_use_subproc is False


def test_build_compute_budget_summary_mid_ram_tier(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    summary = build_compute_budget_summary(settings, physical_ram_gb=16.0)
    assert summary.machine_class == "mid"
    assert summary.recommended_max_symbols == 15
    assert summary.rl_n_envs == 4


def test_build_compute_budget_summary_high_ram_tier(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    summary = build_compute_budget_summary(
        settings,
        physical_ram_gb=32.0,
        n_envs=8,
    )
    assert summary.machine_class == "high"
    assert summary.recommended_max_symbols == 15
    assert summary.rl_n_envs == 8


def test_build_scaling_posture_summary_low_ram_large_basket(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    summary = build_scaling_posture_summary(
        settings,
        basket_count=12,
        physical_ram_gb=8.0,
    )
    assert summary.basket_posture == "small_only"
    assert summary.within_recommended is False
    warning_names = {row.name for row in summary.warnings}
    assert "scaling_basket_exceeds_budget" in warning_names


def test_build_scaling_posture_summary_mid_ram_medium_basket(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    summary = build_scaling_posture_summary(
        settings,
        basket_count=10,
        physical_ram_gb=16.0,
    )
    assert summary.basket_posture == "medium_ok"
    assert summary.within_recommended is True


def test_apply_nightly_report_retention_disabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path, nightly_report_retention_enabled=False)
    report_dir = tmp_path / "reports" / "nightly"
    report_dir.mkdir(parents=True)
    for idx in range(5):
        (report_dir / f"2026060{idx}.json").write_text("{}", encoding="utf-8")
    result = apply_nightly_report_retention(report_dir, settings)
    assert result.enabled is False
    assert result.pruned_count == 0
    assert len(list(report_dir.glob("*.json"))) == 5


def test_apply_nightly_report_retention_prunes_oldest(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        nightly_report_retention_enabled=True,
        nightly_report_retention_count=30,
    )
    report_dir = tmp_path / "reports" / "nightly"
    report_dir.mkdir(parents=True)
    paths = []
    for idx in range(35):
        path = report_dir / f"202601{idx:02d}.json"
        path.write_text("{}", encoding="utf-8")
        md = path.with_suffix(".md")
        md.write_text("# report", encoding="utf-8")
        paths.append(path)
    for offset, path in enumerate(paths):
        path.touch()
        path.with_suffix(".md").touch()
        import os
        import time

        ts = time.time() - (len(paths) - offset) * 60
        os.utime(path, (ts, ts))
        os.utime(path.with_suffix(".md"), (ts, ts))

    result = apply_nightly_report_retention(report_dir, settings)
    assert result.enabled is True
    assert result.pruned_count == 5
    assert result.retained_count == 30
    assert len(list(report_dir.glob("*.json"))) == 30
