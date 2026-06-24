from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "run_nightly_acceptance_dry_run.py"
    spec = importlib.util.spec_from_file_location("fortuna_nightly_acceptance_dry_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_nightly_acceptance_dry_run_cli_exports_bundle(tmp_path, monkeypatch, capsys):
    module = _load_script()
    out_dir = tmp_path / "dry_run"
    bundle_out = out_dir / "acceptance_bundle.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nightly_acceptance_dry_run.py",
            "--out-dir",
            str(out_dir),
            "--bundle-out",
            str(bundle_out),
        ],
    )
    rc = module.main()
    text = capsys.readouterr().out
    assert rc == 0
    assert "acceptance bundle: pass" in text
    assert "nightly_report=" in text
    assert "team_follow_up=discovery=rl research=rl execution=rl" in text
    assert "refresh_context=effective_target=rl" in text
    assert bundle_out.is_file()


def test_run_nightly_acceptance_dry_run_emit_replay_mode(tmp_path, monkeypatch, capsys):
    module = _load_script()
    out_dir = tmp_path / "emit_replay"
    bundle_out = out_dir / "acceptance_bundle.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nightly_acceptance_dry_run.py",
            "--mode",
            "emit-replay",
            "--out-dir",
            str(out_dir),
            "--bundle-out",
            str(bundle_out),
        ],
    )
    rc = module.main()
    text = capsys.readouterr().out
    assert rc == 0
    assert "acceptance bundle: pass" in text
    assert "replay_linkage=status=linked" in text
    assert bundle_out.is_file()


def test_run_nightly_acceptance_dry_run_verify_mode(tmp_path, monkeypatch, capsys):
    module = _load_script()
    out_dir = tmp_path / "dry_run"
    bundle_out = out_dir / "acceptance_bundle.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nightly_acceptance_dry_run.py",
            "--out-dir",
            str(out_dir),
            "--bundle-out",
            str(bundle_out),
        ],
    )
    assert module.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nightly_acceptance_dry_run.py",
            "--mode",
            "verify",
            "--out-dir",
            str(out_dir),
            "--report-dir",
            str(out_dir / "reports" / "nightly"),
            "--bundle-out",
            str(bundle_out),
        ],
    )
    rc = module.main()
    text = capsys.readouterr().out
    assert rc == 0
    assert "verify_report_dir=" in text
    assert "replay_linkage=status=linked" in text


def test_run_nightly_acceptance_dry_run_emit_replay_reports_degraded_linkage(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_script()
    out_dir = tmp_path / "emit_replay"
    bundle_out = out_dir / "acceptance_bundle.md"

    def _no_review(*args, **kwargs):
        return None

    monkeypatch.setattr("fortuna.app.nightly_acceptance.build_promotion_review", _no_review)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nightly_acceptance_dry_run.py",
            "--mode",
            "emit-replay",
            "--out-dir",
            str(out_dir),
            "--bundle-out",
            str(bundle_out),
        ],
    )
    rc = module.main()
    text = capsys.readouterr().out
    assert rc == 0
    assert "acceptance bundle: warn" in text
    assert "replay_linkage=status=partial" in text
    assert "missing=promotion_review" in text
