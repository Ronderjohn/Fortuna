"""Tests for per-symbol checkpoint resolution."""

from __future__ import annotations

from fortuna.rl.inference.signal_generator import (
    resolve_checkpoint_for_symbol,
    resolve_live_checkpoint_dir_for_symbol,
)


def test_resolve_per_symbol_finds_by_symbol_dir(tmp_path):
    sym_dir = tmp_path / "live" / "by_symbol" / "RELIANCE_NS"
    sym_dir.mkdir(parents=True)
    (sym_dir / "policy.zip").write_bytes(b"\x00")

    resolved = resolve_live_checkpoint_dir_for_symbol("RELIANCE.NS", tmp_path)
    assert resolved == sym_dir


def test_resolve_checkpoint_no_global_fallback_by_default(tmp_path):
    global_live = tmp_path / "live"
    global_live.mkdir(parents=True)
    (global_live / "policy.zip").write_bytes(b"\x00")

    resolved = resolve_checkpoint_for_symbol("CROMPTON.NS", tmp_path)
    assert resolved is None


def test_resolve_checkpoint_global_fallback_when_enabled(tmp_path):
    global_live = tmp_path / "live"
    global_live.mkdir(parents=True)
    (global_live / "policy.zip").write_bytes(b"\x00")

    resolved = resolve_checkpoint_for_symbol(
        "CROMPTON.NS",
        tmp_path,
        allow_global_fallback=True,
    )
    assert resolved == global_live
